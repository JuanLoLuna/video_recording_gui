"""Live "has a camera moved?" check against the per-camera reference boards (plan decision 11).

Pure logic plus one blocking measurement function meant for a worker thread;
no Qt. The main window runs measure_references() every few seconds (also while
recording) and feeds the results through MoveHysteresis, whose output is the
`moved` map assess_session() takes.

A reference board that is not visible (hands in front of it) gives no answer:
the last known state stands. A single check is never trusted on its own:
"moved" and "back in place" both need CONFIRMATIONS consecutive results.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Mapping

import numpy as np

from backend import calibration as cal

CHECK_INTERVAL_S = 5.0
FRAMES_PER_CHECK = 5
FRAME_SPACING_S = 0.1
CONFIRMATIONS = 2


@dataclass(frozen=True)
class ReferenceTarget:
    """What one camera's live check needs, taken from the current setup and intrinsics."""
    serial: str
    board: cal.BoardConfig
    R: list
    t: list
    rms_px: float
    K: np.ndarray
    D: np.ndarray


def targets_from(setup, intrinsics_by_serial: Mapping) -> list[ReferenceTarget]:
    """One target per camera that has a reference saved in `setup` and current intrinsics.

    `intrinsics_by_serial`: serial -> IntrinsicsRecord (or None). A camera whose
    intrinsics changed since the setup is skipped: assess_session already
    reports that setup as stale, and its saved pose no longer applies.
    """
    out = []
    if setup is None:
        return out
    for cam in setup.cameras:
        ref = cam.get("reference")
        rec = intrinsics_by_serial.get(cam["serial"])
        if not ref or rec is None or rec.id != cam.get("intrinsics_id"):
            continue
        try:
            out.append(ReferenceTarget(
                serial=cam["serial"], board=cal.BoardConfig.from_dict(ref["board"]),
                R=ref["R"], t=ref["t"], rms_px=float(ref["rms_px"]),
                K=np.asarray(rec.K, dtype=float), D=np.asarray(rec.D, dtype=float)))
        except (KeyError, TypeError, ValueError):
            continue  # a damaged reference entry: no live check for that camera
    return out


def measure_references(targets, grab_frame: Callable[[str], np.ndarray | None],
                       frames: int = FRAMES_PER_CHECK, spacing_s: float = FRAME_SPACING_S,
                       sleep: Callable[[float], None] = time.sleep) -> dict:
    """serial -> ReferenceCheck, or None when the reference board was not usable. Blocking: run in a worker."""
    detectors = {t.serial: cal.BoardDetector(t.board) for t in targets}
    seen: dict[str, list] = {t.serial: [] for t in targets}
    for i in range(frames):
        if i:
            sleep(spacing_s)
        for t in targets:
            frame = grab_frame(t.serial)
            if frame is not None:
                seen[t.serial].append(detectors[t.serial].detect(frame))
    out = {}
    for t in targets:
        averaged = cal.average_detections(seen[t.serial])
        live = None if averaged is None else cal.solve_board_pose(averaged, detectors[t.serial].board, t.K, t.D)
        out[t.serial] = None if live is None else cal.check_reference(t.R, t.t, t.rms_px, live, t.board)
    return out


class MoveHysteresis:
    """Turns noisy per-check results into a stable moved / not-moved state per camera."""

    def __init__(self, confirmations: int = CONFIRMATIONS) -> None:
        self.confirmations = confirmations
        self.state: dict[str, bool] = {}
        self.suspect_rms: dict[str, float] = {}
        self._streak: dict[str, tuple[bool, int]] = {}

    def reset(self) -> None:
        self.state.clear()
        self.suspect_rms.clear()
        self._streak.clear()

    def update(self, results: Mapping) -> bool:
        """Feed one round of measure_references(); True if the visible state changed."""
        changed = False
        for serial, check in results.items():
            if check is None:
                continue  # reference not visible: keep what we knew
            moved = check.moved
            last, count = self._streak.get(serial, (moved, 0))
            count = count + 1 if last == moved else 1
            self._streak[serial] = (moved, count)
            if count >= self.confirmations and self.state.get(serial) != moved:
                self.state[serial] = moved
                changed = True
            rms = check.rms_px if check.suspect else None
            if rms is None:
                changed |= self.suspect_rms.pop(serial, None) is not None
            elif serial not in self.suspect_rms:
                self.suspect_rms[serial] = rms
                changed = True
            else:
                self.suspect_rms[serial] = rms
        return changed
