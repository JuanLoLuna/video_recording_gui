"""One-button auto setup: set each camera's gain from a calibration board (Two-camera setup #2).

Exposure stays fixed (camera_defaults.DEFAULT_EXPOSURE_US); only the gain moves. The
reference is board A's white paper, still resting at its spot right after the setup
calibration: it is the same in every session, unlike the scene around it (a black bench,
white objects, a bright floor), so the same target gives the same exposure level every
time. Press it after the setup, then remove the board and record.

Two references:
- board A (setup capture, and the fallback): its white paper must read TARGET_WHITE;
- each camera's fixed reference board (G1/G2/B1/B2), once the setup has stored how bright it
  read right after the board A step (brightness_reference.json next to the setups, keyed by
  the setup id): the button brings it back to that level, with nothing to place. The
  reference boards never move, so the result does not depend on where board A was left.

Pure logic plus blocking functions for a worker thread; no Qt.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from backend import calibration as cal

# Grey level board A's white paper should read at. It leaves headroom above white paper
# for brighter highlights; skin under the IR light reads darker than the paper.
TARGET_WHITE = 200.0
TOLERANCE = 8.0
MAX_ITERATIONS = 6
MAX_STEP_DB = 12.0
SATURATED = 250.0
# At a gain limit, a white level in this band still counts as set: well exposed, not clipped
# (on the rig the Firefly's board A read 219 at 0 dB, its lowest gain).
ACCEPT_LOW_FRACTION = 0.7      # of the target
ACCEPT_HIGH_FRACTION = 1.25    # of the target, and never above ACCEPT_HIGH_MAX
ACCEPT_HIGH_MAX = 235.0
REFERENCE_FILE = "brightness_reference.json"
# A reference board's stored level is only usable if it is neither clipped nor too dark.
REFERENCE_MIN, REFERENCE_MAX = 60.0, 240.0
FRAMES_PER_MEASURE = 3
FRAME_SPACING_S = 0.1
SETTLE_S = 0.3   # after a gain change: let a few frames with the new gain arrive
# The newest frame the app hands out can lag the camera by a second or more (the stream
# buffer holds seconds of frames, and the setup page's detectors keep the CPU busy). After a
# gain change, re-measure until the level moves to what the new gain predicts (or is no
# longer clipped), at most WAIT_CHECKS x WAIT_STEP_S, before deciding anything: deciding on
# a stale, clipped frame stopped the Firefly at 0 dB with "reduce the light" on the rig.
WAIT_CHECKS = 15
WAIT_STEP_S = 0.2
WAIT_MATCH = 0.15   # within 15 % of the predicted level


@dataclass(frozen=True)
class AutoSetupResult:
    serial: str
    ok: bool
    gain_db: float | None
    white_level: float | None
    iterations: int
    message: str


def board_white_level(gray: np.ndarray, det: cal.Detection) -> float | None:
    """Grey level of the board's white paper: the 95th percentile inside the detected corners.

    Inside the corner hull about a fifth of the pixels are white paper (the rest is black
    squares and markers), so the 95th percentile sits on the paper, not on a highlight.
    """
    if not det.ok or len(det.corners) < 4:
        return None
    hull = cv2.convexHull(det.corners.reshape(-1, 1, 2).astype(np.float32)).astype(np.int32)
    mask = np.zeros(gray.shape[:2], np.uint8)
    cv2.fillConvexPoly(mask, hull, 1)
    vals = gray[mask > 0]
    return float(np.percentile(vals, 95)) if vals.size >= 50 else None


def measure_white(gray: np.ndarray, board: cal.BoardConfig, detector: cal.BoardDetector | None = None) -> float | None:
    """The board's white-paper level in one grey frame, or None if the board isn't found."""
    det = (detector or cal.BoardDetector(board)).detect(gray)
    return board_white_level(gray, det)


def acceptable_at_limit(white: float, target: float) -> bool:
    """A gain limit stopped the loop: is the white level still well exposed and unclipped?"""
    return ACCEPT_LOW_FRACTION * target <= white <= min(ACCEPT_HIGH_MAX, ACCEPT_HIGH_FRACTION * target)


def gain_step_db(white: float, target: float = TARGET_WHITE) -> float:
    """How far to move the gain: brightness scales with gain, 20 dB per 10x."""
    if white >= SATURATED:
        return -6.0  # clipped: the true level is unknown, halve the brightness
    step = 20.0 * math.log10(target / max(white, 1.0))
    return max(-MAX_STEP_DB, min(MAX_STEP_DB, step))


def _wait_for_level(grab_frame, detector, before: float, predicted: float,
                    sleep: Callable[[float], None]) -> None:
    """Return once a frame shows the gain change took effect (see WAIT_CHECKS), or on timeout."""
    for _ in range(WAIT_CHECKS):
        frame = grab_frame()
        if frame is not None:
            gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            level = board_white_level(gray, detector.detect(gray))
            if level is not None:
                if before >= SATURATED and predicted < before:
                    if level < SATURATED:        # was clipped: any unclipped reading is new
                        return
                elif abs(level - min(predicted, 255.0)) <= WAIT_MATCH * max(predicted, 1.0):
                    return
        sleep(WAIT_STEP_S)


def auto_gain(serial: str, board: cal.BoardConfig, grab_frame: Callable[[], np.ndarray | None],
              get_gain: Callable[[], float | None], set_gain: Callable[[float], bool],
              gain_limits: tuple[float, float], target: float = TARGET_WHITE,
              sleep: Callable[[float], None] = time.sleep, board_name: str = "board A",
              tolerance: float | None = None) -> AutoSetupResult:
    """Move one camera's gain until the board's white paper reads `target` (+- tolerance). Blocking."""
    detector = cal.BoardDetector(board)
    tol = TOLERANCE if tolerance is None else tolerance
    lo, hi = gain_limits
    gain = get_gain()
    if gain is None:
        return AutoSetupResult(serial, False, None, None, 0, "gain not readable")
    white = None
    for it in range(1, MAX_ITERATIONS + 1):
        levels = []
        for i in range(FRAMES_PER_MEASURE):
            if i:
                sleep(FRAME_SPACING_S)
            frame = grab_frame()
            if frame is None:
                continue
            gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            level = board_white_level(gray, detector.detect(gray))
            if level is not None:
                levels.append(level)
        if not levels:
            where = "put it at its spot, whole and in view" if board_name == "board A" else "is it covered?"
            return AutoSetupResult(serial, False, gain, None, it, f"{board_name} not found: {where}")
        white = float(np.median(levels))
        if abs(white - target) <= tol:
            return AutoSetupResult(serial, True, gain, white, it,
                                   f"gain {gain:.1f} dB, {board_name} white {white:.0f}")
        new = max(lo, min(hi, gain + gain_step_db(white, target)))
        if abs(new - gain) < 0.05:
            if acceptable_at_limit(white, target):
                return AutoSetupResult(serial, True, gain, white, it,
                                       f"gain {gain:.1f} dB (its limit), {board_name} white {white:.0f}")
            what = "add light" if white < target else "reduce the light"
            return AutoSetupResult(serial, False, gain, white, it,
                                   f"gain at its limit ({gain:.1f} dB) with {board_name} white {white:.0f}: {what}")
        if not set_gain(new):
            return AutoSetupResult(serial, False, gain, white, it, "camera refused the gain")
        actual = get_gain()
        predicted = white * 10 ** (((new if actual is None else actual) - gain) / 20.0)
        gain = new if actual is None else actual
        sleep(SETTLE_S)
        _wait_for_level(grab_frame, detector, white, predicted, sleep)
    return AutoSetupResult(serial, False, gain, white, MAX_ITERATIONS,
                           f"did not settle (gain {gain:.1f} dB, {board_name} white {white:.0f}); press again")


@dataclass(frozen=True)
class Target:
    """What one camera's gain is tuned on: a board, the white level it must read, a name."""
    board: cal.BoardConfig
    white: float = TARGET_WHITE
    name: str = "board A"

    @property
    def tolerance(self) -> float:
        return max(6.0, 0.04 * self.white)


def auto_setup_cameras(cameras, board: cal.BoardConfig, exposure_us: float,
                       sleep: Callable[[float], None] = time.sleep,
                       targets: dict | None = None) -> list[tuple[str, AutoSetupResult]]:
    """Every camera: auto exposure/gain off, exposure to `exposure_us` (capped to the frame
    period), then auto_gain on its target (`targets`: serial -> Target; default board A at
    TARGET_WHITE). `cameras`: (label, serial, CameraController). Blocking: run it in a worker.
    Used by the main window's button and the setup capture."""
    out = []
    for label, serial, ctl in cameras:
        target = (targets or {}).get(serial) or Target(board)
        ctl.set_enum_param("ExposureAuto", "Off")
        ctl.set_enum_param("GainAuto", "Off")
        limits = ctl.get_exposure_time_limits()
        if limits is not None:
            ctl.set_image_param("ExposureTime", min(limits[1], max(limits[0], float(exposure_us))))
        gl = ctl.get_image_param_limits("Gain")
        if gl is None:
            out.append((label, AutoSetupResult(serial, False, None, None, 0, "gain not writable")))
            continue

        def get_gain(ctl=ctl):
            lim = ctl.get_image_param_limits("Gain")
            return None if lim is None else float(lim[2])

        sleep(SETTLE_S)  # frames with the new exposure
        out.append((label, auto_gain(serial, target.board, ctl.get_latest_frame, get_gain,
                                     lambda v, ctl=ctl: ctl.set_image_param("Gain", v),
                                     (float(gl[0]), float(gl[1])), target=target.white, sleep=sleep,
                                     board_name=target.name, tolerance=target.tolerance)))
    return out


# --------------------------------------------------------------------------
# Reference-board brightness, stored per setup
# --------------------------------------------------------------------------

def reference_path(root) -> Path:
    return Path(root) / REFERENCE_FILE


def save_reference(root, setup_id: str, cameras: dict) -> None:
    """cameras: serial -> {"name", "board" (BoardConfig dict), "white", "gain_db", "exposure_us"}.
    Only usable levels (REFERENCE_MIN..REFERENCE_MAX) are kept. Replaces any older file."""
    keep = {s: c for s, c in cameras.items()
            if c and c.get("white") is not None and REFERENCE_MIN <= float(c["white"]) <= REFERENCE_MAX}
    path = reference_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"setup_id": setup_id, "cameras": keep}, indent=1))
    tmp.replace(path)


def load_reference(root, setup_id: str | None) -> dict:
    """serial -> stored entry, for this setup only (a newer setup makes the file stale)."""
    try:
        data = json.loads(reference_path(root).read_text())
    except (OSError, ValueError):
        return {}
    if not setup_id or data.get("setup_id") != setup_id:
        return {}
    return data.get("cameras") or {}


def targets_from_reference(reference: dict) -> dict:
    """serial -> Target on that camera's reference board, from load_reference()."""
    out = {}
    for serial, entry in reference.items():
        try:
            out[serial] = Target(cal.BoardConfig.from_dict(entry["board"]), float(entry["white"]),
                                 f"reference {entry.get('name', 'board')}")
        except (KeyError, TypeError, ValueError):
            continue
    return out


def summary(results: list[tuple[str, AutoSetupResult]]) -> tuple[bool, str]:
    """(all ok, one line per camera) for a status label."""
    ok = all(r.ok for _, r in results)
    text = " · ".join(f"{label}: {'OK, ' if r.ok else ''}{r.message}" for label, r in results)
    return ok, text
