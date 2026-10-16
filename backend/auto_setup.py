"""One-button auto setup: set each camera's gain from calibration board A (Two-camera setup #2).

Exposure stays fixed (camera_defaults.DEFAULT_EXPOSURE_US); only the gain moves. The
reference is board A's white paper, still resting at its spot right after the setup
calibration: it is the same in every session, unlike the scene around it (a black bench,
white objects, a bright floor), so the same target gives the same exposure level every
time. Press it after the setup, then remove the board and record.

Pure logic plus one blocking function for a worker thread; no Qt.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
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
FRAMES_PER_MEASURE = 3
FRAME_SPACING_S = 0.1
SETTLE_S = 0.3   # after a gain change: let a few frames with the new gain arrive


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


def gain_step_db(white: float, target: float = TARGET_WHITE) -> float:
    """How far to move the gain: brightness scales with gain, 20 dB per 10x."""
    if white >= SATURATED:
        return -6.0  # clipped: the true level is unknown, halve the brightness
    step = 20.0 * math.log10(target / max(white, 1.0))
    return max(-MAX_STEP_DB, min(MAX_STEP_DB, step))


def auto_gain(serial: str, board: cal.BoardConfig, grab_frame: Callable[[], np.ndarray | None],
              get_gain: Callable[[], float | None], set_gain: Callable[[float], bool],
              gain_limits: tuple[float, float], target: float = TARGET_WHITE,
              sleep: Callable[[float], None] = time.sleep) -> AutoSetupResult:
    """Move one camera's gain until board A's white paper reads `target` (+- TOLERANCE). Blocking."""
    detector = cal.BoardDetector(board)
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
            return AutoSetupResult(serial, False, gain, None, it,
                                   "board A not found: put it at its spot, whole and in view")
        white = float(np.median(levels))
        if abs(white - target) <= TOLERANCE:
            return AutoSetupResult(serial, True, gain, white, it, f"gain {gain:.1f} dB, board white {white:.0f}")
        new = max(lo, min(hi, gain + gain_step_db(white, target)))
        if abs(new - gain) < 0.05:
            what = "add light" if white < target else "reduce the light"
            return AutoSetupResult(serial, False, gain, white, it,
                                   f"gain at its limit ({gain:.1f} dB) with board white {white:.0f}: {what}")
        if not set_gain(new):
            return AutoSetupResult(serial, False, gain, white, it, "camera refused the gain")
        actual = get_gain()
        gain = new if actual is None else actual
        sleep(SETTLE_S)
    return AutoSetupResult(serial, False, gain, white, MAX_ITERATIONS,
                           f"did not settle (gain {gain:.1f} dB, board white {white:.0f}); press again")


def auto_setup_cameras(cameras, board: cal.BoardConfig, exposure_us: float,
                       sleep: Callable[[float], None] = time.sleep) -> list[tuple[str, AutoSetupResult]]:
    """Every camera: auto exposure/gain off, exposure to `exposure_us` (capped to the frame
    period), then auto_gain from board A. `cameras`: (label, serial, CameraController).
    Blocking: run it in a worker. Used by the main window's button and the setup capture."""
    out = []
    for label, serial, ctl in cameras:
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
        out.append((label, auto_gain(serial, board, ctl.get_latest_frame, get_gain,
                                     lambda v, ctl=ctl: ctl.set_image_param("Gain", v),
                                     (float(gl[0]), float(gl[1])), sleep=sleep)))
    return out


def summary(results: list[tuple[str, AutoSetupResult]]) -> tuple[bool, str]:
    """(all ok, one line per camera) for a status label."""
    ok = all(r.ok for _, r in results)
    text = " · ".join(f"{label}: {'OK, ' if r.ok else ''}{r.message}" for label, r in results)
    return ok, text
