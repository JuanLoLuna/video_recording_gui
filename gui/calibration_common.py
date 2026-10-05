"""Pieces shared by the Calibration window's tasks (camera calibration, session setup).

Kept apart from gui/calibration_window.py so the setup task (gui/calibration_setup.py)
can use them without a circular import.
"""

from __future__ import annotations

import threading
import time
from typing import Sequence

import cv2
import numpy as np

from backend import calibration as cal

DETECT_INTERVAL_S = 0.2       # worker: ~5 detections per second
LIVE_VIEW_SIZE = (720, 540)


class _Detector:
    """Worker thread: latest frame of one camera -> board detection, a few times per second.

    `detector` is one BoardDetector (latest() gives a Detection) or a dict of
    them, all run on the same frame (latest() gives {name: Detection}): the
    setup looks for board A and both reference boards in every frame.
    """

    def __init__(self, controller, detector) -> None:
        self.controller = controller
        self.detector = detector
        self._lock = threading.Lock()
        self._latest: tuple[int, np.ndarray, object] | None = None
        self._seq = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="calibration-detector", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            started = time.monotonic()
            frame = self.controller.get_latest_frame()
            if frame is not None:
                try:
                    if isinstance(self.detector, dict):
                        det = {name: d.detect(frame) for name, d in self.detector.items()}
                    else:
                        det = self.detector.detect(frame)
                except Exception as exc:  # never let a bad frame kill the worker
                    print(f"[calibration] detection failed: {exc}")
                else:
                    with self._lock:
                        self._seq += 1
                        self._latest = (self._seq, frame, det)
            self._stop.wait(max(0.0, DETECT_INTERVAL_S - (time.monotonic() - started)))

    def latest(self):
        with self._lock:
            return self._latest

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)


def draw_overlay(frame: np.ndarray, det: cal.Detection | None, max_size=LIVE_VIEW_SIZE,
                 extra: Sequence = ()) -> np.ndarray:
    """Downscaled BGR copy of the frame with the detected corners drawn.

    `extra`: more (Detection, BGR colour) pairs drawn on top, e.g. a reference board.
    """
    h, w = frame.shape[:2]
    scale = min(max_size[0] / w, max_size[1] / h, 1.0)
    small = cv2.resize(frame, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    if small.ndim == 2:
        small = cv2.cvtColor(small, cv2.COLOR_GRAY2BGR)
    if det is not None and len(det.ids):
        colour = (60, 200, 60) if det.ok else (0, 160, 255)
        for x, y in det.corners * scale:
            cv2.circle(small, (int(x), int(y)), 4, colour, -1, cv2.LINE_AA)
    for other, colour in extra:
        if other is not None and len(other.ids):
            for x, y in other.corners * scale:
                cv2.circle(small, (int(x), int(y)), 3, colour, -1, cv2.LINE_AA)
    return small
