"""Pure helpers for the satellite preview (no Qt, no thalamus, no av; tests: tests/test_preview.py)."""
from __future__ import annotations

import collections
from typing import Optional

H264_FORMAT = 12            # Image.Format.H264 (thalamus.proto, Thalamus 1.0.66)
STALE_PICTURE_S = 5.0       # picture older than this is greyed out
NO_FRAMES_S = 5.0           # no frames for this long while Running -> orange
FPS_WINDOW_S = 5.0


def nal_types(data: bytes) -> list[int]:
    """NAL unit types of an Annex-B access unit (start codes 00 00 01); same scan as h264_to_mp4."""
    out, i = [], data.find(b"\x00\x00\x01")
    while 0 <= i < len(data) - 3:
        out.append(data[i + 3] & 0x1F)
        i = data.find(b"\x00\x00\x01", i + 3)
    return out


def is_keyframe(data: bytes) -> bool:
    """True if the access unit holds an IDR slice (NAL type 5): decodable without earlier frames
    (the satellite repeats SPS/PPS before every IDR). P-frames are never worth decoding here."""
    return 5 in nal_types(data)


class RateLimiter:
    """allow(now) is True about max_fps times per second (gaps of at least 90% of 1/max_fps, so an IDR every
    14 frames at 15 fps = 0.93 s still passes at max_fps 1); max_fps <= 0 never allows."""

    def __init__(self, max_fps: float):
        self.min_gap = 0.9 / max_fps if max_fps and max_fps > 0 else None
        self.last: Optional[float] = None

    def allow(self, now: float) -> bool:
        if self.min_gap is None:
            return False
        if self.last is None or now - self.last >= self.min_gap:
            self.last = now
            return True
        return False


class FpsMeter:
    """Frames per second over the last window_s seconds, and the time since the newest frame."""

    def __init__(self, window_s: float = FPS_WINDOW_S):
        self.window_s = window_s
        self.times: collections.deque = collections.deque()
        self.first: Optional[float] = None
        self.last: Optional[float] = None

    def add(self, now: float) -> None:
        if self.first is None:
            self.first = now
        self.last = now
        self.times.append(now)

    def _trim(self, now: float) -> None:
        while self.times and now - self.times[0] > self.window_s:
            self.times.popleft()

    def fps(self, now: float) -> float:
        self._trim(now)
        if not self.times or self.first is None:
            return 0.0
        span = min(self.window_s, now - self.first)
        if span < 1.0:   # too short to divide by; fewer than 2 frames cannot give a rate
            return float(len(self.times) - 1) / span if len(self.times) > 1 and span > 0 else 0.0
        return len(self.times) / span

    def age_s(self, now: float) -> Optional[float]:
        return None if self.last is None else now - self.last


def picture_state(age_s: Optional[float]) -> str:
    """none (never had one) | ok | stale (older than 5 s: grey it out)."""
    if age_s is None:
        return "none"
    return "stale" if age_s > STALE_PICTURE_S else "ok"


def picture_caption(age_s: Optional[float], error: Optional[str]) -> str:
    if age_s is None:
        return "no picture (%s)" % (error or "waiting for the first keyframe")
    if age_s > STALE_PICTURE_S:
        return "no picture (last one %.0f s ago%s)" % (age_s, ": " + error if error else "")
    return "picture %.1f s old" % age_s


def stream_status(running: Optional[bool], fps: float, frame_age_s: Optional[float]) -> dict:
    """The 'frames arriving' signal for the status line."""
    arriving = frame_age_s is not None and frame_age_s <= NO_FRAMES_S
    return {"running": running, "fps": fps, "age_s": frame_age_s, "arriving": arriving}
