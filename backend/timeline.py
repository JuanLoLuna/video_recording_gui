# backend/timeline.py
"""
Pure per-frame monotonic/wall-clock skew tracking.

time.monotonic() on Windows is backed by GetTickCount64 at ~15.6ms
resolution -- too coarse to measure a ~33ms frame period meaningfully (the
deltas this module exists to expose would themselves be quantized away).
time.perf_counter() does not have that problem, so it backs monotonic_s
here. perf_counter() has no fixed epoch, so a reading is only ever compared
to another perf_counter() reading from the same process run, never across
a restart -- which is exactly how it's used below (as an offset from one
session's own start).
"""

from __future__ import annotations

from dataclasses import dataclass

# Matches sync-service/heartbeat.py's --timeline-break-ms default so a wall/
# monotonic divergence segments identically in both logs.
DEFAULT_BREAK_THRESHOLD_S = 1.0


@dataclass(frozen=True)
class TimelineBaseline:
    session_start_wall_s: float
    session_start_mono_s: float


def compute_wall_mono_skew_s(
    baseline: TimelineBaseline, *, wall_s: float, mono_s: float
) -> float:
    """How far wall-clock elapsed time has drifted from monotonic elapsed
    time since the session started.

    An NTP step or a suspend/resume shows up here as a jump; ordinary clock
    drift shows up as a slow ramp. Near-zero and flat means the two clocks
    agree on how much time has passed.
    """
    elapsed_wall = wall_s - baseline.session_start_wall_s
    elapsed_mono = mono_s - baseline.session_start_mono_s
    return elapsed_wall - elapsed_mono


def is_timeline_break(
    skew_s: float, previous_skew_s: float, *, threshold_s: float = DEFAULT_BREAK_THRESHOLD_S
) -> bool:
    """Whether the skew jumped by at least threshold_s since the last frame."""
    return abs(skew_s - previous_skew_s) >= threshold_s
