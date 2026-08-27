"""Machine-readable "do not fit across this" markers for camera-fault recovery.

When the acquisition watchdog reinitialises the camera after a fault, the
video's record_frame_index stays unbroken (per the session's binding
requirement: one continuous metadata CSV per session), but camera_frame_id
and timestamp_us both jump. A timeline_break record documents exactly
where and why, in the same vocabulary the sibling SyncService heartbeat
already uses (Sleeve/sync-service/heartbeat.py) so the two logs can be
cross-referenced during analysis.

Records go to a JSONL sidecar (`<base>_events.jsonl`), never into the
per-frame metadata CSV -- inserting a non-frame row there would break the
"exactly one row per record_frame_index" lookup downstream tooling
depends on.
"""

from __future__ import annotations

import json
import os
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


TIMELINE_BREAK_THRESHOLD_NS = 1_000_000_000  # matches heartbeat.py's --timeline-break-ms default


@dataclass(frozen=True)
class TimelineBreak:
    segment: int
    segment_uuid: str
    prev_segment: int
    mono_ns: int
    wall_ns: int
    cause: str  # "camera_reinit" | "system_suspend"
    note: str
    gap_s: float | None = None
    frames_lost_estimate: int | None = None
    record_frame_index: int | None = None


def timeline_break_record(brk: TimelineBreak) -> dict[str, object]:
    record: dict[str, object] = {
        "rec": "timeline_break",
        "segment": brk.segment,
        "segment_uuid": brk.segment_uuid,
        "prev_segment": brk.prev_segment,
        "mono_ns": brk.mono_ns,
        "wall_ns": brk.wall_ns,
        "cause": brk.cause,
        "note": brk.note,
    }
    if brk.gap_s is not None:
        record["gap_s"] = brk.gap_s
    if brk.frames_lost_estimate is not None:
        record["frames_lost_estimate"] = brk.frames_lost_estimate
    if brk.record_frame_index is not None:
        record["record_frame_index"] = brk.record_frame_index
    return record


def session_header_record(
    *, mono_ns: int, wall_ns: int, recording_basename: str
) -> dict[str, object]:
    """First record in a session's events sidecar.

    Written even when the session has zero faults, so the file is never
    silently empty -- an events.jsonl with only a header and a stop
    record is the expected, correct shape for a clean session; one with
    nothing at all previously looked identical to "logging is broken".
    """
    return {
        "rec": "header",
        "segment": 0,
        "mono_ns": mono_ns,
        "wall_ns": wall_ns,
        "recording": recording_basename,
    }


def session_stop_record(
    *, mono_ns: int, wall_ns: int, total_segments: int, camera_reinits: int
) -> dict[str, object]:
    """Last record in a session's events sidecar."""
    return {
        "rec": "stop",
        "mono_ns": mono_ns,
        "wall_ns": wall_ns,
        "total_segments": total_segments,
        "camera_reinits": camera_reinits,
    }


def detect_suspend(
    *,
    prev_skew_ns: int,
    wall_ns: int,
    mono_ns: int,
    t0_wall_ns: int,
    t0_mono_ns: int,
    break_threshold_ns: int = TIMELINE_BREAK_THRESHOLD_NS,
) -> tuple[bool, int, int]:
    """Port of heartbeat.py's suspend/clock-step detector.

    Returns (is_break, skew_ns, delta_skew_ns). skew_ns is how far wall
    and monotonic clocks have drifted apart since t0; a sudden jump in
    that drift means one clock moved and the other didn't -- either a
    suspend/resume (monotonic stalls, wall keeps going) or an NTP/manual
    clock step (wall jumps, monotonic doesn't).
    """
    skew_ns = (wall_ns - t0_wall_ns) - (mono_ns - t0_mono_ns)
    delta_skew_ns = skew_ns - prev_skew_ns
    is_break = abs(delta_skew_ns) > break_threshold_ns
    return is_break, skew_ns, delta_skew_ns


def estimate_frames_lost(gap_s: float, fps: float) -> int:
    if fps <= 0:
        return 0
    return max(0, round(gap_s * fps))


class SegmentTracker:
    """Thread-safe segment/uuid counter shared by the GUI and acquisition threads."""

    def __init__(self, *, uuid_factory: Callable[[], str] = lambda: str(uuid.uuid4())) -> None:
        self._lock = threading.Lock()
        self._segment = 0
        self._uuid = uuid_factory()
        self._uuid_factory = uuid_factory

    @property
    def current_segment(self) -> int:
        with self._lock:
            return self._segment

    @property
    def current_uuid(self) -> str:
        with self._lock:
            return self._uuid

    def reset(self) -> None:
        with self._lock:
            self._segment = 0
            self._uuid = self._uuid_factory()

    def begin_break(
        self,
        *,
        cause: str,
        mono_ns: int,
        wall_ns: int,
        note: str,
        gap_s: float | None = None,
        frames_lost_estimate: int | None = None,
        record_frame_index: int | None = None,
    ) -> TimelineBreak:
        with self._lock:
            prev_segment = self._segment
            self._segment += 1
            self._uuid = self._uuid_factory()
            return TimelineBreak(
                segment=self._segment,
                segment_uuid=self._uuid,
                prev_segment=prev_segment,
                mono_ns=mono_ns,
                wall_ns=wall_ns,
                cause=cause,
                note=note,
                gap_s=gap_s,
                frames_lost_estimate=frames_lost_estimate,
                record_frame_index=record_frame_index,
            )


class JsonlEventLog:
    """Append-only JSONL sidecar, one fsync'd line per event.

    Ports heartbeat.py's Log class: `open(path, "x")` so an existing file
    is never silently overwritten, and per-record fsync -- affordable
    here because timeline breaks are rare (unlike the 30 Hz frame
    metadata stream, which deliberately does NOT fsync per row).
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._lock = threading.Lock()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = open(self._path, "x", encoding="utf-8")

    def write(self, rec: dict[str, object]) -> None:
        with self._lock:
            self._handle.write(json.dumps(rec) + "\n")
            self._handle.flush()
            os.fsync(self._handle.fileno())

    def close(self) -> None:
        with self._lock:
            if not self._handle.closed:
                self._handle.close()
