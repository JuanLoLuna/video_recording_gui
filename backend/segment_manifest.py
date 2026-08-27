"""Per-segment manifest: one row per video segment file, not per frame.

Lets any downstream consumer byte-skip into the (multi-GB, session-long)
metadata CSV via skiprows/nrows using first/last_record_frame_index,
instead of parsing the whole thing to find one segment's rows. Also
carries the roll_reason/timeline_break/gap_s that distinguish a routine
scheduled rotation from a camera-fault-forced one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from backend.async_csv_writer import AsyncCsvWriter


MANIFEST_FIELDS = [
    "segment_index",
    "segment_file",
    "first_record_frame_index",
    "last_record_frame_index",
    "frame_count",
    "first_system_time",
    "last_system_time",
    "bytes",
    "opened_at",
    "closed_at",
    "close_duration_s",
    "roll_reason",
    "timeline_break",
    "gap_s",
    "append_errors",
]


@dataclass(frozen=True)
class SegmentManifestEntry:
    segment_index: int
    segment_file: str
    frame_count: int = 0
    first_record_frame_index: int | None = None
    last_record_frame_index: int | None = None
    first_system_time: float | None = None
    last_system_time: float | None = None
    bytes: int | None = None
    opened_at: float | None = None
    closed_at: float | None = None
    close_duration_s: float | None = None
    roll_reason: str = ""
    timeline_break: bool = False
    gap_s: float | None = None
    append_errors: int = 0


def _opt(value: object) -> object:
    return "" if value is None else value


def manifest_row(entry: SegmentManifestEntry | Mapping[str, object]) -> dict[str, object]:
    """Coerce a manifest entry into the exact CSV row shape."""
    if isinstance(entry, SegmentManifestEntry):
        e: Mapping[str, object] = {
            "segment_index": entry.segment_index,
            "segment_file": entry.segment_file,
            "first_record_frame_index": entry.first_record_frame_index,
            "last_record_frame_index": entry.last_record_frame_index,
            "frame_count": entry.frame_count,
            "first_system_time": entry.first_system_time,
            "last_system_time": entry.last_system_time,
            "bytes": entry.bytes,
            "opened_at": entry.opened_at,
            "closed_at": entry.closed_at,
            "close_duration_s": entry.close_duration_s,
            "roll_reason": entry.roll_reason,
            "timeline_break": entry.timeline_break,
            "gap_s": entry.gap_s,
            "append_errors": entry.append_errors,
        }
    else:
        e = entry

    return {
        "segment_index": int(e.get("segment_index", 0)),
        "segment_file": str(e.get("segment_file", "")),
        "first_record_frame_index": _opt(e.get("first_record_frame_index")),
        "last_record_frame_index": _opt(e.get("last_record_frame_index")),
        "frame_count": int(e.get("frame_count", 0)),
        "first_system_time": _opt(e.get("first_system_time")),
        "last_system_time": _opt(e.get("last_system_time")),
        "bytes": _opt(e.get("bytes")),
        "opened_at": _opt(e.get("opened_at")),
        "closed_at": _opt(e.get("closed_at")),
        "close_duration_s": _opt(e.get("close_duration_s")),
        "roll_reason": str(e.get("roll_reason", "")),
        "timeline_break": bool(e.get("timeline_break", False)),
        "gap_s": _opt(e.get("gap_s")),
        "append_errors": int(e.get("append_errors", 0)),
    }


class SegmentManifestWriter(AsyncCsvWriter):
    """One row per segment (~960 over 10 days) -- correctness over throughput.

    Never drops a row (drop_when_full=False): each row is the only record
    of a segment's frame range, and volume is trivial compared to the
    per-frame metadata stream.
    """

    def __init__(self) -> None:
        super().__init__(
            MANIFEST_FIELDS,
            max_pending_rows=64,
            flush_every_rows=1,
            drop_when_full=False,
            thread_name="segment-manifest-writer",
        )
