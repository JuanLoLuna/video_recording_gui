"""Pure per-frame metadata coercion, shared by the acquisition loop and tests.

Lifts the row-building logic that used to live inline in
CameraController._acquisition_loop's batch CSV write, so it can be
exercised without importing PySpin (impossible on macOS) and reused by
the incremental AsyncCsvWriter path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping


LEGACY_METADATA_FIELDS = [
    "record_frame_index",
    "camera_frame_id",
    "timestamp_us",
    "system_time",
    "sync_pulse",
    "sync_label",
    "adl_id",
    "adl_label",
]

# "segment" tracks camera-recovery timeline breaks (Phase 1.4); it starts at
# 0 for every session and increments on each auto-reinit.
#
# "segment_file"/"segment_frame_index" (Phase 2) track video ROTATION
# instead: which physical AVI a frame landed in, and its position within
# that file. These change on every rotation (scheduled or fault-forced),
# while "segment" only changes on a fault -- so "segment_file changed but
# segment did not" means a routine scheduled rotation, and "both changed"
# means a fault-forced one, without needing to cross-reference the
# segment manifest for that distinction.
#
# All three are appended at the end -- every downstream consumer of this
# CSV reads columns by name, so adding trailing columns is additive and
# safe.
#
# "monotonic_s"/"wall_mono_skew_s" (Phase 3.5) are additive in the same way:
# a perf_counter() reading (session-relative, not comparable across a
# restart) and the drift between it and system_time since the session
# started. See backend/timeline.py for why perf_counter() and not
# time.monotonic().
METADATA_FIELDS = LEGACY_METADATA_FIELDS + [
    "segment",
    "segment_file",
    "segment_frame_index",
    "monotonic_s",
    "wall_mono_skew_s",
]


@dataclass(frozen=True)
class FrameMetadata:
    record_frame_index: int
    camera_frame_id: int | None = None
    timestamp_us: int | None = None
    system_time: float = 0.0
    sync_pulse: bool = False
    sync_label: str | None = None
    adl_id: int | None = None
    adl_label: str | None = None
    segment: int = 0
    segment_file: str = ""
    segment_frame_index: int = 0
    monotonic_s: float = 0.0
    wall_mono_skew_s: float = 0.0


def resolve_sync_label(label_event: str | None, sync_label: str | None) -> str | None:
    """Match camera_control.py's existing precedence: label event wins."""
    return label_event if label_event else sync_label


def metadata_row(record: FrameMetadata | Mapping[str, object]) -> dict[str, object]:
    """Coerce a frame record into the exact CSV row shape written today.

    Byte-identical to the coercions previously inline in
    CameraController._acquisition_loop (int/"" for optional ints, real
    bool for sync_pulse, str/"" for optional strings) plus the additive
    `segment` column.
    """
    if isinstance(record, FrameMetadata):
        rec: Mapping[str, object] = {
            "record_frame_index": record.record_frame_index,
            "camera_frame_id": record.camera_frame_id,
            "timestamp_us": record.timestamp_us,
            "system_time": record.system_time,
            "sync_pulse": record.sync_pulse,
            "sync_label": record.sync_label,
            "adl_id": record.adl_id,
            "adl_label": record.adl_label,
            "segment": record.segment,
            "segment_file": record.segment_file,
            "segment_frame_index": record.segment_frame_index,
            "monotonic_s": record.monotonic_s,
            "wall_mono_skew_s": record.wall_mono_skew_s,
        }
    else:
        rec = record

    return {
        "record_frame_index": int(rec.get("record_frame_index", 0)),
        "camera_frame_id": (
            "" if rec.get("camera_frame_id") is None else int(rec.get("camera_frame_id"))
        ),
        "timestamp_us": (
            "" if rec.get("timestamp_us") is None else int(rec.get("timestamp_us"))
        ),
        "system_time": float(rec.get("system_time", 0.0)),
        "sync_pulse": bool(rec.get("sync_pulse", False)),
        "sync_label": "" if rec.get("sync_label") is None else str(rec.get("sync_label")),
        "adl_id": "" if rec.get("adl_id") is None else int(rec.get("adl_id")),
        "adl_label": "" if rec.get("adl_label") is None else str(rec.get("adl_label")),
        "segment": int(rec.get("segment", 0)),
        "segment_file": str(rec.get("segment_file", "")),
        "segment_frame_index": int(rec.get("segment_frame_index", 0)),
        "monotonic_s": round(float(rec.get("monotonic_s", 0.0)), 6),
        "wall_mono_skew_s": round(float(rec.get("wall_mono_skew_s", 0.0)), 6),
    }


def metadata_csv_path(recording_basename: str) -> str:
    """Mirror the existing `<base>_metadata.csv` naming exactly."""
    return recording_basename.rsplit(".", 1)[0] + "_metadata.csv"


def find_frame_index_gaps(indices: Iterable[int]) -> list[tuple[int, int]]:
    """Return (prev, next) pairs where `next` is not `prev + 1`.

    A post-run verifier for the dense 1..N invariant that downstream
    tooling (video_label_metadata.py) depends on.
    """
    gaps: list[tuple[int, int]] = []
    prev: int | None = None
    for idx in indices:
        if prev is not None and idx != prev + 1:
            gaps.append((prev, idx))
        prev = idx
    return gaps
