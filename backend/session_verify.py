"""Post-run checks on one camera's recorded sidecars (no PySpin, no video).

Reads the three per-camera files a recording leaves behind -- the per-frame
metadata CSV, the per-segment manifest and the events JSONL -- and reports
whether they agree with each other. This is the "reconciliation" the phase 2
and 3 hardware reports did by hand: metadata rows == sum of segment frame
counts == last record_frame_index, a dense index, a closed final segment, a
header that names the right files and camera, and no camera_frame_id gaps
inside a continuous stretch.

Used by scripts/multi_controller_smoke.py and usable on any finished session.
"""

from __future__ import annotations

import csv
import json
import statistics
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from backend.frame_metadata import find_frame_index_gaps


@dataclass
class CameraSessionReport:
    metadata_rows: int = 0
    last_record_frame_index: int = 0
    segment_count: int = 0
    segment_frame_sum: int = 0
    roll_reasons: dict[str, int] = field(default_factory=dict)
    frame_index_gaps: list[tuple[int, int]] = field(default_factory=list)
    camera_frame_id_gaps: int = 0
    timeline_breaks: int = 0
    effective_fps: float | None = None
    # Longest wall-clock time between two consecutive captured frames. A big
    # value on a camera with no timeline break means its acquisition stalled.
    max_capture_gap_s: float = 0.0
    max_capture_gap_row: int | None = None
    median_timestamp_delta_ms: float | None = None
    header: dict | None = None
    problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems


# How many rows after a timeline_break's recorded record_frame_index the break's
# discontinuity (a camera_frame_id reset or jump) may appear. The recorded index
# is the controller's frame counter when the fault was handled, which can lag the
# append thread by its queue depth, and the reconnect frame itself is appended
# before the segment roll -- so the discontinuity is near, not exactly at, it.
BREAK_WINDOW_ROWS = 200


def _read_csv(path: Path) -> list[dict[str, str]]:
    # utf-8 is what AsyncCsvWriter writes; the platform default (cp1252 on
    # Windows) would mis-decode a non-ASCII adl_label.
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _read_events(path: Path) -> tuple[list[dict], list[int]]:
    """(records, 1-based numbers of lines that are not valid JSON)."""
    records: list[dict] = []
    bad: list[int] = []
    with open(path, encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                bad.append(number)  # e.g. a truncated last line after a crash
    return records, bad


def _int(value: str | None) -> int | None:
    try:
        return int(value) if value not in (None, "") else None
    except ValueError:
        return None


def verify_camera_outputs(
    metadata_csv: Path,
    segments_csv: Path,
    events_jsonl: Path,
    *,
    expect_stem: str | None = None,
    expect_serial: str | None = None,
    expected_fps: float | None = None,
    fps_tolerance: float = 0.001,
) -> CameraSessionReport:
    """Cross-check one camera's sidecars; problems are collected, never raised."""
    report = CameraSessionReport()
    problems = report.problems

    for label, path in (
        ("metadata", metadata_csv),
        ("segments", segments_csv),
        ("events", events_jsonl),
    ):
        if not Path(path).exists():
            problems.append(f"{label} file missing: {Path(path).name}")
    if problems:
        return report

    rows = _read_csv(metadata_csv)
    segments = _read_csv(segments_csv)
    events, bad_event_lines = _read_events(events_jsonl)
    if bad_event_lines:
        problems.append(f"events log has unparsable line(s): {bad_event_lines[:5]} (truncated?)")

    # ---- metadata: dense index, row count
    indices = [i for i in (_int(r.get("record_frame_index")) for r in rows) if i is not None]
    report.metadata_rows = len(rows)
    report.last_record_frame_index = indices[-1] if indices else 0
    report.frame_index_gaps = find_frame_index_gaps(indices)
    if not rows:
        problems.append("metadata has no rows")
    if report.frame_index_gaps:
        problems.append(f"record_frame_index not dense: {len(report.frame_index_gaps)} gap(s), first {report.frame_index_gaps[0]}")
    if indices and indices[0] != 1:
        problems.append(f"record_frame_index starts at {indices[0]}, not 1")

    # ---- camera_frame_id gaps. A camera fault legitimately breaks the camera's
    # own counter (it resets, or jumps) around a reinit, and the segment
    # column flips one row AFTER the reconnect frame, so neither "same segment"
    # nor an exact row match identifies it. Instead each timeline_break in the
    # events log is allowed to explain ONE discontinuity in the rows that follow
    # its recorded index; any other missing frame id is a real loss.
    break_indices = sorted(
        e["record_frame_index"]
        for e in events
        if e.get("rec") == "timeline_break" and isinstance(e.get("record_frame_index"), int)
    )
    pending_breaks = list(break_indices)
    armed_until: list[int] = []
    gaps = 0
    deltas_ms: list[float] = []
    prev = None
    have_frame_ids = False
    for row in rows:
        idx = _int(row.get("record_frame_index"))
        fid, seg = _int(row.get("camera_frame_id")), row.get("segment")
        ts = _int(row.get("timestamp_us"))
        have_frame_ids |= fid is not None
        if prev is not None:
            prev_idx, prev_fid, prev_seg, prev_ts = prev
            while pending_breaks and prev_idx is not None and prev_idx >= pending_breaks[0]:
                armed_until.append(pending_breaks.pop(0) + BREAK_WINDOW_ROWS)
            armed_until = [e for e in armed_until if prev_idx is None or prev_idx <= e]
            discontinuity = fid is not None and prev_fid is not None and fid != prev_fid + 1
            if discontinuity and armed_until:
                armed_until.pop(0)  # explained by a recorded fault
            elif prev_seg == seg:
                if fid is not None and prev_fid is not None and fid > prev_fid + 1:
                    gaps += fid - prev_fid - 1
                if ts is not None and prev_ts is not None and ts > prev_ts:
                    deltas_ms.append((ts - prev_ts) / 1000.0)
        prev = (idx, fid, seg, ts)
    report.camera_frame_id_gaps = gaps
    if rows and not have_frame_ids:
        problems.append("no camera_frame_id values in the metadata: frame loss cannot be checked")
    if gaps:
        problems.append(f"{gaps} camera_frame_id gap(s) not explained by a timeline break (frames lost)")
    if deltas_ms:
        report.median_timestamp_delta_ms = statistics.median(deltas_ms)

    # ---- segments manifest
    report.segment_count = len(segments)
    report.segment_frame_sum = sum(_int(s.get("frame_count")) or 0 for s in segments)
    report.roll_reasons = dict(Counter(s.get("roll_reason", "") for s in segments))
    if not segments:
        problems.append("segments manifest has no rows")
    else:
        if segments[-1].get("roll_reason") != "session_stop":
            problems.append(
                f"last segment roll_reason is {segments[-1].get('roll_reason')!r}, not 'session_stop' (not closed cleanly?)"
            )
        if report.segment_frame_sum != report.metadata_rows:
            problems.append(
                f"segment frame_count sum {report.segment_frame_sum} != metadata rows {report.metadata_rows}"
            )
    if indices and report.last_record_frame_index != report.metadata_rows:
        problems.append(
            f"last record_frame_index {report.last_record_frame_index} != metadata rows {report.metadata_rows}"
        )
    if segments:
        seg_indices = [_int(sg.get("segment_index")) for sg in segments]
        if seg_indices != list(range(len(segments))):
            problems.append(f"segment_index is not 0..{len(segments) - 1} without holes: {seg_indices[:8]}")
        for before, after in zip(segments, segments[1:]):
            last, first = _int(before.get("last_record_frame_index")), _int(after.get("first_record_frame_index"))
            if last is not None and first is not None and first != last + 1:
                problems.append(
                    f"segment frame ranges do not join: {before.get('segment_file')} ends at {last}, "
                    f"{after.get('segment_file')} starts at {first}"
                )
                break
        listed = {sg.get("segment_file") for sg in segments}
        stray = {r.get("segment_file") for r in rows} - listed
        if stray:
            problems.append(f"metadata names segment file(s) missing from the manifest: {sorted(stray)[:3]}")
    if expect_stem is not None:
        for seg in segments:
            name = seg.get("segment_file", "")
            if not name.startswith(expect_stem + "-"):
                problems.append(f"segment file {name!r} does not belong to stem {expect_stem!r}")
                break

    # ---- events: header and stop
    headers = [e for e in events if e.get("rec") == "header"]
    stops = [e for e in events if e.get("rec") == "stop"]
    report.timeline_breaks = sum(1 for e in events if e.get("rec") == "timeline_break")
    if len(headers) != 1:
        problems.append(f"expected 1 events header, found {len(headers)}")
    else:
        report.header = headers[0]
        if expect_stem is not None and headers[0].get("recording") != expect_stem:
            problems.append(
                f"events header recording {headers[0].get('recording')!r} != stem {expect_stem!r}"
            )
        if expect_serial is not None and headers[0].get("camera_serial") != expect_serial:
            problems.append(
                f"events header camera_serial {headers[0].get('camera_serial')!r} != {expect_serial!r}"
            )
    if not stops:
        problems.append("events has no stop record (session did not end cleanly)")

    # ---- achieved rate from the wall clock
    times = [float(r["system_time"]) for r in rows if r.get("system_time")]
    stamped = [
        (float(r["system_time"]), _int(r.get("record_frame_index")))
        for r in rows
        if r.get("system_time")
    ]
    for (before, _), (after, frame_index) in zip(stamped, stamped[1:]):
        if after - before > report.max_capture_gap_s:
            report.max_capture_gap_s = after - before
            report.max_capture_gap_row = frame_index  # the first frame after the pause
    if len(times) > 1 and times[-1] > times[0]:
        report.effective_fps = (len(times) - 1) / (times[-1] - times[0])
        if (
            expected_fps
            and report.timeline_breaks == 0
            and report.effective_fps < expected_fps * (1.0 - fps_tolerance)
        ):
            problems.append(
                f"effective {report.effective_fps:.3f} fps < {expected_fps:.3f} fps (-{fps_tolerance * 100:.1f}%)"
            )
    return report
