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
    median_timestamp_delta_ms: float | None = None
    header: dict | None = None
    problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems


def _read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def _read_events(path: Path) -> list[dict]:
    records = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


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
    events = _read_events(events_jsonl)

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

    # ---- camera_frame_id gaps, only inside one continuous stretch (a reinit
    # resets the camera's own counter and bumps `segment`)
    gaps = 0
    deltas_ms: list[float] = []
    prev = None
    for row in rows:
        fid, seg = _int(row.get("camera_frame_id")), row.get("segment")
        ts = _int(row.get("timestamp_us"))
        if prev is not None and prev[1] == seg:
            if fid is not None and prev[0] is not None and fid > prev[0] + 1:
                gaps += fid - prev[0] - 1
            if ts is not None and prev[2] is not None and ts > prev[2]:
                deltas_ms.append((ts - prev[2]) / 1000.0)
        prev = (fid, seg, ts)
    report.camera_frame_id_gaps = gaps
    if gaps:
        problems.append(f"{gaps} camera_frame_id gap(s) inside continuous stretches (frames lost)")
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
