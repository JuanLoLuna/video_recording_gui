"""Lightweight preview-pipeline diagnostics and asynchronous CSV logging."""

from __future__ import annotations

import csv
import queue
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Mapping


DIAGNOSTIC_FIELDS = [
    "wall_time",
    "interval_s",
    "rendered_fps",
    "latest_frame_id",
    "latest_sequence",
    "preview_age_ms",
    "preview_age_p95_ms",
    "preview_age_max_ms",
    "retrieval_to_publish_ms",
    "retrieval_to_publish_p95_ms",
    "publish_to_display_ms",
    "publish_to_display_p95_ms",
    "repeated_frames_skipped",
    "camera_frame_gaps",
    "incomplete_images",
    "acquisition_errors",
]


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = round((len(ordered) - 1) * percentile)
    return float(ordered[index])


def _rounded(value: float | None) -> float | str:
    return "" if value is None else round(float(value), 3)


class PreviewDiagnosticsAccumulator:
    """Aggregate per-display timing into one compact row per interval."""

    def __init__(self) -> None:
        self.reset()

    def reset(
        self,
        acquisition_stats: Mapping[str, int] | None = None,
        *,
        now: float | None = None,
    ) -> None:
        self._interval_started_at = time.monotonic() if now is None else float(now)
        self._preview_ages_ms: list[float] = []
        self._retrieval_to_publish_ms: list[float] = []
        self._publish_to_display_ms: list[float] = []
        self._rendered_frames = 0
        self._repeated_frames = 0
        self._latest_frame_id: int | None = None
        self._latest_sequence: int | None = None
        self._previous_acquisition_stats = dict(acquisition_stats or {})

    def note_repeated_frame(self) -> None:
        self._repeated_frames += 1

    def note_displayed_frame(
        self,
        *,
        frame_id: int | None,
        sequence: int,
        retrieved_at: float,
        published_at: float,
        displayed_at: float,
    ) -> None:
        self._rendered_frames += 1
        self._latest_frame_id = frame_id
        self._latest_sequence = sequence
        self._preview_ages_ms.append(max(0.0, displayed_at - retrieved_at) * 1000.0)
        self._retrieval_to_publish_ms.append(
            max(0.0, published_at - retrieved_at) * 1000.0
        )
        self._publish_to_display_ms.append(
            max(0.0, displayed_at - published_at) * 1000.0
        )

    def sample(
        self,
        acquisition_stats: Mapping[str, int] | None = None,
        *,
        now: float | None = None,
        wall_time: str | None = None,
    ) -> dict[str, object]:
        sampled_at = time.monotonic() if now is None else float(now)
        interval_s = max(0.001, sampled_at - self._interval_started_at)
        stats = dict(acquisition_stats or {})

        def stat_delta(name: str) -> int:
            current = int(stats.get(name, 0))
            previous = int(self._previous_acquisition_stats.get(name, current))
            return max(0, current - previous)

        latest_age = self._preview_ages_ms[-1] if self._preview_ages_ms else None
        latest_retrieval_publish = (
            self._retrieval_to_publish_ms[-1]
            if self._retrieval_to_publish_ms
            else None
        )
        latest_publish_display = (
            self._publish_to_display_ms[-1]
            if self._publish_to_display_ms
            else None
        )

        row: dict[str, object] = {
            "wall_time": wall_time
            or datetime.now().astimezone().isoformat(timespec="milliseconds"),
            "interval_s": round(interval_s, 3),
            "rendered_fps": round(self._rendered_frames / interval_s, 3),
            "latest_frame_id": (
                "" if self._latest_frame_id is None else self._latest_frame_id
            ),
            "latest_sequence": (
                "" if self._latest_sequence is None else self._latest_sequence
            ),
            "preview_age_ms": _rounded(latest_age),
            "preview_age_p95_ms": _rounded(_percentile(self._preview_ages_ms, 0.95)),
            "preview_age_max_ms": _rounded(
                max(self._preview_ages_ms) if self._preview_ages_ms else None
            ),
            "retrieval_to_publish_ms": _rounded(latest_retrieval_publish),
            "retrieval_to_publish_p95_ms": _rounded(
                _percentile(self._retrieval_to_publish_ms, 0.95)
            ),
            "publish_to_display_ms": _rounded(latest_publish_display),
            "publish_to_display_p95_ms": _rounded(
                _percentile(self._publish_to_display_ms, 0.95)
            ),
            "repeated_frames_skipped": self._repeated_frames,
            "camera_frame_gaps": stat_delta("camera_frame_gaps"),
            "incomplete_images": stat_delta("incomplete_images"),
            "acquisition_errors": stat_delta("acquisition_errors"),
        }

        self._interval_started_at = sampled_at
        self._preview_ages_ms.clear()
        self._retrieval_to_publish_ms.clear()
        self._publish_to_display_ms.clear()
        self._rendered_frames = 0
        self._repeated_frames = 0
        self._previous_acquisition_stats = stats
        return row


class AsyncDiagnosticsCsvLogger:
    """Write diagnostics rows off the GUI thread."""

    _STOP = object()

    def __init__(self, max_pending_rows: int = 120) -> None:
        self._max_pending_rows = max_pending_rows
        self._queue: queue.Queue[dict[str, object] | object] | None = None
        self._thread: threading.Thread | None = None
        self._path: Path | None = None
        self._last_error: str | None = None
        self._dropped_rows = 0

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def path(self) -> Path | None:
        return self._path

    @property
    def last_error(self) -> str | None:
        return self._last_error

    @property
    def dropped_rows(self) -> int:
        return self._dropped_rows

    def start(self, path: str | Path) -> None:
        self.stop()
        self._path = Path(path)
        self._last_error = None
        self._dropped_rows = 0
        self._queue = queue.Queue(maxsize=self._max_pending_rows)
        self._thread = threading.Thread(
            target=self._writer_loop,
            name="preview-diagnostics-writer",
            daemon=True,
        )
        self._thread.start()

    def submit(self, row: Mapping[str, object]) -> bool:
        if self._queue is None or not self.is_running:
            return False
        try:
            self._queue.put_nowait(dict(row))
            return True
        except queue.Full:
            self._dropped_rows += 1
            return False

    def stop(self, timeout: float = 3.0) -> None:
        work_queue = self._queue
        worker = self._thread
        if work_queue is not None and worker is not None and worker.is_alive():
            try:
                work_queue.put(self._STOP, timeout=timeout)
            except queue.Full:
                self._last_error = "Diagnostics queue did not drain during shutdown."
            worker.join(timeout=timeout)
            if worker.is_alive():
                self._last_error = "Diagnostics writer did not stop cleanly."
        self._thread = None
        self._queue = None

    def _writer_loop(self) -> None:
        assert self._path is not None
        assert self._queue is not None
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=DIAGNOSTIC_FIELDS)
                writer.writeheader()
                handle.flush()
                while True:
                    item = self._queue.get()
                    try:
                        if item is self._STOP:
                            break
                        writer.writerow(
                            {field: item.get(field, "") for field in DIAGNOSTIC_FIELDS}
                        )
                        handle.flush()
                    finally:
                        self._queue.task_done()
        except Exception as exc:
            self._last_error = f"{exc.__class__.__name__}: {exc}"
