"""Lightweight preview-pipeline diagnostics and asynchronous CSV logging."""

from __future__ import annotations

import time
from datetime import datetime
from typing import Mapping

from backend.async_csv_writer import AsyncCsvWriter


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
    "append_failures",
    "camera_reinits",
    "audio_xruns",
    "audio_reconnects",
    "audio_silence_frames_inserted",
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
            "append_failures": stat_delta("append_failures"),
            "camera_reinits": stat_delta("camera_reinits"),
            "audio_xruns": stat_delta("audio_xruns"),
            "audio_reconnects": stat_delta("audio_reconnects"),
            "audio_silence_frames_inserted": stat_delta("audio_silence_frames_inserted"),
        }

        self._interval_started_at = sampled_at
        self._preview_ages_ms.clear()
        self._retrieval_to_publish_ms.clear()
        self._publish_to_display_ms.clear()
        self._rendered_frames = 0
        self._repeated_frames = 0
        self._previous_acquisition_stats = stats
        return row


class AsyncDiagnosticsCsvLogger(AsyncCsvWriter):
    """Write diagnostics rows off the GUI thread, one row per second.

    A thin, behaviour-preserving specialisation of AsyncCsvWriter: per-row
    flush (this log is small -- ~130 MB over 10 days at 1 Hz) and
    drop-on-full (an occasional missed diagnostics sample is fine to lose;
    unlike frame metadata, nothing downstream requires it to be dense).
    """

    def __init__(self, max_pending_rows: int = 120) -> None:
        super().__init__(
            DIAGNOSTIC_FIELDS,
            max_pending_rows=max_pending_rows,
            flush_every_rows=1,
            drop_when_full=True,
            thread_name="preview-diagnostics-writer",
        )
