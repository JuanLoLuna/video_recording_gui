import csv
import tempfile
import unittest
from pathlib import Path

from backend.preview_diagnostics import (
    DIAGNOSTIC_FIELDS,
    AsyncDiagnosticsCsvLogger,
    PreviewDiagnosticsAccumulator,
)


class PreviewDiagnosticsAccumulatorTests(unittest.TestCase):
    def test_aggregates_pipeline_timings_and_counter_deltas(self):
        accumulator = PreviewDiagnosticsAccumulator()
        accumulator.reset(
            {
                "camera_frame_gaps": 3,
                "incomplete_images": 4,
                "acquisition_errors": 5,
            },
            now=10.0,
        )
        accumulator.note_repeated_frame()
        accumulator.note_repeated_frame()
        accumulator.note_displayed_frame(
            frame_id=101,
            sequence=9,
            retrieved_at=10.000,
            published_at=10.010,
            displayed_at=10.025,
        )

        row = accumulator.sample(
            {
                "camera_frame_gaps": 5,
                "incomplete_images": 5,
                "acquisition_errors": 8,
            },
            now=11.0,
            wall_time="2026-08-25T12:00:00-04:00",
        )

        self.assertEqual(row["wall_time"], "2026-08-25T12:00:00-04:00")
        self.assertEqual(row["rendered_fps"], 1.0)
        self.assertEqual(row["latest_frame_id"], 101)
        self.assertEqual(row["latest_sequence"], 9)
        self.assertEqual(row["preview_age_ms"], 25.0)
        self.assertEqual(row["retrieval_to_publish_ms"], 10.0)
        self.assertEqual(row["publish_to_display_ms"], 15.0)
        self.assertEqual(row["repeated_frames_skipped"], 2)
        self.assertEqual(row["camera_frame_gaps"], 2)
        self.assertEqual(row["incomplete_images"], 1)
        self.assertEqual(row["acquisition_errors"], 3)

        empty_row = accumulator.sample(
            {
                "camera_frame_gaps": 5,
                "incomplete_images": 5,
                "acquisition_errors": 8,
            },
            now=12.0,
            wall_time="2026-08-25T12:00:01-04:00",
        )
        self.assertEqual(empty_row["rendered_fps"], 0.0)
        self.assertEqual(empty_row["preview_age_ms"], "")
        self.assertEqual(empty_row["repeated_frames_skipped"], 0)
        self.assertEqual(empty_row["camera_frame_gaps"], 0)

    def test_append_failures_and_camera_reinits_are_also_deltas(self):
        accumulator = PreviewDiagnosticsAccumulator()
        accumulator.reset(
            {"append_failures": 2, "camera_reinits": 1},
            now=0.0,
        )
        row = accumulator.sample(
            {"append_failures": 5, "camera_reinits": 2},
            now=1.0,
        )
        self.assertEqual(row["append_failures"], 3)
        self.assertEqual(row["camera_reinits"], 1)

    def test_missing_new_stat_keys_default_to_zero_delta(self):
        accumulator = PreviewDiagnosticsAccumulator()
        accumulator.reset({"camera_frame_gaps": 0}, now=0.0)
        row = accumulator.sample({"camera_frame_gaps": 0}, now=1.0)
        self.assertEqual(row["append_failures"], 0)
        self.assertEqual(row["camera_reinits"], 0)

    def test_audio_health_stats_are_deltas_like_camera_stats(self):
        accumulator = PreviewDiagnosticsAccumulator()
        accumulator.reset(
            {"audio_xruns": 1, "audio_reconnects": 0, "audio_silence_frames_inserted": 0},
            now=0.0,
        )
        row = accumulator.sample(
            {"audio_xruns": 4, "audio_reconnects": 1, "audio_silence_frames_inserted": 480},
            now=1.0,
        )
        self.assertEqual(row["audio_xruns"], 3)
        self.assertEqual(row["audio_reconnects"], 1)
        self.assertEqual(row["audio_silence_frames_inserted"], 480)


class AsyncDiagnosticsCsvLoggerTests(unittest.TestCase):
    def test_writes_rows_with_the_expected_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "recording_diagnostics.csv"
            logger = AsyncDiagnosticsCsvLogger()
            logger.start(path)
            submitted = logger.submit(
                {
                    "wall_time": "2026-08-25T12:00:00-04:00",
                    "interval_s": 1.0,
                    "rendered_fps": 30.0,
                    "preview_age_ms": 12.5,
                }
            )
            logger.stop()

            self.assertTrue(submitted)
            self.assertIsNone(logger.last_error)
            with path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                rows = list(reader)

            self.assertEqual(reader.fieldnames, DIAGNOSTIC_FIELDS)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["rendered_fps"], "30.0")
            self.assertEqual(rows[0]["preview_age_ms"], "12.5")
            self.assertEqual(rows[0]["camera_frame_gaps"], "")


if __name__ == "__main__":
    unittest.main()
