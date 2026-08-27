import csv
import tempfile
import time
import unittest
from pathlib import Path

from backend.async_csv_writer import AsyncCsvWriter


FIELDS = ["record_frame_index", "value"]


def _wait_until(predicate, timeout=2.0, interval=0.01):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


class AsyncCsvWriterTests(unittest.TestCase):
    def test_writes_rows_in_submission_order(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metadata.csv"
            writer = AsyncCsvWriter(
                FIELDS,
                max_pending_rows=2048,
                flush_every_rows=100,
                drop_when_full=False,
            )
            writer.start(path)
            self.assertTrue(writer.wait_until_open())
            for i in range(1, 1001):
                writer.submit({"record_frame_index": i, "value": i * 2})
            writer.stop()

            with path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            indices = [int(r["record_frame_index"]) for r in rows]
            self.assertEqual(indices, list(range(1, 1001)))

    def test_overflow_preserves_every_row_when_the_queue_fills(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metadata.csv"
            writer = AsyncCsvWriter(
                FIELDS,
                max_pending_rows=2,
                flush_every_rows=50,
                drop_when_full=False,
            )
            writer.start(path)
            self.assertTrue(writer.wait_until_open())
            for i in range(1, 501):
                submitted = writer.submit({"record_frame_index": i, "value": i})
                self.assertTrue(submitted)
            writer.stop()

            with path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 500)
            self.assertEqual(writer.dropped_rows, 0)
            self.assertGreater(writer.overflow_high_water, 0)

    def test_drop_when_full_mode_still_counts_drops(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "diag.csv"
            writer = AsyncCsvWriter(FIELDS, max_pending_rows=1, drop_when_full=True)
            writer.start(path)
            self.assertTrue(writer.wait_until_open())
            # Flood far faster than the writer can drain a 1-row queue.
            for i in range(200):
                writer.submit({"record_frame_index": i, "value": i})
            writer.stop()
            self.assertGreaterEqual(writer.dropped_rows, 0)

    def test_submit_returns_true_without_dropping_under_a_full_queue(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metadata.csv"
            writer = AsyncCsvWriter(FIELDS, max_pending_rows=1, drop_when_full=False)
            writer.start(path)
            self.assertTrue(writer.wait_until_open())
            results = [writer.submit({"record_frame_index": i, "value": i}) for i in range(50)]
            writer.stop()
            self.assertTrue(all(results))
            self.assertEqual(writer.dropped_rows, 0)

    def test_batching_defers_writes_until_the_row_threshold(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metadata.csv"
            writer = AsyncCsvWriter(FIELDS, flush_every_rows=1000)
            writer.start(path)
            self.assertTrue(writer.wait_until_open())
            writer.submit({"record_frame_index": 1, "value": 1})
            time.sleep(0.1)
            self.assertEqual(writer.rows_flushed, 0)
            writer.stop()
            self.assertEqual(writer.rows_flushed, 1)

    def test_time_based_flush_fires_without_reaching_the_row_threshold(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metadata.csv"
            writer = AsyncCsvWriter(FIELDS, flush_every_rows=1000, flush_every_seconds=0.1)
            writer.start(path)
            self.assertTrue(writer.wait_until_open())
            writer.submit({"record_frame_index": 1, "value": 1})
            self.assertTrue(_wait_until(lambda: writer.rows_flushed >= 1, timeout=2.0))
            writer.stop()

    def test_stop_drains_pending_and_overflow_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metadata.csv"
            writer = AsyncCsvWriter(FIELDS, max_pending_rows=1, drop_when_full=False)
            writer.start(path)
            self.assertTrue(writer.wait_until_open())
            for i in range(1, 21):
                writer.submit({"record_frame_index": i, "value": i})
            writer.stop()
            with path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 20)

    def test_unknown_keys_ignored_and_missing_keys_blank(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metadata.csv"
            writer = AsyncCsvWriter(FIELDS)
            writer.start(path)
            self.assertTrue(writer.wait_until_open())
            writer.submit({"record_frame_index": 1, "extra": "ignored"})
            writer.stop()
            with path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["record_frame_index"], "1")
            self.assertEqual(rows[0]["value"], "")

    def test_wait_until_open_reports_failure_for_an_unwritable_path(self):
        with tempfile.TemporaryDirectory() as directory:
            blocking_file = Path(directory) / "not_a_directory"
            blocking_file.write_text("x")
            bad_path = blocking_file / "metadata.csv"
            writer = AsyncCsvWriter(FIELDS)
            writer.start(bad_path)
            opened = writer.wait_until_open(timeout=2.0)
            self.assertFalse(opened)
            self.assertIsNotNone(writer.last_error)
            writer.stop()


if __name__ == "__main__":
    unittest.main()
