import csv
import tempfile
import unittest
from pathlib import Path

from backend.segment_manifest import (
    MANIFEST_FIELDS,
    SegmentManifestEntry,
    SegmentManifestWriter,
    manifest_row,
)


class ManifestRowTests(unittest.TestCase):
    def test_row_matches_the_expected_shape(self):
        entry = SegmentManifestEntry(
            segment_index=7,
            segment_file="recording_X-0007.avi",
            frame_count=27000,
            first_record_frame_index=189001,
            last_record_frame_index=216000,
            first_system_time=1700000000.0,
            last_system_time=1700000900.0,
            bytes=1_650_000_000,
            opened_at=100.0,
            closed_at=101.5,
            close_duration_s=1.5,
            roll_reason="frame_count",
            timeline_break=False,
            gap_s=None,
            append_errors=0,
        )
        row = manifest_row(entry)
        self.assertEqual(row["segment_index"], 7)
        self.assertEqual(row["segment_file"], "recording_X-0007.avi")
        self.assertEqual(row["frame_count"], 27000)
        self.assertEqual(row["gap_s"], "")
        self.assertIs(row["timeline_break"], False)

    def test_a_timeline_break_row_round_trips_with_gap_s(self):
        entry = SegmentManifestEntry(
            segment_index=8,
            segment_file="recording_X-0008.avi",
            roll_reason="fault",
            timeline_break=True,
            gap_s=12.5,
        )
        row = manifest_row(entry)
        self.assertEqual(row["roll_reason"], "fault")
        self.assertIs(row["timeline_break"], True)
        self.assertEqual(row["gap_s"], 12.5)

    def test_accepts_a_plain_mapping_too(self):
        row = manifest_row({"segment_index": 1, "segment_file": "x.avi"})
        self.assertEqual(row["segment_index"], 1)
        self.assertEqual(row["frame_count"], 0)


class SegmentManifestWriterTests(unittest.TestCase):
    def test_header_and_rows_written_and_flushed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "recording_x_segments.csv"
            writer = SegmentManifestWriter()
            writer.start(path)
            self.assertTrue(writer.wait_until_open())
            writer.submit(
                manifest_row(
                    SegmentManifestEntry(segment_index=0, segment_file="recording_x-0000.avi", frame_count=27000)
                )
            )
            writer.submit(
                manifest_row(
                    SegmentManifestEntry(segment_index=1, segment_file="recording_x-0001.avi", frame_count=27000)
                )
            )
            writer.stop()

            with path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                self.assertEqual(reader.fieldnames, MANIFEST_FIELDS)
                rows = list(reader)
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["segment_file"], "recording_x-0000.avi")
            self.assertEqual(rows[1]["segment_file"], "recording_x-0001.avi")

    def test_never_drops_a_row_under_a_full_queue(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "recording_x_segments.csv"
            writer = SegmentManifestWriter()
            writer.start(path)
            self.assertTrue(writer.wait_until_open())
            for i in range(200):
                writer.submit(
                    manifest_row(
                        SegmentManifestEntry(segment_index=i, segment_file=f"recording_x-{i:04d}.avi")
                    )
                )
            writer.stop()
            self.assertEqual(writer.dropped_rows, 0)
            with path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 200)


if __name__ == "__main__":
    unittest.main()
