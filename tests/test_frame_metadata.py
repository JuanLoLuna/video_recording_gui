import csv
import tempfile
import unittest
from pathlib import Path

from backend.async_csv_writer import AsyncCsvWriter
from backend.frame_metadata import (
    LEGACY_METADATA_FIELDS,
    METADATA_FIELDS,
    find_frame_index_gaps,
    metadata_csv_path,
    metadata_row,
    resolve_sync_label,
)


SAMPLE_RAW_RECORD_PATH = (
    Path(__file__).parent / "raw_videos" / "recording_20251111_175410_metadata.csv"
)


class MetadataRowTests(unittest.TestCase):
    def test_row_matches_the_legacy_coercions_exactly(self):
        row = metadata_row(
            {
                "record_frame_index": 42,
                "camera_frame_id": 100,
                "timestamp_us": 123456,
                "system_time": 1700000000.5,
                "sync_pulse": True,
                "sync_label": "record_start",
                "adl_id": 7,
                "adl_label": "Writing",
            }
        )
        self.assertEqual(
            row,
            {
                "record_frame_index": 42,
                "camera_frame_id": 100,
                "timestamp_us": 123456,
                "system_time": 1700000000.5,
                "sync_pulse": True,
                "sync_label": "record_start",
                "adl_id": 7,
                "adl_label": "Writing",
                "segment": 0,
                "segment_file": "",
                "segment_frame_index": 0,
                "monotonic_s": 0.0,
                "wall_mono_skew_s": 0.0,
            },
        )

    def test_none_optionals_become_empty_strings(self):
        row = metadata_row(
            {
                "record_frame_index": 1,
                "camera_frame_id": None,
                "timestamp_us": None,
                "system_time": 0.0,
                "sync_pulse": False,
                "sync_label": None,
                "adl_id": None,
                "adl_label": None,
            }
        )
        self.assertEqual(row["camera_frame_id"], "")
        self.assertEqual(row["timestamp_us"], "")
        self.assertEqual(row["sync_label"], "")
        self.assertEqual(row["adl_id"], "")
        self.assertEqual(row["adl_label"], "")

    def test_sync_pulse_is_a_real_bool_not_a_string(self):
        row = metadata_row(
            {
                "record_frame_index": 1,
                "system_time": 0.0,
                "sync_pulse": True,
            }
        )
        self.assertIs(row["sync_pulse"], True)

    def test_legacy_field_order_is_unchanged(self):
        self.assertEqual(METADATA_FIELDS[:8], LEGACY_METADATA_FIELDS)
        self.assertEqual(
            METADATA_FIELDS[8:],
            [
                "segment",
                "segment_file",
                "segment_frame_index",
                "monotonic_s",
                "wall_mono_skew_s",
            ],
        )

    def test_monotonic_s_and_skew_pass_through_and_round(self):
        row = metadata_row(
            {
                "record_frame_index": 1,
                "system_time": 0.0,
                "monotonic_s": 12.3456789,
                "wall_mono_skew_s": -0.0001234567,
            }
        )
        self.assertEqual(row["monotonic_s"], 12.345679)
        self.assertEqual(row["wall_mono_skew_s"], -0.000123)

    def test_segment_file_and_frame_index_default_to_empty_and_zero(self):
        row = metadata_row(
            {"record_frame_index": 1, "system_time": 0.0}
        )
        self.assertEqual(row["segment_file"], "")
        self.assertEqual(row["segment_frame_index"], 0)

    def test_segment_file_and_frame_index_pass_through(self):
        row = metadata_row(
            {
                "record_frame_index": 1,
                "system_time": 0.0,
                "segment_file": "recording_X-0007.avi",
                "segment_frame_index": 42,
            }
        )
        self.assertEqual(row["segment_file"], "recording_X-0007.avi")
        self.assertEqual(row["segment_frame_index"], 42)


class ResolveSyncLabelTests(unittest.TestCase):
    def test_label_event_takes_precedence_over_sync_window_label(self):
        self.assertEqual(resolve_sync_label("label_start", "record_start"), "label_start")

    def test_sync_window_label_used_when_no_label_event(self):
        self.assertEqual(resolve_sync_label(None, "record_start"), "record_start")

    def test_both_none_is_none(self):
        self.assertIsNone(resolve_sync_label(None, None))


class MetadataCsvPathTests(unittest.TestCase):
    def test_handles_names_without_an_extension(self):
        self.assertEqual(metadata_csv_path("recording_x"), "recording_x_metadata.csv")

    def test_handles_names_with_an_extension(self):
        self.assertEqual(metadata_csv_path("recording_x.avi"), "recording_x_metadata.csv")


class FindFrameIndexGapsTests(unittest.TestCase):
    def test_dense_sequence_has_no_gaps(self):
        self.assertEqual(find_frame_index_gaps([1, 2, 3, 4]), [])

    def test_reports_a_hole(self):
        self.assertEqual(find_frame_index_gaps([1, 2, 5, 6]), [(2, 5)])

    def test_empty_input_has_no_gaps(self):
        self.assertEqual(find_frame_index_gaps([]), [])


class RoundTripThroughAsyncCsvWriterTests(unittest.TestCase):
    def test_round_trip_matches_the_sample_header(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "recording_x_metadata.csv"
            writer = AsyncCsvWriter(METADATA_FIELDS, flush_every_rows=1, drop_when_full=False)
            writer.start(path)
            self.assertTrue(writer.wait_until_open())
            for i in range(1, 4):
                writer.submit(
                    metadata_row(
                        {
                            "record_frame_index": i,
                            "camera_frame_id": i,
                            "timestamp_us": i * 1000,
                            "system_time": float(i),
                            "sync_pulse": False,
                            "sync_label": None,
                            "adl_id": None,
                            "adl_label": None,
                        }
                    )
                )
            writer.stop()

            with path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                header = reader.fieldnames
                rows = list(reader)

            with SAMPLE_RAW_RECORD_PATH.open(newline="", encoding="utf-8") as handle:
                sample_header = csv.DictReader(handle).fieldnames

            self.assertEqual(header[:4], list(sample_header))
            self.assertEqual([int(r["record_frame_index"]) for r in rows], [1, 2, 3])


if __name__ == "__main__":
    unittest.main()
