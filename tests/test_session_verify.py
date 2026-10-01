import csv
import json
import tempfile
import unittest
from pathlib import Path

from backend.frame_metadata import METADATA_FIELDS
from backend.segment_manifest import MANIFEST_FIELDS
from backend.session_verify import verify_camera_outputs

STEM = "recording_20261001_101500_cam222"


def write_session(
    directory,
    *,
    frames_per_segment=(3, 2),
    fps=30.0,
    stem=STEM,
    serial="222",
    drop_stop=False,
    last_roll="session_stop",
    skip_indices=(),
    skip_ids=(),
    break_after_row=None,
    header_recording=None,
):
    directory = Path(directory)
    rows, index, seg_rows = [], 0, []
    frame_id = 0
    for seg_no, count in enumerate(frames_per_segment):
        first = index + 1
        for _ in range(count):
            index += 1
            frame_id += 1
            if index in skip_ids:
                frame_id += 1  # the camera produced a frame we never received
            segment = 1 if (break_after_row is not None and index > break_after_row) else 0
            if break_after_row is not None and index == break_after_row + 1:
                frame_id = 1  # reinit resets the camera's own counter
            if index in skip_indices:
                continue
            rows.append({
                "record_frame_index": index,
                "camera_frame_id": frame_id,
                "timestamp_us": int(index * 1_000_000 / fps),
                "system_time": 1000.0 + index / fps,
                "sync_pulse": False, "sync_label": "", "adl_id": "", "adl_label": "",
                "segment": segment, "segment_file": f"{stem}-{seg_no:04d}.avi",
                "segment_frame_index": index - first + 1,
                "monotonic_s": index / fps, "wall_mono_skew_s": 0.0,
            })
        seg_rows.append({
            "segment_index": seg_no, "segment_file": f"{stem}-{seg_no:04d}.avi",
            "first_record_frame_index": first, "last_record_frame_index": index,
            "frame_count": count, "bytes": 1,
            "roll_reason": last_roll if seg_no == len(frames_per_segment) - 1 else "frame_count",
        })
    with open(directory / "m.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=METADATA_FIELDS)
        w.writeheader()
        w.writerows(rows)
    with open(directory / "s.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(seg_rows)
    records = [{"rec": "header", "recording": header_recording or stem, "camera_serial": serial}]
    if break_after_row is not None:
        records.append({"rec": "timeline_break", "cause": "camera_reinit"})
    if not drop_stop:
        records.append({"rec": "stop", "total_segments": len(frames_per_segment)})
    with open(directory / "e.jsonl", "w") as f:
        f.write("\n".join(json.dumps(r) for r in records) + "\n")
    return directory / "m.csv", directory / "s.csv", directory / "e.jsonl"


class VerifyCameraOutputsTests(unittest.TestCase):
    def verify(self, **kw):
        expect = {k: kw.pop(k) for k in ("expect_stem", "expect_serial", "expected_fps") if k in kw}
        with tempfile.TemporaryDirectory() as d:
            m, s, e = write_session(d, **kw)
            return verify_camera_outputs(m, s, e, **expect)

    def test_a_clean_session_has_no_problems(self):
        report = self.verify(expect_stem=STEM, expect_serial="222", expected_fps=30.0)
        self.assertTrue(report.ok, report.problems)
        self.assertEqual(report.metadata_rows, 5)
        self.assertEqual(report.segment_frame_sum, 5)
        self.assertEqual(report.last_record_frame_index, 5)
        self.assertEqual(report.segment_count, 2)
        self.assertAlmostEqual(report.effective_fps, 30.0, places=3)
        self.assertAlmostEqual(report.median_timestamp_delta_ms, 33.333, places=2)
        self.assertEqual(report.roll_reasons, {"frame_count": 1, "session_stop": 1})

    def test_missing_files_are_reported_not_raised(self):
        with tempfile.TemporaryDirectory() as d:
            report = verify_camera_outputs(Path(d) / "a", Path(d) / "b", Path(d) / "c")
        self.assertFalse(report.ok)
        self.assertEqual(len(report.problems), 3)

    def test_a_missing_row_breaks_the_dense_index(self):
        report = self.verify(skip_indices=(3,))
        self.assertFalse(report.ok)
        self.assertEqual(report.frame_index_gaps, [(2, 4)])
        self.assertTrue(any("not dense" in p for p in report.problems))
        self.assertTrue(any("segment frame_count sum" in p for p in report.problems))

    def test_lost_camera_frames_are_counted(self):
        report = self.verify(skip_ids=(3,))
        self.assertEqual(report.camera_frame_id_gaps, 1)
        self.assertTrue(any("camera_frame_id gap" in p for p in report.problems))

    def test_a_camera_counter_reset_at_a_reinit_is_not_a_gap(self):
        report = self.verify(break_after_row=2)
        self.assertEqual(report.camera_frame_id_gaps, 0)
        self.assertEqual(report.timeline_breaks, 1)
        self.assertTrue(report.ok, report.problems)

    def test_an_unclosed_final_segment_is_a_problem(self):
        report = self.verify(last_roll="frame_count")
        self.assertTrue(any("session_stop" in p for p in report.problems))

    def test_a_missing_stop_record_is_a_problem(self):
        self.assertTrue(any("stop record" in p for p in self.verify(drop_stop=True).problems))

    def test_header_must_name_this_cameras_files(self):
        report = self.verify(expect_stem=STEM, header_recording="recording_20261001_101500")
        self.assertTrue(any("events header recording" in p for p in report.problems))

    def test_header_must_name_this_camera(self):
        report = self.verify(expect_serial="111")
        self.assertTrue(any("camera_serial" in p for p in report.problems))

    def test_segments_must_belong_to_the_stem(self):
        report = self.verify(expect_stem="recording_20261001_101500")  # the untagged stem
        self.assertTrue(any("does not belong to stem" in p for p in report.problems))

    def test_a_slow_effective_rate_is_flagged_only_without_a_fault(self):
        slow = self.verify(fps=29.0, expected_fps=30.0)
        self.assertTrue(any("fps" in p for p in slow.problems))
        # A timeline break legitimately explains a lower average rate.
        faulted = self.verify(fps=29.0, expected_fps=30.0, break_after_row=2)
        self.assertFalse(any("effective" in p for p in faulted.problems))


if __name__ == "__main__":
    unittest.main()
