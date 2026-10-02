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
    real_layout=False,
    break_record_index=True,
    break_record_offset=0,
    adl_label='',
    no_fids=False,
    truncate_events=False,
    manifest_hole=False,
    bad_join=False,
    stray_segment_file=False,
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
            flip_after = break_after_row + (1 if real_layout else 0) if break_after_row is not None else None
            segment = 1 if (flip_after is not None and index > flip_after) else 0
            if break_after_row is not None and index == break_after_row + 1:
                # A reinit either resets the camera's own counter, or (a stall
                # without a power cycle) leaves it counting past the lost frames.
                frame_id = frame_id + 5 if real_layout else 1
            if index in skip_indices:
                continue
            rows.append({
                "record_frame_index": index,
                "camera_frame_id": "" if no_fids else frame_id,
                "timestamp_us": int(index * 1_000_000 / fps),
                "system_time": 1000.0 + index / fps,
                "sync_pulse": False, "sync_label": "", "adl_id": "", "adl_label": adl_label,
                "segment": segment,
                "segment_file": "other-0000.avi" if (stray_segment_file and index == 2) else f"{stem}-{seg_no:04d}.avi",
                "segment_frame_index": index - first + 1,
                "monotonic_s": index / fps, "wall_mono_skew_s": 0.0,
            })
        seg_rows.append({
            "segment_index": 5 if (manifest_hole and seg_no == 1) else seg_no,
            "segment_file": f"{stem}-{seg_no:04d}.avi",
            "first_record_frame_index": first + (1 if (bad_join and seg_no == 1) else 0),
            "last_record_frame_index": index,
            "frame_count": count, "bytes": 1,
            "roll_reason": last_roll if seg_no == len(frames_per_segment) - 1 else "frame_count",
        })
    with open(directory / "m.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=METADATA_FIELDS)
        w.writeheader()
        w.writerows(rows)
    with open(directory / "s.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(seg_rows)
    records = [{"rec": "header", "recording": header_recording or stem, "camera_serial": serial}]
    if break_after_row is not None:
        brk = {"rec": "timeline_break", "cause": "camera_reinit"}
        if break_record_index:
            brk["record_frame_index"] = break_after_row + break_record_offset
        records.append(brk)
    if not drop_stop:
        records.append({"rec": "stop", "total_segments": len(frames_per_segment)})
    with open(directory / "e.jsonl", "w", encoding="utf-8") as f:
        f.write("\n".join(json.dumps(r) for r in records) + "\n")
        if truncate_events:
            f.write('{"rec": "sto')
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

    def test_the_real_fault_layout_is_not_a_false_failure(self):
        # The reconnect frame is appended to the OLD segment (segment column
        # flips one row later) and the camera's counter may keep counting past
        # the lost frames instead of resetting.
        report = self.verify(frames_per_segment=(4, 4), break_after_row=2, real_layout=True)
        self.assertEqual(report.camera_frame_id_gaps, 0)
        self.assertTrue(report.ok, report.problems)

    def test_a_forward_jump_without_any_timeline_break_is_still_a_loss(self):
        report = self.verify(frames_per_segment=(4, 4), break_after_row=2, real_layout=True,
                             break_record_index=False)
        self.assertGreater(report.camera_frame_id_gaps, 0)

    def test_a_break_explains_one_discontinuity_not_every_later_loss(self):
        report = self.verify(frames_per_segment=(4, 6), break_after_row=2, real_layout=True,
                             skip_ids=(8,))
        self.assertEqual(report.camera_frame_id_gaps, 1)  # only the loss at row 8

    def test_a_recorded_index_a_little_behind_the_true_break_still_matches(self):
        # The controller records its frame counter when it handled the fault,
        # which can lag the append thread.
        report = self.verify(frames_per_segment=(6, 4), break_after_row=4, real_layout=True,
                             break_record_offset=-2)
        self.assertTrue(report.ok, report.problems)

    def test_missing_frame_ids_are_flagged_not_silently_passed(self):
        report = self.verify(no_fids=True)
        self.assertTrue(any("no camera_frame_id" in p for p in report.problems))

    def test_a_truncated_events_line_is_a_problem_not_an_exception(self):
        report = self.verify(truncate_events=True)
        self.assertTrue(any("unparsable" in p for p in report.problems))

    def test_a_hole_in_segment_indices_is_flagged(self):
        self.assertTrue(any("segment_index" in p for p in self.verify(manifest_hole=True).problems))

    def test_segment_ranges_must_join(self):
        self.assertTrue(any("do not join" in p for p in self.verify(bad_join=True).problems))

    def test_metadata_naming_a_segment_the_manifest_lacks_is_flagged(self):
        report = self.verify(stray_segment_file=True)
        self.assertTrue(any("missing from the manifest" in p for p in report.problems))

    def test_non_ascii_labels_are_read_as_utf8(self):
        # The writer emits utf-8; reading with a locale default (cp1252 on
        # Windows) would mis-decode or raise on a label like this one.
        report = self.verify(adl_label="Écrire \u2014 \u66f8\u304f")
        self.assertTrue(report.ok, report.problems)

    def test_the_longest_capture_gap_is_reported(self):
        clean = self.verify()
        self.assertAlmostEqual(clean.max_capture_gap_s, 1 / 30.0, places=4)
        stalled = self.verify(frames_per_segment=(3, 2), skip_indices=(3,))  # one row missing = one 2-frame gap
        self.assertAlmostEqual(stalled.max_capture_gap_s, 2 / 30.0, places=4)
        self.assertEqual(stalled.max_capture_gap_row, 4)

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
        slow = self.verify(frames_per_segment=(300, 300), fps=29.0, expected_fps=30.0)
        self.assertTrue(any("fps" in p for p in slow.problems))
        # A timeline break legitimately explains a lower average rate.
        faulted = self.verify(frames_per_segment=(300, 300), fps=29.0, expected_fps=30.0, break_after_row=2)
        self.assertFalse(any("effective" in p for p in faulted.problems))

    def test_a_very_short_run_gets_two_frames_of_slack(self):
        # 76 frames in 2.5 s: one frame short is -1.3%, which is rounding, not a fault.
        ok = self.verify(frames_per_segment=(38, 38), fps=29.6, expected_fps=30.0)
        self.assertFalse(any("effective" in p for p in ok.problems), ok.problems)
        # ...but a genuinely slow short run is still caught.
        slow = self.verify(frames_per_segment=(38, 38), fps=24.0, expected_fps=30.0)
        self.assertTrue(any("effective" in p for p in slow.problems))


if __name__ == "__main__":
    unittest.main()
