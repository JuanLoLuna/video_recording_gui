import json
import tempfile
import threading
import unittest
from pathlib import Path

from backend.timeline_break import (
    JsonlEventLog,
    SegmentTracker,
    TimelineBreak,
    detect_suspend,
    estimate_frames_lost,
    session_header_record,
    session_stop_record,
    timeline_break_record,
)


class TimelineBreakRecordTests(unittest.TestCase):
    def test_record_matches_the_sync_service_vocabulary(self):
        brk = TimelineBreak(
            segment=1,
            segment_uuid="abc-123",
            prev_segment=0,
            mono_ns=1_000_000_000,
            wall_ns=2_000_000_000,
            cause="camera_reinit",
            note="do not fit across this",
        )
        record = timeline_break_record(brk)
        self.assertEqual(record["rec"], "timeline_break")
        for key in ("segment", "segment_uuid", "prev_segment", "mono_ns", "wall_ns", "note"):
            self.assertIn(key, record)

    def test_optional_fields_omitted_when_none(self):
        brk = TimelineBreak(
            segment=1,
            segment_uuid="abc",
            prev_segment=0,
            mono_ns=0,
            wall_ns=0,
            cause="camera_reinit",
            note="x",
        )
        record = timeline_break_record(brk)
        self.assertNotIn("gap_s", record)
        self.assertNotIn("frames_lost_estimate", record)
        self.assertNotIn("record_frame_index", record)

    def test_optional_fields_included_when_present(self):
        brk = TimelineBreak(
            segment=1,
            segment_uuid="abc",
            prev_segment=0,
            mono_ns=0,
            wall_ns=0,
            cause="camera_reinit",
            note="x",
            gap_s=12.0,
            frames_lost_estimate=360,
            record_frame_index=5000,
        )
        record = timeline_break_record(brk)
        self.assertEqual(record["gap_s"], 12.0)
        self.assertEqual(record["frames_lost_estimate"], 360)
        self.assertEqual(record["record_frame_index"], 5000)


class SessionBookendRecordTests(unittest.TestCase):
    def test_header_record_shape(self):
        record = session_header_record(
            mono_ns=1, wall_ns=2, recording_basename="recording_20260101_000000"
        )
        self.assertEqual(record["rec"], "header")
        self.assertEqual(record["segment"], 0)
        self.assertEqual(record["recording"], "recording_20260101_000000")

    def test_stop_record_shape(self):
        record = session_stop_record(mono_ns=1, wall_ns=2, total_segments=3, camera_reinits=1)
        self.assertEqual(record["rec"], "stop")
        self.assertEqual(record["total_segments"], 3)
        self.assertEqual(record["camera_reinits"], 1)

    def test_a_clean_session_still_produces_a_non_empty_log(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            log = JsonlEventLog(path)
            log.write(session_header_record(mono_ns=0, wall_ns=0, recording_basename="x"))
            log.write(session_stop_record(mono_ns=1, wall_ns=1, total_segments=0, camera_reinits=0))
            log.close()
            lines = path.read_text().strip().split("\n")
            self.assertEqual(len(lines), 2)
            self.assertGreater(path.stat().st_size, 0)


class DetectSuspendTests(unittest.TestCase):
    def test_no_drift_is_not_a_break(self):
        is_break, skew, delta = detect_suspend(
            prev_skew_ns=0, wall_ns=1_000_000_000, mono_ns=1_000_000_000,
            t0_wall_ns=0, t0_mono_ns=0,
        )
        self.assertFalse(is_break)
        self.assertEqual(skew, 0)

    def test_flags_a_monotonic_stall_wall_advances_mono_does_not(self):
        # Machine suspended for an hour: wall clock jumps, monotonic barely moves.
        is_break, skew, delta = detect_suspend(
            prev_skew_ns=0,
            wall_ns=3600_000_000_000,
            mono_ns=100_000_000,
            t0_wall_ns=0,
            t0_mono_ns=0,
        )
        self.assertTrue(is_break)

    def test_flags_an_ntp_step(self):
        # Wall clock stepped backward 10 minutes; monotonic kept going normally.
        is_break, skew, delta = detect_suspend(
            prev_skew_ns=0,
            wall_ns=-600_000_000_000,
            mono_ns=1_000_000_000,
            t0_wall_ns=0,
            t0_mono_ns=0,
        )
        self.assertTrue(is_break)

    def test_sub_threshold_skew_is_not_a_break(self):
        is_break, skew, delta = detect_suspend(
            prev_skew_ns=0,
            wall_ns=1_000_500_000,  # 500 us of drift, well under the 1s threshold
            mono_ns=1_000_000_000,
            t0_wall_ns=0,
            t0_mono_ns=0,
        )
        self.assertFalse(is_break)


class EstimateFramesLostTests(unittest.TestCase):
    def test_uses_the_configured_frame_rate(self):
        self.assertEqual(estimate_frames_lost(12.0, 30.0), 360)

    def test_zero_fps_is_zero_frames(self):
        self.assertEqual(estimate_frames_lost(12.0, 0.0), 0)

    def test_negative_gap_clamped_to_zero(self):
        self.assertEqual(estimate_frames_lost(-1.0, 30.0), 0)


class SegmentTrackerTests(unittest.TestCase):
    def test_first_segment_is_zero_and_the_first_break_is_one(self):
        tracker = SegmentTracker(uuid_factory=lambda: "u")
        self.assertEqual(tracker.current_segment, 0)
        brk = tracker.begin_break(cause="camera_reinit", mono_ns=0, wall_ns=0, note="x")
        self.assertEqual(brk.segment, 1)
        self.assertEqual(brk.prev_segment, 0)
        self.assertEqual(tracker.current_segment, 1)

    def test_each_break_issues_a_fresh_uuid(self):
        counter = iter(["u0", "u1", "u2"])
        tracker = SegmentTracker(uuid_factory=lambda: next(counter))
        b1 = tracker.begin_break(cause="camera_reinit", mono_ns=0, wall_ns=0, note="x")
        b2 = tracker.begin_break(cause="camera_reinit", mono_ns=0, wall_ns=0, note="x")
        self.assertNotEqual(b1.segment_uuid, b2.segment_uuid)

    def test_segment_tracker_is_safe_under_concurrent_breaks(self):
        tracker = SegmentTracker()
        seen: list[int] = []
        lock = threading.Lock()

        def worker():
            for _ in range(100):
                brk = tracker.begin_break(cause="camera_reinit", mono_ns=0, wall_ns=0, note="x")
                with lock:
                    seen.append(brk.segment)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(seen), 800)
        self.assertEqual(len(set(seen)), 800)
        self.assertEqual(sorted(seen), list(range(1, 801)))


class JsonlEventLogTests(unittest.TestCase):
    def test_writes_one_json_object_per_line(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            log = JsonlEventLog(path)
            log.write({"rec": "header", "segment": 0})
            log.write({"rec": "timeline_break", "segment": 1})
            log.close()

            lines = path.read_text().strip().split("\n")
            self.assertEqual(len(lines), 2)
            self.assertEqual(json.loads(lines[0])["rec"], "header")
            self.assertEqual(json.loads(lines[1])["rec"], "timeline_break")

    def test_refuses_to_overwrite_an_existing_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            first = JsonlEventLog(path)
            first.close()
            with self.assertRaises(FileExistsError):
                JsonlEventLog(path)


if __name__ == "__main__":
    unittest.main()
