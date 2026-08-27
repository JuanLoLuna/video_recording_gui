import tempfile
import unittest
from pathlib import Path

from backend.segment_policy import (
    PREPARE_LEAD_FRAMES,
    RenamePlan,
    plan_renames,
    reconcile_part_files,
    segment_frames_for,
    should_prepare,
    should_roll,
)


class SegmentFramesForTests(unittest.TestCase):
    def test_30_fps_900_seconds_is_27000_frames(self):
        self.assertEqual(segment_frames_for(30.0, 900.0), 27000)

    def test_falls_back_to_30fps_for_zero(self):
        self.assertEqual(segment_frames_for(0.0), segment_frames_for(30.0))

    def test_falls_back_to_30fps_for_negative(self):
        self.assertEqual(segment_frames_for(-5.0), segment_frames_for(30.0))

    def test_falls_back_to_30fps_for_nan(self):
        self.assertEqual(segment_frames_for(float("nan")), segment_frames_for(30.0))

    def test_falls_back_to_30fps_for_none(self):
        self.assertEqual(segment_frames_for(None), segment_frames_for(30.0))

    def test_never_returns_zero_or_less(self):
        self.assertGreaterEqual(segment_frames_for(0.001, 1.0), 1)


class ShouldRollTests(unittest.TestCase):
    def test_no_roll_below_all_thresholds(self):
        decision = should_roll(
            frames_in_segment=100, bytes_in_segment=1000, max_frames=27000
        )
        self.assertFalse(decision.should_roll)
        self.assertIsNone(decision.reason)

    def test_rolls_exactly_at_max_frames(self):
        decision = should_roll(
            frames_in_segment=27000, bytes_in_segment=0, max_frames=27000
        )
        self.assertTrue(decision.should_roll)
        self.assertEqual(decision.reason, "frame_count")

    def test_does_not_roll_one_frame_before_max(self):
        decision = should_roll(
            frames_in_segment=26999, bytes_in_segment=0, max_frames=27000
        )
        self.assertFalse(decision.should_roll)

    def test_rolls_past_max_frames_too(self):
        decision = should_roll(
            frames_in_segment=27001, bytes_in_segment=0, max_frames=27000
        )
        self.assertTrue(decision.should_roll)

    def test_rolls_on_bytes_with_low_frame_count(self):
        decision = should_roll(
            frames_in_segment=10, bytes_in_segment=4_000_000_000, max_frames=27000
        )
        self.assertTrue(decision.should_roll)
        self.assertEqual(decision.reason, "bytes")

    def test_fault_forces_a_roll_regardless_of_counters(self):
        decision = should_roll(
            frames_in_segment=0, bytes_in_segment=0, max_frames=27000, fault=True
        )
        self.assertTrue(decision.should_roll)
        self.assertEqual(decision.reason, "fault")

    def test_fault_takes_precedence_over_other_reasons(self):
        decision = should_roll(
            frames_in_segment=27000, bytes_in_segment=4_000_000_000,
            max_frames=27000, fault=True,
        )
        self.assertEqual(decision.reason, "fault")


class ShouldPrepareTests(unittest.TestCase):
    def test_false_well_before_the_roll(self):
        self.assertFalse(
            should_prepare(frames_in_segment=100, max_frames=27000)
        )

    def test_true_within_the_lead_window(self):
        self.assertTrue(
            should_prepare(
                frames_in_segment=27000 - PREPARE_LEAD_FRAMES, max_frames=27000
            )
        )

    def test_true_at_max_frames(self):
        self.assertTrue(should_prepare(frames_in_segment=27000, max_frames=27000))

    def test_false_one_frame_before_the_window_opens(self):
        self.assertFalse(
            should_prepare(
                frames_in_segment=27000 - PREPARE_LEAD_FRAMES - 1, max_frames=27000
            )
        )

    def test_lead_frames_larger_than_max_frames_clamps_to_zero(self):
        # Degenerate config: should_prepare becomes true immediately, not an error.
        self.assertTrue(
            should_prepare(frames_in_segment=0, max_frames=10, lead_frames=999)
        )


class ReconcilePartFilesTests(unittest.TestCase):
    def test_single_part_is_found(self):
        with tempfile.TemporaryDirectory() as directory:
            part_base = Path(directory) / "recording_X_part0007"
            (Path(directory) / "recording_X_part0007-0000.avi").touch()
            result = reconcile_part_files(part_base)
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0].name, "recording_X_part0007-0000.avi")

    def test_sdk_net_firing_produces_two_parts_in_order(self):
        with tempfile.TemporaryDirectory() as directory:
            part_base = Path(directory) / "recording_X_part0007"
            (Path(directory) / "recording_X_part0007-0001.avi").touch()
            (Path(directory) / "recording_X_part0007-0000.avi").touch()
            result = reconcile_part_files(part_base)
            self.assertEqual([p.name for p in result], [
                "recording_X_part0007-0000.avi",
                "recording_X_part0007-0001.avi",
            ])

    def test_missing_part_base_returns_empty_not_raise(self):
        with tempfile.TemporaryDirectory() as directory:
            part_base = Path(directory) / "never_written"
            self.assertEqual(reconcile_part_files(part_base), [])

    def test_does_not_match_a_different_parts_prefix(self):
        with tempfile.TemporaryDirectory() as directory:
            part_base = Path(directory) / "recording_X_part0007"
            (Path(directory) / "recording_X_part0008-0000.avi").touch()
            self.assertEqual(reconcile_part_files(part_base), [])

    def test_nonexistent_parent_directory_returns_empty(self):
        part_base = Path("/definitely/does/not/exist/recording_X_part0000")
        self.assertEqual(reconcile_part_files(part_base), [])


class PlanRenamesTests(unittest.TestCase):
    def test_single_part_maps_to_the_requested_index(self):
        plans = plan_renames(
            [Path("/tmp/incomplete/recording_X_part0007-0000.avi")],
            next_segment_index=7,
            video_final=lambda i: Path(f"/tmp/recording_X-{i:04d}.avi"),
        )
        self.assertEqual(plans, [
            RenamePlan(
                source=Path("/tmp/incomplete/recording_X_part0007-0000.avi"),
                destination=Path("/tmp/recording_X-0007.avi"),
            )
        ])

    def test_two_parts_consume_two_consecutive_indices(self):
        plans = plan_renames(
            [
                Path("/tmp/incomplete/recording_X_part0007-0000.avi"),
                Path("/tmp/incomplete/recording_X_part0007-0001.avi"),
            ],
            next_segment_index=7,
            video_final=lambda i: Path(f"/tmp/recording_X-{i:04d}.avi"),
        )
        destinations = [p.destination.name for p in plans]
        self.assertEqual(destinations, ["recording_X-0007.avi", "recording_X-0008.avi"])

    def test_never_maps_two_sources_to_one_destination(self):
        plans = plan_renames(
            [Path("a-0000.avi"), Path("a-0001.avi"), Path("a-0002.avi")],
            next_segment_index=0,
            video_final=lambda i: Path(f"final-{i}.avi"),
        )
        destinations = [p.destination for p in plans]
        self.assertEqual(len(destinations), len(set(destinations)))

    def test_empty_input_produces_no_plans(self):
        self.assertEqual(
            plan_renames([], next_segment_index=0, video_final=lambda i: Path(f"{i}")),
            [],
        )


if __name__ == "__main__":
    unittest.main()
