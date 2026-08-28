import unittest

from backend.timeline import TimelineBaseline, compute_wall_mono_skew_s, is_timeline_break


class ComputeWallMonoSkewSTests(unittest.TestCase):
    def test_clocks_agreeing_exactly_yields_zero_skew(self):
        baseline = TimelineBaseline(session_start_wall_s=1000.0, session_start_mono_s=50.0)
        skew = compute_wall_mono_skew_s(baseline, wall_s=1010.0, mono_s=60.0)
        self.assertEqual(skew, 0.0)

    def test_a_wall_clock_step_forward_shows_up_as_positive_skew(self):
        baseline = TimelineBaseline(session_start_wall_s=1000.0, session_start_mono_s=50.0)
        # 10s of monotonic time passed, but wall clock jumped by 15s (NTP step).
        skew = compute_wall_mono_skew_s(baseline, wall_s=1015.0, mono_s=60.0)
        self.assertEqual(skew, 5.0)

    def test_a_wall_clock_step_backward_shows_up_as_negative_skew(self):
        baseline = TimelineBaseline(session_start_wall_s=1000.0, session_start_mono_s=50.0)
        skew = compute_wall_mono_skew_s(baseline, wall_s=1005.0, mono_s=60.0)
        self.assertEqual(skew, -5.0)

    def test_skew_is_independent_of_the_absolute_baseline_values(self):
        b1 = TimelineBaseline(session_start_wall_s=0.0, session_start_mono_s=0.0)
        b2 = TimelineBaseline(session_start_wall_s=1_700_000_000.0, session_start_mono_s=99.0)
        skew1 = compute_wall_mono_skew_s(b1, wall_s=10.0, mono_s=9.5)
        skew2 = compute_wall_mono_skew_s(
            b2, wall_s=1_700_000_010.0, mono_s=99.0 + 9.5
        )
        self.assertEqual(skew1, skew2)


class IsTimelineBreakTests(unittest.TestCase):
    def test_sub_threshold_change_is_not_a_break(self):
        self.assertFalse(is_timeline_break(0.5, 0.0, threshold_s=1.0))

    def test_at_threshold_change_is_a_break(self):
        self.assertTrue(is_timeline_break(1.0, 0.0, threshold_s=1.0))

    def test_a_negative_going_jump_is_also_a_break(self):
        self.assertTrue(is_timeline_break(-2.0, 0.0, threshold_s=1.0))

    def test_default_threshold_matches_heartbeat_pys_default(self):
        self.assertFalse(is_timeline_break(0.9, 0.0))
        self.assertTrue(is_timeline_break(1.0, 0.0))


if __name__ == "__main__":
    unittest.main()
