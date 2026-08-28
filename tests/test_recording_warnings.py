import unittest

from backend.recording_warnings import RecordingWarningTracker, format_duration_s


class RecordingWarningTrackerTests(unittest.TestCase):
    def test_no_events_is_hidden(self):
        tracker = RecordingWarningTracker()
        state = tracker.summarize(now_s=0.0)
        self.assertFalse(state.visible)
        self.assertEqual(state.level, "hidden")

    def test_issue_makes_the_banner_active_and_visible(self):
        tracker = RecordingWarningTracker()
        tracker.note_issue("2 frame gap(s)", now_s=10.0)
        state = tracker.summarize(now_s=10.5)
        self.assertTrue(state.visible)
        self.assertEqual(state.level, "active")
        self.assertIn("2 frame gap(s)", state.headline)

    def test_recovers_after_the_window_with_no_new_events(self):
        tracker = RecordingWarningTracker(recovered_window_s=15.0)
        tracker.note_issue("boom", now_s=0.0)
        self.assertEqual(tracker.summarize(now_s=14.9).level, "active")
        state = tracker.summarize(now_s=15.1)
        self.assertEqual(state.level, "recovered")
        self.assertTrue(state.visible)

    def test_dismiss_hides_while_still_in_the_same_level(self):
        tracker = RecordingWarningTracker(recovered_window_s=15.0)
        tracker.note_issue("boom", now_s=0.0)
        tracker.dismiss(now_s=1.0)
        state = tracker.summarize(now_s=2.0)
        self.assertFalse(state.visible)
        self.assertEqual(state.level, "active")  # still tracked, just hidden

    def test_dismissed_active_reopens_on_transition_to_recovered(self):
        tracker = RecordingWarningTracker(recovered_window_s=15.0)
        tracker.note_issue("boom", now_s=0.0)
        tracker.dismiss(now_s=1.0)
        # No new issues; once it crosses into "recovered" it should
        # resurface once, rather than staying hidden forever.
        state = tracker.summarize(now_s=15.1)
        self.assertEqual(state.level, "recovered")
        self.assertTrue(state.visible)

    def test_dismissed_recovered_stays_hidden_across_repeated_polls(self):
        tracker = RecordingWarningTracker(recovered_window_s=15.0)
        tracker.note_issue("boom", now_s=0.0)
        tracker.dismiss(now_s=20.0)  # dismissed while already "recovered"
        self.assertFalse(tracker.summarize(now_s=21.0).visible)
        self.assertFalse(tracker.summarize(now_s=25.0).visible)

    def test_new_issue_after_full_quiet_reopens_a_dismissed_banner(self):
        tracker = RecordingWarningTracker(recovered_window_s=15.0)
        tracker.note_issue("boom", now_s=0.0)
        tracker.dismiss(now_s=20.0)
        tracker.note_issue("boom again", now_s=100.0)
        state = tracker.summarize(now_s=100.5)
        self.assertTrue(state.visible)
        self.assertEqual(state.level, "active")
        self.assertIn("boom again", state.headline)

    def test_total_events_accumulates_across_the_session(self):
        tracker = RecordingWarningTracker(recovered_window_s=15.0)
        tracker.note_issue("a", now_s=0.0)
        tracker.note_issue("b", now_s=1.0)
        tracker.note_issue("c", now_s=2.0)
        self.assertEqual(tracker.summarize(now_s=2.5).total_events, 3)

    def test_reset_clears_everything(self):
        tracker = RecordingWarningTracker()
        tracker.note_issue("boom", now_s=0.0)
        tracker.dismiss(now_s=1.0)
        tracker.reset()
        state = tracker.summarize(now_s=2.0)
        self.assertFalse(state.visible)
        self.assertEqual(state.level, "hidden")
        self.assertEqual(state.total_events, 0)

    def test_continued_activity_keeps_it_active_not_recovered(self):
        tracker = RecordingWarningTracker(recovered_window_s=15.0)
        tracker.note_issue("boom", now_s=0.0)
        tracker.note_issue("boom", now_s=10.0)
        # 10s after the *second* event, still well within the window.
        state = tracker.summarize(now_s=20.0)
        self.assertEqual(state.level, "active")

    def test_healthy_for_s_is_none_without_a_known_session_start(self):
        tracker = RecordingWarningTracker()
        self.assertIsNone(tracker.summarize(now_s=100.0).healthy_for_s)

    def test_healthy_for_s_counts_from_session_start_with_no_issues(self):
        tracker = RecordingWarningTracker()
        tracker.reset(now_s=1000.0)
        state = tracker.summarize(now_s=1090.0)
        self.assertEqual(state.healthy_for_s, 90.0)

    def test_healthy_for_s_is_zero_while_active(self):
        tracker = RecordingWarningTracker(recovered_window_s=15.0)
        tracker.reset(now_s=0.0)
        tracker.note_issue("boom", now_s=5.0)
        state = tracker.summarize(now_s=10.0)
        self.assertEqual(state.level, "active")
        self.assertEqual(state.healthy_for_s, 0.0)

    def test_healthy_for_s_counts_from_the_last_issue_once_recovered(self):
        tracker = RecordingWarningTracker(recovered_window_s=15.0)
        tracker.reset(now_s=0.0)
        tracker.note_issue("boom", now_s=5.0)
        state = tracker.summarize(now_s=25.0)
        self.assertEqual(state.level, "recovered")
        self.assertEqual(state.healthy_for_s, 20.0)


class FormatDurationSTests(unittest.TestCase):
    def test_seconds_only(self):
        self.assertEqual(format_duration_s(45), "45s")

    def test_minutes_and_seconds(self):
        self.assertEqual(format_duration_s(90), "1m 30s")

    def test_hours_and_minutes_drops_seconds(self):
        self.assertEqual(format_duration_s(3 * 3600 + 61), "3h 1m")

    def test_days_and_hours_drops_minutes(self):
        self.assertEqual(format_duration_s(3 * 86_400 + 4 * 3600 + 30 * 60), "3d 4h")

    def test_negative_clamps_to_zero(self):
        self.assertEqual(format_duration_s(-5), "0s")


if __name__ == "__main__":
    unittest.main()
