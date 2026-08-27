import unittest

from backend.acquisition_watchdog import (
    AcquisitionWatchdog,
    WatchdogConfig,
    watchdog_config_for_frame_rate,
)


def make_watchdog(now=0.0, **overrides):
    config = WatchdogConfig(
        frame_period_s=overrides.pop("frame_period_s", 1 / 30),
        grab_timeout_s=overrides.pop("grab_timeout_s", 0.2),
        stall_timeout_s=overrides.pop("stall_timeout_s", 5.0),
        consecutive_error_limit=overrides.pop("consecutive_error_limit", 10),
        backoff_initial_s=overrides.pop("backoff_initial_s", 0.05),
        backoff_max_s=overrides.pop("backoff_max_s", 5.0),
        backoff_factor=overrides.pop("backoff_factor", 2.0),
    )
    return AcquisitionWatchdog(config, now=now)


class WatchdogConfigForFrameRateTests(unittest.TestCase):
    def test_config_for_30_fps_bounds_the_grab_timeout(self):
        config = watchdog_config_for_frame_rate(30.0)
        self.assertGreaterEqual(config.grab_timeout_s, 0.2)
        self.assertLessEqual(config.grab_timeout_s, 2.0)

    def test_config_clamps_extreme_frame_rates(self):
        low_fps = watchdog_config_for_frame_rate(1.0)
        self.assertEqual(low_fps.grab_timeout_s, 2.0)
        high_fps = watchdog_config_for_frame_rate(200.0)
        self.assertEqual(high_fps.grab_timeout_s, 0.2)

    def test_grab_timeout_is_always_shorter_than_stall_timeout(self):
        for fps in (1.0, 5.0, 30.0, 60.0, 120.0, 240.0):
            config = watchdog_config_for_frame_rate(fps)
            self.assertLess(config.grab_timeout_s, config.stall_timeout_s)

    def test_zero_or_negative_fps_falls_back_to_a_sane_default(self):
        config = watchdog_config_for_frame_rate(0.0)
        self.assertGreater(config.frame_period_s, 0.0)
        negative = watchdog_config_for_frame_rate(-5.0)
        self.assertGreater(negative.frame_period_s, 0.0)


class AcquisitionWatchdogTests(unittest.TestCase):
    def test_healthy_frames_never_request_recovery(self):
        wd = make_watchdog(now=0.0)
        for t in range(1, 100):
            wd.note_frame_ok(now=float(t) * 0.033)
            decision = wd.poll(now=float(t) * 0.033)
            self.assertEqual(decision.action, "continue")

    def test_a_single_error_sleeps_instead_of_spinning(self):
        wd = make_watchdog(now=0.0, backoff_initial_s=0.05)
        decision = wd.note_error(now=0.01, error="boom")
        self.assertEqual(decision.action, "sleep")
        self.assertGreater(decision.sleep_s, 0.0)
        self.assertEqual(decision.sleep_s, 0.05)

    def test_backoff_grows_exponentially_and_is_capped(self):
        wd = make_watchdog(
            now=0.0,
            backoff_initial_s=0.05,
            backoff_max_s=5.0,
            backoff_factor=2.0,
            consecutive_error_limit=100,
        )
        sleeps = []
        t = 0.0
        for _ in range(10):
            decision = wd.note_error(now=t, error="boom")
            sleeps.append(decision.sleep_s)
            t += 0.01
        self.assertEqual(sleeps[0], 0.05)
        self.assertEqual(sleeps[1], 0.1)
        self.assertEqual(sleeps[2], 0.2)
        self.assertEqual(sleeps[-1], 5.0)

    def test_backoff_resets_after_a_good_frame(self):
        wd = make_watchdog(now=0.0, backoff_initial_s=0.05, consecutive_error_limit=100)
        wd.note_error(now=0.1, error="boom")
        wd.note_error(now=0.2, error="boom")
        wd.note_frame_ok(now=0.3)
        decision = wd.note_error(now=0.4, error="boom")
        self.assertEqual(decision.sleep_s, 0.05)

    def test_consecutive_error_limit_triggers_reinit(self):
        wd = make_watchdog(now=0.0, consecutive_error_limit=3)
        t = 0.0
        for _ in range(2):
            decision = wd.note_error(now=t, error="boom")
            self.assertEqual(decision.action, "sleep")
            t += 0.1
        decision = wd.note_error(now=t, error="boom")
        self.assertEqual(decision.action, "reinit")

    def test_a_silent_stall_with_no_errors_triggers_reinit(self):
        wd = make_watchdog(now=0.0, stall_timeout_s=5.0)
        wd.note_frame_ok(now=0.0)
        decision = wd.poll(now=5.1)
        self.assertEqual(decision.action, "reinit")
        self.assertGreaterEqual(decision.stalled_for_s, 5.0)

    def test_stall_below_threshold_does_not_reinit(self):
        wd = make_watchdog(now=0.0, stall_timeout_s=5.0)
        wd.note_frame_ok(now=0.0)
        decision = wd.poll(now=4.9)
        self.assertEqual(decision.action, "continue")

    def test_failed_reinit_retries_forever_and_never_gives_up(self):
        wd = make_watchdog(now=0.0, consecutive_error_limit=1, backoff_max_s=5.0)
        wd.note_error(now=0.0, error="boom")  # triggers reinit at limit=1
        t = 0.0
        for _ in range(10_000):
            decision = wd.note_reinit_result(now=t, ok=False)
            self.assertIn(decision.action, ("sleep", "continue"))
            self.assertNotEqual(decision.action, "give_up")
            t += decision.sleep_s or 0.01

    def test_config_property_returns_the_configured_values(self):
        wd = make_watchdog(now=0.0, grab_timeout_s=0.25)
        self.assertEqual(wd.config.grab_timeout_s, 0.25)

    def test_successful_reinit_returns_to_healthy_and_resets_counters(self):
        wd = make_watchdog(now=0.0, consecutive_error_limit=1)
        wd.note_error(now=0.0, error="boom")
        decision = wd.note_reinit_result(now=1.0, ok=True)
        self.assertEqual(decision.action, "continue")
        self.assertEqual(decision.consecutive_errors, 0)
        self.assertEqual(decision.reinit_attempt, 0)
        # Watchdog is healthy again -- a poll well within stall_timeout is fine.
        self.assertEqual(wd.poll(now=1.1).action, "continue")


if __name__ == "__main__":
    unittest.main()
