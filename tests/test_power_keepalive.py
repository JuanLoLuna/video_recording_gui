import unittest

from backend.power_keepalive import (
    ES_AWAYMODE_REQUIRED,
    ES_CONTINUOUS,
    ES_DISPLAY_REQUIRED,
    ES_SYSTEM_REQUIRED,
    KeepAwakeRequest,
    KeepAwakeState,
    assess_keepalive,
    build_execution_state_flags,
    release_execution_state_flags,
)


class ExecutionStateFlagsTests(unittest.TestCase):
    def test_recording_request_sets_continuous_and_system_flags(self):
        flags = build_execution_state_flags(
            KeepAwakeRequest(system=True, display=False, away_mode=False)
        )
        self.assertEqual(flags, ES_CONTINUOUS | ES_SYSTEM_REQUIRED)

    def test_away_mode_adds_its_flag(self):
        flags = build_execution_state_flags(
            KeepAwakeRequest(system=True, display=False, away_mode=True)
        )
        self.assertEqual(flags, ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_AWAYMODE_REQUIRED)

    def test_display_flag_is_opt_in(self):
        default_flags = build_execution_state_flags(KeepAwakeRequest())
        self.assertEqual(default_flags & ES_DISPLAY_REQUIRED, 0)

        explicit_flags = build_execution_state_flags(
            KeepAwakeRequest(system=True, display=True, away_mode=False)
        )
        self.assertEqual(explicit_flags, ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED)

    def test_release_flags_are_continuous_only(self):
        self.assertEqual(release_execution_state_flags(), ES_CONTINUOUS)


class AssessKeepaliveTests(unittest.TestCase):
    def test_unsupported_platform_is_neutral_and_never_blocks(self):
        result = assess_keepalive(KeepAwakeState(supported=False), recording=True)
        self.assertEqual(result.level, "neutral")

    def test_inactive_while_recording_is_danger(self):
        result = assess_keepalive(
            KeepAwakeState(supported=True, active=False), recording=True
        )
        self.assertEqual(result.level, "danger")
        self.assertIn("NOT active", result.summary)

    def test_inactive_while_idle_is_not_a_warning(self):
        result = assess_keepalive(
            KeepAwakeState(supported=True, active=False), recording=False
        )
        self.assertEqual(result.level, "neutral")

    def test_active_with_away_mode_while_recording_is_safe(self):
        result = assess_keepalive(
            KeepAwakeState(supported=True, active=True, away_mode_granted=True),
            recording=True,
        )
        self.assertEqual(result.level, "safe")

    def test_away_mode_denied_downgrades_to_warning_mentioning_modern_standby(self):
        result = assess_keepalive(
            KeepAwakeState(supported=True, active=True, away_mode_granted=False),
            recording=True,
        )
        self.assertEqual(result.level, "warning")
        self.assertIn("Modern Standby", result.reason)

    def test_error_state_is_warning(self):
        result = assess_keepalive(
            KeepAwakeState(supported=True, error="boom"), recording=True
        )
        self.assertEqual(result.level, "warning")
        self.assertEqual(result.reason, "boom")


if __name__ == "__main__":
    unittest.main()
