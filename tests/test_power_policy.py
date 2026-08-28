import unittest

from backend.power_policy import PowerPolicyState, next_power_action


class NextPowerActionTests(unittest.TestCase):
    def test_not_recording_and_not_paused_returns_none_and_resets_state(self):
        state = PowerPolicyState(consecutive_blocked=5)
        new_state, decision = next_power_action(
            state, recording_blocked=True, recording=False
        )
        self.assertEqual(decision.action, "none")
        self.assertEqual(new_state, PowerPolicyState())

    def test_stays_none_below_debounce_threshold_while_blocked(self):
        state = PowerPolicyState()
        for _ in range(19):
            state, decision = next_power_action(
                state, recording_blocked=True, recording=True, debounce_samples=20
            )
            self.assertEqual(decision.action, "none")
        self.assertEqual(state.consecutive_blocked, 19)

    def test_pauses_after_debounce_threshold_of_consecutive_blocked_samples(self):
        state = PowerPolicyState()
        for _ in range(19):
            state, _ = next_power_action(
                state, recording_blocked=True, recording=True, debounce_samples=20
            )
        state, decision = next_power_action(
            state,
            recording_blocked=True,
            recording=True,
            reason="Energy Saver is on",
            debounce_samples=20,
        )
        self.assertEqual(decision.action, "pause")
        self.assertEqual(decision.reason, "Energy Saver is on")
        self.assertTrue(state.paused)
        self.assertEqual(state.consecutive_blocked, 0)

    def test_a_single_safe_sample_resets_the_blocked_counter(self):
        state = PowerPolicyState()
        for _ in range(15):
            state, _ = next_power_action(
                state, recording_blocked=True, recording=True, debounce_samples=20
            )
        state, decision = next_power_action(
            state, recording_blocked=False, recording=True, debounce_samples=20
        )
        self.assertEqual(decision.action, "none")
        self.assertEqual(state.consecutive_blocked, 0)

    def test_resume_requires_debounce_threshold_of_consecutive_safe_samples_while_paused(
        self,
    ):
        state = PowerPolicyState(paused=True)
        for _ in range(19):
            state, decision = next_power_action(
                state, recording_blocked=False, recording=False, debounce_samples=20
            )
            self.assertEqual(decision.action, "none")
            self.assertTrue(state.paused)
        state, decision = next_power_action(
            state, recording_blocked=False, recording=False, debounce_samples=20
        )
        self.assertEqual(decision.action, "resume")
        self.assertEqual(state, PowerPolicyState())

    def test_a_single_blocked_sample_resets_the_safe_counter_while_paused(self):
        state = PowerPolicyState(paused=True)
        for _ in range(15):
            state, _ = next_power_action(
                state, recording_blocked=False, recording=False, debounce_samples=20
            )
        state, decision = next_power_action(
            state, recording_blocked=True, recording=False, debounce_samples=20
        )
        self.assertEqual(decision.action, "none")
        self.assertEqual(state.consecutive_safe, 0)
        self.assertTrue(state.paused)

    def test_custom_debounce_samples_is_honored(self):
        state = PowerPolicyState()
        state, decision = next_power_action(
            state, recording_blocked=True, recording=True, debounce_samples=1
        )
        self.assertEqual(decision.action, "pause")

    def test_paused_state_ignores_the_recording_flag(self):
        # `recording` is naturally False for the whole pause window; the
        # policy's own `paused` flag -- not the caller's `recording` flag --
        # must be what drives the resume-debounce branch.
        state = PowerPolicyState(paused=True)
        state, decision = next_power_action(
            state, recording_blocked=False, recording=True, debounce_samples=20
        )
        self.assertEqual(decision.action, "none")
        self.assertTrue(state.paused)


if __name__ == "__main__":
    unittest.main()
