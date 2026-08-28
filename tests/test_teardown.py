import unittest

from backend.teardown import assess_teardown_readiness


class AssessTeardownReadinessTests(unittest.TestCase):
    def test_a_dead_thread_is_safe_to_release(self):
        decision = assess_teardown_readiness(acquisition_thread_alive=False)
        self.assertTrue(decision.safe_to_release)
        self.assertEqual(decision.reason, "")

    def test_a_still_alive_thread_is_not_safe_to_release(self):
        decision = assess_teardown_readiness(acquisition_thread_alive=True)
        self.assertFalse(decision.safe_to_release)
        self.assertIn("acquisition thread", decision.reason)


if __name__ == "__main__":
    unittest.main()
