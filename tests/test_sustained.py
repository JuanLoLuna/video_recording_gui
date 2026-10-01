import unittest

from backend.sustained import SustainedCondition


class SustainedConditionTests(unittest.TestCase):
    def test_a_brief_spike_never_fires(self):
        # The dock stall: high for ~5 s, then drained.
        cond = SustainedCondition(10.0)
        fired = [cond.update(t < 5, float(t)) for t in range(30)]
        self.assertFalse(any(fired))

    def test_fires_only_after_holding_for_the_full_time(self):
        cond = SustainedCondition(10.0)
        results = [cond.update(True, float(t)) for t in range(0, 13)]
        self.assertEqual(results.index(True), 10)
        self.assertTrue(all(results[10:]))

    def test_any_dip_restarts_the_clock(self):
        cond = SustainedCondition(10.0)
        for t in range(0, 9):
            cond.update(True, float(t))
        cond.update(False, 9.0)  # drained for one sample
        self.assertFalse(cond.update(True, 10.0))
        self.assertFalse(cond.update(True, 19.0))
        self.assertTrue(cond.update(True, 20.0))

    def test_reset_and_active_since(self):
        cond = SustainedCondition(5.0)
        self.assertIsNone(cond.active_since)
        cond.update(True, 3.0)
        self.assertEqual(cond.active_since, 3.0)
        cond.reset()
        self.assertIsNone(cond.active_since)

    def test_zero_hold_fires_immediately(self):
        self.assertTrue(SustainedCondition(0.0).update(True, 1.0))


if __name__ == "__main__":
    unittest.main()
