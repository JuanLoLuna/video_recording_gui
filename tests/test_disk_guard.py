import unittest

from backend.disk_guard import (
    DEFAULT_BYTES_PER_HOUR,
    DiskSample,
    assess_disk,
    sample_disk_usage,
)


def make_sample(free_gib: float, *, total_gib: float = 2000.0) -> DiskSample:
    free_bytes = int(free_gib * 1024**3)
    total_bytes = int(total_gib * 1024**3)
    return DiskSample(
        total_bytes=total_bytes,
        used_bytes=total_bytes - free_bytes,
        free_bytes=free_bytes,
        at_s=0.0,
    )


class AssessDiskTests(unittest.TestCase):
    def test_plenty_of_space_is_safe(self):
        verdict = assess_disk(make_sample(1000.0))
        self.assertEqual(verdict.level, "safe")
        self.assertFalse(verdict.recording_blocked)
        self.assertFalse(verdict.requires_confirmation)

    def test_below_warn_hours_is_a_warning(self):
        # ~24h at the default rate is ~147 GiB; pick something just under it.
        hours_20 = DEFAULT_BYTES_PER_HOUR * 20 / 1024**3
        verdict = assess_disk(make_sample(hours_20))
        self.assertEqual(verdict.level, "warning")
        self.assertFalse(verdict.recording_blocked)
        self.assertTrue(verdict.requires_confirmation)

    def test_below_critical_hours_is_blocked(self):
        hours_3 = DEFAULT_BYTES_PER_HOUR * 3 / 1024**3
        verdict = assess_disk(make_sample(hours_3))
        self.assertEqual(verdict.level, "danger")
        self.assertTrue(verdict.recording_blocked)

    def test_min_free_bytes_floor_dominates_even_with_high_hours_remaining(self):
        # Tiny bytes_per_hour makes hours_remaining huge, but min_free_bytes
        # should still block an almost-full disk.
        verdict = assess_disk(
            make_sample(1.0), bytes_per_hour=1, min_free_bytes=20 * 1024**3
        )
        self.assertEqual(verdict.level, "danger")
        self.assertTrue(verdict.recording_blocked)

    def test_zero_bytes_per_hour_does_not_raise(self):
        verdict = assess_disk(make_sample(1000.0), bytes_per_hour=0)
        self.assertEqual(verdict.hours_remaining, float("inf"))
        self.assertEqual(verdict.level, "safe")

    def test_hours_remaining_matches_hand_computed_value(self):
        sample = make_sample(free_gib=100.0)
        verdict = assess_disk(sample, bytes_per_hour=1024**3)  # 1 GiB/hr
        self.assertAlmostEqual(verdict.hours_remaining, 100.0, delta=0.01)

    def test_the_real_planning_case_1_2tb_free_needing_1_58tb_is_blocked(self):
        # From the plan: 1.58 TB needed over 10 days, 1.2 TB free.
        verdict = assess_disk(make_sample(1200.0), warn_hours=240.0, critical_hours=48.0)
        self.assertTrue(verdict.recording_blocked or verdict.requires_confirmation)

    def test_sample_disk_usage_wraps_an_injected_callable(self):
        class FakeUsage:
            total = 2000 * 1024**3
            used = 1000 * 1024**3
            free = 1000 * 1024**3

        sample = sample_disk_usage("/fake/path", at_s=5.0, disk_usage=lambda p: FakeUsage())
        self.assertEqual(sample.free_bytes, 1000 * 1024**3)
        self.assertEqual(sample.at_s, 5.0)


if __name__ == "__main__":
    unittest.main()
