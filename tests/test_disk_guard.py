import unittest

from backend.disk_guard import (
    DEFAULT_BYTES_PER_HOUR,
    DEFAULT_CRITICAL_HOURS,
    DEFAULT_WARN_HOURS,
    PLANNED_HOURS_ENV,
    MJPEG_SIZE_FRACTION_ESTIMATE,
    DiskSample,
    StreamRate,
    assess_disk,
    estimate_bytes_per_hour,
    resolve_planned_hours,
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
        # 30h worth at the default rate: safely above DEFAULT_WARN_HOURS
        # regardless of what DEFAULT_BYTES_PER_HOUR currently is.
        free_gib = DEFAULT_BYTES_PER_HOUR * 30 / 1024**3
        verdict = assess_disk(make_sample(free_gib, total_gib=free_gib * 2))
        self.assertEqual(verdict.level, "safe")
        self.assertFalse(verdict.recording_blocked)
        self.assertFalse(verdict.requires_confirmation)

    def test_below_warn_hours_is_a_warning(self):
        # Between the critical and warn thresholds, whatever they are set to.
        between = (DEFAULT_CRITICAL_HOURS + DEFAULT_WARN_HOURS) / 2
        hours_between = DEFAULT_BYTES_PER_HOUR * between / 1024**3
        verdict = assess_disk(make_sample(hours_between))
        self.assertEqual(verdict.level, "warning")
        self.assertFalse(verdict.recording_blocked)
        self.assertTrue(verdict.requires_confirmation)

    def test_below_critical_hours_requires_confirmation_but_does_not_block(self):
        # A rate-based projection is a judgment call, not a fact about the
        # disk right now (bytes_per_hour is an estimate) -- confirmable,
        # not a hard block. Only the absolute min_free_bytes floor blocks.
        hours_1 = DEFAULT_BYTES_PER_HOUR * (DEFAULT_CRITICAL_HOURS / 2) / 1024**3
        verdict = assess_disk(make_sample(hours_1))
        self.assertEqual(verdict.level, "danger")
        self.assertFalse(verdict.recording_blocked)
        self.assertTrue(verdict.requires_confirmation)

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


# The two cameras on the rig: Firefly 720x540 and Blackfly S 1280x1024, Mono8, 30 fps.
FIREFLY_30 = StreamRate(720, 540, 1, 30.0)
BLACKFLY_30 = StreamRate(1280, 1024, 1, 30.0)


class EstimateBytesPerHourTests(unittest.TestCase):
    def test_single_uncompressed_camera_matches_hand_computation(self):
        self.assertEqual(estimate_bytes_per_hour([FIREFLY_30]), 720 * 540 * 30 * 3600)
        self.assertEqual(estimate_bytes_per_hour([BLACKFLY_30]), 141_557_760_000)

    def test_two_different_sensors_are_summed(self):
        # The probe measured ~184 GB/h for this pair.
        total = estimate_bytes_per_hour([FIREFLY_30, BLACKFLY_30])
        self.assertEqual(total, 41_990_400_000 + 141_557_760_000)
        self.assertAlmostEqual(total / 1e9, 183.5, delta=0.1)

    def test_compressed_streams_use_the_mjpeg_fraction(self):
        raw = estimate_bytes_per_hour([BLACKFLY_30])
        compressed = estimate_bytes_per_hour([StreamRate(1280, 1024, 1, 30.0, compressed=True)])
        self.assertAlmostEqual(compressed, raw * MJPEG_SIZE_FRACTION_ESTIMATE, delta=1)

    def test_mixed_codecs_per_camera(self):
        total = estimate_bytes_per_hour(
            [FIREFLY_30, StreamRate(1280, 1024, 1, 30.0, compressed=True)],
            mjpeg_fraction=0.5,
        )
        self.assertEqual(total, 41_990_400_000 + 70_778_880_000)

    def test_rate_scales_with_fps(self):
        double = estimate_bytes_per_hour([StreamRate(1280, 1024, 1, 60.0)])
        self.assertEqual(double, 2 * 141_557_760_000)

    def test_no_streams_falls_back_to_the_default(self):
        self.assertEqual(estimate_bytes_per_hour([]), DEFAULT_BYTES_PER_HOUR)


class RigScenarioTests(unittest.TestCase):
    """The thresholds against the drives this study actually uses."""

    def verdict(self, free_gb, streams):
        rate = estimate_bytes_per_hour(streams)
        return assess_disk(make_sample(free_gb * 1e9 / 1024**3), bytes_per_hour=rate)

    def test_4tb_drive_is_safe_for_two_uncompressed_cameras(self):
        # ~21.8 h of video: used to warn on every start under the 24 h threshold.
        verdict = self.verdict(4000, [FIREFLY_30, BLACKFLY_30])
        self.assertEqual(verdict.level, "safe")
        self.assertFalse(verdict.requires_confirmation)

    def test_internal_drive_with_700gb_asks_for_confirmation_when_uncompressed(self):
        verdict = self.verdict(700, [FIREFLY_30, BLACKFLY_30])  # ~3.8 h
        self.assertEqual(verdict.level, "warning")
        self.assertTrue(verdict.requires_confirmation)
        self.assertFalse(verdict.recording_blocked)

    def test_internal_drive_is_safe_with_mjpeg(self):
        streams = [
            StreamRate(720, 540, 1, 30.0, compressed=True),
            StreamRate(1280, 1024, 1, 30.0, compressed=True),
        ]
        self.assertEqual(self.verdict(700, streams).level, "safe")

    def test_less_than_two_sessions_of_room_is_danger(self):
        verdict = self.verdict(300, [FIREFLY_30, BLACKFLY_30])  # ~1.6 h
        self.assertEqual(verdict.level, "danger")
        self.assertFalse(verdict.recording_blocked)  # still confirmable


class FixedThresholdTests(unittest.TestCase):
    """Concrete numbers, so the tests fail if the defaults are changed by accident."""

    RATE = 100 * 1024**3  # 100 GiB per hour

    def hours(self, hours):
        return assess_disk(make_sample(hours * 100.0), bytes_per_hour=self.RATE)

    def test_defaults_are_8h_warn_and_2h_critical(self):
        self.assertEqual(DEFAULT_WARN_HOURS, 8.0)
        self.assertEqual(DEFAULT_CRITICAL_HOURS, 2.0)

    def test_boundaries(self):
        self.assertEqual(self.hours(8.5).level, "safe")
        self.assertEqual(self.hours(7.5).level, "warning")
        self.assertEqual(self.hours(2.5).level, "warning")
        self.assertEqual(self.hours(1.5).level, "danger")

    def test_messages_say_estimated_not_measured_and_not_multi_day(self):
        for hours in (7.5, 1.5, 9.0):
            reason = self.hours(hours).reason
            self.assertIn("estimated", reason)
            self.assertNotIn("measured", reason)
            self.assertNotIn("multi-day", reason)


class PlannedHoursTests(unittest.TestCase):
    RATE = 100 * 1024**3

    def verdict(self, free_hours, planned):
        return assess_disk(
            make_sample(free_hours * 100.0), bytes_per_hour=self.RATE, planned_hours=planned
        )

    def test_a_long_run_that_will_not_fit_asks_even_when_far_above_warn_hours(self):
        # 10 days planned, ~22 h of room: "safe" by the default thresholds.
        self.assertEqual(self.verdict(22.0, None).level, "safe")
        verdict = self.verdict(22.0, 240.0)
        self.assertEqual(verdict.level, "warning")
        self.assertTrue(verdict.requires_confirmation)
        self.assertFalse(verdict.recording_blocked)
        self.assertIn("240 h", verdict.reason)

    def test_a_run_that_fits_is_not_nagged(self):
        self.assertEqual(self.verdict(300.0, 240.0).level, "safe")

    def test_critical_still_wins_over_planned(self):
        self.assertEqual(self.verdict(1.0, 240.0).level, "danger")

    def test_min_free_floor_still_blocks(self):
        verdict = assess_disk(make_sample(1.0), bytes_per_hour=1, planned_hours=1.0)
        self.assertTrue(verdict.recording_blocked)

    def test_resolve_planned_hours(self):
        self.assertEqual(resolve_planned_hours({PLANNED_HOURS_ENV: "240"}), 240.0)
        self.assertEqual(resolve_planned_hours({PLANNED_HOURS_ENV: " 1.5 "}), 1.5)
        for bad in ("", "abc", "0", "-3", "nan"):
            self.assertIsNone(resolve_planned_hours({PLANNED_HOURS_ENV: bad}), bad)
        self.assertIsNone(resolve_planned_hours({}))


if __name__ == "__main__":
    unittest.main()
