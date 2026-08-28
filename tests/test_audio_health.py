import unittest
from dataclasses import dataclass

from backend.audio_health import (
    AudioHealthCounters,
    plan_reconnect,
    resolve_device_index_by_name,
    silence_frames_for_gap,
)


@dataclass
class _FakeStatus:
    input_underflow: bool = False
    input_overflow: bool = False
    output_underflow: bool = False
    output_overflow: bool = False
    priming_output: bool = False


class AudioHealthCountersTests(unittest.TestCase):
    def test_note_status_counts_each_flag_independently(self):
        counters = AudioHealthCounters()
        counters.note_status(_FakeStatus(input_overflow=True))
        counters.note_status(_FakeStatus(input_overflow=True, output_underflow=True))
        self.assertEqual(counters.input_overflow, 2)
        self.assertEqual(counters.output_underflow, 1)
        self.assertEqual(counters.input_underflow, 0)

    def test_note_status_with_no_flags_set_counts_nothing(self):
        counters = AudioHealthCounters()
        counters.note_status(_FakeStatus())
        snap = counters.snapshot()
        self.assertEqual(snap.total_xruns, 0)

    def test_note_status_tolerates_an_object_missing_flag_attributes(self):
        counters = AudioHealthCounters()
        counters.note_status(object())
        self.assertEqual(counters.snapshot().total_xruns, 0)

    def test_snapshot_is_independent_of_later_mutation(self):
        counters = AudioHealthCounters()
        counters.note_status(_FakeStatus(input_overflow=True))
        snap = counters.snapshot()
        counters.note_status(_FakeStatus(input_overflow=True))
        self.assertEqual(snap.input_overflow, 1)
        self.assertEqual(counters.input_overflow, 2)

    def test_total_xruns_excludes_priming_output(self):
        counters = AudioHealthCounters()
        counters.note_status(_FakeStatus(priming_output=True))
        self.assertEqual(counters.snapshot().total_xruns, 0)
        self.assertEqual(counters.snapshot().priming_output, 1)

    def test_note_reconnect_and_note_silence_frames_accumulate(self):
        counters = AudioHealthCounters()
        counters.note_reconnect()
        counters.note_reconnect()
        counters.note_silence_frames(480)
        counters.note_silence_frames(120)
        snap = counters.snapshot()
        self.assertEqual(snap.reconnects, 2)
        self.assertEqual(snap.silence_frames_inserted, 600)

    def test_note_silence_frames_ignores_negative_values(self):
        counters = AudioHealthCounters()
        counters.note_silence_frames(-5)
        self.assertEqual(counters.snapshot().silence_frames_inserted, 0)


class SilenceFramesForGapTests(unittest.TestCase):
    def test_rounds_to_nearest_frame(self):
        self.assertEqual(silence_frames_for_gap(1.0, 48_000), 48_000)
        self.assertEqual(silence_frames_for_gap(0.5, 48_000), 24_000)

    def test_zero_or_negative_gap_yields_zero_frames(self):
        self.assertEqual(silence_frames_for_gap(0.0, 48_000), 0)
        self.assertEqual(silence_frames_for_gap(-1.0, 48_000), 0)

    def test_zero_samplerate_yields_zero_frames(self):
        self.assertEqual(silence_frames_for_gap(5.0, 0), 0)


class PlanReconnectTests(unittest.TestCase):
    def test_backoff_increases_by_attempt(self):
        p1 = plan_reconnect("Mic", 1)
        p2 = plan_reconnect("Mic", 2)
        self.assertLess(p1.backoff_s, p2.backoff_s)

    def test_backoff_caps_at_last_entry_for_high_attempt_numbers(self):
        capped = plan_reconnect("Mic", 999, backoffs_s=(0.2, 0.5, 1.0))
        self.assertEqual(capped.backoff_s, 1.0)

    def test_attempt_below_one_is_treated_as_attempt_one(self):
        p0 = plan_reconnect("Mic", 0)
        p1 = plan_reconnect("Mic", 1)
        self.assertEqual(p0.backoff_s, p1.backoff_s)

    def test_device_name_and_attempt_are_carried_through(self):
        plan = plan_reconnect("USB Mic (WASAPI)", 3)
        self.assertEqual(plan.device_name, "USB Mic (WASAPI)")
        self.assertEqual(plan.attempt, 3)


class ResolveDeviceIndexByNameTests(unittest.TestCase):
    def test_finds_matching_input_device_by_exact_name(self):
        devices = [
            {"name": "Built-in Mic", "max_input_channels": 2},
            {"name": "USB Mic", "max_input_channels": 1},
        ]
        self.assertEqual(resolve_device_index_by_name("USB Mic", devices), 1)

    def test_returns_none_when_name_not_found(self):
        devices = [{"name": "Built-in Mic", "max_input_channels": 2}]
        self.assertIsNone(resolve_device_index_by_name("USB Mic", devices))

    def test_ignores_a_same_named_device_with_no_input_channels(self):
        devices = [{"name": "USB Mic", "max_input_channels": 0}]
        self.assertIsNone(resolve_device_index_by_name("USB Mic", devices))

    def test_matches_after_reenumeration_shuffles_indices(self):
        before = [
            {"name": "Built-in Mic", "max_input_channels": 2},
            {"name": "USB Mic", "max_input_channels": 1},
        ]
        after_replug = [
            {"name": "USB Mic", "max_input_channels": 1},
            {"name": "Built-in Mic", "max_input_channels": 2},
        ]
        self.assertEqual(resolve_device_index_by_name("USB Mic", before), 1)
        self.assertEqual(resolve_device_index_by_name("USB Mic", after_replug), 0)

    def test_tolerates_whitespace_differences_in_stored_name(self):
        devices = [{"name": "USB Mic", "max_input_channels": 1}]
        self.assertEqual(resolve_device_index_by_name("  USB Mic  ", devices), 0)


if __name__ == "__main__":
    unittest.main()
