import unittest

from backend.audio_format import (
    WAV_RIFF_LIMIT_BYTES,
    AudioFormatChoice,
    AudioWriteBudget,
    choose_audio_format,
    riff_capacity_seconds,
    wav_bytes_per_second,
)


class ChooseAudioFormatTests(unittest.TestCase):
    def test_prefers_rf64_when_available(self):
        choice = choose_audio_format(
            available_formats={"WAV": "WAV", "RF64": "RF64", "W64": "W64"},
            samplerate=44100,
            channels=1,
        )
        self.assertEqual(choice.format, "RF64")
        self.assertTrue(choice.is_streaming_safe)
        self.assertIsNone(choice.max_bytes)

    def test_falls_back_to_w64_when_rf64_missing(self):
        choice = choose_audio_format(
            available_formats={"WAV": "WAV", "W64": "W64"},
            samplerate=44100,
            channels=1,
        )
        self.assertEqual(choice.format, "W64")
        self.assertTrue(choice.is_streaming_safe)

    def test_falls_back_to_plain_wav_when_only_wav_is_available(self):
        choice = choose_audio_format(
            available_formats={"WAV": "WAV"},
            samplerate=44100,
            channels=1,
        )
        self.assertIsNone(choice.format)
        self.assertFalse(choice.is_streaming_safe)
        self.assertIn("4 GiB", choice.reason)

    def test_listed_format_with_unsupported_subtype_is_skipped(self):
        choice = choose_audio_format(
            available_formats={"WAV": "WAV", "RF64": "RF64", "W64": "W64"},
            samplerate=44100,
            channels=1,
            subtype="PCM_16",
            format_checker=lambda fmt, sub: fmt != "RF64",
        )
        self.assertEqual(choice.format, "W64")

    def test_no_streaming_safe_format_supported_falls_back_to_wav(self):
        choice = choose_audio_format(
            available_formats={"WAV": "WAV", "RF64": "RF64", "W64": "W64"},
            samplerate=44100,
            channels=1,
            format_checker=lambda fmt, sub: fmt == "WAV",
        )
        self.assertIsNone(choice.format)
        self.assertFalse(choice.is_streaming_safe)

    def test_wav_bytes_per_second_matches_the_measured_rate(self):
        self.assertEqual(wav_bytes_per_second(44100, 1, "PCM_16"), 88200)

    def test_riff_capacity_is_about_13_5_hours_at_44_1k(self):
        hours = riff_capacity_seconds(44100, 1, "PCM_16") / 3600.0
        self.assertAlmostEqual(hours, 13.5, delta=0.1)

    def test_riff_capacity_is_about_12_4_hours_at_48k(self):
        hours = riff_capacity_seconds(48000, 1, "PCM_16") / 3600.0
        self.assertAlmostEqual(hours, 12.4, delta=0.1)

    def test_ten_day_session_exceeds_the_riff_limit(self):
        bps = wav_bytes_per_second(44100, 1, "PCM_16")
        ten_days_s = 10 * 24 * 3600
        self.assertGreater(ten_days_s * bps, WAV_RIFF_LIMIT_BYTES)


class AudioWriteBudgetTests(unittest.TestCase):
    def test_unlimited_budget_never_stops(self):
        budget = AudioWriteBudget(limit_frames=None)
        budget.note_frames(10_000_000)
        self.assertFalse(budget.should_stop())

    def test_budget_stops_exactly_at_the_limit_and_not_before(self):
        budget = AudioWriteBudget(limit_frames=100)
        budget.note_frames(99)
        self.assertFalse(budget.should_stop())
        budget.note_frames(1)
        self.assertTrue(budget.should_stop())

    def test_for_choice_builds_unlimited_budget_for_streaming_safe_choice(self):
        choice = AudioFormatChoice(
            format="RF64",
            subtype="PCM_16",
            is_streaming_safe=True,
            max_bytes=None,
            max_seconds=None,
            reason="RF64, unlimited size",
        )
        budget = AudioWriteBudget.for_choice(choice, samplerate=44100)
        self.assertFalse(budget.should_stop())

    def test_for_choice_builds_bounded_budget_for_plain_wav(self):
        choice = choose_audio_format(available_formats={"WAV": "WAV"}, samplerate=44100, channels=1)
        budget = AudioWriteBudget.for_choice(choice, samplerate=44100)
        budget.note_frames(int(choice.max_seconds * 44100) + 1)
        self.assertTrue(budget.should_stop())


if __name__ == "__main__":
    unittest.main()
