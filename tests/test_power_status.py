import unittest

from backend.power_status import PowerStatus, assess_power_safety


class PowerSafetyAssessmentTests(unittest.TestCase):
    def test_ac_power_with_saver_off_is_safe(self):
        result = assess_power_safety(
            PowerStatus(
                supported=True,
                ac_online=True,
                battery_percent=75,
                energy_saver_on=False,
            )
        )
        self.assertEqual(result.level, "safe")
        self.assertFalse(result.recording_blocked)
        self.assertFalse(result.requires_confirmation)

    def test_energy_saver_blocks_recording_even_on_ac(self):
        result = assess_power_safety(
            PowerStatus(
                supported=True,
                ac_online=True,
                battery_percent=75,
                energy_saver_on=True,
            )
        )
        self.assertEqual(result.level, "danger")
        self.assertTrue(result.recording_blocked)
        self.assertIn("Energy Saver", result.reason)

    def test_normal_battery_is_warning_but_allowed(self):
        result = assess_power_safety(
            PowerStatus(
                supported=True,
                ac_online=False,
                battery_percent=60,
                energy_saver_on=False,
            )
        )
        self.assertEqual(result.level, "warning")
        self.assertFalse(result.recording_blocked)
        self.assertFalse(result.requires_confirmation)

    def test_low_battery_requires_confirmation(self):
        result = assess_power_safety(
            PowerStatus(
                supported=True,
                ac_online=False,
                battery_percent=15,
                energy_saver_on=False,
            )
        )
        self.assertFalse(result.recording_blocked)
        self.assertTrue(result.requires_confirmation)

    def test_windows_low_flag_does_not_override_configured_percent_threshold(self):
        result = assess_power_safety(
            PowerStatus(
                supported=True,
                ac_online=False,
                battery_percent=25,
                energy_saver_on=False,
                battery_low=True,
            )
        )
        self.assertFalse(result.recording_blocked)
        self.assertFalse(result.requires_confirmation)

    def test_critical_battery_blocks_recording(self):
        result = assess_power_safety(
            PowerStatus(
                supported=True,
                ac_online=False,
                battery_percent=4,
                energy_saver_on=False,
            )
        )
        self.assertTrue(result.recording_blocked)

    def test_unsupported_platform_does_not_block(self):
        result = assess_power_safety(PowerStatus(supported=False))
        self.assertEqual(result.level, "neutral")
        self.assertFalse(result.recording_blocked)


if __name__ == "__main__":
    unittest.main()
