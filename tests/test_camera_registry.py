import unittest

from backend.camera_registry import (
    CAMERA_SERIALS_ENV,
    CameraDescriptor,
    format_camera_summary,
    intersect_ranges,
    parse_serials_env,
    select_cameras,
)

FIREFLY = CameraDescriptor("23227865", "Firefly FFY-U3-04S2M", "FLIR")
BLACKFLY = CameraDescriptor("26134271", "Blackfly S BFS-U3-13Y3M", "FLIR")


class ParseSerialsEnvTests(unittest.TestCase):
    def test_unset_or_blank_is_empty(self):
        self.assertEqual(parse_serials_env({}), [])
        self.assertEqual(parse_serials_env({CAMERA_SERIALS_ENV: "  "}), [])

    def test_order_is_kept(self):
        env = {CAMERA_SERIALS_ENV: "26134271,23227865"}
        self.assertEqual(parse_serials_env(env), ["26134271", "23227865"])

    def test_tolerates_spaces_semicolons_blanks_and_repeats(self):
        env = {CAMERA_SERIALS_ENV: " 26134271 ;; 23227865,,26134271 "}
        self.assertEqual(parse_serials_env(env), ["26134271", "23227865"])


class SelectCamerasTests(unittest.TestCase):
    def test_no_configuration_uses_everything_sorted_by_serial(self):
        # Detection order must not matter.
        for discovered in ([BLACKFLY, FIREFLY], [FIREFLY, BLACKFLY]):
            selection = select_cameras(discovered)
            self.assertEqual([c.serial for c in selection.bound], ["23227865", "26134271"])
            self.assertEqual(selection.missing, ())
            self.assertTrue(selection.multi_camera)

    def test_lowest_serial_is_primary_and_unsuffixed_others_tagged(self):
        selection = select_cameras([BLACKFLY, FIREFLY])
        primary, other = selection.bound
        self.assertTrue(primary.is_primary)
        self.assertIsNone(primary.tag)
        self.assertFalse(other.is_primary)
        self.assertEqual(other.tag, "cam26134271")

    def test_configured_order_decides_the_primary(self):
        selection = select_cameras([FIREFLY, BLACKFLY], ["26134271", "23227865"])
        self.assertEqual([c.serial for c in selection.bound], ["26134271", "23227865"])
        self.assertIsNone(selection.bound[0].tag)
        self.assertEqual(selection.bound[1].tag, "cam23227865")

    def test_single_camera_without_config_is_primary_with_legacy_names(self):
        # A one-camera setup must produce exactly today's file names, even if
        # that one camera happens to have the higher serial.
        selection = select_cameras([BLACKFLY])
        self.assertEqual(len(selection.bound), 1)
        self.assertIsNone(selection.bound[0].tag)
        self.assertTrue(selection.bound[0].is_primary)
        self.assertFalse(selection.multi_camera)

    def test_configured_camera_missing_is_reported_not_silently_dropped(self):
        selection = select_cameras([BLACKFLY], ["23227865", "26134271"])
        self.assertEqual(selection.missing, ("23227865",))
        self.assertEqual([c.serial for c in selection.bound], ["26134271"])

    def test_a_camera_keeps_its_name_while_the_primary_is_missing(self):
        # With explicit config, naming follows the intended order, so the
        # second camera does not silently become "primary" mid-study.
        selection = select_cameras([BLACKFLY], ["23227865", "26134271"])
        self.assertEqual(selection.bound[0].tag, "cam26134271")
        self.assertFalse(selection.bound[0].is_primary)

    def test_detected_but_unconfigured_cameras_are_listed_as_unused(self):
        selection = select_cameras([FIREFLY, BLACKFLY], ["23227865"])
        self.assertEqual([c.serial for c in selection.bound], ["23227865"])
        self.assertEqual([c.serial for c in selection.unused], ["26134271"])

    def test_nothing_detected(self):
        selection = select_cameras([])
        self.assertEqual(selection.bound, ())
        self.assertEqual(selection.missing, ())

    def test_nothing_detected_but_configured_reports_all_missing(self):
        selection = select_cameras([], ["1", "2"])
        self.assertEqual(selection.bound, ())
        self.assertEqual(selection.missing, ("1", "2"))

    def test_duplicate_detection_of_one_serial_counts_once(self):
        selection = select_cameras([FIREFLY, FIREFLY])
        self.assertEqual(len(selection.bound), 1)


class SelectionWarningTests(unittest.TestCase):
    def test_several_cameras_without_the_env_var_warn_about_unstable_names(self):
        selection = select_cameras([FIREFLY, BLACKFLY])
        self.assertEqual(len(selection.warnings), 1)
        self.assertIn(CAMERA_SERIALS_ENV, selection.warnings[0])
        # The suggestion is directly pasteable.
        self.assertIn("23227865,26134271", selection.warnings[0])

    def test_one_camera_without_the_env_var_is_quiet_legacy_behaviour(self):
        self.assertEqual(select_cameras([BLACKFLY]).warnings, ())

    def test_configured_cameras_do_not_warn_about_pinning(self):
        selection = select_cameras([FIREFLY, BLACKFLY], ["23227865", "26134271"])
        self.assertEqual(selection.warnings, ())

    def test_missing_primary_is_called_out(self):
        selection = select_cameras([BLACKFLY], ["23227865", "26134271"])
        self.assertEqual(len(selection.warnings), 1)
        self.assertIn("#23227865", selection.warnings[0])
        self.assertIn("no untagged", selection.warnings[0])

    def test_a_missing_secondary_is_not_a_primary_warning(self):
        selection = select_cameras([FIREFLY], ["23227865", "26134271"])
        self.assertEqual(selection.warnings, ())
        self.assertEqual(selection.missing, ("26134271",))

    def test_an_unreadable_serial_is_ignored_with_a_warning_not_a_crash(self):
        selection = select_cameras(
            [CameraDescriptor("<unavailable>", "X"), CameraDescriptor("", "Y"), FIREFLY]
        )
        self.assertEqual([c.serial for c in selection.bound], ["23227865"])
        self.assertTrue(any("unreadable" in w for w in selection.warnings))

    def test_serials_sort_numerically_by_length_then_value(self):
        short = CameraDescriptor("9999999", "S")
        long_ = CameraDescriptor("10000000", "L")
        selection = select_cameras([long_, short])
        self.assertEqual([c.serial for c in selection.bound], ["9999999", "10000000"])

    def test_cmd_style_quoted_and_padded_config_still_matches(self):
        env = {CAMERA_SERIALS_ENV: ' "26134271 , 23227865" '}
        configured = parse_serials_env(env)
        self.assertEqual(configured, ["26134271", "23227865"])
        selection = select_cameras([FIREFLY, BLACKFLY], configured)
        self.assertEqual(selection.missing, ())


class FormatSummaryTests(unittest.TestCase):
    def test_single_camera_summary(self):
        text = format_camera_summary(select_cameras([FIREFLY]))
        self.assertEqual(text, "1 camera: Firefly FFY-U3-04S2M #23227865")

    def test_two_cameras_mark_the_primary(self):
        text = format_camera_summary(select_cameras([BLACKFLY, FIREFLY]))
        self.assertIn("2 cameras", text)
        self.assertIn("Firefly FFY-U3-04S2M #23227865 (primary)", text)
        self.assertIn("Blackfly S BFS-U3-13Y3M #26134271", text)

    def test_missing_cameras_are_called_out(self):
        text = format_camera_summary(select_cameras([FIREFLY], ["23227865", "26134271"]))
        self.assertIn("MISSING: 26134271", text)

    def test_nothing_found(self):
        self.assertEqual(format_camera_summary(select_cameras([])), "No cameras detected.")
        self.assertIn(
            "missing: 5", format_camera_summary(select_cameras([], ["5"]))
        )


class IntersectRangesTests(unittest.TestCase):
    def test_overlap(self):
        self.assertEqual(intersect_ranges([(1.0, 120.9), (1.0, 170.6)]), (1.0, 120.9))
        self.assertEqual(intersect_ranges([(2.0, 100.0), (5.0, 80.0)]), (5.0, 80.0))

    def test_disjoint_is_none(self):
        self.assertIsNone(intersect_ranges([(1.0, 10.0), (20.0, 30.0)]))

    def test_a_min_max_current_triple_is_accepted(self):
        # get_frame_rate_limits() returns (min, max, current).
        self.assertEqual(intersect_ranges([(1.0, 120.9, 30.0), (2.0, 170.6, 60.0)]), (2.0, 120.9))

    def test_unreadable_cameras_are_ignored(self):
        self.assertEqual(intersect_ranges([None, (1.0, 50.0)]), (1.0, 50.0))
        self.assertIsNone(intersect_ranges([None, None]))
        self.assertIsNone(intersect_ranges([]))


if __name__ == "__main__":
    unittest.main()
