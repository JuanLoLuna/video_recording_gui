"""backend/auto_setup.py: gain from board A's white paper."""
import unittest

import numpy as np

import test_calibration as synth
from backend import auto_setup as au
from backend import calibration as cal

BOARD = cal.with_measured_square(cal.BOARD_PRESETS[cal.A_BIG_PRESET], 44.82)


class FakeCam:
    """Board A on a table under a fixed light: grey level = scene * 10**(gain/20), clipped."""

    def __init__(self, gain_db, scene_scale=1.0, board_visible=True, limits=(0.0, 24.0)):
        R, t = synth.pose_looking_at_board(0.9, (25, 10, 0), (0.0, 0.0), BOARD)
        img = synth.render_scene([(BOARD, R, t)], D=np.zeros(5), background=40) if board_visible \
            else np.full((540, 720), 40, np.uint8)
        # Scale the scene so that 0 dB puts the paper (the brightest pixels) at 120 * scene_scale.
        self.scene = img.astype(np.float32) / max(1, int(img.max())) * 120.0 * scene_scale
        self.gain, self.limits, self.sets = gain_db, limits, []

    def frame(self):
        return np.clip(self.scene * 10 ** (self.gain / 20), 0, 255).astype(np.uint8)

    def set(self, v):
        self.gain = min(self.limits[1], max(self.limits[0], v)); self.sets.append(v)
        return True

    def run(self, **kw):
        return au.auto_gain("S", BOARD, self.frame, lambda: self.gain, self.set, self.limits,
                            sleep=lambda s: None, **kw)


class WhiteLevelTest(unittest.TestCase):
    def test_reads_the_paper_not_the_markers(self):
        cam = FakeCam(0.0)
        det = cal.BoardDetector(BOARD).detect(cam.frame())
        self.assertTrue(det.ok)
        self.assertAlmostEqual(au.board_white_level(cam.frame(), det), 120.0, delta=3)

    def test_step_follows_the_brightness_ratio(self):
        self.assertAlmostEqual(au.gain_step_db(100.0, 200.0), 6.02, places=2)
        self.assertEqual(au.gain_step_db(255.0), -6.0)
        self.assertEqual(au.gain_step_db(1.0), au.MAX_STEP_DB)


class AutoGainTest(unittest.TestCase):
    def test_converges_from_too_dark_and_from_clipped(self):
        for start in (0.0, 15.0):   # paper at 120 (dark) and clipped
            cam = FakeCam(start)
            r = cam.run()
            self.assertTrue(r.ok, r)
            self.assertAlmostEqual(r.white_level, au.TARGET_WHITE, delta=au.TOLERANCE)
            self.assertAlmostEqual(r.gain_db, 20 * np.log10(au.TARGET_WHITE / 120.0), delta=0.6)
            self.assertLessEqual(r.iterations, 4)

    def test_same_paper_level_whatever_the_light(self):
        gains = [FakeCam(3.0, scene_scale=s).run().gain_db for s in (0.5, 1.0, 1.5)]
        # Half the light needs +6 dB, 1.5x the light -3.5 dB.
        self.assertAlmostEqual(gains[0] - gains[1], 6.0, delta=0.8)
        self.assertAlmostEqual(gains[1] - gains[2], 3.5, delta=0.8)
        # Twice the light: the paper is already at 240 with no gain, so it says to reduce the light.
        r = FakeCam(3.0, scene_scale=2.0).run()
        self.assertFalse(r.ok)
        self.assertIn("reduce the light", r.message)

    def test_board_missing(self):
        r = FakeCam(3.0, board_visible=False).run()
        self.assertFalse(r.ok)
        self.assertIn("board A not found", r.message)

    def test_too_little_light_says_so(self):
        r = FakeCam(0.0, scene_scale=0.3, limits=(0.0, 6.0)).run()   # paper at 36 grey, at most +6 dB
        self.assertFalse(r.ok)
        self.assertIn("add light", r.message)


if __name__ == "__main__":
    unittest.main()


G2 = cal.BOARD_PRESETS[cal.REFERENCE_PRESETS[3]]


class FakeController:
    """Just enough of CameraController for auto_setup_cameras, over a FakeCam scene."""

    def __init__(self, cam):
        self.cam, self.exposure, self.enums = cam, 3000.0, {}

    def set_enum_param(self, name, entry):
        self.enums[name] = entry
        return True

    def get_exposure_time_limits(self):
        return (10.0, 30000.0, self.exposure)

    def get_image_param_limits(self, name):
        return (self.cam.limits[0], self.cam.limits[1], self.cam.gain) if name == "Gain" else None

    def set_image_param(self, name, value):
        if name == "ExposureTime":
            self.exposure = value
            return True
        return self.cam.set(value)

    def get_latest_frame(self):
        return self.cam.frame()


class ReferenceTest(unittest.TestCase):
    def setUp(self):
        import tempfile
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def test_store_is_per_setup_and_keeps_only_usable_levels(self):
        entry = {"name": "G2", "board": G2.to_dict(), "white": 150.0, "gain_db": 2.0, "exposure_us": 8000.0}
        au.save_reference(self.tmp.name, "20261016_154306",
                          {"A": entry, "B": dict(entry, white=252.0), "C": dict(entry, white=None)})
        ref = au.load_reference(self.tmp.name, "20261016_154306")
        self.assertEqual(set(ref), {"A"})                     # clipped and missing levels dropped
        self.assertEqual(au.load_reference(self.tmp.name, "20261017_090000"), {})   # newer setup: stale
        self.assertEqual(au.load_reference(self.tmp.name, None), {})
        t = au.targets_from_reference(ref)["A"]
        self.assertEqual((t.white, t.name, t.board), (150.0, "reference G2", G2))

    def test_button_restores_the_stored_reference_level(self):
        # Session 1: the reference board read 150 at the gain board A chose. Session 2: the light
        # dimmed to 70 %: the gain goes up until the reference board reads 150 again (+3.1 dB).
        def scene(scale):
            R, t = synth.pose_looking_at_board(1.2, (20, 0, 0), (0.0, 0.0), G2)
            cam = FakeCam(4.0)
            img = synth.render_scene([(G2, R, t)], D=np.zeros(5), background=40).astype(np.float32)
            cam.scene = img / img.max() * 150.0 / 10 ** (4.0 / 20) * scale   # 150 at 4 dB, full light
            return cam
        cam = scene(0.7)
        results = au.auto_setup_cameras([("Firefly", "A", FakeController(cam))], BOARD, 8000.0,
                                        sleep=lambda s: None, targets={"A": au.Target(G2, 150.0, "reference G2")})
        (label, r), = results
        self.assertTrue(r.ok, r)
        self.assertIn("reference G2", r.message)
        self.assertAlmostEqual(r.gain_db, 4.0 + 20 * np.log10(1 / 0.7), delta=0.5)

    def test_camera_without_a_reference_falls_back_to_board_a(self):
        ctl = FakeController(FakeCam(0.0))
        (label, r), = au.auto_setup_cameras([("Blackfly", "B", ctl)], BOARD, 8000.0,
                                            sleep=lambda s: None, targets={})
        self.assertTrue(r.ok, r)
        self.assertIn("board A", r.message)
        self.assertEqual(ctl.exposure, 8000.0)
        self.assertEqual(ctl.enums, {"ExposureAuto": "Off", "GainAuto": "Off"})


class GainLimitTest(unittest.TestCase):
    def test_too_bright_at_zero_gain_but_unclipped_counts_as_set(self):
        # The rig's Firefly: board A at 219 with the gain already at 0 dB.
        cam = FakeCam(2.0, scene_scale=219 / 120)
        r = cam.run()
        self.assertTrue(r.ok, r)
        self.assertEqual(r.gain_db, 0.0)
        self.assertAlmostEqual(r.white_level, 219, delta=4)
        self.assertIn("its limit", r.message)

    def test_band_edges(self):
        self.assertTrue(au.acceptable_at_limit(219, 200))
        self.assertFalse(au.acceptable_at_limit(240, 200))   # near clipping
        self.assertFalse(au.acceptable_at_limit(130, 200))   # too dark
        self.assertTrue(au.acceptable_at_limit(180, 150))    # reference target 150: up to 187
        self.assertFalse(au.acceptable_at_limit(190, 150))


class LaggingFramesTest(unittest.TestCase):
    """The app's newest frame can lag the camera: frames show the gain set several grabs ago."""

    def test_waits_for_frames_with_the_new_gain(self):
        cam = FakeCam(2.5, scene_scale=210 / 120)      # the rig's Firefly: board at 210 at 0 dB
        history = []

        def lagging_frame(delay=10):
            history.append(cam.gain)
            seen = history[max(0, len(history) - 1 - delay)]
            return np.clip(cam.scene * 10 ** (seen / 20), 0, 255).astype(np.uint8)

        r = au.auto_gain("S", BOARD, lagging_frame, lambda: cam.gain, cam.set, cam.limits,
                         sleep=lambda s: None)
        self.assertTrue(r.ok, r)                        # not "reduce the light" on a stale clipped frame
        self.assertEqual(r.gain_db, 0.0)
        self.assertAlmostEqual(r.white_level, 210, delta=5)
