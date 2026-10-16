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
