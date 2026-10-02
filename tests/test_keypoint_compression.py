import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import keypoint_compression_test as kct  # noqa: E402


def _skeleton(frames: int = 60, seed: int = 0):
    """Smoothly moving, fully visible 133-keypoint set."""
    rng = np.random.default_rng(seed)
    base = rng.uniform(100, 500, size=(1, kct.K, 2))
    drift = np.cumsum(rng.normal(0, 0.5, size=(frames, 1, 2)), axis=0)
    return base + drift, np.full((frames, kct.K), 0.9)


class AngleSeriesTests(unittest.TestCase):
    def _three_points(self, a, b, c):
        kp = np.zeros((1, kct.K, 2))
        kp[0, 0], kp[0, 1], kp[0, 2] = a, b, c
        return kp, np.ones((1, kct.K), dtype=bool)

    def test_a_right_angle_is_90_degrees(self):
        kp, valid = self._three_points((1, 0), (0, 0), (0, 1))
        self.assertAlmostEqual(kct.angle_series(kp, valid, 0, 1, 2)[0], 90.0)

    def test_a_straight_limb_is_180_degrees(self):
        kp, valid = self._three_points((-1, 0), (0, 0), (1, 0))
        self.assertAlmostEqual(kct.angle_series(kp, valid, 0, 1, 2)[0], 180.0)

    def test_an_invalid_point_makes_the_angle_nan(self):
        kp, valid = self._three_points((1, 0), (0, 0), (0, 1))
        valid[0, 2] = False
        self.assertTrue(np.isnan(kct.angle_series(kp, valid, 0, 1, 2)[0]))

    def test_coincident_points_give_nan_not_a_crash(self):
        kp, valid = self._three_points((0, 0), (0, 0), (0, 1))
        self.assertTrue(np.isnan(kct.angle_series(kp, valid, 0, 1, 2)[0]))


class JointDefinitionTests(unittest.TestCase):
    def test_every_index_is_a_valid_wholebody_keypoint(self):
        for name, idx in kct.joint_definitions().items():
            self.assertEqual(len(set(idx)), 3, name)
            self.assertTrue(all(0 <= i < kct.K for i in idx), name)

    def test_each_hand_angle_stays_inside_its_own_hand(self):
        for name, idx in kct.joint_definitions().items():
            if name.startswith("L_") and ("mcp" in name or "pip" in name or "dip" in name or "thumb" in name):
                self.assertTrue(all(91 <= i <= 111 for i in idx), name)
            if name.startswith("R_") and ("mcp" in name or "pip" in name or "dip" in name or "thumb" in name):
                self.assertTrue(all(112 <= i <= 132 for i in idx), name)


class ComparePairTests(unittest.TestCase):
    def test_identical_keypoints_differ_by_nothing(self):
        kp, sc = _skeleton()
        r = kct.compare_pair((kp, sc), (kp.copy(), sc.copy()), 0.3)
        self.assertEqual(r["displacement_px"]["all"]["max"], 0.0)
        self.assertEqual(r["angle_abs_diff_deg_pooled"]["max"], 0.0)
        self.assertEqual(r["keypoints_valid"]["lost"], 0)

    def test_a_uniform_shift_moves_every_keypoint_by_that_distance_but_not_the_angles(self):
        kp, sc = _skeleton()
        r = kct.compare_pair((kp, sc), (kp + np.array([3.0, 4.0]), sc), 0.3)
        self.assertAlmostEqual(r["displacement_px"]["all"]["median"], 5.0)
        self.assertAlmostEqual(r["angle_abs_diff_deg_pooled"]["max"], 0.0, places=6)

    def test_lost_keypoints_are_counted_and_excluded_from_the_error(self):
        kp, sc = _skeleton()
        sc_v = sc.copy()
        sc_v[:, 91:112] = 0.1  # the whole left hand falls under the threshold
        r = kct.compare_pair((kp, sc), (kp, sc_v), 0.3)
        self.assertEqual(r["keypoints_valid"]["lost"], 21 * len(kp))
        self.assertEqual(r["displacement_px"]["left_hand"]["n"], 0)

    def test_a_frame_with_no_person_in_the_variant_is_a_lost_frame(self):
        kp, sc = _skeleton()
        kp_v = kp.copy()
        kp_v[10] = np.nan
        r = kct.compare_pair((kp, sc), (kp_v, sc), 0.3)
        self.assertEqual(r["frames_with_person"]["lost"], 1)

    def test_mismatched_frame_counts_are_rejected(self):
        kp, sc = _skeleton(60)
        kp2, sc2 = _skeleton(50)
        with self.assertRaises(ValueError):
            kct.compare_pair((kp, sc), (kp2, sc2), 0.3)

    def test_per_pixel_jitter_gives_a_small_nonzero_angle_difference(self):
        kp, sc = _skeleton(200)
        jitter = np.random.default_rng(1).normal(0, 0.3, size=kp.shape)
        r = kct.compare_pair((kp, sc), (kp + jitter, sc), 0.3)
        p95 = r["angle_abs_diff_deg_pooled"]["p95"]
        self.assertGreater(p95, 0.0)
        self.assertLess(p95, 30.0)


class LoadKeypointsTests(unittest.TestCase):
    def test_wrong_shape_is_rejected_with_a_clear_message(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "bad.npz"
            np.savez(path, keypoints=np.zeros((5, 17, 2)), scores=np.zeros((5, 17)))
            with self.assertRaises(ValueError):
                kct.load_keypoints(path)


class EncodeTests(unittest.TestCase):
    def test_encode_writes_readable_variants_with_the_expected_quality_ordering(self):
        try:
            import cv2
        except ImportError:
            self.skipTest("opencv not installed")
        rng = np.random.default_rng(0)
        yy, xx = np.mgrid[0:96, 0:128]
        base = (128 + 100 * np.sin(xx / 9.0) * np.cos(yy / 7.0)).astype(np.float32)
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "clip.avi"
            writer = cv2.VideoWriter(str(src), cv2.CAP_FFMPEG, cv2.VideoWriter_fourcc(*"GREY"), 30, (128, 96), False)
            if not writer.isOpened():
                self.skipTest("this OpenCV build cannot write GREY AVI")
            for i in range(8):
                writer.write(np.clip(base + rng.normal(0, 2, base.shape), 0, 255).astype(np.uint8))
            writer.release()
            out = Path(d) / "out"
            rc = kct.cmd_encode(argparse.Namespace(video=str(src), out_dir=str(out), cv_color=False))
            self.assertEqual(rc, 0)
            report = json.loads((out / "clip_encode_report.json").read_text())
            app, control = report["variants"]["mjpeg_app"], report["variants"]["noise_control"]
            self.assertEqual(app["frames_written_readable"], 8)
            self.assertEqual(control["frames_written_readable"], 8)
            # sigma=1 noise is far gentler than JPEG compression
            self.assertGreater(control["psnr_mean_db"], app["psnr_mean_db"])
            self.assertGreater(app["ratio_vs_raw"], 1.5)


if __name__ == "__main__":
    unittest.main()
