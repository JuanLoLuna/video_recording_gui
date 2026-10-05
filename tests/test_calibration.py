"""backend/calibration.py against synthetic ChArUco images with known K, D and poses.

Each image is rendered by projecting the printed board through a known camera
(homography for the plane, then the lens distortion applied with a remap), so
every number calibration.py reports can be checked against ground truth.
"""
import math
import unittest

import cv2
import numpy as np

from backend import calibration as cal

IMAGE_SIZE = (720, 540)
K_TRUE = np.array([[820.0, 0, 362.0], [0, 815.0, 268.0], [0, 0, 1]])
D_TRUE = np.array([-0.12, 0.05, 0.0, 0.0, 0.0])
CFG = cal.BoardConfig()
BOARD_PX_PER_M = 5000.0
_BOARD_IMG = cal.render_board(CFG, BOARD_PX_PER_M)


def pose_looking_at_board(distance_m, rot_xyz_deg=(0, 0, 0), offset_m=(0.0, 0.0), cfg=CFG):
    """Board -> camera (R, t) with the board centre at (offset, distance) in front of the camera."""
    rvec = np.radians(np.array(rot_xyz_deg, dtype=float))
    R, _ = cv2.Rodrigues(rvec)
    centre = np.array([cfg.size_m[0] / 2, cfg.size_m[1] / 2, 0.0])
    t = np.array([offset_m[0], offset_m[1], distance_m]) - R @ centre
    return R, t


def render_view(R, t, K=K_TRUE, D=D_TRUE, size=IMAGE_SIZE, background=90):
    """Image of the board seen by camera (K, D) at board->camera pose (R, t)."""
    # board pixel (u, v) -> board metres ((u + 0.5) / ppm, (v + 0.5) / ppm)
    S = np.array([[1 / BOARD_PX_PER_M, 0, 0.5 / BOARD_PX_PER_M],
                  [0, 1 / BOARD_PX_PER_M, 0.5 / BOARD_PX_PER_M],
                  [0, 0, 1]])
    H = K @ np.column_stack([R[:, 0], R[:, 1], t]) @ S
    board = np.where(_BOARD_IMG > 127, 230, 25).astype(np.uint8)
    ideal = cv2.warpPerspective(board, H, size, flags=cv2.INTER_LINEAR, borderValue=background)
    mask = cv2.warpPerspective(np.full_like(board, 255), H, size, flags=cv2.INTER_NEAREST, borderValue=0)
    ideal[mask == 0] = background
    if not np.any(D):
        return ideal
    # Distort: for every output pixel find where it lies in the ideal (undistorted) image.
    w, h = size
    grid = np.stack(np.meshgrid(np.arange(w), np.arange(h)), -1).reshape(-1, 1, 2).astype(np.float32)
    und = cv2.undistortPoints(grid, K, D, P=K).reshape(h, w, 2)
    return cv2.remap(ideal, und[..., 0], und[..., 1], cv2.INTER_LINEAR, borderValue=background)


def render_view_a4(cfg, R, t, K=K_TRUE, size=IMAGE_SIZE, ppm=BOARD_PX_PER_M):
    """Undistorted image of any board config (the module's render_view is fixed to CFG)."""
    board_img = cal.render_board(cfg, ppm)
    S = np.array([[1 / ppm, 0, 0.5 / ppm], [0, 1 / ppm, 0.5 / ppm], [0, 0, 1]])
    H = K @ np.column_stack([R[:, 0], R[:, 1], t]) @ S
    board = np.where(board_img > 127, 230, 25).astype(np.uint8)
    img = cv2.warpPerspective(board, H, size, flags=cv2.INTER_LINEAR, borderValue=90)
    mask = cv2.warpPerspective(np.full_like(board, 255), H, size, flags=cv2.INTER_NEAREST, borderValue=0)
    img[mask == 0] = 90
    return img


def varied_views(n=18, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        rot = rng.uniform(-35, 35, size=3) * np.array([1, 1, 0.3])
        dist = rng.uniform(0.35, 0.7)
        off = rng.uniform(-0.12, 0.12, size=2) * dist
        out.append(pose_looking_at_board(dist, rot, off))
    return out


class BoardConfigTest(unittest.TestCase):
    def test_defaults_match_lab_script(self):
        self.assertEqual((CFG.squares_x, CFG.squares_y, CFG.square_length_m, CFG.marker_length_m, CFG.dictionary),
                         (5, 5, 0.04, 0.03, "DICT_5X5_50"))
        self.assertEqual(CFG.corner_count, 16)

    def test_round_trip_and_validation(self):
        self.assertEqual(cal.BoardConfig.from_dict(CFG.to_dict()), CFG)
        with self.assertRaises(ValueError):
            cal.BoardConfig(marker_length_m=0.05)
        with self.assertRaises(ValueError):
            cal.BoardConfig(dictionary="DICT_NOPE")

    def test_render_is_true_size(self):
        img = cal.render_board(CFG, px_per_m=300 / 0.0254, margin_px=10)  # 300 dpi
        self.assertEqual(img.shape, (round(0.2 * 300 / 0.0254) + 20,) * 2)


class DetectionAndPoseTest(unittest.TestCase):
    def setUp(self):
        self.det = cal.BoardDetector(CFG)

    def test_frontal_view_finds_every_corner(self):
        R, t = pose_looking_at_board(0.5)
        d = self.det.detect(render_view(R, t, D=np.zeros(5)))
        self.assertTrue(d.ok)
        self.assertEqual(len(d.ids), CFG.corner_count)
        self.assertEqual(d.image_size, IMAGE_SIZE)

    def test_empty_image(self):
        d = self.det.detect(np.full((540, 720), 90, np.uint8))
        self.assertFalse(d.ok)
        self.assertEqual(len(d.ids), 0)

    def test_pose_recovered_with_known_intrinsics(self):
        R, t = pose_looking_at_board(0.6, (20, -15, 5), (0.03, -0.02))
        d = self.det.detect(render_view(R, t))
        pose = cal.solve_board_pose(d, self.det.board, K_TRUE, D_TRUE)
        self.assertLess(pose.rms_px, 0.3)
        self.assertLess(cal.rotation_angle_deg(pose.R, R), 0.3)
        self.assertLess(np.linalg.norm(pose.t - t) * 1000, 2.0)
        self.assertAlmostEqual(pose.tilt_deg, math.degrees(math.acos(abs((R @ [0, 0, 1])[2]))), delta=0.5)

    def test_average_and_motion(self):
        R, t = pose_looking_at_board(0.5)
        img = render_view(R, t)
        a = self.det.detect(img)
        b = self.det.detect(np.roll(img, 3, axis=1))
        self.assertAlmostEqual(cal.corner_motion_px(a, b), 3.0, delta=0.2)
        avg = cal.average_detections([a, a, b])
        self.assertEqual(len(avg.ids), len(a.ids))
        self.assertAlmostEqual(float(np.mean(avg.corners[:, 0] - a.corners[:, 0])), 1.0, delta=0.1)


def keep_ids(det, keep):
    """The detection restricted to the corners where keep(ids) is True."""
    m = keep(det.ids)
    return cal.Detection(det.corners[m], det.ids[m], det.marker_count, det.image_size, det.corners_per_row)


class DegenerateViewTest(unittest.TestCase):
    """A row of corners (a board edge-on or cut off at the image border) has no pose."""

    def setUp(self):
        # The A4 presets: 6 corners per row = MIN_CORNERS, so one row used to count as ok.
        self.cfg = cal.BOARD_PRESETS[cal.HANDHELD_PRESET]
        self.det = cal.BoardDetector(self.cfg)
        self.full = self.det.detect(render_view_a4(self.cfg, *pose_looking_at_board(0.5, (10, 5, 0), cfg=self.cfg)))
        self.assertTrue(self.full.ok, len(self.full.ids))

    def test_single_row_or_column_is_not_ok(self):
        per_row = self.cfg.squares_x - 1
        one_row = keep_ids(self.full, lambda ids: ids // per_row == 1)
        one_col = keep_ids(self.full, lambda ids: ids % per_row == 2)
        self.assertGreaterEqual(len(one_row.ids), cal.MIN_CORNERS)  # enough corners, wrong shape
        self.assertGreater(one_row.marker_count, cal.MIN_MARKERS)
        self.assertFalse(one_row.ok)
        self.assertFalse(one_col.ok)
        self.assertIsNone(cal.view_signature(one_row, self.det.board))
        two_rows = keep_ids(self.full, lambda ids: ids // per_row < 2)
        self.assertTrue(two_rows.ok)

    def test_a_single_row_is_never_auto_captured(self):
        per_row = self.cfg.squares_x - 1
        one_row = keep_ids(self.full, lambda ids: ids // per_row == 1)
        session = cal.CaptureSession(self.det.board, still_s=0.5)
        states = [session.feed(one_row, t * 0.2).state for t in range(10)]
        self.assertEqual(set(states), {"no board"})
        self.assertFalse(session.capture_now(one_row))
        self.assertEqual(session.views, [])

    def test_unusable_views_are_found_and_calibration_recovers(self):
        rng = np.random.default_rng(4)
        views = []
        for _ in range(14):
            R, t = pose_looking_at_board(rng.uniform(0.45, 0.7), rng.uniform(-30, 30, 3) * [1, 1, 0.3],
                                         rng.uniform(-0.04, 0.04, 2), cfg=self.cfg)
            views.append(self.det.detect(render_view_a4(self.cfg, R, t)))
        self.assertTrue(all(v.ok for v in views))
        # A detection whose layout is unknown (corners_per_row None) cannot be judged by `ok`.
        per_row = self.cfg.squares_x - 1
        row = keep_ids(self.full, lambda ids: ids // per_row == 1)
        line = cal.Detection(row.corners, row.ids, row.marker_count, row.image_size)
        self.assertTrue(line.ok)
        with self.assertRaises(ValueError):  # not a raw cv2.error
            cal.calibrate_intrinsics(views + [line], self.det.board)
        self.assertEqual(cal.unusable_views(views + [line], self.det.board), [len(views)])
        self.assertEqual(cal.unusable_views(views, self.det.board), [])
        self.assertIsNotNone(cal.calibrate_intrinsics(views, self.det.board))


class IntrinsicsTest(unittest.TestCase):
    def test_recovers_camera_matrix_and_predicts_new_views(self):
        det = cal.BoardDetector(CFG)
        views = [det.detect(render_view(R, t)) for R, t in varied_views(40, seed=1)]
        result = cal.calibrate_intrinsics(views, det.board)
        self.assertTrue(result.passed, result.rms_px)
        self.assertEqual(result.image_size, IMAGE_SIZE)
        self.assertEqual(len(result.per_view_rms_px), result.n_views)
        self.assertAlmostEqual(result.K[0, 0], K_TRUE[0, 0], delta=K_TRUE[0, 0] * 0.015)
        self.assertAlmostEqual(result.K[1, 1], K_TRUE[1, 1], delta=K_TRUE[1, 1] * 0.015)
        # A 16-corner board pins the principal point and distortion only loosely
        # (they trade off against each other); what must hold is that the model
        # predicts views it was not fitted on.
        self.assertAlmostEqual(result.K[0, 2], K_TRUE[0, 2], delta=10)
        self.assertAlmostEqual(result.K[1, 2], K_TRUE[1, 2], delta=10)
        for R, t in varied_views(8, seed=99):
            pose = cal.solve_board_pose(det.detect(render_view(R, t)), det.board, result.K, result.D)
            self.assertLess(pose.rms_px, 0.5)

    def test_refuses_too_few_or_mixed_views(self):
        det = cal.BoardDetector(CFG)
        views = [det.detect(render_view(R, t)) for R, t in varied_views(5)]
        with self.assertRaises(ValueError):
            cal.calibrate_intrinsics(views, det.board)
        odd = cal.Detection(views[0].corners, views[0].ids, views[0].marker_count, (640, 480))
        with self.assertRaises(ValueError):
            cal.calibrate_intrinsics(views * 2 + [odd], det.board)


class CoverageTest(unittest.TestCase):
    def test_signatures_and_new_view_rule(self):
        det = cal.BoardDetector(CFG)
        flat = cal.view_signature(det.detect(render_view(*pose_looking_at_board(0.5))), det.board)
        same = cal.view_signature(det.detect(render_view(*pose_looking_at_board(0.5, (2, 0, 0)))), det.board)
        tilted = cal.view_signature(det.detect(render_view(*pose_looking_at_board(0.5, (30, 0, 0)))), det.board)
        near = cal.view_signature(det.detect(render_view(*pose_looking_at_board(0.3))), det.board)
        self.assertEqual(flat.cell, (1, 1))
        self.assertFalse(flat.tilted)
        self.assertTrue(tilted.tilted)
        self.assertTrue(near.near)
        self.assertFalse(cal.is_new_view(same, [flat], IMAGE_SIZE))
        self.assertTrue(cal.is_new_view(tilted, [flat], IMAGE_SIZE))
        self.assertTrue(cal.is_new_view(near, [flat], IMAGE_SIZE))

    def test_tilt_direction_counts_as_a_new_view(self):
        det = cal.BoardDetector(CFG)

        def sig(*rot):
            return cal.view_signature(det.detect(render_view(*pose_looking_at_board(0.5, rot))), det.board)

        left, right, up, down = sig(0, 30, 0), sig(0, -30, 0), sig(30, 0, 0), sig(-30, 0, 0)
        # Same place and same tilt magnitude, different direction.
        for s in (left, right, up, down):
            self.assertAlmostEqual(s.tilt_deg, 30, delta=2)
        self.assertEqual({s.tilt_direction for s in (left, right, up, down)}, set(cal.TILT_DIRECTIONS))
        for taken, others in ((left, (right, up, down)), (right, (left, up, down)),
                              (up, (left, right, down)), (down, (left, right, up))):
            for other in others:
                self.assertTrue(cal.is_new_view(other, [taken], IMAGE_SIZE), (taken.tilt_direction, other.tilt_direction))
        # A genuinely similar view (a couple of degrees off, nothing else changed) is still a duplicate.
        again = sig(2, 29, 1)
        self.assertFalse(cal.is_new_view(again, [left], IMAGE_SIZE))
        self.assertFalse(cal.is_new_view(left, [left], IMAGE_SIZE))

    def test_hints(self):
        cov = cal.Coverage()
        self.assertTrue(any("cover" in h for h in cov.missing()))
        normals = {"left": (0.5, 0, 0.87), "right": (-0.5, 0, 0.87), "up": (0, 0.5, 0.87), "down": (0, -0.5, 0.87)}
        for c in range(3):
            for r in range(3):
                for near, tilt in ((0.4, 30.0), (0.1, 0.0), (0.4, 0.0), (0.1, 30.0)):
                    direction = list(normals)[(c + r) % 4]
                    cov.add(cal.ViewSignature((c, r), (0, 0), near, tilt, normals[direction] if tilt else (0, 0, 1)))
        self.assertEqual(cov.missing(target_views=30), [])
        one_way = cal.Coverage()
        for i in range(9):
            one_way.add(cal.ViewSignature((i % 3, i // 3), (0, 0), 0.4 if i % 2 else 0.1, 30.0, normals["left"]))
        self.assertTrue(any("faces right and up and down" in h for h in one_way.missing()), one_way.missing())


class SetupTest(unittest.TestCase):
    def setUp(self):
        self.det = cal.BoardDetector(CFG)
        self.K_b = np.array([[1150.0, 0, 640], [0, 1150.0, 512], [0, 0, 1]])
        self.D_b = np.array([-0.05, 0.01, 0, 0, 0])
        self.Ra, self.ta = pose_looking_at_board(0.6, (-20, 10, 0))
        self.Rb, self.tb = pose_looking_at_board(0.7, (15, -25, 40))
        self.da = self.det.detect(render_view(self.Ra, self.ta))
        self.db = self.det.detect(render_view(self.Rb, self.tb, self.K_b, self.D_b, (1280, 1024)))
        self.intr = {"A": (K_TRUE, D_TRUE), "B": (self.K_b, self.D_b)}

    def test_two_cameras_fixed_board(self):
        result = cal.compute_setup({"A": self.da, "B": self.db}, self.intr, self.det.board)
        true_baseline = np.linalg.norm((-self.Ra.T @ self.ta) - (-self.Rb.T @ self.tb)) * 1000
        self.assertAlmostEqual(result.baseline_mm, true_baseline, delta=true_baseline * 0.01)
        self.assertLess(result.triangulation_rms_mm, 0.5)
        self.assertEqual(result.triangulated_corners, CFG.corner_count)
        self.assertTrue(result.passed)

    def _verify(self, intr, offset=(0.08, -0.06, -0.22)):
        """Setup from the fixed board, then the board moved by (M, offset) in the fixed board's frame
        (negative z = raised toward the cameras) and verified with the SAVED poses."""
        setup = cal.compute_setup({"A": self.da, "B": self.db}, intr, self.det.board)
        M, _ = cv2.Rodrigues(np.radians([10, -15, 20]))
        m = np.array(offset)
        va = self.det.detect(render_view(self.Ra @ M, self.Ra @ m + self.ta))
        vb = self.det.detect(render_view(self.Rb @ M, self.Rb @ m + self.tb, self.K_b, self.D_b, (1280, 1024)))
        return setup, cal.verify_setup(va, setup.poses["A"], *intr["A"], vb, setup.poses["B"], *intr["B"],
                                       self.det.board)

    def test_verify_with_the_board_raised(self):
        setup, verify = self._verify(self.intr)
        self.assertTrue(setup.passed)
        self.assertTrue(verify.passed, verify.problems)
        self.assertLess(abs(verify.scale_error_pct), 0.5)
        self.assertGreaterEqual(verify.corners, cal.MIN_CORNERS)  # partly out of view is fine
        M, _ = cv2.Rodrigues(np.radians([10, -15, 20]))
        centre = np.array([CFG.size_m[0] / 2, CFG.size_m[1] / 2, 0.0])
        self.assertAlmostEqual(verify.depth_change_m, abs((M @ centre + [0.08, -0.06, -0.22])[2]), delta=0.01)

    def test_wrong_intrinsics_pass_the_setup_but_fail_verification(self):
        bad = {"A": (K_TRUE * np.array([[1.1], [1.1], [1]]), D_TRUE), "B": self.intr["B"]}
        setup, verify = self._verify(bad)
        self.assertTrue(setup.passed)  # self-consistent: exactly why verification exists
        self.assertFalse(verify.passed)
        self.assertTrue(any("real size" in p for p in verify.problems), verify.problems)

    def test_verification_spot_too_close_to_the_setup_board(self):
        _, verify = self._verify(self.intr, offset=(0.1, -0.05, -0.03))
        self.assertFalse(verify.passed)
        self.assertTrue(any("raise the board" in p for p in verify.problems), verify.problems)


class PresetsTest(unittest.TestCase):
    def test_presets_are_valid_boards(self):
        for name, cfg in cal.BOARD_PRESETS.items():
            with self.subTest(name=name):
                self.assertEqual(cal.make_board(cfg).getChessboardSize(), (cfg.squares_x, cfg.squares_y))
        self.assertNotEqual(cal.BOARD_PRESETS[cal.HANDHELD_PRESET].dictionary,
                            cal.BOARD_PRESETS[cal.FIXED_PRESET].dictionary)

    def test_measured_square_rescales_markers_too(self):
        cfg = cal.with_measured_square(cal.BOARD_PRESETS[cal.HANDHELD_PRESET], 35.5)
        self.assertAlmostEqual(cfg.square_length_m, 0.0355)
        self.assertAlmostEqual(cfg.marker_length_m, 0.027 * 35.5 / 36)


class CaptureSessionTest(unittest.TestCase):
    def setUp(self):
        self.det = cal.BoardDetector(CFG)
        self.session = cal.CaptureSession(self.det.board, still_s=0.5)

    def feed_still(self, det, start, seconds=1.0, step=0.2):
        states = []
        t = start
        while t <= start + seconds + 1e-9:
            states.append(self.session.feed(det, t).state)
            t += step
        return states

    def test_takes_a_view_only_after_holding_still(self):
        view = self.det.detect(render_view(*pose_looking_at_board(0.5)))
        states = self.feed_still(view, 0.0)
        self.assertEqual(states[0], "moving")
        self.assertIn("steady", states)
        self.assertEqual(states.count("captured"), 1)
        self.assertEqual(states[-1], "seen already")  # holding on does not add duplicates
        self.assertEqual(len(self.session.views), 1)

    def test_no_board_resets_and_new_poses_are_added(self):
        empty = self.det.detect(np.full((540, 720), 90, np.uint8))
        self.assertEqual(self.session.feed(empty, 0.0).state, "no board")
        t = 1.0
        for R, tv in varied_views(4, seed=5):
            self.feed_still(self.det.detect(render_view(R, tv)), t)
            t += 2.0
        self.assertGreaterEqual(len(self.session.views), 3)
        self.assertEqual(self.session.coverage.views, len(self.session.views))

    def test_manual_capture_undo_and_drop(self):
        view = self.det.detect(render_view(*pose_looking_at_board(0.5)))
        self.assertTrue(self.session.capture_now(view))
        self.assertTrue(self.session.capture_now(view))  # manual allows similar views
        self.session.drop([0])
        self.assertEqual(len(self.session.views), 1)
        self.assertTrue(self.session.undo())
        self.assertFalse(self.session.undo())
        empty = self.det.detect(np.full((540, 720), 90, np.uint8))
        self.assertFalse(self.session.capture_now(empty))

    def test_outlier_views(self):
        self.assertEqual(cal.outlier_views([0.2, 0.25, 0.22, 0.9]), [3])
        self.assertEqual(cal.outlier_views([0.2, 0.9]), [])


if __name__ == "__main__":
    unittest.main()
