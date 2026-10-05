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


_BOARD_CACHE: dict = {}


def render_scene(items, K=K_TRUE, D=D_TRUE, size=IMAGE_SIZE, background=90, ppm=2500.0):
    """One camera image showing several boards: items = [(cfg, R, t)], board -> camera poses.

    Later items are drawn over earlier ones; distortion is applied once to the
    whole image, as a real lens would.
    """
    ideal = np.full((size[1], size[0]), background, np.uint8)
    S = np.array([[1 / ppm, 0, 0.5 / ppm], [0, 1 / ppm, 0.5 / ppm], [0, 0, 1]])
    for cfg, R, t in items:
        if cfg not in _BOARD_CACHE:
            _BOARD_CACHE[cfg] = np.where(cal.render_board(cfg, ppm) > 127, 230, 25).astype(np.uint8)
        board = _BOARD_CACHE[cfg]
        H = K @ np.column_stack([R[:, 0], R[:, 1], t]) @ S
        warped = cv2.warpPerspective(board, H, size, flags=cv2.INTER_LINEAR, borderValue=background)
        mask = cv2.warpPerspective(np.full_like(board, 255), H, size, flags=cv2.INTER_NEAREST, borderValue=0)
        ideal[mask > 0] = warped[mask > 0]
    if not np.any(D):
        return ideal
    w, h = size
    grid = np.stack(np.meshgrid(np.arange(w), np.arange(h)), -1).reshape(-1, 1, 2).astype(np.float32)
    und = cv2.undistortPoints(grid, K, D, P=K).reshape(h, w, 2)
    return cv2.remap(ideal, und[..., 0], und[..., 1], cv2.INTER_LINEAR, borderValue=background)


def in_camera(R_cam, t_cam, R_obj, t_obj):
    """Pose of an object placed at (R_obj, t_obj) in the world (board A) frame, seen from a camera
    whose world -> camera pose is (R_cam, t_cam)."""
    return R_cam @ R_obj, R_cam @ t_obj + t_cam


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

    def test_a_camera_without_a_pose_fails_the_setup(self):
        blank = cal.Detection(np.zeros((0, 2), np.float32), np.zeros((0,), np.int32), 0, IMAGE_SIZE)
        result = cal.compute_setup({"A": self.da, "B": blank}, self.intr, self.det.board)
        self.assertEqual(list(result.poses), ["A"])
        self.assertEqual(result.missing, ["B"])
        self.assertFalse(result.passed)
        result = cal.compute_setup({"A": self.da, "B": None}, self.intr, self.det.board)
        self.assertFalse(result.passed)
        # One camera on its own is still a valid single-camera check.
        self.assertTrue(cal.compute_setup({"A": self.da}, self.intr, self.det.board).passed)

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


class MovedCheckTest(unittest.TestCase):
    """The fixed-board drift check at the rig's working distance (1.2 m, A4 board)."""

    def setUp(self):
        self.cfg = cal.BOARD_PRESETS[cal.FIXED_PRESET]
        self.board = cal.make_board(self.cfg)
        self.K = np.array([[1150.0, 0, 640], [0, 1150.0, 512], [0, 0, 1]])
        self.ids = np.arange(self.cfg.corner_count, dtype=np.int32)
        self.obj = self.board.getChessboardCorners().astype(np.float64)
        self.rng = np.random.default_rng(0)
        self.centre = np.array(self.cfg.centre_m)

    def pose(self, rot_deg, shift=(0.0, 0.0, 0.0), dist=1.2):
        R, _ = cv2.Rodrigues(np.radians(np.array(rot_deg, dtype=float)))
        return R, np.array([0, 0, dist]) - R @ self.centre + np.array(shift)

    def solve(self, R, t, noise_px):
        rvec, _ = cv2.Rodrigues(R)
        pts, _ = cv2.projectPoints(self.obj, rvec, t, self.K, np.zeros(5))
        pts = pts.reshape(-1, 2) + self.rng.normal(0, noise_px, (len(self.obj), 2))
        det = cal.Detection(pts.astype(np.float32), self.ids, 12, (1280, 1024), self.cfg.squares_x - 1)
        return cal.solve_board_pose(det, self.board, self.K, np.zeros(5))

    def test_corner_noise_is_not_a_move(self):
        R, t = self.pose((20, 0, 0))
        saved = self.solve(R, t, 0.0)
        old_style_mm = []
        # At 0.2 px a single frame's rotation (limit 0.5 deg) is itself near the noise: a few %
        # of single frames, which is why the live pose should be an average of many frames.
        for noise, allowed_false_alarms in ((0.1, 0), (0.2, 0.1)):
            checks = [cal.compare_poses(saved, self.solve(R, t, noise), self.centre) for _ in range(200)]
            # Translation is never the reason, even at 0.2 px (the board centre moves ~1-3 mm; limit 5 mm).
            self.assertLess(max(c.translation_mm for c in checks), cal.SETUP_MOVED_TRANSLATION_MM)
            self.assertLessEqual(np.mean([c.moved for c in checks]), allowed_false_alarms, noise)
        # What this replaced: the camera centre in the board frame, at 0.2 px.
        for _ in range(200):
            live = self.solve(R, t, 0.2)
            old_style_mm.append(np.linalg.norm(saved.camera_centre_m - live.camera_centre_m) * 1000)
        self.assertGreater(np.mean(np.array(old_style_mm) > cal.SETUP_MOVED_TRANSLATION_MM), 0.25)

    def test_a_real_turn_or_shift_is_detected(self):
        R0, t0 = self.pose((20, 0, 0))
        saved = self.solve(R0, t0, 0.0)
        turned = cal.compare_poses(saved, self.solve(*self.pose((20, 2, 0)), 0.1), self.centre)
        self.assertTrue(turned.moved)
        self.assertGreater(turned.rotation_deg, 1.5)
        for axis in ((0.01, 0, 0), (0, 0.01, 0), (0, 0, 0.01)):  # 10 mm sideways, up, toward the camera
            shifted = cal.compare_poses(saved, self.solve(*self.pose((20, 0, 0), axis), 0.1), self.centre)
            self.assertTrue(shifted.moved, axis)
            self.assertAlmostEqual(shifted.translation_mm, 10.0, delta=1.5)
        same = cal.compare_poses(saved, saved, self.centre)
        self.assertAlmostEqual(same.translation_mm, 0.0)
        self.assertAlmostEqual(same.rotation_deg, 0.0, places=3)
        self.assertFalse(same.moved)


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


class MarkerIdRangeTest(unittest.TestCase):
    def test_reference_boards_share_a_dictionary_but_not_ids(self):
        b1, b2 = (cal.BOARD_PRESETS[n] for n in cal.REFERENCE_PRESETS)
        self.assertEqual(b1.dictionary, b2.dictionary)
        self.assertEqual((b1.first_marker_id, b2.first_marker_id), (0, 17))
        self.assertEqual(cal.with_measured_square(b2, 35.0).first_marker_id, 17)
        self.assertEqual(cal.BoardConfig.from_dict(b2.to_dict()), b2)
        # A record saved before first_marker_id existed reads as ids from 0.
        legacy = {k: v for k, v in b1.to_dict().items() if k != "first_marker_id"}
        self.assertEqual(cal.BoardConfig.from_dict(legacy), b1)

    def test_ids_must_fit_the_dictionary(self):
        with self.assertRaises(ValueError):
            cal.BoardConfig(7, 5, 0.036, 0.027, "DICT_4X4_50", 40)  # 40..56 > 49
        with self.assertRaises(ValueError):
            cal.BoardConfig(7, 5, 0.036, 0.027, "DICT_4X4_50", -1)

    def test_each_reference_detector_sees_only_its_own_board(self):
        b1, b2 = (cal.BOARD_PRESETS[n] for n in cal.REFERENCE_PRESETS)
        img = render_scene([(b1, *pose_looking_at_board(0.9, (0, 0, 0), (-0.15, 0), b1)),
                            (b2, *pose_looking_at_board(0.9, (0, 0, 0), (0.15, 0), b2))], D=np.zeros(5))
        d1, d2 = cal.BoardDetector(b1).detect(img), cal.BoardDetector(b2).detect(img)
        self.assertTrue(d1.ok and d2.ok)
        self.assertLess(d1.corners[:, 0].max(), IMAGE_SIZE[0] / 2)
        self.assertGreater(d2.corners[:, 0].min(), IMAGE_SIZE[0] / 2)
        self.assertEqual(cal.best_detection({"B1": d1, "B2": None, "x": cal.Detection(
            np.zeros((0, 2), np.float32), np.zeros(0, np.int32), 0, IMAGE_SIZE)})[0], "B1")
        self.assertIsNone(cal.best_detection({"B1": None}))


class ReferenceCheckTest(unittest.TestCase):
    def setUp(self):
        self.cfg = cal.BOARD_PRESETS[cal.REFERENCE_PRESETS[0]]
        self.det = cal.BoardDetector(self.cfg)
        self.R, self.t = pose_looking_at_board(0.9, (25, -10, 5), (0.05, 0.02), self.cfg)
        self.saved = cal.solve_board_pose(self.det.detect(render_scene([(self.cfg, self.R, self.t)])),
                                          self.det.board, K_TRUE, D_TRUE)

    def live(self, R, t, K=K_TRUE):
        return cal.solve_board_pose(self.det.detect(render_scene([(self.cfg, R, t)], K=K)), self.det.board, K_TRUE, D_TRUE)

    def test_still_camera(self):
        check = cal.check_reference(self.saved.R, self.saved.t, self.saved.rms_px, self.live(self.R, self.t), self.cfg)
        self.assertFalse(check.moved)
        self.assertFalse(check.suspect)

    def test_turned_camera(self):
        turn, _ = cv2.Rodrigues(np.radians([0, 2.0, 0]))  # the camera turned 2 degrees
        check = cal.check_reference(self.saved.R, self.saved.t, self.saved.rms_px,
                                    self.live(turn @ self.R, turn @ self.t), self.cfg)
        self.assertTrue(check.moved)
        self.assertAlmostEqual(check.move.rotation_deg, 2.0, delta=0.3)

    def test_touched_lens_is_suspect(self):
        zoomed = K_TRUE * np.array([[1.15], [1.15], [1]])  # focal changed, intrinsics not updated
        check = cal.check_reference(self.saved.R, self.saved.t, self.saved.rms_px,
                                    self.live(self.R, self.t, K=zoomed), self.cfg)
        self.assertTrue(check.suspect or check.moved, (check.rms_px, check.move))


if __name__ == "__main__":
    unittest.main()
