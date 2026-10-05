"""backend/reference_monitor.py: the live "has a camera moved?" check."""
import unittest

import cv2
import numpy as np

import test_calibration as synth
from backend import calibration as cal
from backend import calibration_store as cs
from backend import reference_monitor as rm

B1 = cal.BOARD_PRESETS[cal.REFERENCE_PRESETS[0]]


def check(moved=False, suspect=False, rms=0.3):
    move = cal.MoveCheck(translation_mm=10.0 if moved else 0.5, rotation_deg=0.05)
    return cal.ReferenceCheck(move=move, rms_px=rms, suspect=suspect)


class HysteresisTest(unittest.TestCase):
    def test_needs_two_agreeing_checks(self):
        h = rm.MoveHysteresis()
        self.assertFalse(h.update({"A": check(False)}))
        self.assertEqual(h.state, {})
        self.assertTrue(h.update({"A": check(False)}))
        self.assertEqual(h.state, {"A": False})
        self.assertFalse(h.update({"A": check(True)}))   # one noisy result: no flip
        self.assertFalse(h.update({"A": check(False)}))
        self.assertFalse(h.update({"A": check(True)}))
        self.assertTrue(h.update({"A": check(True)}))
        self.assertEqual(h.state, {"A": True})

    def test_hidden_reference_keeps_the_last_state(self):
        h = rm.MoveHysteresis()
        h.update({"A": check(True)})
        h.update({"A": check(True)})
        self.assertFalse(h.update({"A": None}))
        self.assertEqual(h.state, {"A": True})

    def test_suspect_rms_is_reported_and_cleared(self):
        h = rm.MoveHysteresis()
        self.assertTrue(h.update({"A": check(suspect=True, rms=2.4)}))
        self.assertEqual(h.suspect_rms, {"A": 2.4})
        self.assertTrue(h.update({"A": check()}))
        self.assertEqual(h.suspect_rms, {})

    def test_reset(self):
        h = rm.MoveHysteresis()
        h.update({"A": check(True)})
        h.update({"A": check(True)})
        h.reset()
        self.assertEqual(h.state, {})


def intrinsics(serial, rec_id="i1"):
    return cs.IntrinsicsRecord(serial=serial, model="m", K=synth.K_TRUE.tolist(), D=list(synth.D_TRUE),
                               image_size=[720, 540], fingerprint={}, board={}, rms_px=0.3, n_views=30, id=rec_id)


def setup_with_reference(serial, R, t, rms=0.3, intrinsics_id="i1"):
    ref = {"name": cal.REFERENCE_PRESETS[0], "board": B1.to_dict(), "R": R.tolist(), "t": t.tolist(), "rms_px": rms}
    return cs.SetupRecord(cameras=[{"serial": serial, "R": [], "t": [], "rms_px": 0.3,
                                    "intrinsics_id": intrinsics_id, "reference": ref}],
                          board={}, baseline_mm=None, triangulation_rms_mm=None, passed=True)


class TargetsTest(unittest.TestCase):
    def test_only_cameras_with_a_reference_and_current_intrinsics(self):
        R, t = synth.pose_looking_at_board(0.9, cfg=B1)
        setup = setup_with_reference("A", R, t)
        setup.cameras.append({"serial": "B", "R": [], "t": [], "rms_px": 0.3, "intrinsics_id": "i1",
                              "reference": None})
        self.assertEqual([x.serial for x in rm.targets_from(setup, {"A": intrinsics("A"), "B": intrinsics("B")})],
                         ["A"])
        # Recalibrated since the setup: its saved reference pose no longer applies.
        self.assertEqual(rm.targets_from(setup, {"A": intrinsics("A", "i2")}), [])
        self.assertEqual(rm.targets_from(None, {}), [])
        broken = setup_with_reference("A", R, t)
        broken.cameras[0]["reference"]["board"] = {"squares_x": "x"}
        self.assertEqual(rm.targets_from(broken, {"A": intrinsics("A")}), [])


class MeasureTest(unittest.TestCase):
    def setUp(self):
        self.R, self.t = synth.pose_looking_at_board(0.9, (20, -10, 0), (0.0, 0.03), B1)
        frame = synth.render_scene([(B1, self.R, self.t)])
        saved = cal.solve_board_pose(cal.BoardDetector(B1).detect(frame), cal.make_board(B1),
                                     synth.K_TRUE, synth.D_TRUE)
        self.targets = rm.targets_from(setup_with_reference("A", saved.R, saved.t, saved.rms_px),
                                       {"A": intrinsics("A")})
        self.frame = frame

    def test_still_and_moved_and_hidden(self):
        sleeps = []
        still = rm.measure_references(self.targets, lambda s: self.frame, sleep=sleeps.append)
        self.assertEqual(len(sleeps), rm.FRAMES_PER_CHECK - 1)
        self.assertFalse(still["A"].moved)

        turn, _ = cv2.Rodrigues(np.radians([1.5, 0, 0]))
        moved_frame = synth.render_scene([(B1, turn @ self.R, turn @ self.t)])
        moved = rm.measure_references(self.targets, lambda s: moved_frame, sleep=lambda s: None)
        self.assertTrue(moved["A"].moved)

        blank = np.full((540, 720), 90, np.uint8)
        self.assertIsNone(rm.measure_references(self.targets, lambda s: blank, sleep=lambda s: None)["A"])
        self.assertIsNone(rm.measure_references(self.targets, lambda s: None, sleep=lambda s: None)["A"])


if __name__ == "__main__":
    unittest.main()
