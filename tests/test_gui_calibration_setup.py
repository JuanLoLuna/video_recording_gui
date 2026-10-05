"""Set up for this session (plan step 14), end to end in the real windows.

Two fake cameras with known lenses look at a rendered scene: board A resting in
the world (its frame IS the world), reference boards B1/B2 beside it, and for
verification board A raised on a "box". The test clicks through the setup task,
then drives the main window's live reference check, including a camera turned
by 2 degrees after the setup.
"""
import os
import tempfile
import time
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from fake_spinnaker import FakeCamera, FakeImage, FakeSystem, install_pyspin_stub  # noqa: E402

REAL_PYSPIN = install_pyspin_stub()
try:
    from PySide6.QtWidgets import QApplication
    HAVE_QT = True
except ImportError:  # pragma: no cover
    HAVE_QT = False

import cv2  # noqa: E402
import numpy as np  # noqa: E402

if HAVE_QT:
    import backend.spinnaker_system as spinnaker_system
    import test_calibration as synth
    from backend import calibration as cal
    from backend import calibration_store as cs
    from backend.camera_registry import CAMERA_SERIALS_ENV
    from backend.spinnaker_system import SharedSystemHolder
    from gui.main import AppState, MainWindow

FIREFLY = ("23227865", "Firefly", 720, 540)
BLACKFLY = ("26134271", "Blackfly S", 1280, 1024)
K2 = np.array([[1150.0, 0, 640], [0, 1150.0, 512], [0, 0, 1]])
D2 = np.array([-0.05, 0.01, 0, 0, 0])
_app = None


def app():
    global _app
    if _app is None:
        _app = QApplication.instance() or QApplication([])
    return _app


def pump(seconds):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        app().processEvents()
        time.sleep(0.02)


def pump_until(condition, timeout):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        app().processEvents()
        if condition():
            return True
        time.sleep(0.02)
    return False


@unittest.skipIf(REAL_PYSPIN or not HAVE_QT, "needs PySide6 and no real PySpin")
class SessionSetupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        A = cal.BOARD_PRESETS[cal.HANDHELD_PRESET]
        B1, B2 = (cal.BOARD_PRESETS[n] for n in cal.REFERENCE_PRESETS)
        I = np.eye(3)
        # world (board A) -> camera poses
        cls.R1, cls.t1 = synth.pose_looking_at_board(1.0, (-20, 10, 0), (0.06, 0.0), A)
        cls.R2, cls.t2 = synth.pose_looking_at_board(1.1, (15, -25, 30), (-0.08, 0.0), A)
        refs = [(B1, I, np.array([-0.30, 0.0, 0.0])), (B2, I, np.array([0.30, 0.0, 0.0]))]
        M, _ = cv2.Rodrigues(np.radians([10, -15, 20]))
        a_rest, a_raised = (I, np.zeros(3)), (M, np.array([0.06, -0.04, -0.22]))  # -z = toward the cameras

        def scene(Rc, tc, K, D, size, a_pose):
            items = [(A, *synth.in_camera(Rc, tc, *a_pose))]
            items += [(cfg, *synth.in_camera(Rc, tc, R, t)) for cfg, R, t in refs]
            return synth.render_scene(items, K=K, D=D, size=size)

        turn, _ = cv2.Rodrigues(np.radians([0, 2.0, 0]))  # camera 2 bumped by 2 degrees
        cls.frames = {
            "setup": {720: scene(cls.R1, cls.t1, synth.K_TRUE, synth.D_TRUE, (720, 540), a_rest),
                      1280: scene(cls.R2, cls.t2, K2, D2, (1280, 1024), a_rest)},
            "verify": {720: scene(cls.R1, cls.t1, synth.K_TRUE, synth.D_TRUE, (720, 540), a_raised),
                       1280: scene(cls.R2, cls.t2, K2, D2, (1280, 1024), a_raised)},
        }
        cls.frames["moved"] = {720: cls.frames["setup"][720],
                               1280: scene(turn @ cls.R2, turn @ cls.t2, K2, D2, (1280, 1024), a_rest)}
        cls.true_baseline_mm = np.linalg.norm((-cls.R1.T @ cls.t1) - (-cls.R2.T @ cls.t2)) * 1000

    def setUp(self):
        app()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        keys = ("SLEEVE_VIDEO_GUI_SEGMENT_SECONDS", CAMERA_SERIALS_ENV, cs.CALIBRATION_DIR_ENV)
        saved = {k: os.environ.get(k) for k in keys}
        self.addCleanup(lambda: [os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
                                 for k, v in saved.items()])
        os.environ["SLEEVE_VIDEO_GUI_SEGMENT_SECONDS"] = "1"
        os.environ.pop(CAMERA_SERIALS_ENV, None)
        os.environ[cs.CALIBRATION_DIR_ENV] = str(Path(self.tmp.name) / "cal")
        previous = spinnaker_system._default_holder
        self.addCleanup(lambda: setattr(spinnaker_system, "_default_holder", previous))
        original = FakeImage.GetNDArray
        self.addCleanup(lambda: setattr(FakeImage, "GetNDArray", original))
        self.scene = "setup"
        FakeImage.GetNDArray = lambda image: self.frames[self.scene][image.width]

    def main_window(self):
        self.cameras = [FakeCamera(*BLACKFLY), FakeCamera(*FIREFLY)]
        spinnaker_system._default_holder = SharedSystemHolder(lambda: FakeSystem(self.cameras))
        window = MainWindow()
        window._configured_output_dir = self.tmp.name
        window._confirm_power_safe_to_record = lambda: True
        window._confirm_disk_safe_to_record = lambda *a, **k: True
        self.addCleanup(window.close)
        window.on_detect_clicked()
        window.on_preview_clicked()
        self.assertEqual(window.state, AppState.PREVIEWING, window.status_label.text())
        pump(0.3)
        # Both cameras calibrated with their true lenses (camera calibration is tested elsewhere).
        for slot in window._slots():
            fp = slot.controller.get_sensor_fingerprint()
            K, D = (synth.K_TRUE, synth.D_TRUE) if slot.serial == FIREFLY[0] else (K2, D2)
            window._calibration_store.save_intrinsics(cs.IntrinsicsRecord(
                serial=slot.serial, model=slot.model, K=K.tolist(), D=[float(d) for d in D],
                image_size=[fp["Width"], fp["Height"]], fingerprint=fp, board={}, rms_px=0.3, n_views=30))
        window._on_calibration_saved()
        self.assertIn(cs.NO_SETUP, window.calibration_bar.label.text())
        return window

    def run_setup(self, window):
        window._open_calibration_window()
        cw = window._calibration_window
        cw.setup_button.click()
        self.assertIs(cw.stack.currentWidget(), cw.page_setup_check)
        self.assertIn("calibrated", cw.setup_cameras_label.text())
        self.assertFalse(cw.setup_check_next.isEnabled())
        for box in cw.setup_checks:
            box.setChecked(True)
        cw.setup_check_next.click()
        self.assertIs(cw.stack.currentWidget(), cw.page_setup_live)

        self.assertTrue(pump_until(cw.setup_capture_button.isEnabled, timeout=10), "board A never seen by both")
        cw.setup_capture_button.click()
        self.assertTrue(pump_until(lambda: cw._setup_result is not None, timeout=10), cw.setup_text.text())
        self.assertTrue(cw._setup_result.passed, cw.setup_text.text())
        self.assertIn("PASS", cw.setup_verdict.text())
        self.assertAlmostEqual(cw._setup_result.baseline_mm, self.true_baseline_mm,
                               delta=self.true_baseline_mm * 0.02)
        self.assertTrue(all(cw._setup_refs[s] is not None for s in (FIREFLY[0], BLACKFLY[0])))

        cw.setup_next_button.click()
        self.assertFalse(cw.setup_save_button.isEnabled())
        self.scene = "verify"
        pump(0.8)  # let both workers see the raised board before capturing
        self.assertTrue(pump_until(cw.verify_button.isEnabled, timeout=10))
        cw.verify_button.click()
        self.assertTrue(pump_until(lambda: cw._verify_result is not None, timeout=10), cw.setup_text.text())
        self.assertTrue(cw._verify_result.passed, cw.setup_text.text())
        self.assertLess(abs(cw._verify_result.scale_error_pct), 1.0)
        cw.setup_save_button.click()
        self.assertIs(cw.stack.currentWidget(), cw.page_home)
        self.assertIn("Setup saved", cw.saved_label.text())
        self.assertEqual(cw._setup_detectors, {})
        self.scene = "setup"  # board A back on the table
        return cw

    def test_setup_verify_save_then_the_live_check(self):
        window = self.main_window()
        self.run_setup(window)
        setup = window._calibration_store.current_setup()
        self.assertTrue(setup.passed and setup.verified)
        self.assertEqual(sorted(setup.serials), sorted([FIREFLY[0], BLACKFLY[0]]))
        for cam in setup.cameras:
            self.assertIsNotNone(cam["reference"])
            self.assertIn(cam["reference"]["name"], cal.REFERENCE_PRESETS)
        self.assertEqual(window.calibration_bar.label.text(), "3D pose: ready")

        # Live check: nothing moved (needs two agreeing rounds).
        for _ in range(2):
            window._apply_reference_results(window._measure_reference_boards())
        self.assertEqual(window._calibration_moved, {FIREFLY[0]: False, BLACKFLY[0]: False})
        self.assertEqual(window.calibration_bar.label.text(), "3D pose: ready")

        # Camera 2 gets bumped.
        self.scene = "moved"
        window._apply_reference_results(window._measure_reference_boards())
        self.assertEqual(window.calibration_bar.label.text(), "3D pose: ready")  # one round is not enough
        window._apply_reference_results(window._measure_reference_boards())
        text = window.calibration_bar.label.text()
        self.assertIn("camera moved since setup (Blackfly S #26134271)", text)
        self.assertNotIn("Firefly", text)

    def test_a_new_detect_needs_the_live_check_before_reusing_the_setup(self):
        window = self.main_window()
        self.run_setup(window)
        window._close_calibration_window()
        window.on_preview_clicked()   # stop preview
        window.on_detect_clicked()    # cameras may have been re-mounted
        window.on_preview_clicked()
        pump(0.3)
        window._refresh_calibration_status()
        self.assertIn(cs.SETUP_OLD_SESSION, window.calibration_bar.label.text())
        for _ in range(2):
            window._apply_reference_results(window._measure_reference_boards())
        self.assertEqual(window.calibration_bar.label.text(), "3D pose: ready")

    def test_setup_needs_both_cameras_calibrated(self):
        window = self.main_window()
        store = window._calibration_store
        (store.intrinsics_dir(BLACKFLY[0]) / "current.json").unlink()
        window._open_calibration_window()
        cw = window._calibration_window
        cw.setup_button.click()
        self.assertIn("not calibrated: calibrate it first", cw.setup_cameras_label.text())
        for box in cw.setup_checks:
            box.setChecked(True)
        self.assertFalse(cw.setup_check_next.isEnabled())

    def test_diagnosis_snapshot_writes_frames_and_a_report(self):
        import json
        window = self.main_window()
        window._open_calibration_window()
        cw = window._calibration_window
        cw.setup_button.click()
        for box in cw.setup_checks:
            box.setChecked(True)
        cw.setup_check_next.click()
        self.assertTrue(pump_until(cw.setup_capture_button.isEnabled, timeout=10))
        cw.setup_snapshot_button.click()
        folders = list((window._calibration_store.root / "diagnostics").iterdir())
        self.assertEqual(len(folders), 1)
        names = sorted(p.name for p in folders[0].iterdir())
        self.assertEqual(names, sorted([f"{FIREFLY[0]}.png", f"{BLACKFLY[0]}.png", "report.json", "report.txt"]))
        report = json.loads((folders[0] / "report.json").read_text())
        self.assertTrue(report["cameras"][FIREFLY[0]]["boards"]["A"]["usable"])
        self.assertIn("Snapshot saved", cw.setup_text.text())
        self.assertIn("A: markers", (folders[0] / "report.txt").read_text())

    def test_closing_during_setup_stops_the_workers(self):
        window = self.main_window()
        window._open_calibration_window()
        cw = window._calibration_window
        cw.setup_button.click()
        for box in cw.setup_checks:
            box.setChecked(True)
        cw.setup_check_next.click()
        self.assertEqual(len(cw._setup_detectors), 2)
        cw.reject()  # Esc
        self.assertEqual(cw._setup_detectors, {})


if __name__ == "__main__":
    unittest.main()
