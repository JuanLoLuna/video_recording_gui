"""The "3D pose" line in the real MainWindow (Qt offscreen, fake cameras).

Checks plan step 12: the status follows the stored calibrations and the live
camera settings, never blocks recording, and every recording gets a
<basename>_calibration.json with what was known at start.
"""
import json
import os
import tempfile
import time
import unittest
from datetime import datetime, timedelta
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from fake_spinnaker import FakeCamera, FakeSystem, install_pyspin_stub  # noqa: E402

REAL_PYSPIN = install_pyspin_stub()
try:
    from PySide6.QtWidgets import QApplication
    HAVE_QT = True
except ImportError:  # pragma: no cover
    HAVE_QT = False

if HAVE_QT:
    import backend.spinnaker_system as spinnaker_system
    from backend import calibration_store as cs
    from backend.camera_registry import CAMERA_SERIALS_ENV
    from backend.spinnaker_system import SharedSystemHolder
    from gui.main import AppState, MainWindow

FIREFLY = ("23227865", "Firefly", 32, 24)
BLACKFLY = ("26134271", "Blackfly S", 64, 48)
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


@unittest.skipIf(REAL_PYSPIN or not HAVE_QT, "needs PySide6 and no real PySpin")
class CalibrationStatusTests(unittest.TestCase):
    def setUp(self):
        app()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.out_dir = Path(self.tmp.name) / "out"
        self.cal_dir = Path(self.tmp.name) / "cal"
        keys = ("SLEEVE_VIDEO_GUI_SEGMENT_SECONDS", CAMERA_SERIALS_ENV, cs.CALIBRATION_DIR_ENV)
        saved = {k: os.environ.get(k) for k in keys}
        self.addCleanup(lambda: [os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
                                 for k, v in saved.items()])
        os.environ["SLEEVE_VIDEO_GUI_SEGMENT_SECONDS"] = "1"
        os.environ.pop(CAMERA_SERIALS_ENV, None)
        os.environ[cs.CALIBRATION_DIR_ENV] = str(self.cal_dir)
        previous = spinnaker_system._default_holder
        self.addCleanup(lambda: setattr(spinnaker_system, "_default_holder", previous))
        self.store = cs.CalibrationStore(self.cal_dir)

    def window(self, *specs):
        self.cameras = [FakeCamera(*spec) for spec in specs]
        spinnaker_system._default_holder = SharedSystemHolder(lambda: FakeSystem(self.cameras))
        window = MainWindow()
        window._configured_output_dir = str(self.out_dir)
        window._confirm_power_safe_to_record = lambda: True
        window._confirm_disk_safe_to_record = lambda *a, **k: True
        self.addCleanup(window.close)
        return window

    def preview(self, window):
        window.on_detect_clicked()
        self.assertEqual(window.state, AppState.CAMERA_DETECTED, window.status_label.text())
        window.on_preview_clicked()
        self.assertEqual(window.state, AppState.PREVIEWING, window.status_label.text())
        self.addCleanup(lambda: window.preview_running and window.state != AppState.RECORDING
                        and window.on_preview_clicked())

    def calibrate_all(self, window, *, setup_at=None, verified=True):
        """Store intrinsics matching each live camera, and a setup over them."""
        ids = {}
        for slot in window._slots():
            rec = cs.IntrinsicsRecord(
                serial=slot.serial, model=slot.model, K=[[1, 0, 0], [0, 1, 0], [0, 0, 1]], D=[0] * 5,
                image_size=[0, 0], fingerprint=slot.controller.get_sensor_fingerprint(), board={},
                rms_px=0.3, n_views=30)
            self.store.save_intrinsics(rec)
            ids[slot.serial] = rec.id
        setup = cs.SetupRecord(
            cameras=[{"serial": s, "R": [], "t": [], "rms_px": 0.4, "intrinsics_id": i} for s, i in ids.items()],
            board={}, baseline_mm=500.0, triangulation_rms_mm=0.9, passed=True,
            verify={"passed": verified})
        self.store.save_setup(setup, now=setup_at or datetime.now() + timedelta(seconds=1))

    def test_before_detect_and_nothing_calibrated(self):
        window = self.window(BLACKFLY, FIREFLY)
        self.assertIn("detect cameras first", window.calibration_bar.label.text())
        self.assertFalse(window.calibration_bar.calibrate_button.isEnabled())
        window.on_detect_clicked()
        text = window.calibration_bar.label.text()
        self.assertTrue(text.startswith("3D pose not available"), text)
        self.assertIn("[Firefly #23227865] camera not calibrated", text)
        self.assertIn("[Blackfly S #26134271] camera not calibrated", text)
        self.assertIn(cs.NO_SETUP, text)
        self.assertTrue(window.calibration_bar.calibrate_button.isEnabled())

    def test_one_camera_needs_two(self):
        window = self.window(FIREFLY)
        window.on_detect_clicked()
        self.assertIn("needs two cameras", window.calibration_bar.label.text())

    def test_ready_then_recording_carries_the_snapshot(self):
        window = self.window(BLACKFLY, FIREFLY)
        self.preview(window)
        self.calibrate_all(window)
        window._refresh_calibration_status()
        self.assertEqual(window.calibration_bar.label.text(), "3D pose: ready")

        pump(0.4)
        self.assertTrue(window._begin_recording_session(bypass_confirmation=True), window.status_label.text())
        self.assertFalse(window.calibration_bar.calibrate_button.isEnabled())  # not while recording
        pump(0.6)
        window._stop_recording_session("stopped")
        snapshots = list(self.out_dir.glob("recording_*_calibration.json"))
        self.assertEqual(len(snapshots), 1, list(self.out_dir.iterdir()))
        snap = json.loads(snapshots[0].read_text())
        self.assertTrue(snap["status"]["ready"])
        self.assertEqual(sorted(snap["intrinsics"]), ["23227865", "26134271"])
        self.assertEqual(sorted(c["serial"] for c in snap["setup"]["cameras"]), ["23227865", "26134271"])

    def test_not_ready_still_records_and_says_why(self):
        window = self.window(BLACKFLY, FIREFLY)
        self.preview(window)
        pump(0.4)
        self.assertTrue(window._begin_recording_session(bypass_confirmation=True))
        pump(0.4)
        window._stop_recording_session("stopped")
        snap = json.loads(next(self.out_dir.glob("recording_*_calibration.json")).read_text())
        self.assertFalse(snap["status"]["ready"])
        self.assertIsNone(snap["setup"])

    def test_changed_camera_settings_invalidate_intrinsics(self):
        window = self.window(BLACKFLY, FIREFLY)
        self.preview(window)
        self.calibrate_all(window)
        self.cameras[1]._nodemap.values["Width"] = 16  # Firefly now reads a different width
        window._refresh_calibration_status()
        text = window.calibration_bar.label.text()
        self.assertIn("[Firefly #23227865] camera settings changed since calibration (Width 32 -> 16)", text)

    def test_setup_from_before_detect_is_stale_until_the_board_check(self):
        window = self.window(BLACKFLY, FIREFLY)
        self.preview(window)
        self.calibrate_all(window, setup_at=datetime.now() - timedelta(days=1))
        window._refresh_calibration_status()
        self.assertIn(cs.SETUP_OLD_SESSION, window.calibration_bar.label.text())
        window._calibration_moved = {"23227865": False, "26134271": False}
        window._refresh_calibration_status()
        self.assertEqual(window.calibration_bar.label.text(), "3D pose: ready")

    def test_a_broken_calibration_folder_does_not_break_the_window(self):
        window = self.window(BLACKFLY, FIREFLY)
        window._calibration_store.current_setup = lambda: (_ for _ in ()).throw(OSError("disk gone"))
        window.on_detect_clicked()
        self.assertIn("could not read calibrations", window.calibration_bar.label.text())
        self.assertEqual(window.state, AppState.CAMERA_DETECTED)


if __name__ == "__main__":
    unittest.main()
