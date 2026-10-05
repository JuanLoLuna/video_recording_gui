"""The Calibration window end to end (Qt offscreen, fake cameras showing a rendered board).

A fake camera "sees" a ChArUco board rendered through a known lens, held in a
new pose every second. The test clicks through the real window: checklist ->
automatic capture -> compute -> save, and checks the record and the main
window's 3D status.
"""
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from fake_spinnaker import FakeCamera, FakeImage, FakeSystem, install_pyspin_stub  # noqa: E402

REAL_PYSPIN = install_pyspin_stub()
try:
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication
    HAVE_QT = True
except ImportError:  # pragma: no cover
    HAVE_QT = False

if HAVE_QT:
    import backend.spinnaker_system as spinnaker_system
    import test_calibration as synth
    from backend import calibration as cal
    from backend import calibration_store as cs
    from backend.camera_registry import CAMERA_SERIALS_ENV
    from backend.spinnaker_system import SharedSystemHolder
    from gui.calibration_window import write_board_pdf
    from gui.main import AppState, MainWindow

FIREFLY = ("23227865", "Firefly", 720, 540)
BLACKFLY = ("26134271", "Blackfly S", 64, 48)
LAB_PRESET = "Lab 5x5 board, 40 mm squares"
HOLD_S = 0.8
_app = None


def app():
    global _app
    if _app is None:
        _app = QApplication.instance() or QApplication([])
    return _app


def pump_until(condition, timeout):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        app().processEvents()
        if condition():
            return True
        time.sleep(0.02)
    return False


@unittest.skipIf(REAL_PYSPIN or not HAVE_QT, "needs PySide6 and no real PySpin")
class CalibrationWindowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Board in a new pose every HOLD_S seconds (rendered once, replayed).
        cls.frames = [synth.render_view(R, t) for R, t in synth.varied_views(26, seed=3)]

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
        start = time.monotonic()
        frames = self.frames

        def board_frame(image):
            if image.width != FIREFLY[2]:
                return original(image)
            return frames[int((time.monotonic() - start) / HOLD_S) % len(frames)]

        FakeImage.GetNDArray = board_frame

    def main_window(self):
        self.cameras = [FakeCamera(*BLACKFLY), FakeCamera(*FIREFLY)]
        spinnaker_system._default_holder = SharedSystemHolder(lambda: FakeSystem(self.cameras))
        window = MainWindow()
        window._configured_output_dir = self.tmp.name
        window._confirm_power_safe_to_record = lambda: True
        window._confirm_disk_safe_to_record = lambda *a, **k: True
        self.addCleanup(window.close)
        window.on_detect_clicked()
        self.assertEqual(window.state, AppState.CAMERA_DETECTED)
        return window

    def open(self, window):
        window._open_calibration_window()
        cw = window._calibration_window
        self.assertIsNotNone(cw)
        self.assertTrue(cw.isVisible())
        return cw

    def test_calibrate_a_camera_end_to_end(self):
        window = self.main_window()
        cw = self.open(window)
        self.assertIn("23227865", cw.camera_state_label.text())
        self.assertIn("not calibrated", cw.camera_state_label.text())

        cw.calibrate_camera_button.click()
        self.assertIs(cw.stack.currentWidget(), cw.page_check)
        cw.camera_combo.setCurrentIndex(cw.camera_combo.findData(FIREFLY[0]))
        cw.board_combo.setCurrentText(LAB_PRESET)
        self.assertAlmostEqual(cw.square_spin.value(), 40.0)
        self.assertFalse(cw.check_next.isEnabled())
        for box in cw.checks:
            box.setChecked(True)
        self.assertTrue(cw.check_next.isEnabled())
        cw.check_next.click()
        self.assertTrue(window.preview_running)  # the window started Preview itself
        self.assertIs(cw.stack.currentWidget(), cw.page_capture)

        self.assertTrue(pump_until(lambda: len(cw._session.views) >= 12, timeout=40),
                        f"only {len(cw._session.views)} views: {cw.capture_state.text()}")
        self.assertTrue(cw.compute_button.isEnabled())
        cw.compute_button.click()
        self.assertTrue(pump_until(lambda: cw._result is not None, timeout=60), cw.result_text.text())
        self.assertTrue(cw._result.passed, cw.result_verdict.text())
        self.assertIn("PASS", cw.result_verdict.text())
        self.assertAlmostEqual(cw._result.K[0, 0], synth.K_TRUE[0, 0], delta=synth.K_TRUE[0, 0] * 0.03)
        self.assertTrue(cw.save_button.isEnabled())
        self.assertFalse(cw.save_loose_button.isEnabled())

        cw.save_button.click()
        rec = window._calibration_store.current_intrinsics(FIREFLY[0])
        self.assertIsNotNone(rec)
        self.assertEqual(rec.image_size, [720, 540])
        self.assertEqual(rec.fingerprint["Width"], 720)
        self.assertEqual(rec.board["square_length_m"], 0.04)
        self.assertFalse(rec.loose)
        text = window.calibration_bar.label.text()
        self.assertNotIn("[Firefly #23227865]", text)
        self.assertIn("[Blackfly S #26134271] camera not calibrated", text)
        self.assertIs(cw.stack.currentWidget(), cw.page_home)
        self.assertIn("Saved Firefly #23227865", cw.saved_label.text())
        self.assertIn("23227865: calibrated", cw.camera_state_label.text())

    def test_recording_closes_the_window(self):
        window = self.main_window()
        cw = self.open(window)
        cw.calibrate_camera_button.click()
        for box in cw.checks:
            box.setChecked(True)
        cw.check_next.click()
        self.assertIsNotNone(cw._detector)
        self.assertTrue(window._begin_recording_session(bypass_confirmation=True))
        self.assertIsNone(window._calibration_window)
        self.assertIsNone(cw._detector)  # detection thread stopped
        self.assertFalse(window.calibration_bar.calibrate_button.isEnabled())
        window._stop_recording_session("stopped")

    def test_back_from_capture_stops_detection(self):
        window = self.main_window()
        cw = self.open(window)
        cw.calibrate_camera_button.click()
        for box in cw.checks:
            box.setChecked(True)
        cw.check_next.click()
        cw._cancel_capture()
        self.assertIsNone(cw._detector)
        self.assertIs(cw.stack.currentWidget(), cw.page_check)


    def start_capture(self, cw):
        cw.calibrate_camera_button.click()
        for box in cw.checks:
            box.setChecked(True)
        cw.check_next.click()
        self.assertIsNotNone(cw._detector)

    @staticmethod
    def detector_threads():
        return [t for t in threading.enumerate() if t.name == "calibration-detector" and t.is_alive()]

    def assert_cleaned_up(self, cw):
        self.assertIsNone(cw._detector)
        self.assertFalse(cw.ui_timer.isActive())
        self.assertFalse(cw.compute_timer.isActive())
        self.assertEqual(self.detector_threads(), [])

    def test_escape_cleans_up(self):
        # Esc reaches reject() -> done() and, in Qt >= 6.3, never a closeEvent.
        window = self.main_window()
        cw = self.open(window)
        self.start_capture(cw)
        QTest.keyClick(cw, Qt.Key.Key_Escape)
        self.assertFalse(cw.isVisible())
        self.assert_cleaned_up(cw)
        self.assertIsNone(window._calibration_window)

    def test_reject_hide_and_repeated_shutdown_clean_up(self):
        window = self.main_window()
        cw = self.open(window)
        self.start_capture(cw)
        cw.reject()
        self.assert_cleaned_up(cw)
        cw._shutdown()  # idempotent
        cw2 = self.open(window)
        self.start_capture(cw2)
        cw2.hide()
        self.assert_cleaned_up(cw2)

    def test_reopening_after_escape_leaves_one_detector(self):
        window = self.main_window()
        cw = self.open(window)
        self.start_capture(cw)
        cw.reject()
        cw2 = self.open(window)
        self.assertIsNot(cw2, cw)
        self.start_capture(cw2)
        self.assertEqual(len(self.detector_threads()), 1)
        window._close_calibration_window()
        self.assertEqual(self.detector_threads(), [])

    def test_hidden_but_referenced_window_is_shut_down_when_replaced(self):
        # A window that was only hidden (so no finished/closeEvent) must not be orphaned.
        window = self.main_window()
        cw = self.open(window)
        self.start_capture(cw)
        cw.isVisible = lambda: False
        cw2 = self.open(window)
        self.assertIsNot(cw2, cw)
        self.assertIsNone(cw._detector)
        self.assertFalse(cw.ui_timer.isActive())

    def test_escape_then_recording_runs_no_detector(self):
        window = self.main_window()
        cw = self.open(window)
        self.start_capture(cw)
        QTest.keyClick(cw, Qt.Key.Key_Escape)
        self.assertTrue(window._begin_recording_session(bypass_confirmation=True))
        self.assertEqual(self.detector_threads(), [])
        window._stop_recording_session("stopped")


@unittest.skipIf(not HAVE_QT, "needs PySide6")
class BoardPdfTests(unittest.TestCase):
    def test_writes_a_pdf_and_refuses_oversize_boards(self):
        app()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "board.pdf"
            write_board_pdf(str(path), cal.BOARD_PRESETS[cal.HANDHELD_PRESET], "A")
            self.assertTrue(path.read_bytes().startswith(b"%PDF"))
            with self.assertRaises(ValueError):
                write_board_pdf(str(path), cal.BoardConfig(10, 8, 0.04, 0.03), "big")


if __name__ == "__main__":
    unittest.main()
