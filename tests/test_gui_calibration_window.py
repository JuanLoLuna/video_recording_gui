"""The Calibration window end to end (Qt offscreen, fake cameras showing a rendered board).

A fake camera "sees" a ChArUco board rendered through a known lens, held in a
new pose every second. The test clicks through the real window: checklist ->
automatic capture -> compute -> save, and checks the record and the main
window's 3D status.
"""
import contextlib
import io
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

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
    import numpy as np
    import backend.spinnaker_system as spinnaker_system
    import test_calibration as synth
    from backend import calibration as cal
    from backend import calibration_store as cs
    from backend.camera_registry import CAMERA_SERIALS_ENV
    from backend.spinnaker_system import SharedSystemHolder
    from gui.calibration_window import CalibrationWindow, write_board_pdf
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

    def test_a_degenerate_view_is_removed_so_compute_still_works(self):
        window = self.main_window()
        cw = self.open(window)
        cw.calibrate_camera_button.click()
        cw.camera_combo.setCurrentIndex(cw.camera_combo.findData(FIREFLY[0]))
        for box in cw.checks:
            box.setChecked(True)
        cw.check_next.click()  # the A4 handheld board: 6 corners per row = MIN_CORNERS
        cfg = cw._board_cfg
        det = cal.BoardDetector(cfg)
        rng = np.random.default_rng(4)
        good = []
        for _ in range(14):
            R, t = synth.pose_looking_at_board(rng.uniform(0.45, 0.7), rng.uniform(-30, 30, 3) * [1, 1, 0.3],
                                               rng.uniform(-0.04, 0.04, 2), cfg=cfg)
            good.append(det.detect(synth.render_view_a4(cfg, R, t)))
        row = good[0]
        m = row.ids // (cfg.squares_x - 1) == 1
        # One board row, from a Detection that does not know the board layout (so `ok` cannot tell).
        line = cal.Detection(row.corners[m], row.ids[m], row.marker_count, row.image_size)
        self.assertTrue(all(v.ok for v in good) and line.ok)
        cw._session.views[:] = good + [line]
        cw._session.signatures[:] = [cal.ViewSignature((1, 1), (0, 0), 0.1, 0.0)] * len(cw._session.views)
        cw._start_compute()
        self.assertTrue(pump_until(lambda: cw._result is not None, timeout=60), cw.result_text.text())
        self.assertEqual(len(cw._session.views), len(good))
        self.assertIn("1 unusable view(s) were removed", cw.result_text.text())
        self.assertEqual(cw._result.n_views, len(good))

    def computed_window(self, window):
        """A window on its result page for the Firefly, from synthetic views (no live capture)."""
        cw = self.open(window)
        cw.calibrate_camera_button.click()
        cw.camera_combo.setCurrentIndex(cw.camera_combo.findData(FIREFLY[0]))
        cw.board_combo.setCurrentText(LAB_PRESET)
        for box in cw.checks:
            box.setChecked(True)
        cw.check_next.click()
        det = cal.BoardDetector(synth.CFG)
        views = [det.detect(synth.render_view(R, t)) for R, t in synth.varied_views(16, seed=11)]
        cw._session.views[:] = views
        cw._session.signatures[:] = [cal.ViewSignature((1, 1), (0, 0), 0.1, 0.0)] * len(views)
        cw._start_compute()
        self.assertTrue(pump_until(lambda: cw._result is not None, timeout=60), cw.result_text.text())
        self.assertIs(cw.stack.currentWidget(), cw.page_result)
        return cw

    def saved_records(self, window):
        return sorted(p.name for p in window._calibration_store.intrinsics_dir(FIREFLY[0]).glob("2*.json"))

    def test_save_is_refused_without_the_camera_settings(self):
        window = self.main_window()
        cw = self.computed_window(window)
        controller = cw._selected_slot().controller
        real = controller.get_sensor_fingerprint
        controller.get_sensor_fingerprint = lambda: None   # Preview stopped / camera recovering
        cw._save(loose=not cw._result.passed)
        self.assertIn("Start Preview", cw.save_error.text())
        self.assertIs(cw.stack.currentWidget(), cw.page_result)
        self.assertEqual(self.saved_records(window), [])
        controller.get_sensor_fingerprint = lambda: dict(real(), Width=None)
        cw._save(loose=not cw._result.passed)
        self.assertIn("Start Preview", cw.save_error.text())
        controller.get_sensor_fingerprint = lambda: dict(real(), Width=1440)
        cw._save(loose=not cw._result.passed)
        self.assertIn("resolution changed", cw.save_error.text())
        self.assertEqual(self.saved_records(window), [])
        controller.get_sensor_fingerprint = real
        cw._save(loose=not cw._result.passed)
        self.assertEqual(len(self.saved_records(window)), 1)

    def test_a_failed_write_is_reported_in_the_window_and_can_be_retried(self):
        window = self.main_window()
        cw = self.computed_window(window)
        loose = not cw._result.passed
        store = cw.store
        real_save = store.save_intrinsics
        store.save_intrinsics = lambda *a, **k: (_ for _ in ()).throw(PermissionError("locked by a scanner"))
        cw._save(loose)
        self.assertIn("locked by a scanner", cw.save_error.text())
        self.assertIs(cw.stack.currentWidget(), cw.page_result)
        self.assertIsNotNone(cw._result)
        self.assertEqual(self.saved_records(window), [])
        store.save_intrinsics = real_save
        # The record is written but the pointer is not: the retry must not write a second record.
        real_pointer = store.set_current_intrinsics
        store.set_current_intrinsics = lambda *a, **k: (_ for _ in ()).throw(PermissionError("current.json busy"))
        cw._save(loose)
        self.assertIn("current.json busy", cw.save_error.text())
        self.assertEqual(len(self.saved_records(window)), 1)
        store.set_current_intrinsics = real_pointer
        cw._save(loose)
        self.assertEqual(cw.save_error.text(), "")
        self.assertEqual(len(self.saved_records(window)), 1)
        self.assertIsNotNone(store.current_intrinsics(FIREFLY[0]))
        self.assertIs(cw.stack.currentWidget(), cw.page_home)

    def test_after_a_save_the_checklist_is_reset_and_the_next_camera_selected(self):
        window = self.main_window()
        cw = self.computed_window(window)
        cw._save(loose=not cw._result.passed)
        self.assertFalse(any(box.isChecked() for box in cw.checks))
        self.assertFalse(cw.check_next.isEnabled())
        self.assertEqual(cw.camera_combo.currentData(), BLACKFLY[0])

    def test_stopped_preview_is_said_in_the_capture_page(self):
        window = self.main_window()
        cw = self.open(window)
        self.start_capture(cw)
        self.assertTrue(pump_until(lambda: cw._last_seq > 0, timeout=10))
        cw._selected_slot().controller.get_latest_frame = lambda: None
        with mock.patch("gui.calibration_window.NO_FRAMES_S", 0.3):
            self.assertTrue(pump_until(lambda: cw._last_state == "preview stopped", timeout=10))
        self.assertIn("Preview stopped: start it again in the main window", cw.live_view.text())
        self.assertIn("Preview stopped", cw.capture_state.text())

    def test_recording_start_says_so_when_it_discards_a_result(self):
        window = self.main_window()
        cw = self.computed_window(window)
        self.assertTrue(cw.has_unsaved_result())
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            window.on_preview_clicked() if not window.preview_running else None
            self.assertTrue(window._begin_recording_session(bypass_confirmation=True))
        self.assertIn("not saved yet is discarded", out.getvalue())
        window._stop_recording_session("stopped")

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
    def test_lossless_and_failures_are_reported(self):
        app()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "board.pdf"
            write_board_pdf(str(path), cal.BOARD_PRESETS[cal.HANDHELD_PRESET], "A")
            self.assertNotIn(b"DCTDecode", path.read_bytes())  # Qt's default is JPEG
            with self.assertRaises(OSError):
                write_board_pdf(str(Path(tmp) / "no_such_folder" / "board.pdf"),
                                cal.BOARD_PRESETS[cal.HANDHELD_PRESET], "A")

    def test_the_window_shows_a_write_failure_without_a_dialog(self):
        app()
        with tempfile.TemporaryDirectory() as tmp:
            cw = CalibrationWindow(None, [], cs.CalibrationStore(Path(tmp) / "cal"),
                                   ensure_preview=lambda: True, on_saved=lambda: None)
            self.addCleanup(cw._shutdown)
            bad = str(Path(tmp) / "no_such_folder" / "board.pdf")
            with mock.patch("gui.calibration_window.QFileDialog.getSaveFileName", return_value=(bad, "")):
                cw._on_print_board()
            self.assertIn("Could not write the board", cw.home_error.text())
            good = str(Path(tmp) / "board.pdf")
            with mock.patch("gui.calibration_window.QFileDialog.getSaveFileName", return_value=(good, "")):
                cw._on_print_board()
            self.assertEqual(cw.home_error.text(), "")
            self.assertIn("Saved", cw.saved_label.text())

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
