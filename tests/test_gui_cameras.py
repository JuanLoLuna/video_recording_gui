"""The real MainWindow, headless (Qt offscreen), driving one or two fake cameras.

Everything above the Spinnaker layer is the real code: Detect through the camera
registry, the camera group, preview tiles, per-camera diagnostics, recording
with per-camera file names, label fan-out, and shutdown. Skipped when PySide6 is
missing or when real PySpin is present (use the rig for that).
"""
import csv
import os
import tempfile
import time
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from fake_spinnaker import FakeCamera, FakeSystem, install_pyspin_stub  # noqa: E402

REAL_PYSPIN = install_pyspin_stub()
try:
    from PySide6.QtCore import QEvent, Qt
    from PySide6.QtGui import QKeyEvent
    from PySide6.QtWidgets import QApplication
    HAVE_QT = True
except ImportError:  # pragma: no cover
    HAVE_QT = False

if HAVE_QT:
    import backend.spinnaker_system as spinnaker_system
    from backend.camera_registry import CAMERA_SERIALS_ENV
    from backend.session_verify import verify_camera_outputs
    from backend.spinnaker_system import SharedSystemHolder
    from gui.main import AppState, MainWindow

_app = None


def app():
    global _app
    if _app is None:
        _app = QApplication.instance() or QApplication([])
    return _app


def pump(seconds):
    """Run the Qt event loop (preview/diagnostics timers) for a while."""
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        app().processEvents()
        time.sleep(0.02)


FIREFLY = ("23227865", "Firefly", 32, 24)
BLACKFLY = ("26134271", "Blackfly S", 64, 48)


@unittest.skipIf(REAL_PYSPIN or not HAVE_QT, "needs PySide6 and no real PySpin")
class MainWindowCameraTests(unittest.TestCase):
    def setUp(self):
        app()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.addCleanup(lambda: os.environ.pop("SLEEVE_VIDEO_GUI_SEGMENT_SECONDS", None))
        self.addCleanup(lambda: os.environ.pop(CAMERA_SERIALS_ENV, None))
        os.environ["SLEEVE_VIDEO_GUI_SEGMENT_SECONDS"] = "1"
        os.environ.pop(CAMERA_SERIALS_ENV, None)
        previous = spinnaker_system._default_holder
        self.addCleanup(lambda: setattr(spinnaker_system, "_default_holder", previous))

    def window(self, *specs):
        self.cameras = [FakeCamera(*spec) for spec in specs]
        self.holder = SharedSystemHolder(lambda: FakeSystem(self.cameras))
        spinnaker_system._default_holder = self.holder
        window = MainWindow()
        window._configured_output_dir = self.tmp.name
        # Dialogs would block a headless run: power / disk confirmations are
        # exercised by their own tests.
        window._confirm_power_safe_to_record = lambda: True
        window._confirm_disk_safe_to_record = lambda *a, **k: True
        self.addCleanup(window.close)
        return window

    def run_session(self, window, seconds=2.5, keys=()):
        window.on_detect_clicked()
        self.assertEqual(window.state, AppState.CAMERA_DETECTED, window.status_label.text())
        window.on_preview_clicked()
        self.assertEqual(window.state, AppState.PREVIEWING, window.status_label.text())
        pump(0.6)
        self.assertTrue(window._begin_recording_session(bypass_confirmation=True), window.status_label.text())
        self.assertEqual(window.state, AppState.RECORDING)
        pump(seconds / 2)
        for key in keys:
            window.keyPressEvent(QKeyEvent(QEvent.Type.KeyPress, key, Qt.KeyboardModifier.NoModifier))
        pump(seconds / 2)
        window._stop_recording_session("Recording stopped.")
        self.assertEqual(window.state, AppState.PREVIEWING)
        window.on_preview_clicked()
        self.assertEqual(window.state, AppState.CAMERA_DETECTED)

    # ---------------------------------------------------------------- detect
    def test_one_camera_keeps_the_legacy_detect_message_and_layout(self):
        window = self.window(FIREFLY)
        window.on_detect_clicked()
        self.assertEqual(window.status_label.text(), "Camera: FAKE Firefly (S/N: 23227865)")
        self.assertEqual(window.state, AppState.CAMERA_DETECTED)
        self.assertEqual(len(window._slot_rt), 1)
        tile = next(iter(window._slot_rt.values())).tile
        self.assertTrue(tile.caption.isHidden())
        self.assertTrue(window.tuning_camera_row.isHidden())

    def test_two_cameras_get_two_tiles_and_a_camera_selector(self):
        window = self.window(BLACKFLY, FIREFLY)  # listed in an unsorted order
        window.on_detect_clicked()
        self.assertEqual(window.state, AppState.CAMERA_DETECTED)
        self.assertEqual(len(window._slot_rt), 2)
        self.assertFalse(window.tuning_camera_row.isHidden())
        self.assertEqual(window.tuning_camera_combo.count(), 2)
        self.assertIn("2 cameras", window.status_label.text())
        # No env var: the user is told how to pin the names.
        self.assertIn(CAMERA_SERIALS_ENV, window.status_label.text())
        tags = {slot.serial: slot.tag for slot in window.cameras.slots}
        self.assertEqual(tags, {"23227865": None, "26134271": "cam26134271"})

    def test_a_configured_camera_that_is_missing_refuses_a_manual_start(self):
        os.environ[CAMERA_SERIALS_ENV] = "23227865,26134271"
        window = self.window(FIREFLY)
        window.on_detect_clicked()
        self.assertEqual(window.state, AppState.IDLE)
        self.assertIn("MISSING: 26134271", window.status_label.text())
        self.assertIsNone(window.cameras)
        self.assertFalse(window.preview_button.isEnabled())

    def test_no_cameras_stays_idle(self):
        window = self.window()
        window.on_detect_clicked()
        self.assertEqual(window.state, AppState.IDLE)
        self.assertEqual(window.status_label.text(), "No cameras detected.")

    def test_preview_without_detect_is_a_polite_no_op(self):
        window = self.window(FIREFLY)
        window.on_preview_clicked()
        self.assertEqual(window.state, AppState.IDLE)
        self.assertIn("Detect", window.status_label.text())

    # ------------------------------------------------------------- recording
    def test_two_camera_session_through_the_gui_writes_reconciled_per_camera_files(self):
        window = self.window(BLACKFLY, FIREFLY)
        self.run_session(window, seconds=3.0)
        names = sorted(os.listdir(self.tmp.name))
        primary_meta = [n for n in names if n.endswith("_metadata.csv") and "_cam" not in n]
        tagged_meta = [n for n in names if n.endswith("_cam26134271_metadata.csv")]
        self.assertEqual(len(primary_meta), 1, names)
        self.assertEqual(len(tagged_meta), 1, names)
        # One diagnostics CSV per camera.
        self.assertEqual(len([n for n in names if n.endswith("_diagnostics.csv")]), 2, names)
        for meta, serial in ((primary_meta[0], "23227865"), (tagged_meta[0], "26134271")):
            stem = meta[: -len("_metadata.csv")]
            report = verify_camera_outputs(
                Path(self.tmp.name) / meta,
                Path(self.tmp.name) / f"{stem}_segments.csv",
                Path(self.tmp.name) / f"{stem}_events.jsonl",
                expect_stem=stem, expect_serial=serial,
            )
            self.assertTrue(report.ok, f"{serial}: {report.problems}")
            self.assertGreater(report.metadata_rows, 40)
            self.assertGreaterEqual(report.segment_count, 2)  # 1 s segments rotated

    def test_one_camera_session_uses_exactly_the_legacy_file_names(self):
        window = self.window(FIREFLY)
        self.run_session(window, seconds=1.5)
        names = sorted(os.listdir(self.tmp.name))
        self.assertTrue(all("_cam" not in n for n in names), names)
        for suffix in ("_metadata.csv", "_segments.csv", "_events.jsonl", "_diagnostics.csv", "-0000.avi"):
            self.assertEqual(len([n for n in names if n.endswith(suffix)]), 1, (suffix, names))

    def test_label_keys_reach_every_cameras_metadata(self):
        window = self.window(BLACKFLY, FIREFLY)
        self.run_session(window, seconds=2.0, keys=(Qt.Key.Key_S,))
        metas = [n for n in os.listdir(self.tmp.name) if n.endswith("_metadata.csv")]
        self.assertEqual(len(metas), 2)
        for meta in metas:
            with open(Path(self.tmp.name) / meta, newline="", encoding="utf-8") as handle:
                labels = [row["sync_label"] for row in csv.DictReader(handle)]
            self.assertIn("label_start", labels, meta)

    def test_both_tiles_show_live_frames_and_stay_blank_after_stop(self):
        window = self.window(BLACKFLY, FIREFLY)
        window.on_detect_clicked()
        window.on_preview_clicked()
        pump(0.8)
        for rt in window._slot_rt.values():
            self.assertFalse(rt.tile.image_label.pixmap().isNull())
            self.assertIsNotNone(rt.last_seq)
        window.on_preview_clicked()
        for rt in window._slot_rt.values():
            self.assertEqual(rt.tile.image_label.text(), "No video")

    def test_per_camera_health_shows_in_the_status_line_and_tile_captions(self):
        window = self.window(BLACKFLY, FIREFLY)
        window.on_detect_clicked()
        window.on_preview_clicked()
        pump(1.6)  # at least one diagnostics tick
        text = window.preview_health_label.text()
        self.assertIn("Firefly #23227865", text)
        self.assertIn("Blackfly S #26134271", text)
        for rt in window._slot_rt.values():
            self.assertIn("displayed fps", rt.tile.caption.text())

    def test_frame_rate_and_compression_controls_reach_both_cameras(self):
        window = self.window(BLACKFLY, FIREFLY)
        window.on_detect_clicked()
        window.on_preview_clicked()
        pump(0.3)
        window.compression_checkbox.setChecked(True)  # a genuine click path
        window._on_compression_toggled(True)
        self.assertTrue(all(s.controller.get_compression_enabled() for s in window.cameras.slots))
        window._on_compression_toggled(False)
        self.assertFalse(any(s.controller.get_compression_enabled() for s in window.cameras.slots))

    def test_closing_the_window_releases_every_camera_and_the_system(self):
        window = self.window(BLACKFLY, FIREFLY)
        window.on_detect_clicked()
        window.on_preview_clicked()
        pump(0.4)
        window.close()
        self.assertEqual(self.holder.count, 0)
        self.assertFalse(any(s.controller.acquiring for s in window.cameras.slots))

    def test_selecting_another_camera_retargets_the_tuning_controls(self):
        window = self.window(BLACKFLY, FIREFLY)
        window.on_detect_clicked()
        first = window.camera.serial
        other = next(s.serial for s in window.cameras.slots if s.serial != first)
        window.tuning_camera_combo.setCurrentIndex(window.tuning_camera_combo.findData(other))
        self.assertEqual(window.camera.serial, other)


if __name__ == "__main__":
    unittest.main()
