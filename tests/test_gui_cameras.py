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
        saved = {k: os.environ.get(k) for k in ("SLEEVE_VIDEO_GUI_SEGMENT_SECONDS", CAMERA_SERIALS_ENV)}

        def restore_environ():
            for key, value in saved.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

        self.addCleanup(restore_environ)
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

    def previewing(self, *specs):
        window = self.window(*specs)
        window.on_detect_clicked()
        window.on_preview_clicked()
        self.assertEqual(window.state, AppState.PREVIEWING, window.status_label.text())
        pump(0.3)
        return window

    def exposure_modes(self, window):
        return {s.serial: s.controller.get_enum_param("ExposureAuto")[0] for s in window.cameras.slots}

    def test_compression_changes_reach_both_cameras(self):
        window = self.previewing(BLACKFLY, FIREFLY)
        window._on_compression_toggled(True)
        self.assertTrue(all(s.controller.get_compression_enabled() for s in window.cameras.slots))
        window._on_compression_toggled(False)
        self.assertFalse(any(s.controller.get_compression_enabled() for s in window.cameras.slots))

    def test_one_camera_defaults_to_mjpeg_at_30_fps(self):
        window = self.previewing(FIREFLY)
        self.assertTrue(window.compression_checkbox.isChecked())
        self.assertTrue(window.camera.get_compression_enabled())
        self.assertIn("30 fps", window.compression_hint.text())

    def test_two_cameras_default_to_uncompressed_even_at_30_fps(self):
        window = self.previewing(BLACKFLY, FIREFLY)
        self.assertFalse(window.compression_checkbox.isChecked())
        self.assertFalse(any(s.controller.get_compression_enabled() for s in window.cameras.slots))
        self.assertIn("several cameras", window.compression_hint.text())

    def test_the_user_can_still_choose_mjpeg_with_two_cameras_and_it_sticks(self):
        window = self.previewing(BLACKFLY, FIREFLY)
        window.compression_checkbox.setChecked(True)  # a real click path -> toggled signal
        self.assertTrue(all(s.controller.get_compression_enabled() for s in window.cameras.slots))
        # A later frame-rate change must not override the user's choice.
        window.frame_rate_spin.setValue(25.0)
        self.assertTrue(all(s.controller.get_compression_enabled() for s in window.cameras.slots))

    def test_there_is_no_quality_control_because_it_never_did_anything(self):
        window = self.window(FIREFLY)
        self.assertFalse(hasattr(window, "compression_quality_spin"))
        self.assertFalse(hasattr(window.camera, "set_compression_quality"))

    def test_the_two_camera_default_can_be_raised_without_touching_the_gui(self):
        import backend.compression_policy as policy

        previous = policy.MULTI_CAMERA_MJPEG_MAX_FPS
        self.addCleanup(lambda: setattr(policy, "MULTI_CAMERA_MJPEG_MAX_FPS", previous))
        policy.MULTI_CAMERA_MJPEG_MAX_FPS = 30.0
        window = self.previewing(BLACKFLY, FIREFLY)
        self.assertTrue(window.compression_checkbox.isChecked())

    def test_a_codec_change_one_camera_refuses_is_undone_on_the_others(self):
        window = self.previewing(BLACKFLY, FIREFLY)
        window._on_compression_toggled(False)
        refusing = window.cameras.slot_for_serial("26134271").controller
        refusing.set_compression_enabled = lambda enabled: False  # e.g. still closing its last recording
        window._on_compression_toggled(True)
        # No camera may end up on a different codec from the others.
        self.assertEqual(
            {s.controller.get_compression_enabled() for s in window.cameras.slots}, {False}
        )

    def test_a_frame_rate_change_reaches_both_cameras(self):
        window = self.previewing(BLACKFLY, FIREFLY)
        self.assertTrue(window.frame_rate_spin.isEnabled())
        window.frame_rate_spin.setValue(20.0)
        for slot in window.cameras.slots:
            self.assertAlmostEqual(slot.controller.get_acquisition_frame_rate(), 20.0, places=1)

    def test_a_frame_rate_one_camera_refuses_is_rolled_back_everywhere(self):
        window = self.previewing(BLACKFLY, FIREFLY)
        before = {s.serial: s.controller.get_acquisition_frame_rate() for s in window.cameras.slots}
        window.cameras.slot_for_serial("26134271").controller.set_frame_rate = lambda value: False
        window.frame_rate_spin.setValue(20.0)
        after = {s.serial: s.controller.get_acquisition_frame_rate() for s in window.cameras.slots}
        self.assertEqual(before, after)  # one shared rate, unchanged
        self.assertIn("NOT changed", window.status_label.text())

    # ------------------------------------------------- exposure lock (audit H1/H2)
    def test_the_exposure_lock_forces_every_camera_to_off_not_just_the_selected_one(self):
        window = self.window(BLACKFLY, FIREFLY)
        # The primary is already Off (an earlier session); the other camera is on
        # its power-on default, Continuous. The combo only shows the primary.
        window.on_detect_clicked()
        window.cameras.slot_for_serial("23227865").controller.set_enum_param("ExposureAuto", "Off")
        self.assertEqual(self.exposure_modes_pre(window)["26134271"], "Continuous")
        window.on_preview_clicked()
        pump(0.2)
        self.assertEqual(set(self.exposure_modes(window).values()), {"Off"})

    def exposure_modes_pre(self, window):
        """ExposureAuto of each fake camera before anything started it."""
        return {
            cam.serial: cam.GetNodeMap().GetNode("ExposureAuto").GetCurrentEntry().GetSymbolic()
            for cam in self.cameras
        }

    def test_the_gui_lock_itself_reaches_a_camera_the_combo_does_not_show(self):
        # The controller also forces Off when it configures a camera, so this
        # drives the GUI's own lock: a non-selected camera drifts back to
        # Continuous while previewing, and the combo (which shows the PRIMARY,
        # already Off) gives no hint of it.
        window = self.previewing(BLACKFLY, FIREFLY)
        node = next(c for c in self.cameras if c.serial == "26134271").GetNodeMap().GetNode("ExposureAuto")
        node.SetIntValue(node.GetEntryByName("Continuous").GetValue())
        self.assertEqual(self.exposure_modes(window)["26134271"], "Continuous")
        self.assertEqual(window._auto_mode_meta["ExposureAuto"]["combo"].currentText(), "Off")
        window._apply_exposure_auto_lock_for_fps(30.0)
        self.assertEqual(set(self.exposure_modes(window).values()), {"Off"})

    def test_switching_the_adjust_camera_does_not_unlock_exposure_mode(self):
        window = self.previewing(BLACKFLY, FIREFLY)
        combo = window._auto_mode_meta["ExposureAuto"]["combo"]
        self.assertFalse(combo.isEnabled())  # locked at 30 fps
        other = next(s.serial for s in window.cameras.slots if s.serial != window.camera.serial)
        window.tuning_camera_combo.setCurrentIndex(window.tuning_camera_combo.findData(other))
        self.assertFalse(combo.isEnabled())
        self.assertEqual(set(self.exposure_modes(window).values()), {"Off"})

    def test_a_replugged_camera_comes_back_with_exposure_off(self):
        # The controller itself re-applies the lock whenever it configures a camera
        # (start AND every reinit), because a replugged camera powers on in Continuous.
        window = self.previewing(BLACKFLY, FIREFLY)
        slot = window.cameras.slot_for_serial("26134271")
        camera = next(c for c in self.cameras if c.serial == "26134271")
        camera.GetNodeMap().GetNode("ExposureAuto").SetIntValue(
            camera.GetNodeMap().GetNode("ExposureAuto").GetEntryByName("Continuous").GetValue()
        )
        ok, message = slot.controller._reinitialize_camera()
        self.assertTrue(ok, message)
        self.assertEqual(self.exposure_modes(window)["26134271"], "Off")

    # ------------------------------------------- a camera that is not recording (H3)
    def test_a_camera_whose_segment_zero_fails_is_called_out_not_silently_healthy(self):
        import contextlib, io

        window = self.previewing(BLACKFLY, FIREFLY)

        def cannot_open(index):
            raise OSError("codec missing")

        window.cameras.slot_for_serial("26134271").controller._open_segment_writer = cannot_open
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertTrue(window._begin_recording_session(bypass_confirmation=True))
            pump(2.4)  # two diagnostics ticks
        state = window._recording_warnings.summarize(now_s=time.monotonic())
        self.assertTrue(state.visible)
        self.assertEqual(state.level, "active")
        self.assertIn("Blackfly S #26134271", state.detail + state.headline)
        self.assertIn("NOT recording", state.detail + state.headline)
        self.assertIn("codec missing", state.detail + state.headline)
        window._stop_recording_session("done")

    def test_a_resume_that_leaves_a_camera_out_keeps_warning_loudly(self):
        window = self.previewing(BLACKFLY, FIREFLY)
        window.cameras.slot_for_serial("26134271").controller.prepare_recording = (
            lambda *a, **k: (False, "Cannot open metadata CSV: disk full")
        )
        self.assertTrue(window._begin_recording_session(bypass_confirmation=True))  # best effort
        pump(2.4)
        state = window._recording_warnings.summarize(now_s=time.monotonic())
        self.assertEqual(state.level, "active")
        self.assertIn("#26134271", state.detail + state.headline)
        # Only the camera that started gets a diagnostics CSV.
        names = os.listdir(self.tmp.name)
        self.assertEqual(len([n for n in names if n.endswith("_diagnostics.csv")]), 1, names)
        window._stop_recording_session("done")

    # --------------------------------------- transient writer stalls are not faults
    def spike_depth(self, window, serial, depths):
        controller = window.cameras.slot_for_serial(serial).controller
        original = controller.get_diagnostics_camera_state
        ticks = iter(depths)
        state = {"last": depths[-1]}

        def patched():
            try:
                state["last"] = next(ticks)
            except StopIteration:
                pass
            return {**original(), "append_queue_depth": state["last"]}

        controller.get_diagnostics_camera_state = patched

    def test_a_brief_writer_stall_that_drains_raises_no_warning(self):
        # The dock stall seen on the rig: queue 150 for ~4 s, then 0, nothing lost.
        window = self.previewing(BLACKFLY, FIREFLY)
        window.cameras.slot_for_serial("26134271").controller.set_compression_enabled(True)
        self.spike_depth(window, "26134271", [150, 150, 150, 150, 0, 0, 0, 0])
        self.assertTrue(window._begin_recording_session(bypass_confirmation=True))
        for _ in range(7):
            window._sample_preview_diagnostics()
        state = window._recording_warnings.summarize(now_s=time.monotonic())
        self.assertFalse(state.visible, state.headline + state.detail)
        window._stop_recording_session("done")

    def test_a_backlog_that_stays_high_does_warn(self):
        window = self.previewing(BLACKFLY, FIREFLY)
        window._slot_rt["26134271"].backlog.hold_seconds = 0.3
        window._slot_rt["26134271"].mjpeg_behind.hold_seconds = 0.3
        self.spike_depth(window, "26134271", [150] * 50)
        self.assertTrue(window._begin_recording_session(bypass_confirmation=True))
        for _ in range(4):
            window._sample_preview_diagnostics()
            time.sleep(0.2)
        state = window._recording_warnings.summarize(now_s=time.monotonic())
        self.assertTrue(state.visible)
        self.assertIn("waiting to be written", state.detail + state.headline)
        window._stop_recording_session("done")

    # ------------------------------------------------- one camera reads as before
    def test_one_camera_record_start_and_failure_texts_read_as_before(self):
        window = self.previewing(FIREFLY)
        self.assertTrue(window._begin_recording_session(bypass_confirmation=True))
        self.assertTrue(
            window.status_label.text().startswith("Recording requested: recording_"),
            window.status_label.text(),
        )
        window._stop_recording_session("done")

    def test_a_manual_start_goes_through_the_all_or_nothing_path(self):
        window = self.previewing(BLACKFLY, FIREFLY)
        window.cameras.slot_for_serial("26134271").controller.prepare_recording = (
            lambda *a, **k: (False, "Cannot open metadata CSV: disk full")
        )
        window.on_record_clicked()  # NOT bypass_confirmation
        self.assertEqual(window.state, AppState.PREVIEWING)
        self.assertEqual(window.status_label.text(), "Blackfly S #26134271: Cannot open metadata CSV: disk full")
        self.assertEqual(os.listdir(self.tmp.name), [])  # the other camera was not started either

    def test_redetect_after_a_recorded_session_builds_a_fresh_group(self):
        window = self.window(BLACKFLY, FIREFLY)
        self.run_session(window, seconds=1.2)
        first = window.cameras
        window.on_detect_clicked()
        self.assertIsNot(window.cameras, first)
        self.assertEqual(window.state, AppState.CAMERA_DETECTED)
        self.assertEqual(len(window._slot_rt), 2)
        self.assertEqual(self.holder.count, 0)

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
