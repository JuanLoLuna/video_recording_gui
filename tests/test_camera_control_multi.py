"""CameraController behaviour added for multi-camera support.

Hand-written fakes stand in for the Spinnaker camera, camera list and System.
PySpin itself is stubbed only when it is not installed (macOS dev machine);
on the rig the real module is used and none of these tests touch it.
"""
import json
import sys
import tempfile
import types
import unittest
from datetime import datetime
from pathlib import Path

import numpy as np

try:
    import PySpin  # noqa: F401
except ImportError:
    sys.modules["PySpin"] = types.ModuleType("PySpin")

from backend.camera_control import CameraController  # noqa: E402
from backend.recording_paths import SessionPaths  # noqa: E402
from backend.spinnaker_system import SharedSystemHolder  # noqa: E402


class FakeCamera:
    def __init__(self, serial, log=None):
        self.serial = serial
        self.log = log if log is not None else []
        self.inited = False
        self.acquiring = False

    def IsValid(self):
        return True

    def Init(self):
        self.inited = True
        self.log.append(f"init:{self.serial}")

    def BeginAcquisition(self):
        self.acquiring = True
        self.log.append(f"begin:{self.serial}")

    def EndAcquisition(self):
        self.acquiring = False

    def DeInit(self):
        self.inited = False


class FakeCamList:
    def __init__(self, cameras, log=None, get_by_serial_raises=False):
        self.cameras = list(cameras)
        self.log = log if log is not None else []
        self.cleared = False
        self.get_by_serial_raises = get_by_serial_raises

    def GetSize(self):
        return len(self.cameras)

    def __getitem__(self, index):
        return self.cameras[index]

    def GetBySerial(self, serial):
        if self.get_by_serial_raises:
            raise RuntimeError("no such camera")
        for cam in self.cameras:
            if cam.serial == serial:
                return cam
        raise RuntimeError(f"serial {serial} not found")

    def Clear(self):
        self.cleared = True
        self.log.append("list-clear")


class FakeSystem:
    """A Spinnaker System whose attached cameras can change between calls."""

    def __init__(self, log, name, cameras, hidden=None):
        self.log = log
        self.name = name
        self.cameras = cameras  # shared by reference: the bus, not this System object
        # Cameras GetCameras() cannot see until UpdateCameras().
        self.hidden = hidden if hidden is not None else []
        self.released = False

    def GetCameras(self):
        self.log.append(f"get-cameras:{self.name}")
        return FakeCamList(self.cameras, self.log)

    def UpdateCameras(self):
        self.log.append(f"update-cameras:{self.name}")
        self.cameras.extend(self.hidden)
        self.hidden.clear()

    def ReleaseInstance(self):
        self.released = True
        self.log.append(f"release-system:{self.name}")


class SystemFactory:
    def __init__(self, log, cameras, fail_on=()):
        self.log = log
        self.cameras = cameras
        self.made = []
        self.hidden = []
        self.fail_on = set(fail_on)

    def __call__(self):
        number = len(self.made) + 1
        if number in self.fail_on:
            self.made.append(None)
            raise RuntimeError("no system")
        system = FakeSystem(self.log, str(number), self.cameras, self.hidden)
        self.made.append(system)
        return system


def make_controller(serial="222", tag="cam222", cameras=None, log=None, **factory_kw):
    log = log if log is not None else []
    cameras = cameras if cameras is not None else [FakeCamera("111", log), FakeCamera("222", log)]
    factory = SystemFactory(log, cameras, **factory_kw)
    holder = SharedSystemHolder(factory)
    controller = CameraController(serial=serial, tag=tag, system_holder=holder)
    controller._configure_camera_nodes = lambda: None  # node-map work needs real PySpin
    return controller, holder, factory, log


def attach(controller, holder, factory, camera_serial):
    """Put the controller in the state start() leaves it in, without PySpin."""
    controller.system = holder.acquire(controller)
    controller.cam_list = controller.system.GetCameras()
    controller.cam = controller._pick_camera(controller.cam_list)
    controller.cam.Init()
    controller.acquiring = True


class IdentityTests(unittest.TestCase):
    def test_default_controller_keeps_legacy_log_prefix_and_no_serial(self):
        controller = CameraController(system_holder=SharedSystemHolder(lambda: None))
        self.assertIsNone(controller.serial)
        self.assertEqual(controller._log_prefix, "[camera]")
        self.assertEqual(controller._thread_suffix, "")

    def test_a_pinned_controller_names_itself_in_logs_and_threads(self):
        controller, *_ = make_controller(serial="222")
        self.assertEqual(controller.serial, "222")
        self.assertEqual(controller._log_prefix, "[camera 222]")
        self.assertEqual(controller._thread_suffix, "-222")

    def test_serial_may_be_given_as_an_int(self):
        controller, *_ = make_controller(serial=26134271)
        self.assertEqual(controller.serial, "26134271")


class PickCameraTests(unittest.TestCase):
    def setUp(self):
        self.a, self.b = FakeCamera("111"), FakeCamera("222")

    def test_binds_by_serial_not_by_position(self):
        controller, *_ = make_controller(serial="222")
        # Enumeration order differs from run to run: "222" listed first here.
        self.assertIs(controller._pick_camera(FakeCamList([self.b, self.a])), self.b)
        self.assertIs(controller._pick_camera(FakeCamList([self.a, self.b])), self.b)

    def test_missing_serial_is_none_not_a_different_camera(self):
        controller, *_ = make_controller(serial="999")
        self.assertIsNone(controller._pick_camera(FakeCamList([self.a, self.b])))

    def test_a_raising_lookup_is_none(self):
        controller, *_ = make_controller(serial="222")
        self.assertIsNone(controller._pick_camera(FakeCamList([self.b], get_by_serial_raises=True)))

    def test_no_serial_uses_the_first_camera_for_legacy_single_camera(self):
        controller = CameraController(system_holder=SharedSystemHolder(lambda: None))
        self.assertIs(controller._pick_camera(FakeCamList([self.a, self.b])), self.a)
        self.assertIsNone(controller._pick_camera(FakeCamList([])))


class ReinitializeTests(unittest.TestCase):
    def test_sole_owner_rebuilds_the_whole_system(self):
        controller, holder, factory, log = make_controller(serial="222")
        attach(controller, holder, factory, "222")
        old_system = controller.system
        ok, message = controller._reinitialize_camera()
        self.assertTrue(ok, message)
        self.assertTrue(old_system.released)
        self.assertIsNot(controller.system, old_system)
        self.assertEqual(controller.cam.serial, "222")
        self.assertTrue(controller.cam.acquiring)

    def test_with_another_camera_streaming_the_system_is_never_touched(self):
        log = []
        cameras = [FakeCamera("111", log), FakeCamera("222", log)]
        factory = SystemFactory(log, cameras)
        holder = SharedSystemHolder(factory)
        a = CameraController(serial="111", tag=None, system_holder=holder)
        b = CameraController(serial="222", tag="cam222", system_holder=holder)
        for c in (a, b):
            c._configure_camera_nodes = lambda: None
            attach(c, holder, factory, c.serial)
        system = holder.system
        a_list = a.cam_list
        ok, message = b._reinitialize_camera()
        self.assertTrue(ok, message)
        self.assertFalse(system.released)  # camera A's System survived B's recovery
        self.assertIs(holder.system, system)
        self.assertEqual(len(factory.made), 1)
        self.assertIs(a.cam_list, a_list)
        self.assertFalse(a_list.cleared)  # A's CameraList untouched
        self.assertEqual(b.cam.serial, "222")

    def test_recovery_always_refinds_the_same_serial_whatever_the_order(self):
        log = []
        cameras = [FakeCamera("222", log), FakeCamera("111", log)]  # B now listed first
        controller, holder, factory, _ = make_controller(serial="111", cameras=cameras, log=log)
        attach(controller, holder, factory, "111")
        ok, _ = controller._reinitialize_camera()
        self.assertTrue(ok)
        self.assertEqual(controller.cam.serial, "111")

    def test_camera_not_visible_yet_is_found_after_update_cameras(self):
        log = []
        controller, holder, factory, _ = make_controller(
            serial="222", cameras=[FakeCamera("111", log)], log=log
        )
        holder.acquire(controller)
        factory.hidden.append(FakeCamera("222", log))  # replugged, bus not refreshed yet
        ok, message = controller._reinitialize_camera()
        self.assertTrue(ok, message)
        self.assertTrue(any(entry.startswith("update-cameras") for entry in log), log)
        self.assertEqual(controller.cam.serial, "222")

    def test_camera_still_missing_reports_failure_and_keeps_the_reference(self):
        log = []
        controller, holder, factory, _ = make_controller(
            serial="222", cameras=[FakeCamera("111", log)], log=log
        )
        holder.acquire(controller)
        ok, message = controller._reinitialize_camera()
        self.assertFalse(ok)
        self.assertIn("222", message)
        self.assertIsNone(controller.cam)
        self.assertTrue(holder.holds(controller))  # can retry on the next backoff tick

    def test_a_failed_system_rebuild_is_a_retryable_failure(self):
        controller, holder, factory, log = make_controller(serial="222", fail_on={2})
        attach(controller, holder, factory, "222")
        ok, message = controller._reinitialize_camera()
        self.assertFalse(ok)
        self.assertIsNone(controller.cam)
        self.assertTrue(holder.holds(controller))
        ok, message = controller._reinitialize_camera()  # next backoff attempt
        self.assertTrue(ok, message)
        self.assertEqual(controller.cam.serial, "222")

    def test_camera_handle_is_dropped_before_its_list_is_cleared(self):
        controller, holder, factory, log = make_controller(serial="222")
        attach(controller, holder, factory, "222")
        old_list = controller.cam_list
        seen = {}
        original_clear = old_list.Clear

        def clear_and_record():
            seen["cam_at_clear"] = controller.cam
            original_clear()

        old_list.Clear = clear_and_record
        controller._reinitialize_camera()
        self.assertIsNone(seen["cam_at_clear"])


class CleanupTests(unittest.TestCase):
    def test_cleanup_releases_only_this_controllers_reference_and_is_idempotent(self):
        log = []
        factory = SystemFactory(log, [FakeCamera("111", log), FakeCamera("222", log)])
        holder = SharedSystemHolder(factory)
        a = CameraController(serial="111", system_holder=holder)
        b = CameraController(serial="222", system_holder=holder)
        holder.acquire(a)
        holder.acquire(b)
        a.system = b.system = holder.system
        b._cleanup_system()
        b._cleanup_system()  # closeEvent after preview-stop
        self.assertEqual(holder.count, 1)
        self.assertTrue(holder.holds(a))
        self.assertFalse(factory.made[0].released)
        a._cleanup_system()
        self.assertTrue(factory.made[0].released)


class RecordingLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.paths = SessionPaths.for_session(
            self.tmp.name, datetime(2026, 10, 1, 10, 15, 0), camera_tag="cam222"
        )
        self.controller, self.holder, self.factory, self.log = make_controller(serial="222")
        attach(self.controller, self.holder, self.factory, "222")
        self.controller.model = "Blackfly S"
        self.addCleanup(self.controller.abort_prepared)

    def header(self):
        first = self.paths.events_jsonl.read_text().splitlines()[0]
        return json.loads(first)

    def test_prepare_opens_the_sidecars_but_does_not_start_recording(self):
        ok, message = self.controller.prepare_recording(self.paths, fps=30.0)
        self.assertTrue(ok, message)
        self.assertFalse(self.controller.record_start_requested)
        self.assertFalse(self.controller.recording_active)
        self.assertTrue(self.paths.metadata_csv.exists())
        self.assertTrue(self.paths.segments_csv.exists())
        self.assertTrue(self.paths.events_jsonl.exists())

    def test_begin_raises_the_start_flag(self):
        self.controller.prepare_recording(self.paths, fps=30.0)
        ok, message = self.controller.begin_recording()
        self.assertTrue(ok, message)
        self.assertTrue(self.controller.record_start_requested)
        self.assertFalse(self.controller.record_stop_requested)

    def test_start_recording_is_prepare_plus_begin(self):
        ok, message = self.controller.start_recording(self.paths, fps=30.0)
        self.assertTrue(ok, message)
        self.assertTrue(self.controller.record_start_requested)
        self.assertIn("cam222", message)  # the per-camera stem, not the shared basename

    def test_header_names_this_cameras_files_and_identity(self):
        self.controller.prepare_recording(self.paths, fps=30.0)
        header = self.header()
        self.assertEqual(header["recording"], "recording_20261001_101500_cam222")
        self.assertEqual(header["session"], "recording_20261001_101500")
        self.assertEqual(header["camera_serial"], "222")
        self.assertEqual(header["camera_model"], "Blackfly S")

    def test_abort_leaves_nothing_started_and_allows_a_fresh_prepare(self):
        self.controller.prepare_recording(self.paths, fps=30.0)
        self.controller.abort_prepared()
        self.assertFalse(self.controller.record_start_requested)
        ok, message = self.controller.prepare_recording(self.paths, fps=30.0)
        self.assertTrue(ok, message)

    def test_a_second_prepare_without_begin_or_abort_is_refused(self):
        self.controller.prepare_recording(self.paths, fps=30.0)
        ok, message = self.controller.prepare_recording(self.paths, fps=30.0)
        self.assertFalse(ok)
        self.assertIn("already prepared", message)

    def test_begin_without_prepare_is_refused(self):
        ok, _ = self.controller.begin_recording()
        self.assertFalse(ok)

    def test_stop_then_immediate_start_is_refused_while_the_old_one_is_closing(self):
        # Existing bug on main: this used to report success and record nothing.
        self.controller.recording_active = True
        self.controller.record_stop_requested = True
        ok, message = self.controller.start_recording(self.paths, fps=30.0)
        self.assertFalse(ok)
        self.assertIn("still closing", message)
        self.assertFalse(self.paths.metadata_csv.exists())  # no sidecars were touched

    def test_starting_while_already_recording_is_still_a_harmless_success(self):
        self.controller.recording_active = True
        ok, message = self.controller.start_recording(self.paths, fps=30.0)
        self.assertTrue(ok)
        self.assertIn("already", message)
        self.assertFalse(self.paths.metadata_csv.exists())

    def test_begin_after_an_already_recording_prepare_is_not_a_failure(self):
        self.controller.recording_active = True
        ok, _ = self.controller.prepare_recording(self.paths, fps=30.0)
        self.assertTrue(ok)
        ok, _ = self.controller.begin_recording()
        self.assertTrue(ok)

    def test_not_acquiring_cannot_prepare(self):
        self.controller.acquiring = False
        ok, message = self.controller.prepare_recording(self.paths, fps=30.0)
        self.assertFalse(ok)
        self.assertIn("not acquiring", message)

    def test_abort_removes_the_empty_sidecars_so_a_same_second_retry_works(self):
        self.controller.prepare_recording(self.paths, fps=30.0)
        self.controller.abort_prepared()
        for path in (self.paths.metadata_csv, self.paths.segments_csv, self.paths.events_jsonl):
            self.assertFalse(path.exists(), path.name)

    def test_stale_label_and_sync_state_are_cleared_for_a_new_session(self):
        self.controller.notify_label_event("label_start", 3, "Writing")
        self.controller.notify_sync_pulse_window(width_s=60.0, label="old")
        self.controller.prepare_recording(self.paths, fps=30.0)
        self.assertIsNone(self.controller._pending_label_event)
        self.assertIsNone(self.controller._pending_adl_id)
        self.assertEqual(self.controller._sync_window_end, 0.0)
        self.assertIsNone(self.controller._sync_label)


class PreviewFrameTests(unittest.TestCase):
    def controller_with_frame(self, shape=(1024, 1280)):
        from backend.camera_control import PreviewFrame

        controller, *_ = make_controller()
        image = np.arange(shape[0] * shape[1], dtype=np.uint32).reshape(shape).astype(np.uint8)
        controller._latest_preview_frame = PreviewFrame(
            image=image, sequence=7, frame_id=70, camera_timestamp=1,
            retrieved_at=1.0, published_at=1.1,
        )
        return controller, image

    def test_unchanged_sequence_returns_none(self):
        controller, _ = self.controller_with_frame()
        self.assertIsNone(controller.get_latest_preview_frame(after_sequence=7))

    def test_full_size_is_an_owned_copy(self):
        controller, image = self.controller_with_frame()
        frame = controller.get_latest_preview_frame()
        self.assertTrue(np.array_equal(frame.image, image))
        self.assertIsNot(frame.image, image)
        frame.image[0, 0] = 255 - frame.image[0, 0]
        self.assertFalse(np.array_equal(frame.image, image))  # caller's edits cannot reach the camera's frame

    def test_max_size_downscales_keeping_aspect_and_metadata(self):
        controller, image = self.controller_with_frame()
        frame = controller.get_latest_preview_frame(max_size=(640, 640))
        self.assertEqual(frame.image.shape, (512, 640))
        self.assertEqual((frame.sequence, frame.frame_id), (7, 70))
        self.assertEqual(image.shape, (1024, 1280))  # source untouched

    def test_max_size_never_upscales(self):
        controller, _ = self.controller_with_frame(shape=(540, 720))
        frame = controller.get_latest_preview_frame(max_size=(4000, 4000))
        self.assertEqual(frame.image.shape, (540, 720))


class StreamRateTests(unittest.TestCase):
    def test_none_until_the_camera_is_acquiring(self):
        controller, *_ = make_controller()
        self.assertIsNone(controller.get_stream_rate())


if __name__ == "__main__":
    unittest.main()
