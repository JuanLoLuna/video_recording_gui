"""Two real CameraControllers recording end to end against fake cameras.

The fake cameras produce frames at a steady rate and stand in for PySpin's
objects, so everything above that layer is the real code: the acquisition and
append threads, segment rotation and the closer thread, the metadata CSV,
segment manifest and events log, the CameraGroup's two-phase start and ordered
stop, per-camera file naming, and the post-run reconciliation. Skipped on a
machine with real PySpin installed (the rig) -- there the real cameras are the
test, via scripts/multi_controller_smoke.py.
"""
import os
import tempfile
import time
import unittest
from datetime import datetime

REAL_PYSPIN = __import__("fake_spinnaker").install_pyspin_stub()

from backend.camera_control import CameraController, enumerate_cameras  # noqa: E402
from backend.camera_group import CameraGroup, CameraSlot  # noqa: E402
from backend.recording_paths import SessionPaths  # noqa: E402
from backend.session_verify import verify_camera_outputs  # noqa: E402
from backend.spinnaker_system import SharedSystemHolder  # noqa: E402
from fake_spinnaker import FPS, FakeCamera, FakeImage, FakeSystem  # noqa: E402


class BrokenTLCamera(FakeCamera):
    def GetTLDeviceNodeMap(self):
        raise RuntimeError("TL read failed")


@unittest.skipIf(REAL_PYSPIN, "real PySpin present: use scripts/multi_controller_smoke.py")
class StartStopTests(unittest.TestCase):
    """start()/stop() against fake cameras: shared-System bookkeeping and identity."""

    def make(self, cameras, **kw):
        self.cameras = cameras
        holder = SharedSystemHolder(lambda: FakeSystem(self.cameras))
        controller = CameraController(system_holder=holder, **kw)
        controller._configure_camera_nodes = lambda: None
        self.addCleanup(controller.stop)
        return controller, holder

    def test_a_legacy_controller_follows_a_swapped_camera_after_stop(self):
        # No serial given (the existing GUI): the first start pins whatever it
        # finds, but a stop must release the pin, or swapping in a different
        # camera and pressing Preview again looks for the old one forever.
        a, b = FakeCamera("111", "A", 8, 6), FakeCamera("222", "B", 8, 6)
        controller, holder = self.make([a])
        ok, message = controller.start()
        self.assertTrue(ok, message)
        self.assertEqual(controller.serial, "111")
        self.assertEqual(controller.model, "A")
        ok, _ = controller.stop()
        self.assertTrue(ok)
        self.assertIsNone(controller.serial)
        self.cameras[:] = [b]
        ok, message = controller.start()
        self.assertTrue(ok, message)
        self.assertEqual(controller.serial, "222")

    def test_a_pinned_serial_survives_stop_because_it_was_requested(self):
        controller, _ = self.make([FakeCamera("111", "A", 8, 6)], serial="111")
        controller.start()
        controller.stop()
        self.assertEqual(controller.serial, "111")

    def test_no_cameras_releases_the_shared_system(self):
        controller, holder = self.make([])
        ok, message = controller.start()
        self.assertFalse(ok)
        self.assertIn("No cameras", message)
        self.assertEqual(holder.count, 0)

    def test_a_requested_serial_that_is_absent_is_not_found_and_releases(self):
        controller, holder = self.make([FakeCamera("111", "A", 8, 6)], serial="999")
        ok, message = controller.start()
        self.assertFalse(ok)
        self.assertIn("999", message)
        self.assertEqual(holder.count, 0)

    def test_stop_twice_is_harmless(self):
        controller, holder = self.make([FakeCamera("111", "A", 8, 6)], serial="111")
        controller.start()
        self.assertTrue(controller.stop()[0])
        self.assertTrue(controller.stop()[0])
        self.assertEqual(holder.count, 0)

    def test_enumerate_lists_every_camera_and_leaves_the_system_released(self):
        holder = SharedSystemHolder(lambda: FakeSystem([FakeCamera("111", "A", 8, 6), FakeCamera("222", "B", 8, 6)]))
        found = enumerate_cameras(holder)
        self.assertEqual([(c.serial, c.model, c.vendor) for c in found],
                         [("111", "A", "FAKE"), ("222", "B", "FAKE")])
        self.assertEqual(holder.count, 0)

    def test_enumerate_keeps_other_cameras_when_one_cannot_be_read(self):
        holder = SharedSystemHolder(
            lambda: FakeSystem([BrokenTLCamera("111", "A", 8, 6), FakeCamera("222", "B", 8, 6)])
        )
        found = enumerate_cameras(holder)
        self.assertEqual([c.serial for c in found], ["<unavailable>", "222"])
        self.assertEqual(holder.count, 0)

    def test_enumerate_does_not_disturb_a_running_controller(self):
        cams = [FakeCamera("111", "A", 8, 6)]
        controller, holder = self.make(cams, serial="111")
        controller.start()
        self.assertEqual(len(enumerate_cameras(holder)), 1)
        self.assertEqual(holder.count, 1)  # the controller's reference is untouched
        self.assertTrue(controller.acquiring)

    def test_a_failed_segment_zero_open_cleans_up_and_reports(self):
        controller, _ = self.make([FakeCamera("111", "A", 8, 6)], serial="111")
        controller.start()
        with tempfile.TemporaryDirectory() as tmp:
            paths = SessionPaths.for_session(tmp, datetime(2026, 10, 1, 10, 15, 0))

            def cannot_open(index):
                raise OSError("codec missing")

            controller._open_segment_writer = cannot_open
            import contextlib, io
            with contextlib.redirect_stdout(io.StringIO()):
                ok, _ = controller.start_recording(paths, fps=FPS)
                self.assertTrue(ok)  # the start was accepted; the failure is asynchronous
                deadline = time.monotonic() + 3.0
                while controller.record_start_requested and time.monotonic() < deadline:
                    time.sleep(0.05)
            self.assertFalse(controller.recording_active)
            self.assertIn("codec missing", controller.last_start_error or "")
            self.assertIsNone(controller._event_log)
            for path in (paths.metadata_csv, paths.segments_csv, paths.events_jsonl):
                self.assertFalse(path.exists(), path.name)
            # ...and the controller is reusable, with the same name, immediately.
            del controller._open_segment_writer
            ok, message = controller.prepare_recording(paths, fps=FPS)
            self.assertTrue(ok, message)
            controller.abort_prepared()


@unittest.skipIf(REAL_PYSPIN, "real PySpin present: use scripts/multi_controller_smoke.py")
class TwoCameraRecordingTests(unittest.TestCase):
    def setUp(self):
        # 1 s segments at 30 fps -> several rotations in a few seconds.
        previous = os.environ.get("SLEEVE_VIDEO_GUI_SEGMENT_SECONDS")
        os.environ["SLEEVE_VIDEO_GUI_SEGMENT_SECONDS"] = "1"
        self.addCleanup(
            lambda: os.environ.pop("SLEEVE_VIDEO_GUI_SEGMENT_SECONDS", None)
            if previous is None
            else os.environ.__setitem__("SLEEVE_VIDEO_GUI_SEGMENT_SECONDS", previous)
        )
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

        # Listed in an order unlike either sorted order, to prove binding is by serial.
        self.cameras = [
            FakeCamera("26134271", "Blackfly S", 64, 48),
            FakeCamera("23227865", "Firefly", 32, 24),
        ]
        holder = SharedSystemHolder(lambda: FakeSystem(self.cameras))
        specs = [("23227865", "Firefly", None, True), ("26134271", "Blackfly S", "cam26134271", False)]
        self.slots = []
        for serial, model, tag, primary in specs:
            controller = CameraController(serial=serial, tag=tag, system_holder=holder)
            controller.target_frame_rate = FPS
            controller._configure_camera_nodes = lambda: None  # needs the real node API
            self.slots.append(CameraSlot(controller, serial=serial, model=model, tag=tag, is_primary=primary))
        self.group = CameraGroup(self.slots)
        self.addCleanup(self.group.stop_all)
        self.holder = holder

    def record(self, seconds):
        started = datetime(2026, 10, 1, 10, 15, 0)
        paths = {s.serial: SessionPaths.for_session(self.tmp.name, started, camera_tag=s.tag)
                 for s in self.slots}
        result = self.group.start_all()
        self.assertTrue(result.ok, result.message)
        result = self.group.start_recording_all(lambda s: paths[s.serial], lambda s: FPS)
        self.assertTrue(result.ok, result.message)
        time.sleep(seconds)
        stopped = self.group.stop_all()
        self.assertTrue(stopped.ok, stopped.message)
        return paths

    def test_two_cameras_record_to_separate_reconciled_sessions(self):
        paths = self.record(3.5)
        reports = {}
        for slot in self.slots:
            p = paths[slot.serial]
            report = verify_camera_outputs(
                p.metadata_csv, p.segments_csv, p.events_jsonl,
                expect_stem=p.stem, expect_serial=slot.serial,
            )
            reports[slot.serial] = report
            self.assertTrue(report.ok, f"{slot.serial}: {report.problems}")
            self.assertGreaterEqual(report.metadata_rows, 80)  # ~3.5 s x 30 fps
            self.assertGreaterEqual(report.segment_count, 3)  # rotation happened
            self.assertEqual(report.camera_frame_id_gaps, 0)
            self.assertEqual(report.timeline_breaks, 0)
            for index in range(report.segment_count):
                self.assertTrue(p.video_final(index).exists(), p.video_final(index).name)
            self.assertEqual(list(p.incomplete_dir.glob("*")), [])

    def test_each_camera_writes_only_its_own_files_with_its_own_names(self):
        paths = self.record(1.5)
        primary, other = paths["23227865"], paths["26134271"]
        self.assertEqual(primary.stem, "recording_20261001_101500")
        self.assertEqual(other.stem, "recording_20261001_101500_cam26134271")
        names = sorted(os.listdir(self.tmp.name))
        self.assertIn("recording_20261001_101500_metadata.csv", names)
        self.assertIn("recording_20261001_101500_cam26134271_metadata.csv", names)
        self.assertIn("recording_20261001_101500-0000.avi", names)
        self.assertIn("recording_20261001_101500_cam26134271-0000.avi", names)

    def test_binding_is_by_serial_and_frame_sizes_stay_per_camera(self):
        import cv2

        paths = self.record(1.5)
        sizes = {}
        for serial, p in paths.items():
            cap = cv2.VideoCapture(str(p.video_final(0)))
            sizes[serial] = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            cap.release()
        self.assertEqual(sizes["23227865"], (32, 24))
        self.assertEqual(sizes["26134271"], (64, 48))
        self.assertEqual([s.controller.model for s in self.slots], ["Firefly", "Blackfly S"])

    def test_stop_releases_the_shared_system_exactly_when_the_last_camera_stops(self):
        self.record(1.0)
        self.assertEqual(self.holder.count, 0)
        self.assertIsNone(self.holder.system)

    def test_a_second_session_can_follow_without_restarting_anything(self):
        self.record(1.0)
        self.tmp2 = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp2.cleanup)
        started = datetime(2026, 10, 1, 11, 0, 0)
        paths = {s.serial: SessionPaths.for_session(self.tmp2.name, started, camera_tag=s.tag)
                 for s in self.slots}
        self.assertTrue(self.group.start_all().ok)
        result = self.group.start_recording_all(lambda s: paths[s.serial], lambda s: FPS)
        self.assertTrue(result.ok, result.message)
        time.sleep(1.0)
        self.assertTrue(self.group.stop_all().ok)
        for slot in self.slots:
            p = paths[slot.serial]
            report = verify_camera_outputs(p.metadata_csv, p.segments_csv, p.events_jsonl,
                                           expect_stem=p.stem, expect_serial=slot.serial)
            self.assertTrue(report.ok, report.problems)


@unittest.skipIf(REAL_PYSPIN, "real PySpin present: use scripts/multi_controller_smoke.py")
class BlackSegmentDetectionTests(unittest.TestCase):
    """A segment that is finalized correctly but decodes black must be reported by the app."""

    def make(self, black):
        import numpy as np

        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        previous = os.environ.get("SLEEVE_VIDEO_GUI_SEGMENT_SECONDS")
        os.environ["SLEEVE_VIDEO_GUI_SEGMENT_SECONDS"] = "1"
        self.addCleanup(
            lambda: os.environ.pop("SLEEVE_VIDEO_GUI_SEGMENT_SECONDS", None)
            if previous is None else os.environ.__setitem__("SLEEVE_VIDEO_GUI_SEGMENT_SECONDS", previous)
        )
        camera = FakeCamera("111", "Cam", 64, 48)
        holder = SharedSystemHolder(lambda: FakeSystem([camera]))
        controller = CameraController(serial="111", system_holder=holder)
        controller._configure_camera_nodes = lambda: None
        controller.target_frame_rate = FPS
        if black:
            real_open = controller._open_segment_writer

            class BlackWriter:
                """Reports success, finalizes normally, but writes a black picture."""

                def __init__(self, inner):
                    self.inner = inner

                def write(self, frame):
                    return self.inner.write(np.zeros_like(frame))

                def release(self):
                    return self.inner.release()

                def isOpened(self):
                    return self.inner.isOpened()

            controller._open_segment_writer = lambda index: BlackWriter(real_open(index))
        # Bright, time-varying frames so the recorded first-frame mean is well above the dark limit.
        original = FakeImage.GetNDArray
        FakeImage.GetNDArray = lambda self: np.full((self.height, self.width), 120, dtype=np.uint8)
        self.addCleanup(lambda: setattr(FakeImage, "GetNDArray", original))
        group = CameraGroup([CameraSlot(controller, serial="111", model="Cam", tag=None, is_primary=True)])
        self.addCleanup(group.stop_all)
        return controller, group

    def record(self, controller, group, seconds=3.2):
        paths = SessionPaths.for_session(self.tmp.name, datetime(2026, 10, 2, 14, 0, 0))
        self.assertTrue(group.start_all().ok)
        self.assertTrue(group.start_recording_all(lambda s: paths, lambda s: FPS).ok)
        time.sleep(seconds)
        self.assertTrue(group.stop_all().ok)

    def test_good_segments_raise_no_pixel_problem(self):
        controller, group = self.make(black=False)
        self.record(controller, group)
        self.assertEqual(controller.segment_pixel_problems, [])

    def test_black_segments_are_reported_by_the_app(self):
        import contextlib, io

        controller, group = self.make(black=True)
        with contextlib.redirect_stdout(io.StringIO()) as printed:
            self.record(controller, group)
        self.assertGreaterEqual(len(controller.segment_pixel_problems), 3)  # every rotated segment
        self.assertTrue(all("black" in reason for _, reason in controller.segment_pixel_problems))
        self.assertIn("ERROR: segment", printed.getvalue())


if __name__ == "__main__":
    unittest.main()
