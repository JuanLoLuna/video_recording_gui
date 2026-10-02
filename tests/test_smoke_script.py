"""scripts/multi_controller_smoke.py end to end against fake cameras.

The script is what runs on the rig, so it gets the same treatment as the app:
both codecs through the real controllers, the group, the verifier and the video
decode checks. Skipped when real PySpin is present (there, run the script).
"""
import contextlib
import importlib.util
import io
import os
import sys
import tempfile
import unittest
from pathlib import Path

from fake_spinnaker import FakeCamera, FakeSystem, install_pyspin_stub

REAL_PYSPIN = install_pyspin_stub()

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import backend.spinnaker_system as spinnaker_system  # noqa: E402
from backend.camera_registry import CAMERA_SERIALS_ENV  # noqa: E402
from backend.spinnaker_system import SharedSystemHolder  # noqa: E402


def load_smoke():
    spec = importlib.util.spec_from_file_location("multi_controller_smoke", SCRIPTS / "multi_controller_smoke.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@unittest.skipIf(REAL_PYSPIN, "real PySpin present: run the script on the rig")
class SmokeScriptTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        previous = spinnaker_system._default_holder
        self.addCleanup(lambda: setattr(spinnaker_system, "_default_holder", previous))
        saved = os.environ.get(CAMERA_SERIALS_ENV)
        os.environ.pop(CAMERA_SERIALS_ENV, None)
        self.addCleanup(lambda: os.environ.__setitem__(CAMERA_SERIALS_ENV, saved) if saved else None)
        self.cameras = [FakeCamera("26134271", "Blackfly S", 64, 48), FakeCamera("23227865", "Firefly", 32, 24)]
        spinnaker_system._default_holder = SharedSystemHolder(lambda: FakeSystem(self.cameras))

    def run_script(self, *extra):
        module = load_smoke()
        argv = ["multi_controller_smoke.py", "--seconds", "2.5", "--segment-seconds", "1",
                "--output-dir", self.tmp.name, *extra]
        out = io.StringIO()
        old_argv = sys.argv
        sys.argv = argv
        try:
            with contextlib.redirect_stdout(out):
                code = module.main()
        finally:
            sys.argv = old_argv
        return code, out.getvalue()

    def test_mjpeg_run_passes_and_proves_the_codec_and_the_pixels(self):
        code, text = self.run_script("--codec", "mjpg")
        self.assertEqual(code, 0, text)
        self.assertIn("RESULT: PASS", text)
        self.assertIn("MJPG", text)  # the codec was read back from the files, not assumed
        self.assertNotIn("BLACK", text)
        self.assertEqual(text.count("PASS"), 3, text)  # two cameras + the overall result

    def test_uncompressed_run_passes_and_reads_back_as_raw(self):
        code, text = self.run_script("--codec", "grey")
        self.assertEqual(code, 0, text)
        self.assertIn("raw/uncompressed", text)

    def test_asking_for_a_codec_the_files_do_not_have_fails(self):
        # Corrupt the expectation to prove the check can fail: claim MJPG but record GREY.
        module = load_smoke()
        module.EXPECTED_CODEC = {"grey": "MJPG", "mjpg": "MJPG"}
        out = io.StringIO()
        old_argv = sys.argv
        sys.argv = ["x", "--seconds", "2.0", "--segment-seconds", "1", "--output-dir", self.tmp.name,
                    "--codec", "grey"]
        try:
            with contextlib.redirect_stdout(out):
                code = module.main()
        finally:
            sys.argv = old_argv
        self.assertEqual(code, 1)
        self.assertIn("expected MJPG", out.getvalue())

    def test_a_missing_configured_camera_refuses_to_run(self):
        code, text = self.run_script("--serials", "23227865", "26134271", "99999999")
        self.assertEqual(code, 2, text)
        self.assertIn("refusing to run", text)

    def test_a_crash_is_written_to_the_log_file_with_the_phase_markers(self):
        # A crash after the cameras started must leave its traceback in the log.
        module = load_smoke()
        module.check_videos = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("decode exploded"))
        out = io.StringIO()
        old_argv = sys.argv
        sys.argv = ["x", "--seconds", "2.0", "--segment-seconds", "1", "--output-dir", self.tmp.name,
                    "--codec", "grey"]
        try:
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
                code = module.main()
        finally:
            sys.argv = old_argv
        self.assertEqual(code, 1)
        log = sorted(Path(self.tmp.name).glob("smoke_log_*.txt"))[0].read_text(encoding="utf-8")
        self.assertIn("Traceback", log)
        self.assertIn("decode exploded", log)
        self.assertIn("capture finished; stopping the cameras", log)
        self.assertIn("verifying #", log)

    def test_the_log_file_holds_the_final_result_and_is_written_line_by_line(self):
        code, text = self.run_script("--codec", "grey")
        logs = sorted(Path(self.tmp.name).glob("smoke_log_*.txt"))
        self.assertEqual(len(logs), 1)
        content = logs[0].read_text(encoding="utf-8")
        self.assertIn("RESULT:", content)
        self.assertIn("start_recording_all", content)
        self.assertEqual(code, 0)

    def test_progress_reaches_the_file_before_the_run_ends(self):
        # Simulates a run that is killed mid-way: the lines printed so far must
        # already be on disk, not sitting in a buffer.
        import io
        module = load_smoke()
        out_dir = Path(self.tmp.name)
        log_path = out_dir / "partial.txt"
        with open(log_path, "w", encoding="utf-8", buffering=1) as log_file:
            tee = module._Tee(io.StringIO(), log_file)
            tee.write("first line\n")
            tee.write("second line\n")
            self.assertEqual(log_path.read_text(encoding="utf-8"), "first line\nsecond line\n")


@unittest.skipIf(REAL_PYSPIN, "real PySpin present: run the script on the rig")
class SegmentSpotCheckTests(unittest.TestCase):
    """A bad segment in the MIDDLE of a run must be found, not only the first and last."""

    def build(self, middle_black):
        import csv
        from types import SimpleNamespace

        import cv2
        import numpy as np

        from backend.recording_paths import SessionPaths

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        paths = SessionPaths.for_session(tmp.name, __import__("datetime").datetime(2026, 10, 2, 14, 0, 0))
        good = np.tile(np.linspace(40, 200, 64, dtype=np.uint8), (48, 1))
        rows = []
        for index in range(5):
            frames = [np.zeros_like(good) if (index == 2 and middle_black) else good for _ in range(10)]
            writer = cv2.VideoWriter(str(paths.video_final(index)), cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (64, 48), isColor=False)
            for frame in frames:
                writer.write(frame)
            writer.release()
            rows.append({"segment_file": paths.video_final(index).name, "frame_count": 10})
        with open(paths.segments_csv, "w", newline="", encoding="utf-8") as handle:
            w = csv.DictWriter(handle, fieldnames=["segment_file", "frame_count"])
            w.writeheader()
            w.writerows(rows)
        return paths, SimpleNamespace(segment_count=5)

    def run_check(self, middle_black):
        module = load_smoke()
        paths, report = self.build(middle_black)
        args = SimpleNamespace_args(codec="mjpg", video_stride=1)
        return module.check_videos(args, paths, report)

    def test_a_black_segment_in_the_middle_is_reported(self):
        problems, lines = self.run_check(middle_black=True)
        self.assertTrue(any("0002.avi" in p and "BLACK" in p for p in problems), problems)
        self.assertTrue(any("quick-checked 3 other" in line for line in lines), lines)

    def test_all_good_segments_pass(self):
        problems, _ = self.run_check(middle_black=False)
        self.assertEqual(problems, [])


def SimpleNamespace_args(**kw):
    from types import SimpleNamespace

    return SimpleNamespace(**kw)


class QueueLimitTests(unittest.TestCase):
    def test_the_limit_is_a_duration_so_it_scales_with_frame_rate(self):
        module = load_smoke()
        self.assertEqual(module.queue_limit("grey", 30.0), 5)
        self.assertEqual(module.queue_limit("mjpg", 30.0), 7)
        self.assertEqual(module.queue_limit("grey", 60.0), 10)
        self.assertEqual(module.queue_limit("mjpg", 59.98), 14)
        # Never stricter than at 30 fps for slower cameras.
        self.assertEqual(module.queue_limit("mjpg", 15.0), 7)

    def test_the_rig_peak_of_7_frames_at_60_fps_is_within_the_limit(self):
        # Run 150639: a retried writer open queued 7 frames at 60 fps (~0.12 s), nothing lost.
        module = load_smoke()
        self.assertLess(7, module.queue_limit("mjpg", 59.98))


class StopHeartbeatTests(unittest.TestCase):
    class Controller:
        recording_active = True
        acquiring = True

        class _Q:
            def __init__(self, n):
                self.n = n
                self.unfinished_tasks = n

            def qsize(self):
                return self.n

        def __init__(self):
            self._append_queue = self._Q(3)
            self._closer_queue = self._Q(1)

    def test_it_names_what_each_camera_is_waiting_for(self):
        from types import SimpleNamespace

        module = load_smoke()
        lines = []
        beat = module.StopHeartbeat(
            [SimpleNamespace(serial="111", controller=self.Controller())], interval_s=0.05, out=lines.append
        ).start()
        import time

        time.sleep(0.3)
        beat.stop()
        self.assertGreaterEqual(len(lines), 2)
        self.assertIn("still stopping after", lines[0])
        self.assertIn("#111: recording=True acquiring=True appendQ=3 closerQ=1", lines[0])

    def test_it_stays_silent_when_the_stop_is_quick(self):
        from types import SimpleNamespace

        module = load_smoke()
        lines = []
        beat = module.StopHeartbeat(
            [SimpleNamespace(serial="111", controller=self.Controller())], interval_s=5.0, out=lines.append
        ).start()
        beat.stop()
        self.assertEqual(lines, [])


if __name__ == "__main__":
    unittest.main()
