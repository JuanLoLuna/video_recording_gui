import importlib.util
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

_spec = importlib.util.spec_from_file_location(
    "verify_avi", Path(__file__).resolve().parent.parent / "scripts" / "verify_avi.py"
)
verify_avi = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(verify_avi)


def write_video(path, codec, frames):
    height, width = frames[0].shape
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec), 30.0, (width, height), isColor=False)
    assert writer.isOpened(), f"cannot open writer for {codec}"
    for frame in frames:
        writer.write(frame)
    writer.release()


def moving_frames(count, height=48, width=64):
    base = np.linspace(40, 200, width, dtype=np.uint8)[None, :].repeat(height, axis=0)
    return [np.roll(base, shift=i * 2, axis=1).copy() for i in range(count)]


class ScanVideoTests(unittest.TestCase):
    def scan(self, codec, frames, **kw):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / f"{codec}.avi"
            write_video(path, codec, frames)
            return verify_avi.scan_video(str(path), **kw)

    def test_reports_the_codec_actually_used(self):
        self.assertEqual(self.scan("MJPG", moving_frames(10))["fourcc"], "MJPG")
        # Uncompressed GREY reads back as a code of 0 in OpenCV.
        self.assertEqual(self.scan("GREY", moving_frames(10))["fourcc"], "raw/uncompressed")

    def test_a_normal_video_decodes_every_frame_and_is_not_black(self):
        for codec in ("MJPG", "GREY"):
            info = self.scan(codec, moving_frames(12))
            self.assertTrue(info["opened"])
            self.assertEqual(info["decoded"], 12)
            self.assertEqual(info["decoded"], info["container_frame_count"])
            self.assertFalse(info["all_black"], codec)
            self.assertEqual(info["black_frames"], 0)
            self.assertGreater(info["mean_min"], 30)

    def test_a_video_of_black_frames_is_flagged_even_though_the_frame_count_is_right(self):
        black = [np.zeros((48, 64), dtype=np.uint8) for _ in range(10)]
        info = self.scan("MJPG", black)
        self.assertEqual(info["decoded"], info["container_frame_count"])  # looks fine by count alone
        self.assertTrue(info["all_black"])

    def test_one_dark_frame_among_good_ones_is_counted_but_not_all_black(self):
        frames = moving_frames(10)
        frames[4] = np.zeros_like(frames[4])
        info = self.scan("GREY", frames)
        self.assertEqual(info["black_frames"], 1)
        self.assertFalse(info["all_black"])

    def test_identical_frames_are_reported_as_a_frozen_run(self):
        frame = moving_frames(1)[0]
        info = self.scan("GREY", [frame.copy() for _ in range(12)], max_frozen_run=5)
        self.assertGreaterEqual(info["max_frozen_run_seen"], 5)
        self.assertGreaterEqual(info["frozen_warnings"], 1)

    def test_stride_only_checks_every_nth_frame(self):
        info = self.scan("GREY", moving_frames(20), stride=5)
        self.assertEqual(info["decoded"], 20)
        self.assertEqual(info["checked"], 4)

    def test_missing_or_unreadable_files_are_not_opened(self):
        self.assertFalse(verify_avi.scan_video("/no/such/file.avi")["opened"])
        with tempfile.TemporaryDirectory() as d:
            junk = Path(d) / "junk.avi"
            junk.write_bytes(b"not a video")
            self.assertFalse(verify_avi.scan_video(str(junk))["opened"])


class QuickCheckTests(unittest.TestCase):
    def check(self, codec, frames):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "q.avi"
            write_video(path, codec, frames)
            return verify_avi.quick_check(str(path))

    def test_a_good_video_reports_frames_codec_and_brightness(self):
        info = self.check("MJPG", moving_frames(20))
        self.assertTrue(info["opened"])
        self.assertEqual(info["frames"], 20)
        self.assertEqual(info["fourcc"], "MJPG")
        self.assertEqual(len(info["means"]), 3)
        self.assertFalse(info["all_black"])

    def test_a_black_video_is_flagged_by_the_cheap_check_too(self):
        info = self.check("MJPG", [np.zeros((48, 64), dtype=np.uint8) for _ in range(12)])
        self.assertTrue(info["all_black"])

    def test_missing_file_is_not_opened(self):
        self.assertFalse(verify_avi.quick_check("/no/such.avi")["opened"])


class ScanFolderTests(unittest.TestCase):
    def folder(self, kinds):
        d = tempfile.TemporaryDirectory()
        self.addCleanup(d.cleanup)
        good = moving_frames(10)
        for i, kind in enumerate(kinds):
            frames = [np.zeros_like(good[0]) for _ in good] if kind == "black" else good
            write_video(Path(d.name) / f"seg-{i:04d}.avi", "MJPG", frames)
            if kind == "junk":
                (Path(d.name) / f"seg-{i:04d}.avi").write_bytes(b"x")
        return d.name

    def run_scan(self, folder):
        import contextlib, io

        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = verify_avi.scan_folder(folder)
        return code, out.getvalue()

    def test_a_clean_folder_returns_zero(self):
        code, text = self.run_scan(self.folder(["good", "good", "good"]))
        self.assertEqual(code, 0, text)
        self.assertIn("3 file(s) checked, 0 with problems", text)

    def test_a_black_file_in_the_middle_is_listed_and_fails_the_scan(self):
        code, text = self.run_scan(self.folder(["good", "black", "good"]))
        self.assertEqual(code, 1)
        self.assertIn("1 with problems", text)
        self.assertRegex(text, r"seg-0001\.avi.*BLACK")

    def test_an_undecodable_file_fails_the_scan(self):
        code, text = self.run_scan(self.folder(["good", "junk"]))
        self.assertEqual(code, 1)
        self.assertIn("CANNOT DECODE", text)

    def test_an_empty_folder_is_a_failure_not_a_pass(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(self.run_scan(d)[0], 1)


if __name__ == "__main__":
    unittest.main()
