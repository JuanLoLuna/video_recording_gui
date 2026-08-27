import re
import stat
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from backend.recording_paths import (
    MAX_SEGMENT_INDEX,
    OUTPUT_DIR_ENV,
    SessionPaths,
    check_writable,
    resolve_output_dir,
    session_basename,
)


class ResolveOutputDirTests(unittest.TestCase):
    def test_explicit_wins_over_everything(self):
        with tempfile.TemporaryDirectory() as directory:
            explicit = Path(directory) / "explicit"
            result = resolve_output_dir(
                explicit=explicit,
                env={OUTPUT_DIR_ENV: str(Path(directory) / "env")},
                cwd=str(Path(directory) / "cwd"),
            )
            self.assertEqual(result, explicit)
            self.assertTrue(result.is_dir())

    def test_env_wins_over_default_and_cwd(self):
        with tempfile.TemporaryDirectory() as directory:
            env_dir = Path(directory) / "from_env"
            result = resolve_output_dir(
                env={OUTPUT_DIR_ENV: str(env_dir)},
                cwd=str(Path(directory) / "cwd"),
            )
            self.assertEqual(result, env_dir)

    def test_falls_back_to_cwd_when_nothing_else_set(self):
        with tempfile.TemporaryDirectory() as directory:
            cwd_dir = Path(directory) / "cwd"
            result = resolve_output_dir(env={}, cwd=str(cwd_dir))
            self.assertEqual(result, cwd_dir)
            self.assertTrue(result.is_dir())

    def test_env_pointing_at_a_missing_dir_creates_it(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "does" / "not" / "exist"
            result = resolve_output_dir(env={OUTPUT_DIR_ENV: str(missing)})
            self.assertTrue(result.is_dir())

    def test_create_false_does_not_make_the_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "not_created"
            result = resolve_output_dir(explicit=missing, create=False)
            self.assertEqual(result, missing)
            self.assertFalse(result.exists())


class SessionBasenameTests(unittest.TestCase):
    def test_matches_the_existing_naming_scheme(self):
        self.assertEqual(
            session_basename(datetime(2026, 8, 27, 14, 30, 12)),
            "recording_20260827_143012",
        )


class SessionPathsTests(unittest.TestCase):
    def test_all_artifacts_share_one_stem(self):
        paths = SessionPaths(output_dir=Path("/tmp/out"), basename="recording_X")
        self.assertEqual(paths.wav.name, "recording_X.wav")
        self.assertEqual(paths.metadata_csv.name, "recording_X_metadata.csv")
        self.assertEqual(paths.diagnostics_csv.name, "recording_X_diagnostics.csv")
        self.assertEqual(paths.segments_csv.name, "recording_X_segments.csv")
        self.assertEqual(paths.events_jsonl.name, "recording_X_events.jsonl")

    def test_video_final_matches_the_downstream_contract(self):
        paths = SessionPaths(output_dir=Path("/tmp/out"), basename="recording_20260827_143012")
        name = paths.video_final(7).name
        self.assertEqual(name, "recording_20260827_143012-0007.avi")
        self.assertRegex(name, r"^recording_\d{8}_\d{6}(?:-\d{4})?\.avi$")

    def test_metadata_csv_matches_the_downstream_contract(self):
        paths = SessionPaths(output_dir=Path("/tmp/out"), basename="recording_20260827_143012")
        name = paths.metadata_csv.name
        self.assertRegex(
            name, r"^recording_(?P<date>\d{8})_(?P<time>\d{6})(?:-\d{4})?_metadata(?:.*)?\.csv$"
        )

    def test_video_part_base_lives_under_incomplete(self):
        paths = SessionPaths(output_dir=Path("/tmp/out"), basename="recording_X")
        part = paths.video_part_base(7)
        self.assertEqual(part.parent, paths.incomplete_dir)
        self.assertEqual(part.name, "recording_X_part0007")
        self.assertNotIn(".", part.name)

    def test_segment_index_zero_pads_to_four_digits(self):
        paths = SessionPaths(output_dir=Path("/tmp/out"), basename="recording_X")
        self.assertEqual(paths.video_final(0).name, "recording_X-0000.avi")
        self.assertEqual(paths.video_final(9999).name, "recording_X-9999.avi")

    def test_segment_index_above_the_limit_raises(self):
        paths = SessionPaths(output_dir=Path("/tmp/out"), basename="recording_X")
        with self.assertRaises(ValueError):
            paths.video_final(MAX_SEGMENT_INDEX + 1)
        with self.assertRaises(ValueError):
            paths.video_part_base(MAX_SEGMENT_INDEX + 1)

    def test_negative_segment_index_raises(self):
        paths = SessionPaths(output_dir=Path("/tmp/out"), basename="recording_X")
        with self.assertRaises(ValueError):
            paths.video_final(-1)

    def test_for_session_builds_the_expected_basename(self):
        paths = SessionPaths.for_session("/tmp/out", datetime(2026, 8, 27, 14, 30, 12))
        self.assertEqual(paths.basename, "recording_20260827_143012")
        self.assertEqual(paths.output_dir, Path("/tmp/out"))


class CheckWritableTests(unittest.TestCase):
    def test_writable_directory_is_ok(self):
        with tempfile.TemporaryDirectory() as directory:
            ok, reason = check_writable(directory)
            self.assertTrue(ok)
            self.assertEqual(reason, "")

    @unittest.skipIf(sys.platform.startswith("win"), "chmod permissions differ on Windows")
    def test_unwritable_directory_is_not_ok(self):
        with tempfile.TemporaryDirectory() as directory:
            locked = Path(directory) / "locked"
            locked.mkdir()
            locked.chmod(stat.S_IREAD | stat.S_IEXEC)
            try:
                ok, reason = check_writable(locked)
                self.assertFalse(ok)
                self.assertNotEqual(reason, "")
            finally:
                locked.chmod(stat.S_IRWXU)


if __name__ == "__main__":
    unittest.main()
