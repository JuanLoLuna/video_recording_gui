import re
import stat
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from backend.recording_paths import (
    CAMERA_TAG_RE,
    MAX_SEGMENT_INDEX,
    OUTPUT_DIR_ENV,
    SessionPaths,
    camera_tag_for_serial,
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


# The downstream pipeline's current video-name pattern
# (smart_sleeve_data_processing pipelines/audit/rules.yaml video_raw) and the
# pattern proposed for the first two-camera ingest.
LEGACY_VIDEO_RE = re.compile(r"^recording_\d{8}_\d{6}(?:-\d{4})?\.avi$")
PROPOSED_VIDEO_RE = re.compile(
    r"^recording_\d{8}_\d{6}(?:_cam[A-Za-z0-9]+)?(?:-\d{4})?\.avi$"
)


class CameraTagTests(unittest.TestCase):
    STARTED = datetime(2026, 10, 1, 10, 15, 0)

    def paths(self, tag=None):
        return SessionPaths.for_session("/data", self.STARTED, camera_tag=tag)

    def test_no_tag_leaves_every_name_exactly_as_before(self):
        paths = self.paths()
        base = "recording_20261001_101500"
        self.assertEqual(paths.stem, base)
        self.assertEqual(paths.video_final(0), Path("/data") / f"{base}-0000.avi")
        self.assertEqual(paths.video_final(12), Path("/data") / f"{base}-0012.avi")
        self.assertEqual(paths.metadata_csv, Path("/data") / f"{base}_metadata.csv")
        self.assertEqual(paths.diagnostics_csv, Path("/data") / f"{base}_diagnostics.csv")
        self.assertEqual(paths.segments_csv, Path("/data") / f"{base}_segments.csv")
        self.assertEqual(paths.events_jsonl, Path("/data") / f"{base}_events.jsonl")
        self.assertEqual(paths.wav, Path("/data") / f"{base}.wav")
        self.assertEqual(
            paths.video_part_base(3), Path("/data/.incomplete") / f"{base}_part0003"
        )

    def test_tag_goes_inside_the_stem_for_every_per_camera_artifact(self):
        paths = self.paths("cam26134271")
        stem = "recording_20261001_101500_cam26134271"
        self.assertEqual(paths.video_final(0), Path("/data") / f"{stem}-0000.avi")
        self.assertEqual(paths.metadata_csv, Path("/data") / f"{stem}_metadata.csv")
        self.assertEqual(paths.diagnostics_csv, Path("/data") / f"{stem}_diagnostics.csv")
        self.assertEqual(paths.segments_csv, Path("/data") / f"{stem}_segments.csv")
        self.assertEqual(paths.events_jsonl, Path("/data") / f"{stem}_events.jsonl")
        self.assertEqual(
            paths.video_part_base(1), Path("/data/.incomplete") / f"{stem}_part0001"
        )

    def test_wav_is_never_tagged_because_the_microphone_is_shared(self):
        self.assertEqual(self.paths("cam26134271").wav, self.paths().wav)

    def test_two_cameras_never_share_any_output_path(self):
        a, b = self.paths(), self.paths("cam26134271")
        for index in (0, 1, 9999):
            self.assertNotEqual(a.video_final(index), b.video_final(index))
            self.assertNotEqual(a.video_part_base(index), b.video_part_base(index))
        for name in ("metadata_csv", "diagnostics_csv", "segments_csv", "events_jsonl"):
            self.assertNotEqual(getattr(a, name), getattr(b, name), name)

    def test_with_camera_keeps_the_session_and_changes_only_the_names(self):
        primary = self.paths()
        other = primary.with_camera("cam23227865")
        self.assertEqual(other.output_dir, primary.output_dir)
        self.assertEqual(other.basename, primary.basename)
        self.assertEqual(other.camera_tag, "cam23227865")
        self.assertIsNone(other.with_camera(None).camera_tag)

    def test_invalid_tags_are_rejected(self):
        # "_" and "-" are the separators downstream matchers split on.
        for bad in ("", "cam_1", "cam-1", "cam 1", "cam/1", "..", "cam.1"):
            with self.assertRaises(ValueError, msg=repr(bad)):
                SessionPaths(output_dir=Path("/data"), basename="b", camera_tag=bad)

    def test_camera_tag_for_serial(self):
        self.assertEqual(camera_tag_for_serial("26134271"), "cam26134271")
        self.assertEqual(camera_tag_for_serial(23227865), "cam23227865")
        self.assertTrue(CAMERA_TAG_RE.match(camera_tag_for_serial("abc123")))
        with self.assertRaises(ValueError):
            camera_tag_for_serial("12_34")

    def test_segment_index_range_still_enforced_with_a_tag(self):
        with self.assertRaises(ValueError):
            self.paths("cam1").video_final(MAX_SEGMENT_INDEX + 1)

    def test_primary_names_still_match_the_legacy_downstream_pattern(self):
        self.assertTrue(LEGACY_VIDEO_RE.match(self.paths().video_final(0).name))
        self.assertTrue(PROPOSED_VIDEO_RE.match(self.paths().video_final(0).name))

    def test_tagged_names_match_only_the_proposed_pattern(self):
        # Documents the known, deliberately deferred downstream gap: until the
        # pipeline's pattern gains "(?:_cam[A-Za-z0-9]+)?", a second camera's
        # video is archive-only.
        name = self.paths("cam26134271").video_final(7).name
        self.assertIsNone(LEGACY_VIDEO_RE.match(name))
        self.assertTrue(PROPOSED_VIDEO_RE.match(name))

    def test_csv_stem_pairs_with_its_video_by_stripping_metadata(self):
        # Downstream pairs "<stem>_metadata.csv" with "<stem>-NNNN.avi".
        paths = self.paths("cam26134271")
        stem = paths.metadata_csv.name[: -len("_metadata.csv")]
        self.assertTrue(paths.video_final(0).name.startswith(stem + "-"))
