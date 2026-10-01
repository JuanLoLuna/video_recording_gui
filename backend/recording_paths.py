"""Output directory resolution and per-session artifact naming.

Today every recording artifact lands in whatever directory the process
happened to be launched from, built by scattered f-strings across
gui/main.py and camera_control.py. Over a 10-day, ~1.9 TB session that
matters a lot more than it used to, and the scattered construction risks
the four (soon six) sibling files drifting out of sync with each other.

SessionPaths is the single place that knows the naming scheme, including
the two downstream filename contracts it must satisfy:
  - video (final):  recording_YYYYMMDD_HHMMSS-NNNN.avi
    matches smart_sleeve_data_processing/pipelines/audit/rules.yaml:83
    (^recording_\\d{8}_\\d{6}(?:-\\d{4})?\\.avi$)
  - metadata CSV:    recording_YYYYMMDD_HHMMSS_metadata.csv (no -NNNN --
    one continuous CSV per session, per the study's binding decision)
    matches pipelines/video_metadata/router.py:43-46
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Mapping


OUTPUT_DIR_ENV = "SLEEVE_VIDEO_GUI_OUTPUT_DIR"
# None means "fall back to the process CWD", preserving today's behaviour
# for anyone who hasn't set the env var or picked a folder in the GUI.
DEFAULT_OUTPUT_DIR: str | None = None

MAX_SEGMENT_INDEX = 9999  # 4-digit suffix is the downstream filename contract

# A camera tag sits INSIDE the stem, before "-NNNN" / "_metadata". Alphanumeric
# only: "_" and "-" are the separators downstream matchers split on, so a tag
# containing either would make the stem ambiguous.
CAMERA_TAG_RE = re.compile(r"^[A-Za-z0-9]+$")


def camera_tag_for_serial(serial: str | int) -> str:
    """Filename tag for a camera, e.g. 26134271 -> "cam26134271"."""
    tag = f"cam{serial}"
    if not str(serial) or not CAMERA_TAG_RE.match(tag):
        raise ValueError(f"camera serial {serial!r} cannot be used in a filename tag")
    return tag


def resolve_output_dir(
    explicit: str | Path | None = None,
    *,
    env: Mapping[str, str] | None = None,
    cwd: str | Path | None = None,
    create: bool = True,
) -> Path:
    """Resolve the recording output directory.

    Precedence: explicit > env var > DEFAULT_OUTPUT_DIR constant > cwd.
    `env` defaults to os.environ; pass a plain dict in tests. `create`
    makes (and returns) the directory, matching how every other output
    path in this app is used without a separate mkdir step.
    """
    if explicit:
        chosen = Path(explicit)
    else:
        env_map = os.environ if env is None else env
        env_value = env_map.get(OUTPUT_DIR_ENV)
        if env_value:
            chosen = Path(env_value)
        elif DEFAULT_OUTPUT_DIR is not None:
            chosen = Path(DEFAULT_OUTPUT_DIR)
        else:
            chosen = Path(cwd) if cwd is not None else Path.cwd()

    if create:
        chosen.mkdir(parents=True, exist_ok=True)
    return chosen


def session_basename(started_at: datetime) -> str:
    """e.g. recording_20260827_143012 -- matches the existing naming exactly."""
    return f"recording_{started_at.strftime('%Y%m%d_%H%M%S')}"


def check_writable(output_dir: str | Path) -> tuple[bool, str]:
    """Cheap permissions probe: create+delete a marker file in output_dir."""
    directory = Path(output_dir)
    probe = directory / ".write_check"
    try:
        directory.mkdir(parents=True, exist_ok=True)
        probe.touch()
        probe.unlink()
        return True, ""
    except OSError as exc:
        return False, f"{exc.__class__.__name__}: {exc}"


def _require_valid_segment_index(segment_index: int) -> None:
    if not (0 <= segment_index <= MAX_SEGMENT_INDEX):
        raise ValueError(
            f"segment_index {segment_index} out of range "
            f"[0, {MAX_SEGMENT_INDEX}] -- the 4-digit suffix is a "
            "downstream filename contract (rules.yaml/router.py)"
        )


@dataclass(frozen=True)
class SessionPaths:
    """Every artifact path for one recording session, derived from one stem.

    Consolidates what used to be four independent f-strings (gui/main.py's
    video base and wav path, camera_control.py's metadata CSV path, and
    the diagnostics CSV path) so they cannot drift apart from each other
    or from the events/segments sidecars added since.
    """

    output_dir: Path
    basename: str
    # None = the primary / only camera: every name is exactly as it was before
    # multi-camera support. Set for the additional cameras of a session.
    camera_tag: str | None = None

    def __post_init__(self) -> None:
        if self.camera_tag is not None and not CAMERA_TAG_RE.match(self.camera_tag):
            raise ValueError(
                f"camera_tag {self.camera_tag!r} must be alphanumeric "
                "('_' and '-' are filename separators)"
            )

    @property
    def stem(self) -> str:
        """basename plus the camera tag: the stem of every per-camera artifact."""
        if self.camera_tag is None:
            return self.basename
        return f"{self.basename}_{self.camera_tag}"

    def with_camera(self, camera_tag: str | None) -> "SessionPaths":
        """Same session (output dir + basename), a different camera's names."""
        return SessionPaths(
            output_dir=self.output_dir, basename=self.basename, camera_tag=camera_tag
        )

    @property
    def incomplete_dir(self) -> Path:
        """Staging directory for in-progress segment writes.

        Keeps a concurrent Box/rclone copy of output_dir from ever seeing
        a half-written AVI: only fully-closed, canonically-named segments
        exist directly under output_dir.
        """
        return self.output_dir / ".incomplete"

    def video_part_base(self, segment_index: int) -> Path:
        """Base path (no extension) SpinVideo writes to while a segment is open.

        SpinVideo.Open() always appends its own "-0000" suffix, so the
        actual file on disk is f"{this}-0000.avi"; rename it to
        video_final(segment_index) after Close() succeeds.
        """
        _require_valid_segment_index(segment_index)
        return self.incomplete_dir / f"{self.stem}_part{segment_index:04d}"

    def video_final(self, segment_index: int) -> Path:
        _require_valid_segment_index(segment_index)
        return self.output_dir / f"{self.stem}-{segment_index:04d}.avi"

    @property
    def wav(self) -> Path:
        # One microphone per session, shared by every camera: never tagged.
        return self.output_dir / f"{self.basename}.wav"

    @property
    def metadata_csv(self) -> Path:
        return self.output_dir / f"{self.stem}_metadata.csv"

    @property
    def diagnostics_csv(self) -> Path:
        return self.output_dir / f"{self.stem}_diagnostics.csv"

    @property
    def segments_csv(self) -> Path:
        return self.output_dir / f"{self.stem}_segments.csv"

    @property
    def events_jsonl(self) -> Path:
        return self.output_dir / f"{self.stem}_events.jsonl"

    @classmethod
    def for_session(
        cls,
        output_dir: str | Path,
        started_at: datetime,
        camera_tag: str | None = None,
    ) -> "SessionPaths":
        return cls(
            output_dir=Path(output_dir),
            basename=session_basename(started_at),
            camera_tag=camera_tag,
        )
