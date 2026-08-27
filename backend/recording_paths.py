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
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Mapping


OUTPUT_DIR_ENV = "SLEEVE_VIDEO_GUI_OUTPUT_DIR"
# None means "fall back to the process CWD", preserving today's behaviour
# for anyone who hasn't set the env var or picked a folder in the GUI.
DEFAULT_OUTPUT_DIR: str | None = None

MAX_SEGMENT_INDEX = 9999  # 4-digit suffix is the downstream filename contract


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
        return self.incomplete_dir / f"{self.basename}_part{segment_index:04d}"

    def video_final(self, segment_index: int) -> Path:
        _require_valid_segment_index(segment_index)
        return self.output_dir / f"{self.basename}-{segment_index:04d}.avi"

    @property
    def wav(self) -> Path:
        return self.output_dir / f"{self.basename}.wav"

    @property
    def metadata_csv(self) -> Path:
        return self.output_dir / f"{self.basename}_metadata.csv"

    @property
    def diagnostics_csv(self) -> Path:
        return self.output_dir / f"{self.basename}_diagnostics.csv"

    @property
    def segments_csv(self) -> Path:
        return self.output_dir / f"{self.basename}_segments.csv"

    @property
    def events_jsonl(self) -> Path:
        return self.output_dir / f"{self.basename}_events.jsonl"

    @classmethod
    def for_session(
        cls, output_dir: str | Path, started_at: datetime
    ) -> "SessionPaths":
        return cls(output_dir=Path(output_dir), basename=session_basename(started_at))
