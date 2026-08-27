"""When to roll video recording to a new segment file.

Naming lives in backend/recording_paths.py (SessionPaths); this module
only decides WHEN to roll and reconciles the rare case where Spinnaker's
own SetMaximumFileSize net fires mid-segment.

Three ceilings, defence in depth:
  - max_frames (primary): frame count, not wall clock -- over a 10-day
    session a wall-clock policy is exposed to NTP steps, and time.time()
    moving backwards could stall rotation indefinitely.
  - max_bytes: sampled periodically by the caller, catches a bitrate
    spike (subject moving, night IR gain noise) that would otherwise blow
    the 4 GiB AVI RIFF ceiling before max_frames is reached.
  - fault: an external signal (the acquisition watchdog's timeline break)
    forces an immediate roll regardless of frame/byte counts, so a
    wall-clock hole always lands between segments, never inside one.
  - SpinVideo's own SetMaximumFileSize is set as a hard SDK-level net
    below both of the above and should never fire; reconcile_part_files
    handles it gracefully if it ever does.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence


DEFAULT_SEGMENT_SECONDS = 900.0  # 15 minutes
DEFAULT_MAX_BYTES = 3_000_000_000  # ~2.6x headroom under the 4 GiB RIFF ceiling
DEFAULT_SDK_MAX_FILE_SIZE_MB = 3584  # SetMaximumFileSize() argument -- the SDK-level net
BYTES_SAMPLE_INTERVAL_FRAMES = 300
PREPARE_LEAD_FRAMES = 60  # pre-open the next writer this many frames before the roll


def segment_frames_for(fps: float, segment_seconds: float = DEFAULT_SEGMENT_SECONDS) -> int:
    """Frames per segment for a given fps, e.g. 30 fps * 900 s = 27000.

    Falls back to 30 fps for a non-positive or NaN fps rather than
    producing a zero/negative segment length.
    """
    if fps is None or fps != fps or fps <= 0:  # None, NaN, or <= 0
        fps = 30.0
    return max(1, round(fps * segment_seconds))


@dataclass(frozen=True)
class SegmentRollDecision:
    should_roll: bool
    reason: str | None  # "frame_count" | "bytes" | "fault" | None


def should_roll(
    *,
    frames_in_segment: int,
    bytes_in_segment: int,
    max_frames: int,
    max_bytes: int = DEFAULT_MAX_BYTES,
    fault: bool = False,
) -> SegmentRollDecision:
    """Precedence when multiple conditions fire at once: fault, then frame count, then bytes."""
    if fault:
        return SegmentRollDecision(True, "fault")
    if frames_in_segment >= max_frames:
        return SegmentRollDecision(True, "frame_count")
    if bytes_in_segment >= max_bytes:
        return SegmentRollDecision(True, "bytes")
    return SegmentRollDecision(False, None)


def should_prepare(
    *, frames_in_segment: int, max_frames: int, lead_frames: int = PREPARE_LEAD_FRAMES
) -> bool:
    """True once frames_in_segment is within lead_frames of rolling.

    A range condition (>=), not a one-shot: robust to a resumed/faulted
    segment whose frame count doesn't start at 0. The caller is
    responsible for tracking whether it has already acted on this (i.e.
    already opened the next writer) so it does so exactly once per
    segment rather than on every frame in the window.
    """
    threshold = max(0, max_frames - lead_frames)
    return frames_in_segment >= threshold


_PART_SUFFIX_RE = re.compile(r"^(?P<prefix>.+)-(?P<index>\d{4})\.avi$")


def reconcile_part_files(part_base: Path) -> list[Path]:
    """Find the SDK-numbered files SpinVideo actually wrote under part_base.

    Normally exactly one: f"{part_base}-0000.avi" (SpinVideo always
    appends its own -0000 suffix). If Spinnaker's own SetMaximumFileSize
    net ever fires mid-segment, a second f"{part_base}-0001.avi" (etc.)
    appears alongside it. Returns all of them, sorted by their own
    numeric suffix, so the caller can rename each into its own final
    segment slot instead of silently orphaning the extras in .incomplete/.

    Returns an empty list (does not raise) if nothing matches -- e.g. the
    writer never successfully opened.
    """
    parent = part_base.parent
    prefix = part_base.name
    if not parent.is_dir():
        return []
    candidates: list[tuple[int, Path]] = []
    for entry in parent.iterdir():
        m = _PART_SUFFIX_RE.match(entry.name)
        if m and m.group("prefix") == prefix:
            candidates.append((int(m.group("index")), entry))
    candidates.sort(key=lambda pair: pair[0])
    return [path for _, path in candidates]


@dataclass(frozen=True)
class RenamePlan:
    source: Path
    destination: Path


def plan_renames(
    part_files: Sequence[Path],
    *,
    next_segment_index: int,
    video_final: Callable[[int], Path],
) -> list[RenamePlan]:
    """Map reconciled part files to final segment slots, one index each.

    Consumes one segment index per part file found -- normally just one,
    but two if the SDK's own file-size net fired mid-segment.
    """
    plans: list[RenamePlan] = []
    index = next_segment_index
    for part_file in part_files:
        plans.append(RenamePlan(source=part_file, destination=video_final(index)))
        index += 1
    return plans
