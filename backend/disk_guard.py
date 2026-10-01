"""Disk-space assessment for long recordings, mirroring power_status.py's shape.

Nothing in this app previously called shutil.disk_usage: when the output
volume filled, Append() started raising, the exception was caught and
printed, and the app kept "recording" indefinitely while producing
nothing. This module classifies free space against the recording
bitrate so the GUI can warn early and refuse to start a run that
can't possibly fit.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping


# Uncompressed 1280x1024 Mono8 (1.31 MB/frame) at the 100 fps target:
# 1,310,720 B/frame * 100 fps * 3600 s/h ~= 471.9 GB/h. Recording switched
# from MJPGOption to AVIOption (uncompressed) after profiling showed
# JPEG encoding inside Append() costing ~17-20ms/frame, enough by itself
# to cap throughput around 55fps regardless of camera settings. The old
# value here (6,593,000,000 B/h, measured from a real MJPEG sample) would
# now be off by ~70x -- badly overestimating how much recording time a
# given amount of free space actually buys.
DEFAULT_BYTES_PER_HOUR = 471_900_000_000
# Sized for the study's actual use: ~1 h sessions, ~2 h/day. Critical = less than
# about two 1-hour sessions' worth of room; warn = under a working day's worth
# of recording. (Was 24 h / 6 h, tuned for the 10-day unattended run; a 4 TB
# drive holds only ~22 h of two-camera uncompressed video, so 24 h warned on
# every start.)
#
# A long unattended run must not lose its early warning to this change: set
# PLANNED_HOURS_ENV (e.g. 240 for 10 days) and assess_disk() asks for
# confirmation whenever the free space is projected to run out sooner than
# that, whatever warn_hours says.
DEFAULT_WARN_HOURS = 8.0
DEFAULT_CRITICAL_HOURS = 2.0
PLANNED_HOURS_ENV = "SLEEVE_VIDEO_GUI_PLANNED_HOURS"


def resolve_planned_hours(env: Mapping[str, str] | None = None) -> float | None:
    """Planned recording duration in hours from PLANNED_HOURS_ENV, else None.

    Ignores blank, non-numeric and non-positive values (a typo should not
    silently disable or break the disk check).
    """
    env_map = os.environ if env is None else env
    raw = (env_map.get(PLANNED_HOURS_ENV) or "").strip()
    try:
        value = float(raw)
    except ValueError:
        return None
    return value if value > 0 else None
DEFAULT_MIN_FREE_BYTES = 20 * 1024**3  # 20 GiB


# MJPEG output relative to the raw frame size. Measured on the rig at ~0.07
# (Blackfly 2.9 MB/s of 39.3 MB/s; Firefly 0.7 of 11.7); rounded up for margin.
# This depends on the encoder's quality setting, which is being reworked
# separately -- revisit when that lands.
MJPEG_SIZE_FRACTION_ESTIMATE = 0.10


@dataclass(frozen=True)
class StreamRate:
    """What one camera writes to disk: its frame geometry, rate and codec."""

    width: int
    height: int
    bytes_per_pixel: int
    fps: float
    compressed: bool = False


def estimate_bytes_per_hour(
    streams: Iterable[StreamRate],
    *,
    mjpeg_fraction: float = MJPEG_SIZE_FRACTION_ESTIMATE,
) -> int:
    """Total bytes/hour for every camera that will record.

    Uses each camera's real frame size, so two different sensors are summed
    correctly instead of assuming one 1280x1024 camera at 100 fps. With no
    streams (camera not yet read) falls back to DEFAULT_BYTES_PER_HOUR.
    """
    total = 0.0
    seen = False
    for stream in streams:
        seen = True
        rate = stream.width * stream.height * stream.bytes_per_pixel * stream.fps * 3600.0
        total += rate * (mjpeg_fraction if stream.compressed else 1.0)
    return int(round(total)) if seen else DEFAULT_BYTES_PER_HOUR


@dataclass(frozen=True)
class DiskSample:
    total_bytes: int
    used_bytes: int
    free_bytes: int
    at_s: float


@dataclass(frozen=True)
class DiskVerdict:
    level: str  # "safe" | "warning" | "danger"
    free_bytes: int
    hours_remaining: float
    reason: str
    recording_blocked: bool
    requires_confirmation: bool


def sample_disk_usage(
    path: str | Path,
    *,
    at_s: float,
    disk_usage: Callable[[str], object] = shutil.disk_usage,
) -> DiskSample:
    """Thin wrapper so callers can inject a fake disk_usage in tests."""
    usage = disk_usage(str(path))
    return DiskSample(
        total_bytes=int(usage.total),
        used_bytes=int(usage.used),
        free_bytes=int(usage.free),
        at_s=at_s,
    )


def assess_disk(
    sample: DiskSample,
    *,
    bytes_per_hour: int = DEFAULT_BYTES_PER_HOUR,
    warn_hours: float = DEFAULT_WARN_HOURS,
    critical_hours: float = DEFAULT_CRITICAL_HOURS,
    min_free_bytes: int = DEFAULT_MIN_FREE_BYTES,
    planned_hours: float | None = None,
) -> DiskVerdict:
    """Classify free space for recording, without touching the filesystem."""
    free_gib = sample.free_bytes / 1024**3
    hours_remaining = (
        sample.free_bytes / bytes_per_hour if bytes_per_hour > 0 else float("inf")
    )

    # Absolute floor, independent of bytes_per_hour: the disk is genuinely
    # almost full right now. This is the actual "Append() started raising
    # and the app kept recording nothing" case from the module docstring,
    # so it stays a hard block.
    if sample.free_bytes <= min_free_bytes:
        return DiskVerdict(
            level="danger",
            free_bytes=sample.free_bytes,
            hours_remaining=hours_remaining,
            reason=(
                f"Only {free_gib:.1f} GiB free. Free up space or point the "
                "output directory at a larger volume before starting."
            ),
            recording_blocked=True,
            requires_confirmation=False,
        )

    # Rate-based projection: a judgment call about whether THIS session is
    # likely to outrun free space, not a fact about the disk right now --
    # bytes_per_hour is an estimate (and can be badly wrong for a short,
    # deliberate test recording, or after a codec change shifts the real
    # rate), so this asks rather than blocks.
    if hours_remaining <= critical_hours:
        return DiskVerdict(
            level="danger",
            free_bytes=sample.free_bytes,
            hours_remaining=hours_remaining,
            reason=(
                f"Only {free_gib:.1f} GiB free (~{hours_remaining:.1f} h at the "
                "estimated recording rate). This may not be enough for the "
                "planned session."
            ),
            recording_blocked=False,
            requires_confirmation=True,
        )

    planned_short = planned_hours is not None and hours_remaining < planned_hours
    if hours_remaining <= warn_hours or planned_short:
        if planned_short:
            advice = (
                f"The planned run is {planned_hours:.0f} h, which this will not "
                "cover. Free space or choose a larger volume."
            )
        else:
            advice = "Consider freeing space before a long session."
        return DiskVerdict(
            level="warning",
            free_bytes=sample.free_bytes,
            hours_remaining=hours_remaining,
            reason=(
                f"{free_gib:.1f} GiB free (~{hours_remaining:.1f} h at the "
                f"estimated recording rate). {advice}"
            ),
            recording_blocked=False,
            requires_confirmation=True,
        )

    return DiskVerdict(
        level="safe",
        free_bytes=sample.free_bytes,
        hours_remaining=hours_remaining,
        reason=f"{free_gib:.1f} GiB free (~{hours_remaining:.1f} h at the estimated rate).",
        recording_blocked=False,
        requires_confirmation=False,
    )
