"""Disk-space assessment for long recordings, mirroring power_status.py's shape.

Nothing in this app previously called shutil.disk_usage: when the output
volume filled, Append() started raising, the exception was caught and
printed, and the app kept "recording" indefinitely while producing
nothing. This module classifies free space against the measured MJPEG
bitrate so the GUI can warn early and refuse to start a run that
can't possibly fit.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


# Measured from a real sample (3,480,960,398 B / 1900.9 s) -- see the
# hardening plan's "corrected facts" section.
DEFAULT_BYTES_PER_HOUR = 6_593_000_000
DEFAULT_WARN_HOURS = 24.0
DEFAULT_CRITICAL_HOURS = 6.0
DEFAULT_MIN_FREE_BYTES = 20 * 1024**3  # 20 GiB


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
) -> DiskVerdict:
    """Classify free space for recording, without touching the filesystem."""
    free_gib = sample.free_bytes / 1024**3
    hours_remaining = (
        sample.free_bytes / bytes_per_hour if bytes_per_hour > 0 else float("inf")
    )

    if sample.free_bytes <= min_free_bytes or hours_remaining <= critical_hours:
        return DiskVerdict(
            level="danger",
            free_bytes=sample.free_bytes,
            hours_remaining=hours_remaining,
            reason=(
                f"Only {free_gib:.1f} GiB free (~{hours_remaining:.1f} h at the "
                "measured recording rate). Free up space or point the output "
                "directory at a larger volume before starting."
            ),
            recording_blocked=True,
            requires_confirmation=False,
        )

    if hours_remaining <= warn_hours:
        return DiskVerdict(
            level="warning",
            free_bytes=sample.free_bytes,
            hours_remaining=hours_remaining,
            reason=(
                f"{free_gib:.1f} GiB free (~{hours_remaining:.1f} h at the "
                "measured recording rate). Consider freeing space for a "
                "multi-day session."
            ),
            recording_blocked=False,
            requires_confirmation=True,
        )

    return DiskVerdict(
        level="safe",
        free_bytes=sample.free_bytes,
        hours_remaining=hours_remaining,
        reason=f"{free_gib:.1f} GiB free (~{hours_remaining:.1f} h at the measured rate).",
        recording_blocked=False,
        requires_confirmation=False,
    )
