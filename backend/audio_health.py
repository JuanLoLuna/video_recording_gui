# backend/audio_health.py
"""
Pure audio-stream health tracking: PortAudio xrun counting, reconnect
planning, and silence-gap calculation for a dropped/reopened input stream.

No sounddevice/soundfile import here so this stays testable on macOS.
"""

from __future__ import annotations

from dataclasses import dataclass, field

_STATUS_FLAGS = (
    "input_underflow",
    "input_overflow",
    "output_underflow",
    "output_overflow",
    "priming_output",
)


@dataclass
class AudioHealthCounters:
    """Running totals of PortAudio callback `status` flags and reconnects."""

    input_underflow: int = 0
    input_overflow: int = 0
    output_underflow: int = 0
    output_overflow: int = 0
    priming_output: int = 0
    reconnects: int = 0
    silence_frames_inserted: int = 0

    def note_status(self, status: object) -> None:
        """Count whichever PortAudio xrun flags are set on a callback's `status`.

        `status` is whatever sounddevice hands the callback (a CallbackFlags-like
        object); missing attributes count as False so this never raises.
        """
        for flag in _STATUS_FLAGS:
            if getattr(status, flag, False):
                setattr(self, flag, getattr(self, flag) + 1)

    def note_reconnect(self) -> None:
        self.reconnects += 1

    def note_silence_frames(self, n_frames: int) -> None:
        self.silence_frames_inserted += max(0, int(n_frames))

    def snapshot(self) -> "AudioHealthSnapshot":
        return AudioHealthSnapshot(
            input_underflow=self.input_underflow,
            input_overflow=self.input_overflow,
            output_underflow=self.output_underflow,
            output_overflow=self.output_overflow,
            priming_output=self.priming_output,
            reconnects=self.reconnects,
            silence_frames_inserted=self.silence_frames_inserted,
        )


@dataclass(frozen=True)
class AudioHealthSnapshot:
    """Immutable point-in-time copy of AudioHealthCounters, safe to hand to a GUI timer."""

    input_underflow: int = 0
    input_overflow: int = 0
    output_underflow: int = 0
    output_overflow: int = 0
    priming_output: int = 0
    reconnects: int = 0
    silence_frames_inserted: int = 0

    @property
    def total_xruns(self) -> int:
        return (
            self.input_underflow
            + self.input_overflow
            + self.output_underflow
            + self.output_overflow
        )


def silence_frames_for_gap(gap_s: float, samplerate: int) -> int:
    """How many zero-sample frames to write so wall-clock time stays aligned
    with sample count across a dropout of gap_s seconds."""
    if gap_s <= 0 or samplerate <= 0:
        return 0
    return round(gap_s * samplerate)


@dataclass
class ReconnectPlan:
    """One attempt to reopen the input stream after it dropped."""

    device_name: str
    attempt: int
    backoff_s: float


_DEFAULT_BACKOFFS_S: tuple[float, ...] = (0.2, 0.5, 1.0, 2.0, 5.0)


def plan_reconnect(
    device_name: str,
    attempt: int,
    *,
    backoffs_s: tuple[float, ...] = _DEFAULT_BACKOFFS_S,
) -> ReconnectPlan:
    """Backoff for reconnect attempt N (1-indexed), capped at the last entry."""
    idx = min(max(attempt, 1), len(backoffs_s)) - 1
    return ReconnectPlan(device_name=device_name, attempt=attempt, backoff_s=backoffs_s[idx])


def resolve_device_index_by_name(
    device_name: str, devices: list[dict], *, hostapi: int | None = None,
) -> int | None:
    """Find a device's current index by its captured name (and, optionally,
    its host API) -- but only if exactly one candidate matches.

    PortAudio device indices shift after USB re-enumeration; matching by name
    (as sounddevice's query_devices() reports it) survives that shuffle where
    matching by the original index would not. Name alone is not always
    unique, though: some drivers report the same generic name (e.g.
    "Microphone Array") for several physically different devices -- observed
    in practice with over a dozen input entries on one test box, some of
    them non-functional. hostapi narrows that in the common case, but
    real hardware can still collide on both name AND host API.

    Silently guessing among multiple candidates risks locking onto a
    non-functional device and recording silence for the rest of the run
    with no indication anything is wrong -- worse than staying disconnected,
    which is at least visible (the reconnect loop keeps retrying and the
    gap keeps growing in the logs). So: return None on any genuine
    ambiguity rather than picking the first match, even though that means
    a colliding name may never auto-resolve.
    """
    target_name = device_name.strip()
    candidates = [
        i
        for i, d in enumerate(devices)
        if isinstance(d, dict)
        and str(d.get("name", "")).strip() == target_name
        and int(d.get("max_input_channels", 0) or 0) >= 1
        and (hostapi is None or d.get("hostapi") == hostapi)
    ]
    return candidates[0] if len(candidates) == 1 else None
