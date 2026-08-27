"""Choose a WAV container that survives a multi-day recording.

Plain WAV (RIFF) uses a 32-bit chunk size, so a file cannot exceed 4 GiB.
At 44.1 kHz mono PCM16 that is ~13.5 hours -- well inside an overnight
session, let alone a 10-day one. RF64 (and the older Sony Wave64) lift
that limit and are both readable by libsndfile/soundfile, so this module
probes what the installed libsndfile actually supports at runtime and
picks the best container, rather than assuming RF64 is present.

The chosen format is written with `format=...` passed explicitly to
soundfile, which does NOT infer the container from the file extension in
that case -- so an RF64 file can still be named "*.wav", matching every
downstream tool that identifies audio by extension alone.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass


WAV_RIFF_LIMIT_BYTES = 4 * 1024**3

BITS_PER_SAMPLE = {
    "PCM_16": 16,
    "PCM_24": 24,
    "PCM_32": 32,
    "FLOAT": 32,
}

DEFAULT_PREFERRED_FORMATS = ("RF64", "W64", "WAV")


@dataclass(frozen=True)
class AudioFormatChoice:
    format: str | None
    subtype: str
    is_streaming_safe: bool
    max_bytes: int | None
    max_seconds: float | None
    reason: str


def wav_bytes_per_second(samplerate: int, channels: int, subtype: str = "PCM_16") -> int:
    """Raw PCM byte rate for a given samplerate/channels/subtype."""
    bits = BITS_PER_SAMPLE.get(subtype, 16)
    return int(samplerate) * int(channels) * (bits // 8)


def riff_capacity_seconds(samplerate: int, channels: int, subtype: str = "PCM_16") -> float:
    """How many seconds of audio fit under the 4 GiB RIFF ceiling."""
    bps = wav_bytes_per_second(samplerate, channels, subtype)
    if bps <= 0:
        return float("inf")
    return WAV_RIFF_LIMIT_BYTES / bps


def choose_audio_format(
    *,
    available_formats: Mapping[str, str],
    samplerate: int,
    channels: int,
    subtype: str = "PCM_16",
    preferred: tuple[str, ...] = DEFAULT_PREFERRED_FORMATS,
    format_checker: Callable[[str, str], bool] | None = None,
) -> AudioFormatChoice:
    """Pick the best container this libsndfile build actually supports.

    available_formats: typically sf.available_formats() -- {format_id: description}.
    format_checker: typically sf.check_format -- (format_id, subtype) -> bool.
        If omitted, any format present in available_formats is accepted.
    """
    checker = format_checker or (lambda fmt, sub: True)

    for fmt in preferred:
        if fmt == "WAV":
            continue
        if fmt not in available_formats:
            continue
        if not checker(fmt, subtype):
            continue
        return AudioFormatChoice(
            format=fmt,
            subtype=subtype,
            is_streaming_safe=True,
            max_bytes=None,
            max_seconds=None,
            reason=f"{fmt}, unlimited size",
        )

    max_seconds = riff_capacity_seconds(samplerate, channels, subtype)
    max_bytes = (
        None if max_seconds == float("inf") else int(max_seconds * wav_bytes_per_second(samplerate, channels, subtype))
    )
    hours = max_seconds / 3600.0 if max_seconds != float("inf") else float("inf")
    return AudioFormatChoice(
        format=None,
        subtype=subtype,
        is_streaming_safe=False,
        max_bytes=max_bytes,
        max_seconds=max_seconds,
        reason=(
            "plain WAV only (RF64/W64 unavailable) -- recording will stop "
            f"cleanly at the 4 GiB RIFF limit (~{hours:.1f} h)"
        ),
    )


class AudioWriteBudget:
    """Tracks frames written against a hard limit, for the plain-WAV fallback.

    Used only when choose_audio_format() could not find a streaming-safe
    container. Stopping cleanly at the budget keeps the WAV header valid;
    letting the RIFF size field overflow instead produces a file most
    readers cannot open.
    """

    def __init__(self, limit_frames: int | None) -> None:
        self._limit_frames = limit_frames
        self._frames_written = 0

    def note_frames(self, n: int) -> None:
        self._frames_written += int(n)

    @property
    def frames_written(self) -> int:
        return self._frames_written

    def should_stop(self) -> bool:
        if self._limit_frames is None:
            return False
        return self._frames_written >= self._limit_frames

    @classmethod
    def for_choice(cls, choice: AudioFormatChoice, samplerate: int) -> "AudioWriteBudget":
        if choice.is_streaming_safe or choice.max_seconds is None:
            return cls(limit_frames=None)
        return cls(limit_frames=int(choice.max_seconds * samplerate))
