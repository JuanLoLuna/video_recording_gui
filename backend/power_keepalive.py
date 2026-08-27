"""Keep Windows from sleeping during a recording via SetThreadExecutionState.

SetThreadExecutionState is per-thread: its ES_CONTINUOUS state is cleared
the moment the calling thread exits. Only call apply_keep_awake /
release_keep_awake from the Qt GUI main thread, which outlives the
recording session.

This only suppresses the *idle* sleep timer. It is not honoured on Modern
Standby (S0ix) machines, and never overrides lid-close or a user-initiated
sleep. It is a belt, not a replacement for the braces: also configure
`powercfg /change standby-timeout-ac 0` (and hibernate/disk timeouts, and
lid-close = Do Nothing) on the recording machine.
"""

from __future__ import annotations

import sys
import ctypes
from dataclasses import dataclass


ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001
ES_DISPLAY_REQUIRED = 0x00000002
ES_AWAYMODE_REQUIRED = 0x00000040


@dataclass(frozen=True)
class KeepAwakeRequest:
    system: bool = True
    display: bool = False
    away_mode: bool = True


@dataclass(frozen=True)
class KeepAwakeState:
    supported: bool
    active: bool = False
    flags: int = 0
    away_mode_granted: bool = False
    error: str | None = None


@dataclass(frozen=True)
class KeepAwakeAssessment:
    level: str
    summary: str
    reason: str | None = None


def build_execution_state_flags(request: KeepAwakeRequest) -> int:
    """Return the ES_* flag combination for a keep-awake request."""
    flags = ES_CONTINUOUS
    if request.system:
        flags |= ES_SYSTEM_REQUIRED
    if request.display:
        flags |= ES_DISPLAY_REQUIRED
    if request.away_mode:
        flags |= ES_AWAYMODE_REQUIRED
    return flags


def release_execution_state_flags() -> int:
    """Return the flag value that clears any prior keep-awake request."""
    return ES_CONTINUOUS


def assess_keepalive(
    state: KeepAwakeState,
    *,
    recording: bool,
) -> KeepAwakeAssessment:
    """Classify keep-awake state for display, without touching the system."""
    if not state.supported:
        return KeepAwakeAssessment(
            level="neutral",
            summary="Sleep prevention: available on Windows only",
        )
    if state.error:
        return KeepAwakeAssessment(
            level="warning",
            summary="Sleep prevention: could not be applied",
            reason=state.error,
        )
    if recording and not state.active:
        return KeepAwakeAssessment(
            level="danger",
            summary="Sleep prevention: NOT active while recording",
            reason=(
                "Windows may sleep and interrupt the recording. "
                "Verify power settings (standby/hibernate timeouts, lid action)."
            ),
        )
    if not recording:
        return KeepAwakeAssessment(
            level="neutral",
            summary="Sleep prevention: applied while recording",
        )
    if not state.away_mode_granted:
        return KeepAwakeAssessment(
            level="warning",
            summary="Sleep prevention: active (away mode not granted)",
            reason=(
                "This system did not grant away mode. On Modern Standby "
                "(S0ix) machines, or for lid-close/user-initiated sleep, "
                "this keep-awake request is not honoured by Windows. "
                "Set standby/hibernate timeouts to 0 and lid action to "
                "Do Nothing as a backstop."
            ),
        )
    return KeepAwakeAssessment(
        level="safe",
        summary="Sleep prevention: active",
    )


def apply_keep_awake(
    request: KeepAwakeRequest = KeepAwakeRequest(),
) -> KeepAwakeState:
    """Ask Windows not to sleep/hibernate while this thread holds the request.

    Must be called from the thread that will remain alive for the whole
    recording (the Qt GUI main thread) -- see module docstring.
    """
    if not sys.platform.startswith("win"):
        return KeepAwakeState(supported=False)

    flags = build_execution_state_flags(request)
    try:
        result = ctypes.windll.kernel32.SetThreadExecutionState(flags)
    except Exception as exc:
        return KeepAwakeState(
            supported=True,
            error=f"{exc.__class__.__name__}: {exc}",
        )

    if result != 0:
        return KeepAwakeState(supported=True, active=True, flags=flags, away_mode_granted=request.away_mode)

    if request.away_mode:
        # Not every system grants away mode; retry without it.
        retry_request = KeepAwakeRequest(
            system=request.system, display=request.display, away_mode=False
        )
        retry_flags = build_execution_state_flags(retry_request)
        try:
            retry_result = ctypes.windll.kernel32.SetThreadExecutionState(retry_flags)
        except Exception as exc:
            return KeepAwakeState(
                supported=True,
                error=f"{exc.__class__.__name__}: {exc}",
            )
        if retry_result != 0:
            return KeepAwakeState(
                supported=True, active=True, flags=retry_flags, away_mode_granted=False
            )

    return KeepAwakeState(
        supported=True,
        error="Windows refused the keep-awake request (SetThreadExecutionState returned 0).",
    )


def release_keep_awake() -> KeepAwakeState:
    """Clear any keep-awake request held by the calling thread."""
    if not sys.platform.startswith("win"):
        return KeepAwakeState(supported=False)

    flags = release_execution_state_flags()
    try:
        result = ctypes.windll.kernel32.SetThreadExecutionState(flags)
    except Exception as exc:
        return KeepAwakeState(
            supported=True,
            error=f"{exc.__class__.__name__}: {exc}",
        )
    if result != 0:
        return KeepAwakeState(supported=True, active=False, flags=flags)
    return KeepAwakeState(
        supported=True,
        error="Windows refused the release request (SetThreadExecutionState returned 0).",
    )
