# backend/power_policy.py
"""
Pure, debounced power pause/resume policy.

Sits on top of the existing PowerSafetyAssessment.recording_blocked boolean
(backend/power_status.py, unchanged). Debounces on a sample count rather than
a clock, matching the GUI's fixed 500ms poll cadence -- no clock injection
needed for deterministic tests.

On sustained-unsafe power while recording: pause (stop the current session
cleanly, camera keeps previewing). On sustained-safe power while paused:
resume (start a brand-new session -- never reopen the interrupted one).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

DEFAULT_DEBOUNCE_SAMPLES = 20  # 20 * 500ms poll interval = 10s sustained


@dataclass(frozen=True)
class PowerPolicyState:
    consecutive_blocked: int = 0
    consecutive_safe: int = 0
    paused: bool = False


@dataclass(frozen=True)
class PowerPolicyDecision:
    action: str  # "none" | "pause" | "resume"
    reason: str = ""


def next_power_action(
    state: PowerPolicyState,
    *,
    recording_blocked: bool,
    recording: bool,
    reason: str = "",
    debounce_samples: int = DEFAULT_DEBOUNCE_SAMPLES,
) -> tuple[PowerPolicyState, PowerPolicyDecision]:
    """Advance the policy by one poll sample.

    `recording` is True only while actively recording (not while paused) --
    the GUI's own AppState already tracks this. `state.paused` is the
    policy's own memory of being in a power-triggered pause, independent of
    `recording`, since `recording` naturally goes False for the whole pause
    window.
    """
    if state.paused:
        if recording_blocked:
            return replace(state, consecutive_safe=0), PowerPolicyDecision("none")
        new_safe = state.consecutive_safe + 1
        if new_safe >= debounce_samples:
            return PowerPolicyState(), PowerPolicyDecision("resume")
        return replace(state, consecutive_safe=new_safe), PowerPolicyDecision("none")

    if not recording:
        return PowerPolicyState(), PowerPolicyDecision("none")

    if recording_blocked:
        new_blocked = state.consecutive_blocked + 1
        if new_blocked >= debounce_samples:
            return (
                PowerPolicyState(paused=True),
                PowerPolicyDecision("pause", reason=reason),
            )
        return replace(state, consecutive_blocked=new_blocked), PowerPolicyDecision("none")

    return replace(state, consecutive_blocked=0), PowerPolicyDecision("none")
