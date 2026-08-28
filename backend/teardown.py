# backend/teardown.py
"""
Pure logic for the camera teardown safety gate.

Releasing the Spinnaker system (or DeInit'ing the camera handle) while the
acquisition thread might still be touching those same native objects is an
access violation in native code, not a catchable Python exception -- there
is no safe way to interrupt it after the fact. The only safe move if the
thread hasn't exited by the time teardown needs to release native resources
is to skip release entirely and leak the handle: a leaked handle at process
exit is recoverable (the OS reclaims it), a use-after-free mid-teardown is
not.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TeardownDecision:
    safe_to_release: bool
    reason: str


def assess_teardown_readiness(*, acquisition_thread_alive: bool) -> TeardownDecision:
    """Whether it's safe to DeInit the camera / release the Spinnaker system."""
    if acquisition_thread_alive:
        return TeardownDecision(
            safe_to_release=False,
            reason=(
                "acquisition thread did not exit in time; leaking the "
                "camera handle rather than risking a use-after-free"
            ),
        )
    return TeardownDecision(safe_to_release=True, reason="")
