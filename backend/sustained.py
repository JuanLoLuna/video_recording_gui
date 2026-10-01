"""A condition that only counts once it has held continuously for a while.

Used for writer-backlog warnings: on the rig a USB device plugged into the
shared dock stalled the SSD for ~5 s (queue peaked at 150 frames, drained on its
own, nothing lost). A one-sample threshold would raise a false alarm and, for
the MJPEG warning, wrong advice. Only a backlog that PERSISTS is a real problem.
"""

from __future__ import annotations


class SustainedCondition:
    def __init__(self, hold_seconds: float) -> None:
        self.hold_seconds = float(hold_seconds)
        self._since: float | None = None

    def update(self, active: bool, now_s: float) -> bool:
        """Feed one sample; True once `active` has held for hold_seconds."""
        if not active:
            self._since = None
            return False
        if self._since is None:
            self._since = now_s
        return now_s - self._since >= self.hold_seconds

    def reset(self) -> None:
        self._since = None

    @property
    def active_since(self) -> float | None:
        return self._since
