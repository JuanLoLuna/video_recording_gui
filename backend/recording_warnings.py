"""Dismissible, auto-recovering banner state for recording integrity warnings.

The GUI previously latched a single "integrity warning" boolean the
moment any frame gap/incomplete image/acquisition error occurred during a
recording, and never cleared it -- so after the acquisition watchdog
recovered from a camera fault, the warning stayed red for the rest of the
(possibly multi-day) session with no way to tell "still broken" from
"fixed an hour ago".

RecordingWarningTracker instead tracks whether a problem is CURRENTLY
active vs recently recovered vs fully quiet, and separately whether the
user has dismissed what they've already read -- so the banner can go
quiet after both the underlying problem clears AND the user acknowledges
it, while still reopening on a genuinely new episode.
"""

from __future__ import annotations

from dataclasses import dataclass


DEFAULT_RECOVERED_WINDOW_S = 15.0


@dataclass(frozen=True)
class WarningEvent:
    at_s: float
    message: str


@dataclass(frozen=True)
class WarningBannerState:
    visible: bool
    level: str  # "active" | "recovered" | "hidden"
    headline: str
    detail: str
    total_events: int


class RecordingWarningTracker:
    """Not thread-safe -- call only from the GUI thread."""

    def __init__(self, *, recovered_window_s: float = DEFAULT_RECOVERED_WINDOW_S) -> None:
        self._recovered_window_s = recovered_window_s
        self._events: list[WarningEvent] = []
        self._last_event_at: float | None = None
        self._dismissed = False
        self._dismissed_level: str | None = None

    def reset(self) -> None:
        """Call at the start of each new recording session."""
        self._events.clear()
        self._last_event_at = None
        self._dismissed = False
        self._dismissed_level = None

    def note_issue(self, message: str, *, now_s: float) -> None:
        self._events.append(WarningEvent(at_s=now_s, message=message))
        self._last_event_at = now_s

    def dismiss(self, *, now_s: float) -> None:
        """Hide the banner until its level changes (active<->recovered) or a
        fresh episode starts after a period of full quiet."""
        state = self.summarize(now_s=now_s)
        self._dismissed = True
        self._dismissed_level = state.level

    def summarize(self, *, now_s: float) -> WarningBannerState:
        if not self._events:
            return WarningBannerState(
                visible=False, level="hidden", headline="", detail="", total_events=0
            )

        elapsed = now_s - self._last_event_at
        level = "active" if elapsed < self._recovered_window_s else "recovered"

        if self._dismissed and level == self._dismissed_level:
            visible = False
        else:
            visible = True
            if self._dismissed and level != self._dismissed_level:
                # The situation changed since the user last dismissed it
                # (e.g. active -> recovered, or a fresh fault after a
                # fully-quiet recovered period) -- worth resurfacing, and
                # stays unsuppressed until dismissed again.
                self._dismissed = False
                self._dismissed_level = None

        recent = self._events[-1].message
        if level == "active":
            headline = f"Recording issue: {recent}"
            detail = f"{len(self._events)} issue(s) logged this session."
        else:
            headline = f"Recovered — last issue: {recent}"
            detail = (
                f"{len(self._events)} issue(s) logged this session, "
                f"none in the last {int(self._recovered_window_s)}s."
            )
        return WarningBannerState(
            visible=visible,
            level=level,
            headline=headline,
            detail=detail,
            total_events=len(self._events),
        )
