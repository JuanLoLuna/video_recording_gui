"""Pure decision logic for recovering the acquisition loop from camera faults.

Two independent failure modes previously had no recovery at all:

  - GetNextImage() raising -> the loop's `except: continue` busy-spun at
    100% CPU forever with no backoff and no reconnect.
  - GetNextImage() blocking forever with no exception at all (the camera
    stops delivering but the call never returns) -> silently records
    nothing, undetected.

AcquisitionWatchdog tracks both from injected clock values only, so it
never depends on a specific PySpin exception type or error code (neither
of which could be verified without the physical camera). The camera
thread calls note_error()/note_frame_ok() around each grab attempt and
poll() periodically; the watchdog tells it whether to keep going, sleep
and retry, or attempt a full camera reinitialisation.

Recovery never gives up: the user's binding requirement for a 10-day
unattended run is auto-resume, not auto-stop. Backoff caps at
backoff_max_s, so a camera unplugged for hours costs only cheap periodic
retries and self-heals the moment it returns.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WatchdogConfig:
    frame_period_s: float
    grab_timeout_s: float
    stall_timeout_s: float
    consecutive_error_limit: int = 10
    backoff_initial_s: float = 0.05
    backoff_max_s: float = 5.0
    backoff_factor: float = 2.0


def watchdog_config_for_frame_rate(fps: float) -> WatchdogConfig:
    """Derive grab/stall timeouts from the configured acquisition rate.

    grab_timeout_s bounds a single GetNextImage() call; stall_timeout_s
    bounds how long we'll go with no successful frame before declaring a
    silent stall (no exception, just nothing arriving).
    """
    fps = float(fps) if fps and fps > 0 else 30.0
    period = 1.0 / fps
    grab_timeout_s = min(2.0, max(0.2, 5.0 * period))
    stall_timeout_s = max(10.0 * period, 5.0)
    return WatchdogConfig(
        frame_period_s=period,
        grab_timeout_s=grab_timeout_s,
        stall_timeout_s=stall_timeout_s,
    )


@dataclass(frozen=True)
class WatchdogDecision:
    action: str  # "continue" | "sleep" | "reinit"
    sleep_s: float
    reason: str
    consecutive_errors: int
    reinit_attempt: int
    stalled_for_s: float


class AcquisitionWatchdog:
    """State machine: HEALTHY -> DEGRADED (backoff) -> RECOVERING (reinit)."""

    def __init__(self, config: WatchdogConfig, *, now: float) -> None:
        self._config = config
        self._last_good_frame_at = now
        self._consecutive_errors = 0
        self._reinit_attempt = 0
        self._current_backoff_s = config.backoff_initial_s

    def note_frame_ok(self, *, now: float) -> None:
        self._last_good_frame_at = now
        self._consecutive_errors = 0
        self._reinit_attempt = 0
        self._current_backoff_s = self._config.backoff_initial_s

    def note_error(self, *, now: float, error: str) -> WatchdogDecision:
        self._consecutive_errors += 1
        if self._consecutive_errors >= self._config.consecutive_error_limit:
            return self._request_reinit(now=now, reason=f"error limit reached: {error}")
        return self._backoff_decision(now=now, reason=error)

    def note_reinit_result(self, *, now: float, ok: bool) -> WatchdogDecision:
        if ok:
            self.note_frame_ok(now=now)
            return WatchdogDecision(
                action="continue",
                sleep_s=0.0,
                reason="reinit succeeded",
                consecutive_errors=0,
                reinit_attempt=0,
                stalled_for_s=0.0,
            )
        self._reinit_attempt += 1
        return self._backoff_decision(now=now, reason="reinit failed", force_sleep=True)

    @property
    def config(self) -> WatchdogConfig:
        return self._config

    def poll(self, *, now: float) -> WatchdogDecision:
        stalled_for = now - self._last_good_frame_at
        if stalled_for >= self._config.stall_timeout_s:
            return self._request_reinit(
                now=now, reason=f"no frame for {stalled_for:.1f}s", stalled_for_s=stalled_for
            )
        return WatchdogDecision(
            action="continue",
            sleep_s=0.0,
            reason="healthy",
            consecutive_errors=self._consecutive_errors,
            reinit_attempt=self._reinit_attempt,
            stalled_for_s=stalled_for,
        )

    @property
    def reinit_count(self) -> int:
        return self._reinit_attempt

    # -- internals ------------------------------------------------------

    def _backoff_decision(
        self, *, now: float, reason: str, force_sleep: bool = False
    ) -> WatchdogDecision:
        sleep_s = self._current_backoff_s
        self._current_backoff_s = min(
            self._config.backoff_max_s, self._current_backoff_s * self._config.backoff_factor
        )
        return WatchdogDecision(
            action="sleep",
            sleep_s=sleep_s,
            reason=reason,
            consecutive_errors=self._consecutive_errors,
            reinit_attempt=self._reinit_attempt,
            stalled_for_s=now - self._last_good_frame_at,
        )

    def _request_reinit(
        self, *, now: float, reason: str, stalled_for_s: float = 0.0
    ) -> WatchdogDecision:
        return WatchdogDecision(
            action="reinit",
            sleep_s=0.0,
            reason=reason,
            consecutive_errors=self._consecutive_errors,
            reinit_attempt=self._reinit_attempt,
            stalled_for_s=stalled_for_s,
        )
