"""Fan-out over the controllers of a multi-camera session.

Duck-typed on purpose: this module never imports camera_control (and so never
PySpin), which is what lets its ordering and failure handling be tested with
hand-written fake controllers. A controller is expected to provide:

    start() -> (ok, message)
    stop() -> (ok, message)                  ok=False means teardown deferred
    start_recording(session_paths, fps=) -> (ok, message)
    stop_recording() -> None
    notify_sync_pulse_window(width_s=, label=)
    notify_label_event(label, adl_id, adl_label)
    .acquiring, .recording_active            plain booleans

Ordering rules this class owns:
  - Stopping: stop_recording() on EVERY camera first, so all of them close
    their final segments in parallel; only then stop() each one. Otherwise
    CameraController.stop()'s wait for a recording to finish (up to 90 s) would
    run once per camera, back to back.
  - Starting is all-or-nothing: if camera N fails, cameras 0..N-1 that already
    started are stopped again, so a refused start leaves nothing half-running.
  - A fan-out call (sync pulse, label event, a settings broadcast) never lets
    one camera's exception stop the others from receiving it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence


@dataclass
class CameraSlot:
    controller: Any
    serial: str
    model: str = ""
    tag: str | None = None  # SessionPaths camera tag; None = primary
    is_primary: bool = False

    @property
    def label(self) -> str:
        return f"{self.model} #{self.serial}" if self.model else f"#{self.serial}"


@dataclass(frozen=True)
class SlotOutcome:
    serial: str
    ok: bool
    message: str = ""


@dataclass(frozen=True)
class GroupResult:
    ok: bool
    message: str
    outcomes: tuple[SlotOutcome, ...] = ()


@dataclass(frozen=True)
class BroadcastResult:
    slot: CameraSlot
    ok: bool
    value: Any = None  # the return value, or the exception when ok is False


class CameraGroup:
    def __init__(self, slots: Sequence[CameraSlot]) -> None:
        slots = list(slots)
        if not slots:
            raise ValueError("a CameraGroup needs at least one camera")
        serials = [slot.serial for slot in slots]
        if len(set(serials)) != len(serials):
            raise ValueError(f"duplicate camera serials: {serials}")
        if not any(slot.is_primary for slot in slots):
            slots[0].is_primary = True
        self._slots = slots

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------
    @property
    def slots(self) -> list[CameraSlot]:
        return list(self._slots)

    @property
    def primary(self) -> CameraSlot:
        return next(slot for slot in self._slots if slot.is_primary)

    @property
    def controllers(self) -> list[Any]:
        return [slot.controller for slot in self._slots]

    def slot_for_serial(self, serial: str) -> CameraSlot | None:
        return next((s for s in self._slots if s.serial == serial), None)

    @property
    def all_acquiring(self) -> bool:
        return all(bool(slot.controller.acquiring) for slot in self._slots)

    @property
    def any_acquiring(self) -> bool:
        return any(bool(slot.controller.acquiring) for slot in self._slots)

    @property
    def any_recording(self) -> bool:
        return any(bool(slot.controller.recording_active) for slot in self._slots)

    # ------------------------------------------------------------------
    # Preview lifecycle
    # ------------------------------------------------------------------
    def start_all(self) -> GroupResult:
        started: list[CameraSlot] = []
        outcomes: list[SlotOutcome] = []
        for slot in self._slots:
            try:
                ok, message = slot.controller.start()
            except Exception as exc:
                ok, message = False, f"{exc.__class__.__name__}: {exc}"
            outcomes.append(SlotOutcome(slot.serial, bool(ok), str(message)))
            if ok:
                started.append(slot)
                continue
            for done in reversed(started):
                try:
                    done.controller.stop()
                except Exception:
                    pass
            return GroupResult(
                False,
                f"{slot.label}: {message}",
                tuple(outcomes),
            )
        return GroupResult(True, self._join(outcomes) or "Preview started.", tuple(outcomes))

    def stop_all(self) -> GroupResult:
        """Stop recording everywhere first, then stop each camera.

        ok is False if ANY camera's teardown was deferred -- in which case the
        caller must not release the shared Spinnaker System.
        """
        self.stop_recording_all()
        outcomes: list[SlotOutcome] = []
        for slot in self._slots:
            try:
                ok, message = slot.controller.stop()
            except Exception as exc:
                ok, message = False, f"{exc.__class__.__name__}: {exc}"
            outcomes.append(SlotOutcome(slot.serial, bool(ok), str(message)))
        failed = [o for o in outcomes if not o.ok]
        message = "; ".join(f"#{o.serial}: {o.message}" for o in failed)
        return GroupResult(not failed, message, tuple(outcomes))

    # ------------------------------------------------------------------
    # Recording lifecycle
    # ------------------------------------------------------------------
    def start_recording_all(
        self,
        paths_for: Callable[[CameraSlot], Any],
        fps_of: Callable[[CameraSlot], float],
    ) -> GroupResult:
        """Ask every camera to start recording; roll back if any refuses."""
        started: list[CameraSlot] = []
        outcomes: list[SlotOutcome] = []
        for slot in self._slots:
            try:
                ok, message = slot.controller.start_recording(
                    paths_for(slot), fps=fps_of(slot)
                )
            except Exception as exc:
                ok, message = False, f"{exc.__class__.__name__}: {exc}"
            outcomes.append(SlotOutcome(slot.serial, bool(ok), str(message)))
            if ok:
                started.append(slot)
                continue
            for done in started:
                try:
                    done.controller.stop_recording()
                except Exception:
                    pass
            return GroupResult(False, f"{slot.label}: {message}", tuple(outcomes))
        return GroupResult(True, self._join(outcomes), tuple(outcomes))

    def stop_recording_all(self) -> None:
        for slot in self._slots:
            try:
                slot.controller.stop_recording()
            except Exception as exc:
                print(f"[camera {slot.serial}] stop_recording failed: {exc}")

    # ------------------------------------------------------------------
    # Fan-out
    # ------------------------------------------------------------------
    def broadcast(self, method_name: str, *args: Any, **kwargs: Any) -> list[BroadcastResult]:
        """Call controller.<method_name>(...) on every camera, isolating failures."""
        results: list[BroadcastResult] = []
        for slot in self._slots:
            try:
                value = getattr(slot.controller, method_name)(*args, **kwargs)
                results.append(BroadcastResult(slot, True, value))
            except Exception as exc:
                results.append(BroadcastResult(slot, False, exc))
        return results

    def notify_sync_pulse_window(self, width_s: float, label: str) -> int:
        """Mark the sync window on every camera. Returns how many failed."""
        results = self.broadcast("notify_sync_pulse_window", width_s=width_s, label=label)
        return sum(1 for r in results if not r.ok)

    def notify_label_event(self, label: str, adl_id: Any, adl_label: Any) -> int:
        """Record a label event on every camera. Returns how many failed."""
        results = self.broadcast("notify_label_event", label, adl_id, adl_label)
        return sum(1 for r in results if not r.ok)

    @staticmethod
    def _join(outcomes: Iterable[SlotOutcome]) -> str:
        return "; ".join(o.message for o in outcomes if o.message)
