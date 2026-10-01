"""Fan-out over the controllers of a multi-camera session.

Duck-typed on purpose: this module never imports camera_control (and so never
PySpin), which is what lets its ordering and failure handling be tested with
hand-written fake controllers. A controller is expected to provide:

    start() -> (ok, message)
    stop() -> (ok, message)                  ok=False means teardown deferred
    start_recording(session_paths, fps=) -> (ok, message)
    stop_recording() -> None
    optional two-phase start, used when EVERY controller has all three:
      prepare_recording(session_paths, fps=) -> (ok, message)
      begin_recording() -> (ok, message)
      abort_prepared() -> None
    notify_sync_pulse_window(width_s=, label=)
    notify_label_event(label, adl_id, adl_label)
    .acquiring, .recording_active            plain booleans

Ordering rules this class owns:
  - Stopping: stop_recording() on EVERY camera first, so all of them close
    their final segments in parallel; only then stop() each one. Otherwise
    CameraController.stop()'s wait for a recording to finish (up to 90 s) would
    run once per camera, back to back.
  - Starting is all-or-nothing by default: if camera N fails, the cameras that
    this call already started are stopped again, so a refused start leaves
    nothing half-running. best_effort=True (the unattended power-resume path)
    instead keeps whichever cameras did start. A camera that was ALREADY
    running when the call began is left alone and never rolled back.
  - Recording is TWO-PHASE when the controllers support it: every camera first
    prepares (opens its sidecars), and only if all of them succeeded does any
    camera begin. A refusal anywhere aborts the prepared cameras and nothing
    was recorded -- a true undo. stop_recording() cannot do that on the real
    controller (it only sets a flag; the acquisition thread still opens and
    closes segment 0), which is why the single-phase fallback, used only for
    controllers without prepare/begin/abort, starts the PRIMARY camera LAST: a
    refusal elsewhere then happens before the camera whose untagged names are
    visible to the downstream pipeline has been touched.
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
    # Serials whose rollback stop() was deferred (teardown could not complete):
    # the caller must NOT release the shared Spinnaker System for these.
    deferred: tuple[str, ...] = ()


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
        # Two slots sharing a tag (including two untagged ones) would build the
        # same SessionPaths and silently write into each other's files.
        tags = [slot.tag for slot in slots]
        if len(set(tags)) != len(tags):
            raise ValueError(
                f"camera tags must be unique (at most one untagged camera): {tags}"
            )
        self._slots = slots

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------
    @property
    def slots(self) -> list[CameraSlot]:
        return list(self._slots)

    @property
    def primary(self) -> CameraSlot | None:
        """The slot flagged primary, or None.

        None is legitimate: when a configured primary camera is missing, every
        present camera is tagged and the session has no untagged video. The
        group deliberately does not promote a slot, because the primary decides
        file names and that decision belongs to backend/camera_registry.
        """
        return next((slot for slot in self._slots if slot.is_primary), None)

    @property
    def default_slot(self) -> CameraSlot:
        """Where the GUI points by default (the tuning panel's camera)."""
        return self.primary or self._slots[0]

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

    def _fail_text(self, slot: CameraSlot, message: str) -> str:
        """A failure message; names the camera only when there are several (one
        camera's messages read exactly as they did before multi-camera support)."""
        return f"{slot.label}: {message}" if len(self._slots) > 1 else str(message)

    def _recording_start_order(self) -> list[CameraSlot]:
        """Everyone else first, the primary last (see the module docstring)."""
        return [s for s in self._slots if not s.is_primary] + [
            s for s in self._slots if s.is_primary
        ]

    # ------------------------------------------------------------------
    # Preview lifecycle
    # ------------------------------------------------------------------
    def start_all(self, *, best_effort: bool = False) -> GroupResult:
        started: list[CameraSlot] = []
        outcomes: list[SlotOutcome] = []
        failures: list[str] = []
        for slot in self._slots:
            if slot.controller.acquiring:
                # Already running (e.g. a second Start Preview): not ours to
                # start, and never ours to stop if a later camera fails.
                outcomes.append(SlotOutcome(slot.serial, True, "Preview already running."))
                continue
            try:
                ok, message = slot.controller.start()
            except Exception as exc:
                ok, message = False, f"{exc.__class__.__name__}: {exc}"
            outcomes.append(SlotOutcome(slot.serial, bool(ok), str(message)))
            if ok:
                started.append(slot)
                continue
            failures.append(self._fail_text(slot, message))
            if best_effort:
                continue
            deferred = self._roll_back_started(started)
            return GroupResult(False, failures[0], tuple(outcomes), deferred)
        if best_effort and not any(o.ok for o in outcomes):
            return GroupResult(False, "; ".join(failures), tuple(outcomes))
        message = "; ".join(failures) if failures else (self._join(outcomes) or "Preview started.")
        return GroupResult(True, message, tuple(outcomes))

    def _roll_back_started(self, started: Sequence[CameraSlot]) -> tuple[str, ...]:
        deferred: list[str] = []
        for done in reversed(started):
            try:
                ok, _message = done.controller.stop()
            except Exception:
                ok = False
            if not ok:
                deferred.append(done.serial)
        return tuple(deferred)

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
        return GroupResult(
            not failed, message, tuple(outcomes), tuple(o.serial for o in failed)
        )

    # ------------------------------------------------------------------
    # Recording lifecycle
    # ------------------------------------------------------------------
    def start_recording_all(
        self,
        paths_for: Callable[[CameraSlot], Any],
        fps_of: Callable[[CameraSlot], float],
        *,
        best_effort: bool = False,
    ) -> GroupResult:
        """Ask every camera to start recording.

        All-or-nothing by default. best_effort=True keeps whichever cameras
        accepted, for an unattended resume that should record with what is
        present; the result is ok if at least one camera is recording, and its
        message names the ones that were not.
        """
        order = self._recording_start_order()
        two_phase = all(
            hasattr(slot.controller, name)
            for slot in order
            for name in ("prepare_recording", "begin_recording", "abort_prepared")
        )
        if two_phase:
            return self._start_recording_two_phase(order, paths_for, fps_of, best_effort)
        return self._start_recording_single_phase(order, paths_for, fps_of, best_effort)

    def _start_recording_two_phase(self, order, paths_for, fps_of, best_effort) -> GroupResult:
        prepared: list[CameraSlot] = []
        outcomes: list[SlotOutcome] = []
        failures: list[str] = []
        for slot in order:
            try:
                ok, message = slot.controller.prepare_recording(
                    paths_for(slot), fps=fps_of(slot)
                )
            except Exception as exc:
                ok, message = False, f"{exc.__class__.__name__}: {exc}"
            outcomes.append(SlotOutcome(slot.serial, bool(ok), str(message)))
            if ok:
                prepared.append(slot)
                continue
            failures.append(self._fail_text(slot, message))
            if not best_effort:
                self._abort_all(prepared)
                return GroupResult(False, failures[0], tuple(outcomes))
        if not prepared:
            return GroupResult(False, "; ".join(failures), tuple(outcomes))

        begun: list[CameraSlot] = []
        begin_messages: dict[str, str] = {}
        for slot in prepared:
            try:
                ok, message = slot.controller.begin_recording()
            except Exception as exc:
                ok, message = False, f"{exc.__class__.__name__}: {exc}"
            if ok:
                begun.append(slot)
                begin_messages[slot.serial] = str(message)
                continue
            failures.append(self._fail_text(slot, message))
            outcomes.append(SlotOutcome(slot.serial, False, f"begin failed: {message}"))
            if not best_effort:
                # Extremely unlikely (begin only raises a flag), but never
                # leave half a session running.
                for done in begun:
                    try:
                        done.controller.stop_recording()
                    except Exception:
                        pass
                self._abort_all([s for s in prepared if s not in begun])
                return GroupResult(False, failures[-1], tuple(outcomes))
        if not begun:
            return GroupResult(False, "; ".join(failures), tuple(outcomes))
        # What the user sees is what each camera said when it began
        # ("Recording requested: <stem>"), as the single-phase path always did.
        outcomes = [
            SlotOutcome(o.serial, o.ok, begin_messages.get(o.serial, o.message)) for o in outcomes
        ]
        message = "; ".join(failures) if failures else self._join(outcomes)
        return GroupResult(True, message, tuple(outcomes))

    @staticmethod
    def _abort_all(slots: Sequence[CameraSlot]) -> None:
        for slot in reversed(list(slots)):
            try:
                slot.controller.abort_prepared()
            except Exception:
                pass

    def _start_recording_single_phase(self, order, paths_for, fps_of, best_effort) -> GroupResult:
        started: list[CameraSlot] = []
        outcomes: list[SlotOutcome] = []
        failures: list[str] = []
        for slot in order:
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
            failures.append(self._fail_text(slot, message))
            if best_effort:
                continue
            for done in started:
                try:
                    done.controller.stop_recording()
                except Exception:
                    pass
            return GroupResult(False, failures[0], tuple(outcomes))
        if best_effort and not started:
            return GroupResult(False, "; ".join(failures), tuple(outcomes))
        message = "; ".join(failures) if failures else self._join(outcomes)
        return GroupResult(True, message, tuple(outcomes))

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
        # Identical messages (every camera says "Preview started.") collapse to one.
        return "; ".join(dict.fromkeys(o.message for o in outcomes if o.message))
