# backend/pulse_manager.py

from __future__ import annotations
import threading
import time
import queue
from dataclasses import dataclass
from typing import Optional

from backend.ni_control import NIDaqDO, DOLine


@dataclass
class PulseRequest:
    width_s: float
    label: Optional[str] = None  # optional tag, e.g. "record_start" / "manual_sync"


class PulseManager:
    """
    Dedicated thread that controls digital pulses on a NI-DAQ DO line.

    - Non-blocking API:
        request_pulse(width_s=..., label="...")

    - Uses NIDaqDO for the actual hardware.
    - Keeps track of whether a pulse is currently active (is_pulse_active).
    """

    def __init__(self, daq: Optional[NIDaqDO] = None, default_width_s: float = 0.005):
        """
        Args:
            daq: An initialized NIDaqDO instance. If None, a default one is created.
            default_width_s: default pulse width for request_pulse() calls.
        """
        self.default_width_s = float(default_width_s)
        if self.default_width_s <= 0:
            raise ValueError("default_width_s must be > 0.")

        self.daq = daq or NIDaqDO(DOLine(line="Dev1/port0/line0", idle_low=True))

        self._queue: "queue.Queue[PulseRequest]" = queue.Queue()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self._pulse_active = False
        self._state_lock = threading.Lock()
        self._started = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def start(self):
        """
        Start the PulseManager thread and ensure DAQ is started.
        Safe to call multiple times; only starts once.
        """
        if self._started:
            return

        # Start the DAQ hardware (no-op on stub/mac)
        self.daq.start()

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="PulseManagerThread",
            daemon=True,
        )
        self._thread.start()
        self._started = True

    def stop(self):
        """
        Stop the worker thread and stop the DAQ.
        Safe to call multiple times.
        """
        if not self._started:
            return

        self._stop_event.set()

        # Wake the queue if it's waiting
        try:
            self._queue.put_nowait(PulseRequest(width_s=0.0, label="__shutdown__"))
        except Exception:
            pass

        if self._thread is not None:
            try:
                self._thread.join(timeout=1.0)
            except Exception:
                pass
            self._thread = None

        # Ensure line goes to idle and hardware is closed
        try:
            self.daq.stop()
        except Exception:
            pass

        with self._state_lock:
            self._pulse_active = False

        self._started = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def request_pulse(self, width_s: float | None = None, label: str | None = None):
        """
        Queue a pulse request. Returns immediately (non-blocking).

        Args:
            width_s: pulse width in seconds. If None, uses default_width_s.
            label: optional tag to identify the pulse type (for logging/debug).
        """
        if not self._started:
            raise RuntimeError("PulseManager not started. Call .start() first.")

        w = self.default_width_s if width_s is None else float(width_s)
        if w <= 0:
            return  # ignore zero/negative pulses

        req = PulseRequest(width_s=w, label=label)
        self._queue.put(req)

    @property
    def is_pulse_active(self) -> bool:
        """
        True while a pulse is currently being held HIGH.
        (This is mostly for debugging; logging per-frame will
         usually come from camera logic when it requests pulses.)
        """
        with self._state_lock:
            return self._pulse_active

    # ------------------------------------------------------------------
    # Worker loop
    # ------------------------------------------------------------------
    def _run(self):
        idle_low = getattr(self.daq.cfg, "idle_low", True)

        while not self._stop_event.is_set():
            try:
                req = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue

            # Shutdown sentinel
            if self._stop_event.is_set() or req.label == "__shutdown__":
                break

            # Apply the pulse
            try:
                with self._state_lock:
                    self._pulse_active = True

                # HIGH
                try:
                    if idle_low:
                        self.daq.set_high()
                    else:
                        self.daq.set_low()
                except Exception as e:
                    print(f"[PulseManager] Error setting HIGH: {e}")
                    continue

                time.sleep(req.width_s)

            finally:
                # Back to idle
                try:
                    if idle_low:
                        self.daq.set_low()
                    else:
                        self.daq.set_high()
                except Exception as e:
                    print(f"[PulseManager] Error setting LOW: {e}")

                with self._state_lock:
                    self._pulse_active = False
