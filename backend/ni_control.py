# backend/ni_control.py
"""
Cross-platform safe NI-DAQ digital output controller.

- On Windows with NI-DAQmx installed → uses nidaqmx
- On macOS/Linux → provides a no-op stub
"""

import os
import sys
import threading
from dataclasses import dataclass

# --- NI-DAQ ownership ------------------------------------------------------
# Exactly ONE process may own the NI-DAQ: DAQmx device reservation on USB
# devices is exclusive. The SmartSleeve SyncService heartbeat script is that
# owner during recording sessions, so this GUI releases the device by default.
#
# To hand the DAQ back to this GUI (running without SyncService, or falling
# back to the previous setup), either:
#   * flip NI_OUTPUT_ENABLED to True below, or
#   * set the environment variable SLEEVE_VIDEO_GUI_NI=1
#
# Nothing is deleted. Every pulse call site in gui/main.py is already gated on
# the PulseManager existing, and the PulseManager is only built from a line
# returned by list_do_lines(), so this switch is the entire mechanism.
#
# See Sleeve/docs/sync-service-plan.md - "Delivery phasing", phase 0.
NI_OUTPUT_ENABLED = False


def ni_output_enabled() -> bool:
    """True if this GUI is allowed to claim the NI-DAQ."""
    override = os.environ.get("SLEEVE_VIDEO_GUI_NI")
    if override is not None:
        return override.strip().lower() in ("1", "true", "yes", "on")
    return NI_OUTPUT_ENABLED


NI_DISABLED_MESSAGE = (
    "NI output disabled - the SyncService heartbeat owns the DAQ. "
    "Set SLEEVE_VIDEO_GUI_NI=1 to re-enable."
)

@dataclass
class DOLine:
    line: str = "Dev1/port0/line0"
    idle_low: bool = True


def list_do_lines() -> list[str]:
    """Return all digital-output line names across every connected NI device.

    On non-Windows platforms (no nidaqmx driver) this returns an empty list.
    Example return value: ["Dev1/port0/line0", "Dev1/port0/line1", "Dev2/port0/line0"]
    """
    if not ni_output_enabled():
        return []
    if not sys.platform.startswith("win"):
        return []
    try:
        import nidaqmx.system
        system = nidaqmx.system.System.local()
        lines: list[str] = []
        for device in system.devices:
            for line in device.do_lines:
                lines.append(line.name)
        return lines
    except Exception:
        return []


def list_devices() -> list[str]:
    """Return the names of all connected NI devices (e.g. ['Dev1', 'Dev2']).

    On non-Windows platforms this returns an empty list.
    """
    if not ni_output_enabled():
        return []
    if not sys.platform.startswith("win"):
        return []
    try:
        import nidaqmx.system
        system = nidaqmx.system.System.local()
        return [d.name for d in system.devices]
    except Exception:
        return []


if sys.platform.startswith("win"):
    import nidaqmx
    from nidaqmx.constants import LineGrouping

    class NIDaqDO:
        def __init__(self, cfg: DOLine | None = None):
            self.cfg = cfg or DOLine()
            self._task = None
            self._lock = threading.Lock()
            self._started = False

        def start(self):
            if self._started:
                return
            # Defence in depth: refuse to claim the device even if some other
            # path constructs NIDaqDO directly. on_connect_daq_clicked catches
            # this and surfaces the message in the UI.
            if not ni_output_enabled():
                raise RuntimeError(NI_DISABLED_MESSAGE)
            self._task = nidaqmx.Task()
            self._task.do_channels.add_do_chan(
                self.cfg.line,
                line_grouping=LineGrouping.CHAN_PER_LINE,
            )
            self._started = True
            # Set the port to a known state
            self.set_low() if self.cfg.idle_low else self.set_high()

        def stop(self):
            if not self._started:
                return

            # snapshot & mark stopped early (prevents set_* usage elsewhere)
            t = self._task
            self._started = False
            self._task = None

            if t is None:
                return

            try:
                with self._lock:
                    # set known idle state without calling set_high/low (avoids re-entrancy)
                    idle_val = False if self.cfg.idle_low else True
                    try:
                        t.write(idle_val, auto_start=True)
                    except Exception:
                        pass
                    try:
                        t.close()
                    except Exception:
                        pass
            except Exception:
                # swallow everything on shutdown
                pass

        def set_high(self):
            if not self._started or self._task is None:
                return
            with self._lock:
                t = self._task
                if t is None:
                    return
                t.write(True, auto_start=True)

        def set_low(self):
            if not self._started or self._task is None:
                return
            with self._lock:
                t = self._task
                if t is None:
                    return
                t.write(False, auto_start=True)

else:
    # --- Stub version for macOS/Linux ---
    class NIDaqDO:
        """Dummy NI-DAQ controller for non-Windows platforms."""
        def __init__(self, cfg: DOLine | None = None):
            self.cfg = cfg or DOLine()
            self._started = False

        def start(self):
            if not ni_output_enabled():
                raise RuntimeError(NI_DISABLED_MESSAGE)
            self._started = True
            print("[NIDaqDO] (stub) start called — no hardware available on this OS")

        def stop(self):
            self._started = False
            print("[NIDaqDO] (stub) stop called — no hardware available on this OS")

        def set_high(self):
            if not self._started:
                return
            # no actual hardware, just print for debug
            print("[NIDaqDO] (stub) set HIGH")

        def set_low(self):
            if not self._started:
                return
            print("[NIDaqDO] (stub) set LOW")
