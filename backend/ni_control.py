# backend/ni_control.py
"""
Cross-platform safe NI-DAQ digital output controller.

- On Windows with NI-DAQmx installed → uses nidaqmx
- On macOS/Linux → provides a no-op stub
"""

import sys
import threading
from dataclasses import dataclass

@dataclass
class DOLine:
    line: str = "Dev1/port0/line0"
    idle_low: bool = True


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
            with self._lock:
                try:
                    self.set_low() if self.cfg.idle_low else self.set_high()
                except Exception:
                    pass
                try:
                    self._task.close()
                except Exception:
                    pass
            self._task = None
            self._started = False

        def set_high(self):
            with self._lock:
                self._task.write(True, auto_start=True)

        def set_low(self):
            with self._lock:
                self._task.write(False, auto_start=True)

else:
    # --- Stub version for macOS/Linux ---
    class NIDaqDO:
        """Dummy NI-DAQ controller for non-Windows platforms."""
        def __init__(self, cfg: DOLine | None = None):
            self.cfg = cfg or DOLine()
            self._started = False

        def start(self):
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
