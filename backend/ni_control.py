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
