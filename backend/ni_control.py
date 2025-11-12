# backend/ni_control.py
from __future__ import annotations
import sys, threading
from dataclasses import dataclass

def _ensure_windows():
    if not sys.platform.startswith("win"):
        raise RuntimeError("NI-DAQmx digital I/O is supported on Windows only.")

@dataclass
class DOLine:
    line: str = "Dev1/port0/line0"  # e.g., "Dev1/port0/line0"
    idle_low: bool = True           # line state when idle/stop

class NIDaqDO:
    """Keeps a DO task open; set_high()/set_low() are instant and thread-safe."""
    def __init__(self, cfg: DOLine | None = None):
        _ensure_windows()
        self.cfg = cfg or DOLine()
        self._task = None
        self._lock = threading.Lock()
        self._started = False

    def start(self):
        if self._started:
            return
        import nidaqmx
        from nidaqmx.constants import LineGrouping
        self._task = nidaqmx.Task()
        self._task.do_channels.add_do_chan(self.cfg.line, line_grouping=LineGrouping.CHAN_PER_LINE)
        # force known idle state
        self.set_low() if self.cfg.idle_low else self.set_high()
        self._started = True

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
        if not self._started:
            raise RuntimeError("NIDaqDO not started. Call .start() first.")
        with self._lock:
            self._task.write(True, auto_start=True)

    def set_low(self):
        if not self._started:
            raise RuntimeError("NIDaqDO not started. Call .start() first.")
        with self._lock:
            self._task.write(False, auto_start=True)
