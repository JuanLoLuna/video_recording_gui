"""Laptop side of the Raspberry Pi trigger (pi_trigger/pi_trigger_server.py).

The Pi appears as a USB serial gadget with two ports: the first is its login
console, the second is the trigger. find_port() asks every candidate port for a
`ping` and keeps the one that answers as "pi-trigger", so the COM number never
has to be configured. Each command is one line out, one JSON line back.

PySerial is only needed to open a real port (PiTrigger.open / find_port); the
protocol itself works on any stream with write() and readline(), which is what
the tests use.
"""

from __future__ import annotations

import json
import time
from typing import Callable, Iterable

# Linux USB gadget defaults for g_serial (Netchip / "Gadget Serial").
GADGET_VID, GADGET_PID = 0x0525, 0xA4A7
BAUD = 115200   # ignored by a USB CDC port, but pyserial wants one


class PiTriggerError(RuntimeError):
    pass


class PiTrigger:
    def __init__(self, stream, port: str = "") -> None:
        self.stream = stream
        self.port = port

    @classmethod
    def open(cls, port: str, timeout_s: float = 1.0) -> "PiTrigger":
        import serial  # pyserial

        stream = serial.Serial(port, BAUD, timeout=timeout_s, write_timeout=timeout_s)
        trigger = cls(stream, port)
        trigger._resync()
        return trigger

    def _resync(self) -> None:
        """Drop anything half-sent or unread from an earlier session."""
        try:
            self.stream.write(b"\n")
            self.stream.flush()
            time.sleep(0.05)
            reset = getattr(self.stream, "reset_input_buffer", None)
            if reset is not None:
                reset()
        except Exception:
            pass

    def command(self, text: str, attempts: int = 3) -> dict:
        """Send one command and return its JSON reply; raises PiTriggerError on no/invalid reply."""
        self.stream.write((text.strip() + "\n").encode("ascii"))
        self.stream.flush()
        last = b""
        for _ in range(attempts):
            line = self.stream.readline()
            if not line:
                break  # read timeout
            last = line
            try:
                reply = json.loads(line.decode("ascii", "replace"))
            except ValueError:
                continue  # a stray line (e.g. an empty reply to the resync newline)
            if isinstance(reply, dict):
                return reply
        raise PiTriggerError(f"no reply to {text!r} from {self.port or 'the Pi'} (last line: {last[:80]!r})")

    def _ok(self, text: str) -> dict:
        reply = self.command(text)
        if not reply.get("ok"):
            raise PiTriggerError(f"{text!r} refused: {reply.get('error')}")
        return reply

    def ping(self) -> dict:
        reply = self._ok("ping")
        if reply.get("name") != "pi-trigger":
            raise PiTriggerError(f"{self.port} answered but is not the trigger: {reply}")
        return reply

    def status(self) -> dict:
        return self._ok("status")

    def start(self, hz: float, pulse_ms: float = 1.0) -> dict:
        return self._ok(f"start {hz:g} {pulse_ms:g}")

    def stop(self) -> dict:
        return self._ok("stop")

    def close(self) -> None:
        close = getattr(self.stream, "close", None)
        if close is not None:
            close()


def candidate_ports(ports: Iterable) -> list[str]:
    """Gadget-serial ports first (by USB id), then every other port. `ports`: pyserial ListPortInfo-likes."""
    ports = list(ports)
    gadget = [p.device for p in ports if (getattr(p, "vid", None), getattr(p, "pid", None)) == (GADGET_VID, GADGET_PID)]
    others = [p.device for p in ports if p.device not in gadget]
    return gadget + others


def find_port(ports: Iterable | None = None,
              opener: Callable[[str], PiTrigger] | None = None,
              gadget_only: bool = True) -> PiTrigger | None:
    """Open the port that answers `ping` as the Pi trigger (None if none does). The caller closes it."""
    if ports is None:
        from serial.tools import list_ports

        ports = list_ports.comports()
    ports = list(ports)
    names = candidate_ports(ports)
    if gadget_only:
        gadget = {p.device for p in ports
                  if (getattr(p, "vid", None), getattr(p, "pid", None)) == (GADGET_VID, GADGET_PID)}
        names = [n for n in names if n in gadget]
    opener = opener or (lambda port: PiTrigger.open(port, timeout_s=0.5))
    for name in names:
        try:
            trigger = opener(name)
        except Exception:
            continue
        try:
            trigger.ping()
            return trigger
        except Exception:
            trigger.close()  # the login console, or something else entirely
    return None
