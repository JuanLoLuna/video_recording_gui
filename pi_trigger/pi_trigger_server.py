#!/usr/bin/env python3
"""Camera trigger pulse generator on a Raspberry Pi 4 (plan Part A, option H2).

The pulses come from the Pi's HARDWARE PWM block on BCM GPIO18 = PHYSICAL PIN 12
(PWM0_0 via ALT5, enabled by `dtoverlay=pwm,pin=18,func=2`; PWM0 could also use
GPIO12 = physical pin 32 via ALT0): once started, no Linux scheduling is involved,
so the period is exact to the PWM clock and edge jitter is ~ns. This program only
starts, stops and reports it.

Control is a line protocol over a USB serial gadget port (default /dev/ttyGS1;
ttyGS0 keeps the login console). One command per line, one JSON line back:

    ping                     -> {"ok": true, "name": "pi-trigger", ...}
    status                   -> state, rate, read-back period/duty, running time, pulses so far
    start <hz> [pulse_ms]    -> start the pulse train (default pulse 1 ms)
    stop                     -> stop it; the pin idles LOW

Every command and reply is appended to a JSONL log with the Pi's clocks.
Standard library only (nothing to pip install on the Pi). Runs as root (sysfs
PWM is root-only) under systemd: see pi-trigger.service and install.sh.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import termios
import time
import tty
from pathlib import Path

VERSION = 1
MIN_HZ, MAX_HZ = 0.5, 500.0
DEFAULT_PULSE_MS = 1.0


class Pwm:
    """One sysfs PWM channel. `root` is /sys/class/pwm/pwmchipN (a plain folder in tests)."""

    def __init__(self, root: Path, channel: int = 0, wait_s: float = 2.0) -> None:
        self.root = Path(root)
        self.channel = channel
        self.dir = self.root / f"pwm{channel}"
        if not self.dir.exists():
            (self.root / "export").write_text(f"{channel}\n")
            # udev may need a moment to create the channel files
            deadline = time.monotonic() + wait_s
            while not (self.dir / "enable").exists():
                if time.monotonic() > deadline:
                    raise RuntimeError(f"{self.dir} did not appear after export")
                time.sleep(0.02)

    def _write(self, name: str, value: int) -> None:
        (self.dir / name).write_text(f"{value}\n")

    def _read(self, name: str) -> int:
        return int((self.dir / name).read_text().strip())

    def configure(self, period_ns: int, duty_ns: int) -> None:
        # The kernel rejects duty > period at every intermediate step, so: off, duty 0,
        # period, duty, in that order, whatever the previous settings were.
        self._write("enable", 0)
        self._write("duty_cycle", 0)
        self._write("period", period_ns)
        self._write("duty_cycle", duty_ns)

    def enable(self, on: bool) -> None:
        self._write("enable", 1 if on else 0)

    def readback(self) -> dict:
        return {"period_ns": self._read("period"), "duty_ns": self._read("duty_cycle"),
                "enabled": bool(self._read("enable"))}


def find_pwm_chip(base: Path = Path("/sys/class/pwm")) -> Path:
    chips = sorted(base.glob("pwmchip*"))
    if not chips:
        raise RuntimeError("no PWM chip: is dtoverlay=pwm,pin=18,func=2 in /boot/firmware/config.txt?")
    return chips[0]


def clocks() -> dict:
    return {"pi_mono_ns": time.monotonic_ns(), "pi_wall_ns": time.time_ns()}


class Trigger:
    def __init__(self, pwm: Pwm, reset: bool = True) -> None:
        self.pwm = pwm
        self.hz: float | None = None
        self.pulse_ms: float | None = None
        self.started_mono_ns: int | None = None
        if reset:
            self.pwm.enable(False)   # a (re)started service never leaves pulses running unannounced

    def status(self) -> dict:
        rb = self.pwm.readback()
        running = rb["enabled"]
        out = {"ok": True, "state": "running" if running else "stopped", **rb, **clocks()}
        if running and self.started_mono_ns is not None:
            elapsed = (out["pi_mono_ns"] - self.started_mono_ns) / 1e9
            out.update(hz=self.hz, pulse_ms=self.pulse_ms, running_s=round(elapsed, 3),
                       actual_hz=1e9 / rb["period_ns"], pulses_so_far=int(elapsed * 1e9 / rb["period_ns"]) + 1)
        return out

    def start(self, hz: float, pulse_ms: float = DEFAULT_PULSE_MS) -> dict:
        if not MIN_HZ <= hz <= MAX_HZ:
            return {"ok": False, "error": f"rate must be {MIN_HZ}..{MAX_HZ} Hz"}
        period_ns = round(1e9 / hz)
        duty_ns = round(pulse_ms * 1e6)
        if not 0 < duty_ns <= period_ns // 2:
            return {"ok": False, "error": "pulse must be > 0 and at most half the period"}
        self.pwm.configure(period_ns, duty_ns)
        before = clocks()
        self.pwm.enable(True)
        self.hz, self.pulse_ms, self.started_mono_ns = hz, pulse_ms, before["pi_mono_ns"]
        rb = self.pwm.readback()
        return {"ok": True, "state": "running", "hz": hz, "pulse_ms": pulse_ms, **rb,
                "actual_hz": 1e9 / rb["period_ns"], "started": before, **clocks()}

    def stop(self) -> dict:
        report = self.status()
        self.pwm.enable(False)
        self.started_mono_ns = None
        return {"ok": True, "state": "stopped", "pulses_sent": report.get("pulses_so_far", 0), **clocks()}

    def handle(self, line: str) -> dict:
        parts = line.strip().split()
        if not parts:
            return {"ok": False, "error": "empty command"}
        cmd, args = parts[0].lower(), parts[1:]
        try:
            if cmd == "ping":
                return {"ok": True, "name": "pi-trigger", "version": VERSION, **clocks()}
            if cmd == "status":
                return self.status()
            if cmd == "stop":
                return self.stop()
            if cmd == "start":
                if not args:
                    return {"ok": False, "error": "usage: start <hz> [pulse_ms]"}
                return self.start(float(args[0]), float(args[1]) if len(args) > 1 else DEFAULT_PULSE_MS)
        except (ValueError, OSError, RuntimeError) as exc:
            return {"ok": False, "error": f"{cmd}: {exc}"}
        return {"ok": False, "error": f"unknown command {cmd!r} (ping, status, start <hz> [pulse_ms], stop)"}


def open_serial(path: str) -> int:
    fd = os.open(path, os.O_RDWR | os.O_NOCTTY)
    if os.isatty(fd):
        # TCSANOW, not setraw's default TCSAFLUSH: a command that arrived just
        # before the port was opened must not be thrown away.
        tty.setraw(fd, termios.TCSANOW)
        attrs = termios.tcgetattr(fd)
        attrs[3] &= ~termios.ECHO
        termios.tcsetattr(fd, termios.TCSANOW, attrs)
    return fd


def serve(trigger: Trigger, port: str, log_path: Path, once: bool = False) -> None:
    """Read commands from the serial port forever; reopen it when the host goes away."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = open(log_path, "a", buffering=1, encoding="utf-8")
    log.write(json.dumps({"event": "service_start", "port": port, **clocks()}) + "\n")
    while True:
        try:
            fd = open_serial(port)
        except OSError as exc:
            log.write(json.dumps({"event": "open_failed", "error": str(exc), **clocks()}) + "\n")
            if once:
                raise
            time.sleep(1.0)
            continue
        buffer = b""
        try:
            while True:
                chunk = os.read(fd, 256)
                if not chunk:
                    break  # host closed the port
                buffer += chunk
                while b"\n" in buffer or b"\r" in buffer:
                    cut = min(i for i in (buffer.find(b"\n"), buffer.find(b"\r")) if i >= 0)
                    line, buffer = buffer[:cut].decode("ascii", "replace"), buffer[cut + 1:]
                    if not line.strip():
                        continue
                    reply = trigger.handle(line)
                    os.write(fd, (json.dumps(reply) + "\n").encode())
                    log.write(json.dumps({"event": "command", "command": line.strip(), "reply": reply}) + "\n")
        except OSError as exc:
            log.write(json.dumps({"event": "port_error", "error": str(exc), **clocks()}) + "\n")
        finally:
            os.close(fd)
        if once:
            return
        time.sleep(0.2)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--port", default="/dev/ttyGS1", help="serial port to listen on (default /dev/ttyGS1)")
    ap.add_argument("--pwm-chip", default=None, help="PWM chip folder (default: first /sys/class/pwm/pwmchip*)")
    ap.add_argument("--channel", type=int, default=0, help="PWM channel (0 = GPIO18 with the pwm overlay)")
    ap.add_argument("--log", default="/var/log/pi-trigger/pi-trigger.jsonl")
    ap.add_argument("--command", help="run ONE command locally and print the reply (testing on the Pi)")
    args = ap.parse_args(argv)
    chip = Path(args.pwm_chip) if args.pwm_chip else find_pwm_chip()
    # A one-off --command must not stop pulses the service started.
    trigger = Trigger(Pwm(chip, args.channel), reset=not args.command)
    if args.command:
        print(json.dumps(trigger.handle(args.command), indent=2))
        if args.command.split()[0].lower() == "start":
            print("(pulses keep running: run --command stop to end them)", file=sys.stderr)
        return 0
    serve(trigger, args.port, Path(args.log))
    return 0


if __name__ == "__main__":
    sys.exit(main())
