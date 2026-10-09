"""pi_trigger/pi_trigger_server.py against a fake sysfs PWM folder and a pseudo-terminal.

The fake PWM channel enforces the kernel rule that matters (duty_cycle may never
exceed period), so a wrong write order fails here as it would on the Pi.
"""
import json
import os
import sys
import tempfile
import threading
import tty
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pi_trigger"))

import pi_trigger_server as pts  # noqa: E402


class StrictChannel:
    """Watches a fake pwm0 folder and enforces duty <= period on every write."""

    def __init__(self, chip: Path) -> None:
        self.dir = chip / "pwm0"
        self.dir.mkdir()
        for name, value in (("period", 0), ("duty_cycle", 0), ("enable", 0)):
            (self.dir / name).write_text(f"{value}\n")


def make_chip(tmp: str) -> Path:
    chip = Path(tmp) / "pwmchip0"
    chip.mkdir()
    (chip / "export").write_text("")
    return chip


class StrictPwm(pts.Pwm):
    def _write(self, name, value):
        period = self._read("period")
        duty = self._read("duty_cycle")
        if name == "duty_cycle" and value > period:
            raise OSError(22, "Invalid argument (duty > period)")
        if name == "period" and value < duty:
            raise OSError(22, "Invalid argument (period < duty)")
        super()._write(name, value)


class TriggerTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.chip = make_chip(self.tmp.name)
        StrictChannel(self.chip)
        self.pwm = StrictPwm(self.chip)
        self.trigger = pts.Trigger(self.pwm)

    def test_start_status_stop(self):
        reply = self.trigger.handle("start 60")
        self.assertTrue(reply["ok"], reply)
        self.assertEqual((reply["period_ns"], reply["duty_ns"], reply["enabled"]), (16666667, 1000000, True))
        self.assertAlmostEqual(reply["actual_hz"], 60.0, places=4)
        status = self.trigger.handle("status")
        self.assertEqual(status["state"], "running")
        self.assertGreaterEqual(status["pulses_so_far"], 1)
        stopped = self.trigger.handle("stop")
        self.assertEqual(stopped["state"], "stopped")
        self.assertFalse(self.pwm.readback()["enabled"])

    def test_rate_changes_in_both_directions_respect_duty_le_period(self):
        for cmd in ("start 30 5", "start 200 2", "start 1 400", "start 60"):
            with self.subTest(cmd=cmd):
                self.assertTrue(self.trigger.handle(cmd)["ok"])

    def test_bad_commands(self):
        for cmd, err in (("start", "usage"), ("start abc", "start:"), ("start 1000", "rate"),
                         ("start 60 10", "half the period"), ("dance", "unknown"), ("", "empty")):
            with self.subTest(cmd=cmd):
                reply = self.trigger.handle(cmd)
                self.assertFalse(reply["ok"])
                self.assertIn(err, reply["error"])

    def test_service_start_stops_old_pulses_but_a_one_off_command_does_not(self):
        self.trigger.handle("start 60")
        pts.Trigger(self.pwm, reset=False)
        self.assertTrue(self.pwm.readback()["enabled"])
        pts.Trigger(self.pwm)
        self.assertFalse(self.pwm.readback()["enabled"])

    def test_export_creates_the_channel(self):
        with tempfile.TemporaryDirectory() as tmp:
            chip = make_chip(tmp)
            with self.assertRaises(RuntimeError):  # nothing appears: clear error, no hang
                pts.Pwm(chip, wait_s=0.1)
            StrictChannel(chip)
            self.assertTrue(pts.Pwm(chip).readback() == {"period_ns": 0, "duty_ns": 0, "enabled": False})


@unittest.skipUnless(hasattr(os, "openpty"), "needs a pseudo-terminal")
class SerialProtocolTest(unittest.TestCase):
    def test_line_protocol_over_a_tty(self):
        with tempfile.TemporaryDirectory() as tmp:
            chip = make_chip(tmp)
            StrictChannel(chip)
            trigger = pts.Trigger(StrictPwm(chip))
            master, slave = os.openpty()
            # Raw before anything is sent: a fresh tty echoes input until the server has
            # opened it and switched it to raw (on the Pi the host only talks after that).
            tty.setraw(slave)
            port = os.ttyname(slave)
            log = Path(tmp) / "log" / "t.jsonl"
            server = threading.Thread(target=pts.serve, args=(trigger, port, log, True), daemon=True)
            server.start()
            replies = []
            with os.fdopen(master, "r+b", buffering=0) as m:
                for cmd in (b"ping\r\n", b"start 60\n", b"status\n", b"stop\n"):
                    m.write(cmd)
                    line = b""
                    while not line.endswith(b"\n"):
                        line += m.read(1)
                    replies.append(json.loads(line))
            os.close(slave)
            server.join(timeout=3)
            self.assertEqual(replies[0]["name"], "pi-trigger")
            self.assertEqual(replies[1]["period_ns"], 16666667)
            self.assertEqual(replies[2]["state"], "running")
            self.assertEqual(replies[3]["state"], "stopped")
            events = [json.loads(x)["event"] for x in log.read_text().splitlines()]
            self.assertEqual(events[0], "service_start")
            self.assertEqual(events.count("command"), 4)


if __name__ == "__main__":
    unittest.main()
