"""backend/pi_trigger_client.py talking to the REAL server command handler (fake PWM folder)."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pi_trigger"))

import pi_trigger_server as pts  # noqa: E402
from backend import pi_trigger_client as ptc  # noqa: E402
from test_pi_trigger_server import StrictChannel, StrictPwm, make_chip  # noqa: E402


class ServerStream:
    """A serial port wired straight to Trigger.handle(): what the client sees over USB."""

    def __init__(self, trigger):
        self.trigger = trigger
        self.out: list[bytes] = []
        self.closed = False

    def write(self, data: bytes):
        for line in data.decode().splitlines():
            if line.strip():
                self.out.append((json.dumps(self.trigger.handle(line)) + "\n").encode())
        return len(data)

    def flush(self):
        pass

    def readline(self):
        return self.out.pop(0) if self.out else b""  # b"" = read timeout

    def close(self):
        self.closed = True


class LoginConsole:
    """The Pi's FIRST gadget port: a getty that answers anything with a login prompt."""

    def __init__(self):
        self.pending = []
        self.closed = False

    def write(self, data):
        self.pending.append(b"juan-rpi-01 login: \n")
        return len(data)

    def flush(self):
        pass

    def readline(self):
        return self.pending.pop(0) if self.pending else b""

    def close(self):
        self.closed = True


class ClientTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        chip = make_chip(self.tmp.name)
        StrictChannel(chip)
        self.pwm = StrictPwm(chip)
        self.pi = ptc.PiTrigger(ServerStream(pts.Trigger(self.pwm)), "COM7")

    def test_commands(self):
        self.assertEqual(self.pi.ping()["name"], "pi-trigger")
        started = self.pi.start(60)
        self.assertEqual(started["period_ns"], 16666667)
        self.assertEqual(self.pi.status()["state"], "running")
        self.assertEqual(self.pi.stop()["state"], "stopped")
        self.assertFalse(self.pwm.readback()["enabled"])

    def test_refused_and_silent(self):
        with self.assertRaises(ptc.PiTriggerError):
            self.pi.start(5000)
        silent = ptc.PiTrigger(SimpleNamespace(write=lambda d: len(d), flush=lambda: None,
                                               readline=lambda: b""), "COM9")
        with self.assertRaises(ptc.PiTriggerError):
            silent.ping()

    def test_find_port_skips_the_login_console(self):
        ports = [SimpleNamespace(device="COM3", vid=0x8086, pid=0x1234),   # something else
                 SimpleNamespace(device="COM5", vid=ptc.GADGET_VID, pid=ptc.GADGET_PID),  # login
                 SimpleNamespace(device="COM6", vid=ptc.GADGET_VID, pid=ptc.GADGET_PID)]  # trigger
        opened = {}

        def opener(name):
            stream = LoginConsole() if name == "COM5" else ServerStream(pts.Trigger(self.pwm))
            if name == "COM3":
                raise AssertionError("non-gadget ports are not probed by default")
            opened[name] = stream
            return ptc.PiTrigger(stream, name)

        found = ptc.find_port(ports, opener)
        self.assertEqual(found.port, "COM6")
        self.assertTrue(opened["COM5"].closed)
        self.assertIsNone(ptc.find_port(ports[:2], opener))
        self.assertEqual(ptc.candidate_ports(ports), ["COM5", "COM6", "COM3"])


if __name__ == "__main__":
    unittest.main()
