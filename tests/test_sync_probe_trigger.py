"""sync_probe.py trigger mode: analysis maths, camera trigger settings, and a full run on fake cameras.

The fake cameras here only produce frames while the fake Pi is "pulsing", like
real cameras in hardware-trigger mode, so the run's stray-trigger check, missed
triggers and the always-restore-free-running cleanup are exercised end to end.
"""
import random
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

from fake_spinnaker import FakeCamera, FakeImage, FakeSystem, install_pyspin_stub

REAL_PYSPIN = install_pyspin_stub()

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import PySpin  # noqa: E402  (the stub, unless real PySpin is installed)
import sync_probe as sp  # noqa: E402


class AnalysisTest(unittest.TestCase):
    def test_interval_stats(self):
        period = 1 / 60
        ticks = [round(k * period * 1e9) for k in range(600)]
        ticks = ticks[:300] + ticks[301:]  # one missed trigger
        stats = sp.interval_stats_us(ticks, 1e-9, period)
        self.assertEqual(stats["intervals_over_1_5_periods"], 1)
        self.assertLess(stats["max_abs_dev_us"], 1.0)
        self.assertEqual(sp.interval_stats_us([5], 1e-9, period), {})

    def test_paired_gaps_constant_offset_plus_noise(self):
        rng = random.Random(1)
        a = [k / 60 for k in range(1000)]
        b = [t + 35e-6 + rng.gauss(0, 2e-6) for t in a]  # 35 us model offset, 2 us noise
        gaps = sp.paired_gaps_us(a, b, 1 / 60)
        self.assertAlmostEqual(gaps["mean_gap_us"], 35.0, delta=0.5)
        self.assertLess(gaps["spread_us_p99"], 8.0)
        self.assertEqual(gaps["pairs"], 1000)


class Node:
    def __init__(self, v):
        self.v = v


@unittest.skipIf(REAL_PYSPIN, "real PySpin present: run the probe on the rig")
class CameraSettingsTest(unittest.TestCase):
    def test_set_and_clear_trigger(self):
        cam = FakeCamera("26134271", "Blackfly S", 64, 48)
        nm = cam.GetNodeMap()
        nm.GetNode("ExposureTime").value = 20000.0
        applied = sp.set_trigger(PySpin, cam, "Line3", 60.0)
        self.assertEqual((applied["TriggerMode"], applied["TriggerSource"], applied["TriggerActivation"]),
                         ("On", "Line3", "RisingEdge"))
        self.assertLessEqual(applied["exposure_us"], 0.5e6 / 60)
        self.assertFalse(nm.GetNode("AcquisitionFrameRateEnable").value)
        sp.clear_trigger(PySpin, cam)
        self.assertEqual(nm.GetNode("TriggerMode").GetCurrentEntry().GetSymbolic(), "Off")
        self.assertTrue(nm.GetNode("AcquisitionFrameRateEnable").value)

    def test_line_defaults_by_model(self):
        self.assertEqual(sp.trigger_line_for("Firefly FFY-U3-04S2M", {}, "1"), "Line2")
        self.assertEqual(sp.trigger_line_for("Blackfly S BFS-U3-13Y3M", {}, "2"), "Line3")
        self.assertEqual(sp.trigger_line_for("Firefly", {"1": "Line3"}, "1"), "Line3")
        with self.assertRaises(RuntimeError):
            sp.trigger_line_for("Chameleon", {}, "3")


class FakePi:
    """The Pi as the probe sees it; `pulsing` is what the fake cameras listen to."""

    def __init__(self):
        self.pulsing = threading.Event()
        self.started_at = None
        self.hz = None
        self.stops = 0

    def start(self, hz, pulse_ms=1.0):
        self.hz, self.started_at = hz, time.monotonic()
        self.pulsing.set()
        return {"ok": True, "actual_hz": hz}

    def stop(self):
        self.stops += 1
        sent = 0
        if self.pulsing.is_set():
            sent = int((time.monotonic() - self.started_at) * self.hz) + 1
        self.pulsing.clear()
        return {"ok": True, "pulses_sent": sent}


class TriggeredCamera(FakeCamera):
    """Frames only on pulses (and none at all if `stray` is 0); real triggered cameras block otherwise."""

    def __init__(self, pi, *args, stray=0, **kw):
        super().__init__(*args, **kw)
        self.pi, self.stray = pi, stray

    def GetNextImage(self, timeout_ms):
        if self.stray > 0 and not self.pi.pulsing.is_set():
            self.stray -= 1
            self._frame_id += 1
            return FakeImage(self._frame_id, self.height, self.width)
        if not self.pi.pulsing.wait(timeout_ms / 1000):
            raise RuntimeError("timeout")  # what Spinnaker does with no trigger
        k = int((time.monotonic() - self.pi.started_at) * self.pi.hz)
        next_at = self.pi.started_at + (k + 1) / self.pi.hz
        time.sleep(max(0.0, next_at - time.monotonic()))
        if not self.pi.pulsing.is_set():
            raise RuntimeError("timeout")
        self._frame_id += 1
        return FakeImage(self._frame_id, self.height, self.width)


@unittest.skipIf(REAL_PYSPIN, "real PySpin present: run the probe on the rig")
class RunTriggerTest(unittest.TestCase):
    def run_probe(self, stray=0):
        pi = FakePi()
        cams = [TriggeredCamera(pi, "23227865", "Firefly FFY-U3-04S2M", 32, 24, stray=stray),
                TriggeredCamera(pi, "26134271", "Blackfly S BFS-U3-13Y3M", 64, 48)]
        with tempfile.TemporaryDirectory() as tmp:
            result = sp.run_trigger(PySpin, FakeSystem(cams), ["23227865", "26134271"], 30.0, 1.5,
                                    Path(tmp), pi, {}, 1.0)
            files = sorted(p.name for p in Path(tmp).iterdir())
        return pi, cams, result, files

    def test_full_run_restores_free_running(self):
        pi, cams, result, files = self.run_probe()
        for cam in cams:
            nm = cam.GetNodeMap()
            self.assertEqual(nm.GetNode("TriggerMode").GetCurrentEntry().GetSymbolic(), "Off")
            self.assertTrue(nm.GetNode("AcquisitionFrameRateEnable").value)
        self.assertGreaterEqual(pi.stops, 2)  # before configuring, after the run (and in cleanup)
        self.assertEqual(result["trigger_settings"]["23227865"]["line"], "Line2")
        self.assertEqual(result["trigger_settings"]["26134271"]["line"], "Line3")
        for serial, cam in result["cameras"].items():
            self.assertEqual(cam["frames_before_start"], 0)
            self.assertLessEqual(abs(cam["missed_triggers"]), 2, (serial, cam["frames"], cam["pulses_sent"]))
        self.assertIn("trigger_30fps_23227865_frames.csv", files)
        self.assertGreater(result["trigger_pair"]["pairs"], 30)

    def test_stray_triggers_are_reported(self):
        _, _, result, _ = self.run_probe(stray=3)
        self.assertEqual(result["cameras"]["23227865"]["frames_before_start"], 3)
        self.assertEqual(result["cameras"]["26134271"]["frames_before_start"], 0)


if __name__ == "__main__":
    unittest.main()
