"""Pure analysis in scripts/sync_probe.py against synthetic two-camera clocks.

The probe runs on the rig; what can be checked here is that the numbers it
prints are right: tick unit, clock drift, the gap between the cameras and how
fast it drifts, and arrival jitter.
"""
import random
import sys
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import sync_probe as sp  # noqa: E402


def camera(fps, drift_ppm, phase_s, seconds, tick0, rng, arrival_ms=(2.0, 6.0)):
    """Frames from a free-running camera whose crystal is off by drift_ppm.

    Exposure k happens at host time phase + k * period * (1 + drift); the
    camera timestamps it with its own clock, which runs (1 + drift) slow/fast
    relative to the host, so camera ticks advance exactly period-in-ns per frame.
    """
    period = 1.0 / fps
    scale = 1.0 + drift_ppm * 1e-6  # host seconds per camera second
    frames = []
    for k in range(int(seconds * fps)):
        ticks = tick0 + round(k * period * 1e9)
        host = phase_s + (ticks - tick0) * 1e-9 * scale
        arrival = host + rng.uniform(*arrival_ms) / 1e3
        frames.append((arrival, 1000 + k, ticks))
    latches = []
    t = phase_s + 0.01
    while t < phase_s + seconds:
        ticks = tick0 + round((t - phase_s) / scale * 1e9)
        before = t - rng.uniform(0.05e-3, 1.5e-3)
        after = t + rng.uniform(0.05e-3, 1.5e-3)
        latches.append((before, after, ticks))
        t += 1.0
    return frames, latches


def make_run(serial, fps, frames, latches):
    run = sp.CamRun(serial=serial, cam=None, applied_fps=fps, model="Fake")
    run.frames, run.latches, run.latch_supported = frames, latches, True
    return run


class FitClockTest(unittest.TestCase):
    def test_recovers_tick_unit_and_drift(self):
        rng = random.Random(1)
        _, latches = camera(30, 25.0, 100.0, 600, 3_367_778_980_316, rng)
        fit = sp.fit_clock(latches)
        self.assertIsNotNone(fit)
        self.assertAlmostEqual(fit.tick_ns, 1.0, places=4)
        self.assertAlmostEqual(fit.drift_ppm(), 25.0, delta=1.0)
        self.assertLess(fit.residual_us, 600)  # midpoints of +-1.5 ms brackets

    def test_too_few_samples(self):
        self.assertIsNone(sp.fit_clock([(0.0, 0.001, 5)]))

    def test_tick_from_frames(self):
        ticks = [round(k * 1e9 / 60) for k in range(100)]
        self.assertAlmostEqual(sp.tick_ns_from_frames(ticks, 60.0), 1.0, places=3)


class PhaseTest(unittest.TestCase):
    def test_constant_gap_is_found_and_wrapped(self):
        a = [k / 30 for k in range(300)]
        b = [t + 0.030 for t in a]  # 30 ms late == 3.33 ms early after wrapping
        offsets = sp.phase_offsets(a, b, 1 / 30)
        self.assertAlmostEqual(offsets[100][1] * 1e3, 30 - 1000 / 30, places=6)
        summary = sp.summarize_phase(offsets, 1 / 30)
        self.assertAlmostEqual(summary["drift_ms_per_min"], 0.0, places=6)
        self.assertIsNone(summary["full_cycle_min"])

    def test_drift_is_unwrapped_through_the_half_period(self):
        # B's crystal is 50 ppm fast: 3 ms/min of drift at any frame rate.
        a = [k / 60 for k in range(60 * 600)]
        b = [0.008 + (k / 60) * (1 - 50e-6) for k in range(60 * 600)]
        summary = sp.summarize_phase(sp.phase_offsets(a, b, 1 / 60), 1 / 60)
        self.assertAlmostEqual(summary["drift_ms_per_min"], -3.0, delta=0.01)
        self.assertAlmostEqual(summary["full_cycle_min"], (1000 / 60) / 3.0, delta=0.05)
        self.assertLessEqual(summary["abs_gap_ms_max"], summary["half_period_ms"] + 1e-9)


class AnalyseTest(unittest.TestCase):
    def test_two_cameras_end_to_end(self):
        rng = random.Random(7)
        fps, seconds = 60.0, 300
        fa, la = camera(fps, 10.0, 50.0, seconds, 1_000_000, rng)
        fb, lb = camera(fps, -20.0, 50.004, seconds, 9_000_000_000, rng)
        fb = fb[:500] + fb[501:]  # one dropped frame on B
        result = sp.analyse([make_run("A", fps, fa, la), make_run("B", fps, fb, lb)], fps)

        a, b = result["cameras"]["A"], result["cameras"]["B"]
        self.assertEqual(a["frame_gaps"], 0)
        self.assertEqual(b["frame_gaps"], 1)
        self.assertAlmostEqual(a["clock_drift_ppm_vs_laptop"], 10.0, delta=1.5)
        self.assertAlmostEqual(b["clock_drift_ppm_vs_laptop"], -20.0, delta=1.5)
        self.assertLess(a["arrival"]["latency_ms_p99"], 4.5)

        pair = result["pair"]
        self.assertAlmostEqual(pair["gap_ms_start"], 4.0, delta=0.3)
        # B runs 30 ppm faster than A on the host timeline: -1.8 ms/min.
        self.assertAlmostEqual(pair["drift_ms_per_min"], -1.8, delta=0.1)
        self.assertEqual(len(result["_offsets"]), len(fb))

    def test_without_latch_falls_back_to_arrival(self):
        rng = random.Random(3)
        fa, _ = camera(30.0, 0.0, 0.0, 20, 0, rng)
        fb, _ = camera(30.0, 0.0, 0.010, 20, 0, rng)
        ra, rb = make_run("A", 30.0, fa, []), make_run("B", 30.0, fb, [])
        result = sp.analyse([ra, rb], 30.0)
        self.assertIn("host arrival only", result["cameras"]["A"]["timeline"])
        self.assertAlmostEqual(result["pair"]["abs_gap_ms_p50"], 10.0, delta=3.0)


if __name__ == "__main__":
    unittest.main()
