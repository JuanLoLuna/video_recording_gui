"""satellite/stream_check.py statistics core and file parser on synthetic data (no Thalamus needed)."""
import json
import random
import struct
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "satellite"))
import stream_check as sc  # noqa: E402


def _vi(n):
    out = b""
    while True:
        b = n & 0x7F
        n >>= 7
        out += bytes([b | (0x80 if n else 0)])
        if not n:
            return out


def _record(node, t, rt, w=640, h=480, fmt=11):
    img = b"".join([b"\x0a" + _vi(3) + b"abc", b"\x10" + _vi(w), b"\x18" + _vi(h), b"\x20" + _vi(fmt),
                    b"\x40" + _vi(t), b"\x48" + _vi(rt)])
    rec = b"\x32" + _vi(len(img)) + img + b"\x20" + _vi(t) + b"\x2a" + _vi(len(node)) + node.encode()
    return struct.pack(">Q", len(rec)) + rec


class StatsTests(unittest.TestCase):
    def test_perfect_stream(self):
        t = [i / 30 for i in range(301)]
        s = sc.interval_stats(t)
        self.assertAlmostEqual(s["median_interval_ms"], 1000 / 30, places=6)
        self.assertEqual(s["gap_count"], 0)
        self.assertEqual(s["estimated_dropped_frames"], 0)
        self.assertAlmostEqual(s["duration_s"], 10.0)

    def test_gap_and_drops(self):
        t = [i / 30 for i in range(100)]
        t = t[:50] + [x for x in t[54:]]  # frames 50..53 missing: one gap of 5 intervals
        s = sc.interval_stats(t)
        self.assertEqual(s["gap_count"], 1)
        self.assertEqual(s["estimated_dropped_frames"], 4)
        self.assertAlmostEqual(s["first_gaps"][0]["length_ms"], 5000 / 30, places=1)

    def test_gap_list_capped_at_20(self):
        t = []
        x = 0.0
        for i in range(500):
            x += 0.5 if i % 10 == 9 else 0.033
            t.append(x)
        s = sc.interval_stats(t)
        self.assertEqual(s["gap_count"], 50)
        self.assertEqual(len(s["first_gaps"]), 20)

    def test_jitter_percentile(self):
        random.seed(1)
        t, x = [], 0.0
        for _ in range(2000):
            x += 1 / 30 + random.uniform(-0.002, 0.002)
            t.append(x)
        s = sc.interval_stats(t)
        self.assertLess(s["p99_interval_ms"], 33.3 + 2.1)
        self.assertGreater(s["stdev_interval_ms"], 0.5)
        self.assertEqual(s["gap_count"], 0)

    def test_too_few_frames(self):
        self.assertEqual(sc.interval_stats([1.0]), {"count": 1})

    def test_offset_drift_and_jitter(self):
        t = [i / 30 for i in range(601)]
        remote = [x * (1 - 100e-6) + 1234.5 for x in t]  # unrelated start, 100 ppm slower
        o = sc.offset_stats(t, remote)
        self.assertTrue(o["available"])
        self.assertAlmostEqual(o["drift_ppm"], 100.0, delta=1.0)
        self.assertLess(o["jitter_stdev_ms"], 1e-3)
        self.assertIn("unrelated", o["note"])

    def test_offset_unavailable_without_remote_time(self):
        self.assertFalse(sc.offset_stats([0, 1, 2], [0, 0, 0])["available"])

    def test_expected_fps(self):
        t = [i / 30 for i in range(291)]  # 9.67 s at 30 fps -> 291 frames, none missing
        rep = sc.analyze(t, t, expected_fps=30)
        self.assertEqual(rep["expected"]["missing_vs_expected"], 0)


class FileTests(unittest.TestCase):
    def test_end_to_end_writes_reports(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "rec.20260101.1"
            blob = b""
            for i in range(100):
                if i == 60:
                    continue  # drop one frame
                blob += _record("sat_cam1", int(i * 1e9 / 30), int(5e9 + i * 1e9 / 30))
            blob += _record("other", 1, 1)
            p.write_bytes(blob)
            rc = sc.main([str(p), "--expected-fps", "30"])
            self.assertEqual(rc, 0)
            rep = json.loads((Path(d) / "report.json").read_text())
            self.assertEqual(rep["frames"], 99)
            self.assertEqual(rep["format"], {"width": 640, "height": 480, "format": "MJPEG"})
            self.assertEqual(rep["receiver_time"]["estimated_dropped_frames"], 1)
            self.assertTrue((Path(d) / "report.txt").read_text().startswith("Stream check"))

    def test_missing_node_gives_warning_and_nonzero(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "rec.1"
            p.write_bytes(_record("x", 1, 1))
            self.assertEqual(sc.main([str(p), "--out", str(Path(d) / "o")]), 1)
            self.assertIn("no image records", (Path(d) / "o" / "report.txt").read_text())

    def test_truncated_file_flagged(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "rec.1"
            p.write_bytes(_record("sat_cam1", 1, 1) + _record("sat_cam1", 2, 2)[:-5])
            sc.main([str(p)])
            self.assertTrue(json.loads((Path(d) / "report.json").read_text())["file_stats"]["truncated"])


if __name__ == "__main__":
    unittest.main()
