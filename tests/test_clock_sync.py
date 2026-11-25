"""satellite/clock_logger.py math and clock_server.py trailer on synthetic data."""
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "satellite"))
import clock_logger as cl  # noqa: E402
import clock_server as cs  # noqa: E402

try:
    import grpc  # noqa: F401
    from thalamus import thalamus_pb2, thalamus_pb2_grpc  # noqa: F401
    HAVE_GRPC = True
except Exception:  # noqa: BLE001
    HAVE_GRPC = False


class FormulaTests(unittest.TestCase):
    def test_offset_rtt_symmetric_delay(self):
        # true offset 1000, one-way delay 50 each way, server holds 10
        t1 = 10_000
        t2 = t1 + 50 + 1000
        t3 = t2 + 10
        t4 = t3 - 1000 + 50
        off, rtt = cl.offset_rtt(t1, t2, t3, t4)
        self.assertEqual(off, 1000)
        self.assertEqual(rtt, 100)

    def test_offset_asymmetric_delay_error_is_half_difference(self):
        t1 = 0
        t2 = 80 + 500     # outbound 80
        t3 = t2
        t4 = t3 - 500 + 20  # return 20
        off, rtt = cl.offset_rtt(t1, t2, t3, t4)
        self.assertEqual(rtt, 100)
        self.assertEqual(off, 500 + (80 - 20) / 2)

    def test_trailer_roundtrip(self):
        blob = b"CLKP\x01\x00\x00\x00" + cs.build_trailer(0xDEADBEEF12345678, 11, 22, 33)
        self.assertEqual(cl.parse_trailer(blob), (0xDEADBEEF12345678, 11, 22, 33))
        self.assertIsNone(cl.parse_trailer(b"short"))
        self.assertIsNone(cl.parse_trailer(b"x" * 36))
        self.assertEqual(cs.TRAILER.size, 36)


class SelectionAndFitTests(unittest.TestCase):
    def test_min_rtt_per_window(self):
        t1 = np.arange(0, 30e9, 0.2e9)  # 5/s for 30 s
        rtt = np.full(t1.size, 5e6)
        rtt[3] = 1e6          # window 0
        rtt[60] = 2e6         # window 1 (t1 = 12 s)
        rtt[-1] = 0.5e6       # window 2
        k = cl.select_min_rtt(t1, rtt, 10e9)
        self.assertEqual(list(k), [3, 60, t1.size - 1])

    def test_theil_sen_recovers_line_with_outliers(self):
        rng = np.random.default_rng(1)
        x = np.linspace(0, 300e9, 60)
        y = 2.5e6 + 30e-6 * x + rng.normal(0, 20e3, x.size)   # 30 ppm drift, 2.5 ms offset
        y[::7] += 8e6                                          # 15 % gross outliers
        slope, icpt = cl.theil_sen(x, y)
        self.assertAlmostEqual(slope * 1e6, 30, delta=1.0)
        self.assertAlmostEqual(icpt / 1e6, 2.5, delta=0.2)

    def test_fit_new_with_queueing_noise(self):
        rng = np.random.default_rng(2)
        t1 = np.arange(0, 300e9, 0.2e9) + 5e14
        true_off = lambda t: -4e6 + 10e-6 * (t - t1[0])  # noqa: E731
        delay = 0.3e6 + rng.exponential(1.5e6, t1.size)   # one-way delays, out and back differ
        back = 0.3e6 + rng.exponential(1.5e6, t1.size)
        offset = true_off(t1) + (delay - back) / 2
        rtt = delay + back
        out, fn = cl.fit_new(t1, rtt, offset)
        self.assertAlmostEqual(out["slope_ppm"], 10, delta=1.0)
        self.assertAlmostEqual(out["offset_at_start_ms"], -4.0, delta=0.5)
        self.assertEqual(out["n_kept"], 30)
        self.assertLess(out["uncertainty_ms_median"], 3.0)
        mid = np.array([t1[0] + 150e9])
        self.assertAlmostEqual(float(fn(mid)[0] - mid[0]), true_off(mid)[0], delta=0.5e6)

    def test_fit_affine_trims_outliers(self):
        steady = np.arange(0, 100e9, 1e9)
        wall = 1.7e18 + steady * (1 + 20e-6)
        wall[10] += 200e6
        wall[40] -= 90e6
        a, b = cl.fit_affine(np.column_stack([steady, wall]))
        self.assertAlmostEqual((a - 1) * 1e6, 20, delta=0.5)


class WallJumpTests(unittest.TestCase):
    def test_flags_jump(self):
        steady = np.arange(0, 10e9, 1e9)
        wall = 1.7e18 + steady
        wall[6:] += 50e6   # Wall steps forward 50 ms at sample 6
        wall[3] += 2e6     # 2 ms wobble below threshold, up and back = no jump
        self.assertEqual(list(cl.find_wall_jumps(wall, steady)), [6])

    def test_no_jump(self):
        steady = np.arange(0, 10e9, 1e9)
        self.assertEqual(cl.find_wall_jumps(1.7e18 + steady, steady).size, 0)


class AnalyzeDirTests(unittest.TestCase):
    def _write(self, d, label, drift_ppm, offset_ns, wall_jump=False):
        rng = np.random.default_rng(3)
        base = 7_000_000_000_000
        with open(Path(d) / f"{label}_analog.csv", "w") as f:
            f.write(",".join(cl.ANALOG_COLS) + "\n")
            for i in range(120):
                c = base + i * 1_000_000_000
                steady = int(c + offset_ns + drift_ppm * 1e-6 * (c - base))
                wall = 1_700_000_000_000_000_000 + steady + (60_000_000 if wall_jump and i >= 80 else 0)
                f.write(f"{i},{c + int(rng.exponential(1e5))},{wall},{steady}\n")
        with open(Path(d) / f"{label}_ping.csv", "w") as f:
            f.write(",".join(cl.PING_COLS) + "\n")
            for i in range(600):
                t1 = base + i * 200_000_000
                out_d, back_d = 300_000 + int(rng.exponential(1e6)), 300_000 + int(rng.exponential(1e6))
                off = offset_ns + drift_ppm * 1e-6 * (t1 - base)
                t2 = int(t1 + out_d + off)
                t3 = t2 + 20_000
                t4 = int(t3 - off + back_d)
                o, r = cl.offset_rtt(t1, t2, t3, t4)
                f.write(f"{i},{t1},{t2},{t3},{t4},{r},{o:.1f},00000000000000ab\n")

    def test_report_files_and_reference(self):
        with tempfile.TemporaryDirectory() as d:
            self._write(d, "a", 15, 3_000_000, wall_jump=True)
            self._write(d, "b", 15, 3_000_000)
            rep, text = cl.analyze_dir(d, reference="b")
            self.assertTrue((Path(d) / "report.txt").exists())
            self.assertEqual(json.loads((Path(d) / "report.json").read_text())["reference"], "NEW mapping of b")
            a = rep["labels"]["a"]
            self.assertAlmostEqual(a["new"]["slope_ppm"], 15, delta=1.0)
            self.assertAlmostEqual(a["current_steady"]["slope_ppm"], 15, delta=3.0)
            self.assertEqual(a["wall_jumps"]["count"], 1)
            self.assertLess(a["vs_reference"]["new"]["max_ms"], 1.0)
            self.assertNotIn("new", rep["labels"]["b"]["vs_reference"])
            self.assertIn("Wall jumps", text)

    def test_truth_zero(self):
        with tempfile.TemporaryDirectory() as d:
            self._write(d, "loop", 0, 0)
            rep, _ = cl.analyze_dir(d, truth_zero=True)
            vr = rep["labels"]["loop"]["vs_reference"]
            self.assertLess(vr["new"]["max_ms"], 0.5)
            self.assertLess(vr["current_steady"]["max_ms"], 5.0)


@unittest.skipUnless(HAVE_GRPC, "grpc / thalamus-neuro not installed")
class ServerTests(unittest.TestCase):
    def test_ping_and_analog_end_to_end(self):
        import asyncio

        async def go():
            server = grpc.aio.server()
            thalamus_pb2_grpc.add_ThalamusServicer_to_server(
                cs.make_servicer(thalamus_pb2, thalamus_pb2_grpc, grpc, 0xABCDEF, 0.05), server)
            port = server.add_insecure_port("127.0.0.1:0")
            await server.start()
            try:
                async with grpc.aio.insecure_channel(f"127.0.0.1:{port}") as ch:
                    stub = thalamus_pb2_grpc.ThalamusStub(ch)

                    async def reqs():
                        yield thalamus_pb2.Ping(id=7, payload=b"hello")

                    got = None
                    async for pong in stub.ping(reqs()):
                        got = pong
                        break
                    self.assertEqual(got.id, 7)
                    self.assertTrue(got.payload.startswith(b"hello"))
                    inst, t2, t3, wall = cl.parse_trailer(bytes(got.payload))
                    self.assertEqual(inst, 0xABCDEF)
                    self.assertLessEqual(t2, t3)
                    async for resp in stub.analog(thalamus_pb2.AnalogRequest()):
                        self.assertEqual(resp.spans[0].name, "Wall")
                        self.assertEqual(len(resp.ulong_data), 1)
                        self.assertGreater(resp.time, 0)
                        break
                    with self.assertRaises(grpc.aio.AioRpcError) as cm:
                        async for _ in stub.image(thalamus_pb2.ImageRequest()):
                            pass
                    self.assertEqual(cm.exception.code(), grpc.StatusCode.UNIMPLEMENTED)
            finally:
                await server.stop(0)

        asyncio.run(go())


if __name__ == "__main__":
    unittest.main()
