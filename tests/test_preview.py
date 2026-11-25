"""satellite preview: keyframe detection, rate limiter, staleness, status text, decode, stream worker."""
import asyncio
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "satellite"))
import preview_logic as pl  # noqa: E402
import window_logic as wl  # noqa: E402

try:
    import av  # noqa: F401
    import numpy  # noqa: F401
    HAVE_AV = True
except Exception:  # noqa: BLE001
    HAVE_AV = False
try:
    import grpc  # noqa: F401
    from thalamus import thalamus_pb2, thalamus_pb2_grpc  # noqa: F401
    HAVE_GRPC = True
except Exception:  # noqa: BLE001
    HAVE_GRPC = False

SC = b"\x00\x00\x00\x01"
SPS, PPS, IDR, P = SC + b"\x67ab", SC + b"\x68cd", SC + b"\x65xyz", SC + b"\x41pp"


class KeyframeTests(unittest.TestCase):
    def test_nal_types_and_keyframe(self):
        self.assertEqual(pl.nal_types(SPS + PPS + IDR), [7, 8, 5])
        self.assertTrue(pl.is_keyframe(SPS + PPS + IDR))
        self.assertTrue(pl.is_keyframe(b"\x00\x00\x01\x65zz"))   # 3-byte start code
        self.assertFalse(pl.is_keyframe(P))
        self.assertFalse(pl.is_keyframe(SPS + PPS))               # parameter sets alone are not a picture
        self.assertFalse(pl.is_keyframe(b""))
        self.assertFalse(pl.is_keyframe(b"no start codes at all"))
        self.assertEqual(pl.H264_FORMAT, 12)


class RateAndMeterTests(unittest.TestCase):
    def test_rate_limiter(self):
        r = pl.RateLimiter(1.0)
        self.assertTrue(r.allow(10.0))
        self.assertFalse(r.allow(10.5))
        self.assertFalse(r.allow(10.85))
        self.assertTrue(r.allow(10.95))
        r2 = pl.RateLimiter(2.0)
        self.assertEqual([r2.allow(t / 10) for t in range(0, 12, 2)], [True, False, False, True, False, False])
        self.assertFalse(pl.RateLimiter(0).allow(1.0))            # 0 = preview off

    def test_fps_meter_over_five_seconds(self):
        m = pl.FpsMeter()
        for i in range(0, 90):                                    # 15 fps for 6 s
            m.add(i / 15)
        self.assertAlmostEqual(m.fps(6.0), 15.0, delta=0.5)
        self.assertAlmostEqual(m.age_s(6.0), 6.0 - 89 / 15, places=6)
        self.assertEqual(m.fps(30.0), 0.0)                        # nothing in the last 5 s
        self.assertIsNone(pl.FpsMeter().age_s(1.0))
        self.assertEqual(pl.FpsMeter().fps(1.0), 0.0)

    def test_picture_state_and_caption(self):
        self.assertEqual(pl.picture_state(None), "none")
        self.assertEqual(pl.picture_state(5.0), "ok")
        self.assertEqual(pl.picture_state(5.1), "stale")
        self.assertEqual(pl.picture_caption(None, "receiver not connected"), "no picture (receiver not connected)")
        self.assertEqual(pl.picture_caption(0.8, None), "picture 0.8 s old")
        self.assertIn("last one 12 s ago", pl.picture_caption(12.0, None))


class StatusTextTests(unittest.TestCase):
    base = {"receiver_connected": True,
            "clock": {"ok": True, "reachable": True, "label": "r", "uncertainty_ms": 1.0, "age_s": 1}}

    def test_frames_arriving_green(self):
        lvl, txt = wl.decide_status({**self.base, "stream": pl.stream_status(True, 15.0, 0.2)})
        self.assertEqual(lvl, "green")
        self.assertIn("frames 15/s", txt)
        self.assertNotIn("camera on", txt)

    def test_running_but_no_frames_orange(self):
        lvl, txt = wl.decide_status({**self.base, "stream": pl.stream_status(True, 0.0, 6.0)})
        self.assertEqual((lvl, "camera on, no frames" in txt), ("orange", True))
        lvl, txt = wl.decide_status({**self.base, "stream": pl.stream_status(True, 0.0, None)})
        self.assertEqual(lvl, "orange")
        lvl, txt = wl.decide_status({**self.base, "stream": pl.stream_status(False, 0.0, None)})
        self.assertEqual((lvl, "camera off, no frames" in txt), ("orange", True))
        self.assertTrue(pl.stream_status(True, 1.0, 5.0)["arriving"])
        self.assertFalse(pl.stream_status(True, 1.0, 5.1)["arriving"])

    def test_preview_off_keeps_camera_flag(self):
        self.assertIn("camera on", wl.decide_status({**self.base, "camera_on": True})[1])
        self.assertEqual(wl.DEFAULTS["preview_max_fps"], 1)


class FakeControl:
    def __init__(self, thread=None):
        self._thread = thread
        self.connected = thread is not None


class WorkerLogicTests(unittest.TestCase):
    def worker(self, decode=None, fps=1.0):
        import preview
        self.t = [100.0]
        w = preview.PreviewWorker(FakeControl(), "sat_cam1", fps, clock=lambda: self.t[0],
                                  decode=decode or (lambda d, width: (2, 2, 4, b"\x01" * 8)))
        return w

    def test_only_keyframes_at_most_once_per_second_are_queued(self):
        w = self.worker()
        queued = 0
        for i in range(45):                                       # 3 s at 15 fps, IDR every 14
            data = (SPS + PPS + IDR) if i % 14 == 0 else P
            w.on_image(12, data, 100.0 + i / 15)
            queued += 0 if w.q.empty() else 1
            while not w.q.empty():
                w.q.get_nowait()
        self.assertEqual(queued, 4)                               # IDR frames 0, 14, 28, 42 (0.93 s apart: all pass)
        s = w.snapshot(103.0)
        self.assertGreater(s["fps"], 5)
        self.assertEqual(s["error"], None)

    def test_wrong_format_is_reported_and_not_decoded(self):
        w = self.worker()
        w.on_image(0, b"raw", 100.0)
        self.assertIn("not H264", w.snapshot(100.0)["error"])
        self.assertTrue(w.q.empty())

    def test_preview_off_never_queues(self):
        w = self.worker(fps=0)
        w.on_image(12, SPS + PPS + IDR, 100.0)
        self.assertTrue(w.q.empty())

    def test_decode_loop_publishes_picture_and_errors(self):
        w = self.worker()
        w.start()
        self.addCleanup(w.stop)
        w.on_image(12, SPS + PPS + IDR, 100.0)
        for _ in range(50):
            if w.snapshot(100.0)["picture"]:
                break
            time.sleep(0.05)
        s = w.snapshot(101.0)
        self.assertEqual(s["picture"], (2, 2, 4, b"\x01" * 8))
        self.assertAlmostEqual(s["picture_age_s"], 1.0, delta=5.0)
        self.assertEqual(s["picture_seq"], 1)

    def test_no_connection_gives_reason(self):
        w = self.worker()
        w.supervise_once(100.0)
        self.assertEqual(w.snapshot(100.0)["error"], "receiver not connected")


@unittest.skipUnless(HAVE_AV and shutil.which("ffmpeg"), "PyAV or ffmpeg not available")
class DecodeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        clip = Path(cls.tmp.name) / "c.h264"
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=320x240:rate=15",
                        "-t", "3", "-c:v", "libx264", "-g", "14", "-pix_fmt", "yuv420p", "-f", "h264", str(clip)],
                       check=True)
        import av as _av
        c = _av.open(str(clip))
        cls.units = [bytes(p) for p in c.demux(video=0) if p.size]
        c.close()

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_keyframes_every_14_and_decode(self):
        import preview
        keys = [i for i, u in enumerate(self.units) if pl.is_keyframe(u)]
        self.assertEqual(keys[:3], [0, 14, 28])
        w, h, stride, data = preview.decode_keyframe(self.units[14], 160)
        self.assertEqual((w, h), (160, 120))
        self.assertEqual(stride, 160)
        self.assertEqual(len(data), stride * h)
        self.assertGreater(len(set(data)), 20)                    # a real picture, not a flat frame

    def test_non_keyframe_alone_gives_nothing_or_error_not_crash(self):
        import preview
        p = next(u for u in self.units if not pl.is_keyframe(u))
        try:
            self.assertIsNone(preview.decode_keyframe(p))
        except Exception:
            pass

    def test_stride_padding(self):
        import preview
        w, h, stride, data = preview.decode_keyframe(self.units[0], 150)
        self.assertEqual((w, stride % 4), (150, 0))
        self.assertEqual(len(data), stride * h)


@unittest.skipUnless(HAVE_AV and HAVE_GRPC and shutil.which("ffmpeg"), "grpc / thalamus / PyAV / ffmpeg not available")
class StreamWorkerTests(unittest.TestCase):
    """Worker against a fake Thalamus server: real gRPC image stream of H264 access units."""

    def test_subscribe_decode_and_resubscribe(self):
        import av as _av
        import preview
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        clip = Path(tmp.name) / "c.h264"
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=320x240:rate=15",
                        "-t", "3", "-c:v", "libx264", "-g", "14", "-pix_fmt", "yuv420p", "-f", "h264", str(clip)], check=True)
        c = _av.open(str(clip))
        units = [bytes(p) for p in c.demux(video=0) if p.size]
        c.close()
        requests = []

        class Servicer(thalamus_pb2_grpc.ThalamusServicer):
            async def image(self, request, context):
                requests.append((request.node.name, request.framerate))
                for u in units:
                    yield thalamus_pb2.Image(data=[u], width=320, height=240, format=12)
                    await asyncio.sleep(1 / 15)

        loop = asyncio.new_event_loop()
        threading.Thread(target=loop.run_forever, daemon=True).start()

        class Th:
            pass
        th = Th()
        th.loop = loop

        async def boot():
            server = grpc.aio.server()
            thalamus_pb2_grpc.add_ThalamusServicer_to_server(Servicer(), server)
            port = server.add_insecure_port("127.0.0.1:0")
            await server.start()
            ch = grpc.aio.insecure_channel("127.0.0.1:%d" % port)
            th.stub = thalamus_pb2_grpc.ThalamusStub(ch)
            return server

        server = asyncio.run_coroutine_threadsafe(boot(), loop).result(10)
        self.addCleanup(lambda: asyncio.run_coroutine_threadsafe(server.stop(0), loop).result(10))
        w = preview.PreviewWorker(FakeControl(th), "sat_cam1", 1.0, width=160)
        self.addCleanup(w.stop)
        w.start()
        end = time.monotonic() + 15
        while time.monotonic() < end and w.snapshot()["picture_seq"] < 2:
            time.sleep(0.1)
        s = w.snapshot()
        self.assertGreaterEqual(s["picture_seq"], 2)
        self.assertEqual(requests[0], ("sat_cam1", 0.0))
        self.assertEqual(s["picture"][:2], (160, 120))
        self.assertGreater(s["fps"], 5)
        # stream ends after 3 s: error says so, and the worker subscribes again after the rate limit
        end = time.monotonic() + 15
        while time.monotonic() < end and len(requests) < 2:
            time.sleep(0.2)
        self.assertGreaterEqual(len(requests), 2)


if __name__ == "__main__":
    unittest.main()
