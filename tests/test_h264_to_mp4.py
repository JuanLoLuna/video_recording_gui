"""satellite/h264_to_mp4.py on a small synthetic STORAGE2 recording (needs PyAV; skipped without it)."""
import struct
import sys
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "satellite"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import h264_to_mp4  # noqa: E402
import stream_check  # noqa: E402
from test_satellite_stream_check import _record, _vi  # noqa: E402

try:
    import av
except ImportError:  # pragma: no cover
    av = None

W, H, FPS = 64, 48, 15


def _h264_record(node, t, data, fmt=12):
    img = (b"\x0a" + _vi(len(data)) + data + b"\x10" + _vi(W) + b"\x18" + _vi(H) + b"\x20" + _vi(fmt))
    rec = b"\x32" + _vi(len(img)) + img + b"\x20" + _vi(t) + b"\x2a" + _vi(len(node)) + node.encode()
    return struct.pack(">Q", len(rec)) + rec


def _encode(n=12, gop=5):
    """n Annex-B access units (one per packet); keyframes (SPS+PPS+IDR) every `gop` frames."""
    codec = None
    for name in ("libx264", "h264", "libopenh264"):
        try:
            codec = av.CodecContext.create(name, "w")
            break
        except Exception:
            continue
    if codec is None:
        raise unittest.SkipTest("no H264 encoder in this PyAV build")
    codec.width, codec.height, codec.pix_fmt = W, H, "yuv420p"
    codec.time_base, codec.framerate = Fraction(1, FPS), FPS
    codec.gop_size, codec.max_b_frames = gop, 0
    codec.options = {"preset": "ultrafast", "tune": "zerolatency", "bf": "0"}
    codec.open()
    packets = []
    for i in range(n):
        frame = av.VideoFrame(W, H, "yuv420p")
        for plane in frame.planes:
            plane.update(bytes([(i * 20 + 40) % 256]) * plane.buffer_size)
        frame.pts = i
        packets += [bytes(p) for p in codec.encode(frame)]
    packets += [bytes(p) for p in codec.encode(None)]
    return packets


@unittest.skipIf(av is None, "PyAV (pip install av) not installed")
class ConvertTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.packets = _encode()
        self.assertGreaterEqual(len(self.packets), 12)

    def _recording(self, packets, node="sat_cam1", step_ns=int(1e9 / FPS), t0=5_000_000_000):
        blob = _record("other_node", t0, t0)  # an unrelated image record
        blob += struct.pack(">Q", 4) + b"\x28\x01\x20\x01"  # a record without an image
        for i, p in enumerate(packets):
            blob += _h264_record(node, t0 + i * step_ns, p)
        path = self.dir / "rec.thalamus"
        path.write_bytes(blob)
        return path

    def test_converts_all_frames_with_timestamps(self):
        rec = self._recording(self.packets)
        out = self.dir / "out.mp4"
        info = h264_to_mp4.convert(rec, out)
        self.assertTrue(info["ok"], info)
        self.assertEqual(info["frames_written"], len(self.packets))
        self.assertEqual(info["frames_skipped_leading"], 0)
        self.assertAlmostEqual(info["fps"], FPS, places=2)
        self.assertEqual(sorted(p.name for p in self.dir.iterdir()), ["out.mp4", "rec.thalamus"])  # no temp files left
        with av.open(str(out)) as c:
            s = c.streams.video[0]
            self.assertEqual((s.width, s.height, s.codec_context.name), (W, H, "h264"))
            times = [f.time for f in c.decode(video=0)]
        self.assertEqual(len(times), len(self.packets))
        self.assertAlmostEqual(times[1] - times[0], 1 / FPS, places=3)
        self.assertAlmostEqual(info["duration_s"], len(self.packets) / FPS, places=2)

    def test_skips_leading_frames_before_first_keyframe(self):
        rec = self._recording(self.packets[1:])  # first keyframe dropped: 4 P-frames, then a keyframe
        info = h264_to_mp4.convert(rec, self.dir / "out.mp4")
        self.assertTrue(info["ok"], info)
        self.assertEqual(info["frames_skipped_leading"], 4)
        self.assertEqual(info["frames_written"], len(self.packets) - 5)
        with av.open(str(self.dir / "out.mp4")) as c:
            self.assertEqual(sum(1 for _ in c.decode(video=0)), info["frames_written"])

    def test_wrong_node_and_wrong_format_fail_without_output(self):
        rec = self._recording(self.packets)
        info = h264_to_mp4.convert(rec, self.dir / "out.mp4", node="nope")
        self.assertFalse(info["ok"])
        self.assertIn("sat_cam1", info["errors"][0])
        info = h264_to_mp4.convert(self._recording(self.packets[:2]).parent / "rec.thalamus",
                                   self.dir / "o2.mp4", node="other_node")
        self.assertFalse(info["ok"])  # other_node holds MJPEG (format 11)
        self.assertIn("not H264", info["errors"][0])
        self.assertFalse((self.dir / "out.mp4").exists() or (self.dir / "o2.mp4").exists())

    def test_cli(self):
        rec = self._recording(self.packets)
        rc = h264_to_mp4.main([str(rec), str(self.dir / "cli.mp4")])
        self.assertEqual(rc, 0)
        self.assertTrue((self.dir / "cli.mp4").stat().st_size > 0)


class NoAvTests(unittest.TestCase):
    def test_nal_types_and_keyframe_detection(self):
        au = b"\x00\x00\x00\x01\x67aa\x00\x00\x00\x01\x68bb\x00\x00\x00\x01\x65cc"
        self.assertEqual(h264_to_mp4.nal_types(au), [7, 8, 5])
        self.assertTrue(h264_to_mp4.starts_stream(au))
        self.assertFalse(h264_to_mp4.starts_stream(b"\x00\x00\x00\x01\x61dd"))

    def test_payload_reader(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "r"
            p.write_bytes(_h264_record("n", 7, b"\x00\x00\x00\x01\x65x"))
            frames = [f for f, _ in stream_check.iter_image_payloads(p)]
        self.assertEqual(frames, [{"node": "n", "time": 7, "width": W, "height": H, "format": 12,
                                   "data": [b"\x00\x00\x00\x01\x65x"]}])


if __name__ == "__main__":
    unittest.main()
