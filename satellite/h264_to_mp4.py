#!/usr/bin/env python3
"""Convert the H264 video of a Thalamus STORAGE2 recording into an mp4, without re-encoding.

Usage:
  python satellite/h264_to_mp4.py RECORDING OUT.mp4 [--node sat_cam1] [--report FILE.json]

Needs PyAV (`pip install av`) in the interpreter that runs it.

Why this exists: Thalamus 1.0.53's av_muxer only accepts MPEG4 video. What the
recording holds (checked on a real 1280x1024, 15 fps recording): one Annex-B
access unit per record (4-byte start codes), in Image.data[0]; keyframes carry
SPS+PPS+IDR in the same record, and the first records of a recording can be
P-frames that cannot be decoded. The stream has no timestamps of its own, so
each frame gets the record's `time` (nanoseconds, receiver steady clock).

How it works: the access units from the first keyframe on are written back to
back into a temporary .h264 file; PyAV's h264 demuxer splits it into packets
(one per record, no B-frames), and the packets are copied (stream copy) into an
mp4 with pts = dts = record time. ffmpeg's mp4 muxer turns the SPS/PPS into the
avcC box. The mp4 is written under a temporary name and renamed when complete.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import tempfile
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stream_check  # noqa: E402

H264_FORMAT = 12  # Image.Format.H264
DEFAULT_NODE = "sat_cam1"
NANOSECONDS = Fraction(1, 1_000_000_000)


def nal_types(data: bytes) -> list[int]:
    """NAL unit types of an Annex-B access unit (start codes 00 00 01)."""
    out, i = [], data.find(b"\x00\x00\x01")
    while 0 <= i < len(data) - 3:
        out.append(data[i + 3] & 0x1F)
        i = data.find(b"\x00\x00\x01", i + 3)
    return out


def starts_stream(data: bytes) -> bool:
    """True if decoding can start here: the access unit has an SPS (7) or IDR slice (5)."""
    return any(t in (5, 7) for t in nal_types(data))


def convert(recording, out_mp4, node: str = DEFAULT_NODE) -> dict:
    """Write the H264 frames of `node` in `recording` to `out_mp4`.

    Returns {ok, frames_written, frames_skipped_leading, frames_in_recording,
    duration_s, fps, width, height, errors, warnings}. Never raises for bad
    data; check `ok`. The output only appears under its final name when complete."""
    recording, out_mp4 = Path(recording), Path(out_mp4)
    info = {"ok": False, "node": node, "frames_in_recording": 0, "frames_written": 0,
            "frames_skipped_leading": 0, "duration_s": None, "fps": None,
            "width": None, "height": None, "errors": [], "warnings": []}
    try:
        import av
    except ImportError:
        info["errors"].append("PyAV is not installed (pip install av)")
        return info

    tmp_mp4 = out_mp4.with_name(out_mp4.name + ".tmp")
    tmp_h264 = None
    try:
        times: list[int] = []
        nodes_seen: dict = {}
        stats: dict = {}
        fd, tmp_h264 = tempfile.mkstemp(suffix=".h264", dir=str(out_mp4.parent))
        with os.fdopen(fd, "wb") as raw:
            for frame, stats in stream_check.iter_image_payloads(recording):
                nodes_seen[frame["node"]] = nodes_seen.get(frame["node"], 0) + 1
                if frame["node"] != node:
                    continue
                if frame["format"] != H264_FORMAT:
                    info["errors"].append("node %r has %s images, not H264" % (
                        node, stream_check.IMAGE_FORMATS.get(frame["format"], frame["format"])))
                    return info
                info["frames_in_recording"] += 1
                data = b"".join(frame["data"])
                # Nothing before the first SPS/IDR can be decoded.
                if not times and not starts_stream(data):
                    info["frames_skipped_leading"] += 1
                    continue
                raw.write(data)
                times.append(frame["time"])
                info["width"], info["height"] = frame["width"], frame["height"]
        if stats.get("truncated"):
            info["warnings"].append("recording ended mid-record or had an unreadable record")
        if not times:
            info["errors"].append("no decodable H264 frames for node %r (nodes with images: %s)" % (
                node, sorted(nodes_seen) or "none"))
            return info

        # Timestamps must strictly increase; two records with the same time are pushed 1 ns apart.
        base, stamps = times[0], []
        for t in times:
            ts = t - base
            if stamps and ts <= stamps[-1]:
                ts = stamps[-1] + 1
            stamps.append(ts)
        steps = sorted(b - a for a, b in zip(stamps, stamps[1:]))
        last_duration = steps[len(steps) // 2] if steps else 0

        written = 0
        with av.open(tmp_h264, format="h264") as src, \
                av.open(str(tmp_mp4), "w", format="mp4", options={"movflags": "+faststart"}) as dst:
            sin = src.streams.video[0]
            sout = dst.add_stream_from_template(sin)
            for packet in src.demux(sin):
                if packet.size == 0:
                    continue
                if written >= len(stamps):
                    info["errors"].append("the stream has more frames than the recording has records")
                    break
                packet.time_base = NANOSECONDS
                packet.pts = packet.dts = stamps[written]
                nxt = stamps[written + 1] if written + 1 < len(stamps) else stamps[written] + last_duration
                packet.duration = max(1, nxt - stamps[written])
                packet.stream = sout
                dst.mux(packet)
                written += 1
        if written != len(stamps):
            info["warnings"].append("stream had %d frames for %d records" % (written, len(stamps)))
        info["frames_written"] = written
        if info["errors"]:
            return info
        info["duration_s"] = float((stamps[-1] + last_duration) * NANOSECONDS)
        info["fps"] = (float((len(stamps) - 1) / (stamps[-1] * NANOSECONDS))
                       if len(stamps) > 1 and stamps[-1] > 0 else None)
        os.replace(tmp_mp4, out_mp4)
        info["ok"] = True
    except Exception as exc:  # corrupt data must not crash the caller
        info["errors"].append("%s: %s" % (type(exc).__name__, exc))
    finally:
        with contextlib.suppress(OSError):
            if tmp_h264:
                os.unlink(tmp_h264)
        with contextlib.suppress(OSError):
            if not info["ok"]:
                tmp_mp4.unlink()
    return info


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("recording", type=Path)
    ap.add_argument("out_mp4", type=Path)
    ap.add_argument("--node", default=DEFAULT_NODE, help="node name inside the recording (default sat_cam1)")
    ap.add_argument("--report", type=Path, help="also write the result as JSON to this file")
    a = ap.parse_args(argv)
    info = convert(a.recording, a.out_mp4, a.node)
    text = json.dumps(info, indent=2)
    if a.report:
        a.report.write_text(text)
    print(text)
    return 0 if info["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
