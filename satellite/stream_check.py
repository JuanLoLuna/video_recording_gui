#!/usr/bin/env python3
"""Check the quality of a camera stream saved by a Thalamus STORAGE2 node.

Usage:
  python satellite/stream_check.py RECORDING [--node sat_cam1] [--expected-fps 30] [--out DIR]

RECORDING is the file STORAGE2 wrote (<Output File>.YYYYMMDD.R, no extension
other than the recording number).  The report is printed AND written to
report.txt and report.json (default: next to the recording; or into --out).

How the file is read (no Thalamus install needed): a recording is a sequence of
records, each an 8-byte big-endian length followed by a protobuf
`StorageRecord` (thalamus/record_reader2.py, SimpleRecordReader.__read_record;
proto/thalamus.proto).  This script decodes the few fields it needs with a
small built-in protobuf wire-format reader, so it runs on any Python 3.8+.
Fields used: StorageRecord.node (5); StorageRecord.image (6) with Image.width
(2), height (3), format (4), time (8), remote_time (9).

Times are assumed to be nanoseconds from a steady clock (UNVERIFIED unit;
override with --time-unit-seconds if the numbers look wrong).
"""
import argparse
import json
import math
import statistics
import struct
import sys
from pathlib import Path

IMAGE_FORMATS = {0: "Gray", 1: "RGB", 2: "YUYV422", 3: "YUV420P", 4: "YUVJ420P",
                 5: "Gray16", 6: "RGB16", 7: "MPEG1", 8: "MPEG4", 9: "NV12",
                 10: "BGR", 11: "MJPEG", 12: "H264", 13: "VP9"}
MAX_RECORD = 40_000_000  # same guard as record_reader2.MAX_SIZE
CLOCK_NOTE = ("Latency proxy: 'time' is the receiver's steady clock and 'remote_time' is the "
              "satellite's steady clock. The two clocks are unrelated, so their difference has "
              "no absolute meaning. Only its change over the recording (drift) and its "
              "frame-to-frame scatter (jitter) say anything about the link.")


# ---------------------------------------------------------------- parsing core
def _varint(buf, pos):
    result = shift = 0
    while True:
        b = buf[pos]
        pos += 1
        result |= (b & 0x7F) << shift
        if not b & 0x80:
            return result, pos
        shift += 7


def parse_message(buf):
    """Return {field_number: [values]} for one protobuf message (wire format).
    Length-delimited values are returned as memoryview slices (no copy)."""
    mv = memoryview(buf)
    fields, pos, n = {}, 0, len(mv)
    while pos < n:
        tag, pos = _varint(mv, pos)
        num, wire = tag >> 3, tag & 7
        if wire == 0:
            val, pos = _varint(mv, pos)
        elif wire == 1:
            val, pos = bytes(mv[pos:pos + 8]), pos + 8
        elif wire == 2:
            ln, pos = _varint(mv, pos)
            val, pos = mv[pos:pos + ln], pos + ln
        elif wire == 5:
            val, pos = bytes(mv[pos:pos + 4]), pos + 4
        else:
            raise ValueError("unsupported wire type %d" % wire)
        fields.setdefault(num, []).append(val)
    return fields


def iter_image_records(path):
    """Yield (node, record_time, width, height, format, image_time, remote_time)
    for every image record in a STORAGE2 file. Counts other records in stats."""
    stats = {"records": 0, "other_records": 0, "truncated": False}
    with open(path, "rb") as f:
        while True:
            head = f.read(8)
            if len(head) < 8:
                break
            size, = struct.unpack(">Q", head)
            if size > MAX_RECORD:
                stats["truncated"] = True
                break
            data = f.read(size)
            if len(data) < size:
                stats["truncated"] = True
                break
            stats["records"] += 1
            try:
                rec = parse_message(data)
            except (ValueError, IndexError):
                stats["truncated"] = True
                break
            if 6 not in rec:
                stats["other_records"] += 1
                continue
            img = parse_message(rec[6][0])
            node = bytes(rec[5][0]).decode("utf-8", "replace") if 5 in rec else ""
            first = lambda d, k: d[k][0] if k in d else 0
            yield (node, first(rec, 4), first(img, 2), first(img, 3), first(img, 4),
                   first(img, 8), first(img, 9)), stats


# ------------------------------------------------------------- statistics core
def _pct(sorted_vals, q):
    if not sorted_vals:
        return float("nan")
    k = (len(sorted_vals) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (k - lo)


def interval_stats(times, gap_factor=2.0, max_list=20):
    """Statistics of successive differences of a list of times (seconds)."""
    out = {"count": len(times)}
    if len(times) < 2:
        return out
    dts = [b - a for a, b in zip(times, times[1:])]
    s = sorted(dts)
    med = statistics.median(dts)
    gaps = [(times[i], dts[i]) for i in range(len(dts)) if med > 0 and dts[i] > gap_factor * med]
    dropped = sum(max(0, int(round(dt / med)) - 1) for _, dt in gaps) if med > 0 else 0
    out.update({
        "duration_s": times[-1] - times[0],
        "mean_interval_ms": statistics.fmean(dts) * 1e3,
        "median_interval_ms": med * 1e3,
        "p99_interval_ms": _pct(s, 0.99) * 1e3,
        "max_interval_ms": s[-1] * 1e3,
        "stdev_interval_ms": (statistics.pstdev(dts) * 1e3),
        "non_increasing_steps": sum(1 for d in dts if d <= 0),
        "gap_threshold_ms": gap_factor * med * 1e3,
        "gap_count": len(gaps),
        "first_gaps": [{"at_s": round(t - times[0], 4), "length_ms": round(dt * 1e3, 2)}
                       for t, dt in gaps[:max_list]],
        "estimated_dropped_frames": dropped,
    })
    return out


def offset_stats(times, remote_times):
    """Drift and jitter of (time - remote_time). Clocks are unrelated, so only
    the change of the offset is meaningful."""
    pairs = [(t, r) for t, r in zip(times, remote_times) if r]
    if len(pairs) < 2:
        return {"available": False}
    off = [t - r for t, r in pairs]
    d = [b - a for a, b in zip(off, off[1:])]
    span = pairs[-1][0] - pairs[0][0]
    drift = off[-1] - off[0]
    return {
        "available": True,
        "note": CLOCK_NOTE,
        "offset_start_s": off[0], "offset_end_s": off[-1],
        "drift_ms": drift * 1e3,
        "drift_ppm": (drift / span * 1e6) if span > 0 else float("nan"),
        "jitter_stdev_ms": statistics.pstdev(d) * 1e3,
        "jitter_max_abs_ms": max(abs(x) for x in d) * 1e3,
        "offset_range_ms": (max(off) - min(off)) * 1e3,
    }


def analyze(times, remote_times, expected_fps=None, info=None):
    rep = {"frames": len(times), "format": info or {}}
    rep["receiver_time"] = interval_stats(times)
    rep["satellite_time"] = interval_stats([r for r in remote_times if r]) if any(remote_times) \
        else {"count": 0, "note": "remote_time is 0 in every record (no satellite timestamps)"}
    rep["transfer_latency_proxy"] = offset_stats(times, remote_times)
    rt = rep["receiver_time"]
    if expected_fps and "duration_s" in rt and rt["duration_s"] > 0:
        expected = rt["duration_s"] * expected_fps + 1
        rep["expected"] = {"fps": expected_fps, "expected_frames": round(expected),
                           "missing_vs_expected": round(expected - len(times)),
                           "achieved_fps": (len(times) - 1) / rt["duration_s"]}
    return rep


def format_text(rep, source):
    L = ["Stream check: %s" % source, "Node: %s" % rep.get("node", "?"),
         "Frames: %d" % rep["frames"]]
    fmt = rep["format"]
    if fmt:
        L.append("Resolution/format: %sx%s %s" % (fmt.get("width"), fmt.get("height"), fmt.get("format")))
    for key, label in (("receiver_time", "Receiver clock (time)"), ("satellite_time", "Satellite clock (remote_time)")):
        s = rep[key]
        L.append("")
        L.append(label)
        if "duration_s" not in s:
            L.append("  " + s.get("note", "not enough frames"))
            continue
        L += ["  duration: %.3f s" % s["duration_s"],
              "  frame interval ms: mean %.2f  median %.2f  p99 %.2f  max %.2f  stdev %.2f" % (
                  s["mean_interval_ms"], s["median_interval_ms"], s["p99_interval_ms"],
                  s["max_interval_ms"], s["stdev_interval_ms"]),
              "  gaps > 2x median (> %.1f ms): %d   estimated dropped frames: %d   non-increasing steps: %d" % (
                  s["gap_threshold_ms"], s["gap_count"], s["estimated_dropped_frames"], s["non_increasing_steps"])]
        for g in s["first_gaps"]:
            L.append("    gap at %.3f s: %.1f ms" % (g["at_s"], g["length_ms"]))
    L.append("")
    L.append("Transfer latency proxy (time - remote_time)")
    t = rep["transfer_latency_proxy"]
    if t.get("available"):
        L += ["  NOTE: " + t["note"],
              "  drift: %.2f ms over the recording (%.1f ppm)" % (t["drift_ms"], t["drift_ppm"]),
              "  jitter (frame to frame change): stdev %.2f ms, max %.2f ms" % (t["jitter_stdev_ms"], t["jitter_max_abs_ms"]),
              "  offset range: %.2f ms" % t["offset_range_ms"]]
    else:
        L.append("  not available (no remote_time values)")
    if "expected" in rep:
        e = rep["expected"]
        L += ["", "Against expected %.1f fps: expected ~%d frames, got %d (missing ~%d), achieved %.2f fps" % (
            e["fps"], e["expected_frames"], rep["frames"], e["missing_vs_expected"], e["achieved_fps"])]
    for w in rep.get("warnings", []):
        L.append("WARNING: " + w)
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("recording", type=Path)
    ap.add_argument("--node", default="sat_cam1", help="node name in the recording (default sat_cam1)")
    ap.add_argument("--expected-fps", type=float)
    ap.add_argument("--out", type=Path, help="folder for report.txt/report.json (default: next to the recording)")
    ap.add_argument("--time-unit-seconds", type=float, default=1e-9,
                    help="seconds per time tick (default 1e-9 = nanoseconds)")
    a = ap.parse_args(argv)

    times, remote, info, stats, nodes_seen = [], [], {}, {}, {}
    for (node, rtime, w, h, fmt, itime, rem), stats in iter_image_records(a.recording):
        nodes_seen[node] = nodes_seen.get(node, 0) + 1
        if node != a.node:
            continue
        # Image.time (field 8) is the image node's time; fall back to record time.
        times.append((itime or rtime) * a.time_unit_seconds)
        remote.append(rem * a.time_unit_seconds)
        info = {"width": w, "height": h, "format": IMAGE_FORMATS.get(fmt, str(fmt))}
    rep = analyze(times, remote, a.expected_fps, info)
    rep["node"] = a.node
    rep["file"] = str(a.recording)
    rep["image_records_by_node"] = nodes_seen
    rep["file_stats"] = stats
    rep["warnings"] = []
    if not times:
        rep["warnings"].append("no image records for node %r; nodes with images: %s" % (a.node, sorted(nodes_seen) or "none"))
    if stats.get("truncated"):
        rep["warnings"].append("file ended mid-record or had an unreadable record (recording stopped early?)")
    text = format_text(rep, a.recording)
    out = a.out or a.recording.parent
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.txt").write_text(text)
    (out / "report.json").write_text(json.dumps(rep, indent=2))
    print(text)
    print("Wrote %s and %s" % (out / "report.txt", out / "report.json"))
    return 0 if times else 1


if __name__ == "__main__":
    sys.exit(main())
