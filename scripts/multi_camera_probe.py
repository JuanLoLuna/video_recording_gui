#!/usr/bin/env python3
"""Multi-camera acquisition probe: can N cameras run together at a given fps?

Run ON THE RECORDING LAPTOP (needs PySpin), with the recorder GUI and
SpinView CLOSED -- either one holds a camera open and Init() will fail.

    python scripts/multi_camera_probe.py                  # all cameras, 30 and 60 fps, 60 s per run
    python scripts/multi_camera_probe.py --seconds 600    # the real test: 10 min per run
    python scripts/multi_camera_probe.py --fps 30 --serials 23227865 26134271

No disk is written. Each camera is configured like the app configures it
(chunk Timestamp/FrameID, continuous mode, frame-rate enable, exposure Off and
clamped to 90% of the frame period, manual stream buffers sized for 5 s at the
camera's max fps) and grabbed in its own thread, copying each frame into a
numpy array exactly as _acquisition_loop does -- so the memcpy and Python
overhead are part of what's measured.

For every fps it runs each camera ALONE first, then all cameras TOGETHER. The
solo run is the baseline: if a camera gets slower, gappier or burstier when
the others are running, that is contention (USB bandwidth, the GIL, or CPU),
and the comparison below says so. Without a baseline you can't tell a camera
limit from a multi-camera limit.

Per camera and run it reports achieved fps (from frame arrival times, not the
requested rate), FrameID gaps (frames the camera produced but we never got),
incomplete images, GetNextImage exceptions, grab-time and inter-frame-interval
tails, MB/s on the wire, and process CPU. A run PASSes when achieved fps is
>= 99% of the rate the camera actually applied, with zero gaps, zero
incomplete images and zero errors.

Results are also written to probe_output/multi_camera_probe_<timestamp>.csv.
"""
from __future__ import annotations

import argparse
import csv
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

# Mirrors backend/camera_control.py.
EXPOSURE_FRAME_PERIOD_HEADROOM = 0.9
STREAM_BUFFER_SECONDS = 5.0
GRAB_TIMEOUT_MS = 1000

PASS_FPS_FRACTION = 0.99
# "both" may lose at most this fraction of its solo fps before it's flagged.
CONTENTION_TOLERANCE = 0.01

# Transport-layer counters; which of these exist varies by camera/firmware,
# so each is read if present and silently skipped otherwise.
TL_STREAM_COUNTERS = (
    "StreamDeliveredFrameCount",
    "StreamLostFrameCount",
    "StreamDroppedFrameCount",
    "StreamFailedBufferCount",
    "StreamBufferUnderrunCount",
)


# ----------------------------------------------------------------------
# Pure analysis (no PySpin) -- importable and checkable on any machine.
# ----------------------------------------------------------------------
def percentile(values, q: float):
    if not values:
        return None
    ordered = sorted(values)
    return float(ordered[round((len(ordered) - 1) * q)])


def count_frame_gaps(frame_ids) -> int:
    """Frames the camera numbered but we never received.

    A FrameID that goes backwards (counter reset after a reconnect) is not a
    gap and is skipped, as is any missing id.
    """
    gaps = 0
    prev = None
    for fid in frame_ids:
        if fid is None:
            continue
        if prev is not None and fid > prev + 1:
            gaps += fid - prev - 1
        prev = fid
    return gaps


def summarize(arrivals, grab_ms, frame_ids, bytes_per_frame: int) -> dict:
    """Reduce one camera's raw samples to the reported metrics."""
    n = len(arrivals)
    if n < 2:
        return {
            "frames": n, "achieved_fps": 0.0, "mb_per_s": 0.0, "frame_gaps": 0,
            "grab_ms_p95": None, "grab_ms_max": None,
            "interval_ms_p99": None, "interval_ms_max": None,
        }
    span = arrivals[-1] - arrivals[0]
    fps = (n - 1) / span if span > 0 else 0.0
    intervals = [(b - a) * 1000.0 for a, b in zip(arrivals, arrivals[1:])]
    return {
        "frames": n,
        "achieved_fps": fps,
        "mb_per_s": fps * bytes_per_frame / 1e6,
        "frame_gaps": count_frame_gaps(frame_ids),
        "grab_ms_p95": percentile(grab_ms, 0.95),
        "grab_ms_max": max(grab_ms),
        "interval_ms_p99": percentile(intervals, 0.99),
        "interval_ms_max": max(intervals),
    }


def verdict(row: dict) -> tuple[bool, str]:
    reasons = []
    applied = row["applied_fps"]
    if applied > 0 and row["achieved_fps"] < PASS_FPS_FRACTION * applied:
        reasons.append(f"fps {row['achieved_fps']:.2f} < 99% of {applied:.2f}")
    if row["frame_gaps"]:
        reasons.append(f"{row['frame_gaps']} frame gaps")
    if row["incomplete"]:
        reasons.append(f"{row['incomplete']} incomplete")
    if row["errors"]:
        reasons.append(f"{row['errors']} grab errors")
    if row["frames"] < 2:
        reasons.append("no frames")
    return (not reasons, "; ".join(reasons) or "ok")


def contention_note(solo: dict | None, both: dict) -> str:
    """Compare a camera's 'both' run to its own 'solo' baseline."""
    if solo is None or solo["achieved_fps"] <= 0:
        return "no solo baseline"
    drop = 1.0 - both["achieved_fps"] / solo["achieved_fps"]
    notes = []
    if drop > CONTENTION_TOLERANCE:
        notes.append(f"fps down {drop * 100:.1f}% vs solo")
    if both["frame_gaps"] > solo["frame_gaps"]:
        notes.append(f"gaps {solo['frame_gaps']} -> {both['frame_gaps']}")
    p_solo, p_both = solo["interval_ms_p99"], both["interval_ms_p99"]
    if p_solo and p_both and p_both > 1.5 * p_solo:
        notes.append(f"interval p99 {p_solo:.1f} -> {p_both:.1f} ms")
    return "; ".join(notes) or "no contention vs solo"


# ----------------------------------------------------------------------
# Camera access (PySpin)
# ----------------------------------------------------------------------
@dataclass
class Recorder:
    serial: str
    arrivals: list = field(default_factory=list)
    grab_ms: list = field(default_factory=list)
    frame_ids: list = field(default_factory=list)
    incomplete: int = 0
    errors: int = 0
    last_error: str = ""


@dataclass
class CamInfo:
    serial: str
    model: str
    speed: str
    width: int
    height: int
    pixel_format: str
    bytes_per_frame: int
    applied_fps: float
    fps_ceiling: float
    link_limit: int | None
    buffer_count: int | None


def _node(nodemap, name):
    return nodemap.GetNode(name)


def _read_str(PySpin, nodemap, name) -> str:
    node = _node(nodemap, name)
    if node is None or not PySpin.IsReadable(node):
        return "?"
    try:
        enum = PySpin.CEnumerationPtr(node)
        if PySpin.IsReadable(enum):
            return enum.GetCurrentEntry().GetSymbolic()
    except Exception:
        pass
    try:
        return PySpin.CStringPtr(node).GetValue()
    except Exception:
        return "?"


def _read_int(PySpin, nodemap, name):
    node = _node(nodemap, name)
    if node is None or not PySpin.IsReadable(node):
        return None
    try:
        return int(PySpin.CIntegerPtr(node).GetValue())
    except Exception:
        return None


def _set_enum(PySpin, nodemap, name, entry_name) -> bool:
    try:
        node = PySpin.CEnumerationPtr(nodemap.GetNode(name))
        if not PySpin.IsWritable(node):
            return False
        node.SetIntValue(node.GetEntryByName(entry_name).GetValue())
        return True
    except Exception:
        return False


def _enable_chunks(PySpin, cam) -> None:
    nodemap = cam.GetNodeMap()
    mode = PySpin.CBooleanPtr(nodemap.GetNode("ChunkModeActive"))
    if not PySpin.IsWritable(mode):
        return
    mode.SetValue(True)
    selector = PySpin.CEnumerationPtr(nodemap.GetNode("ChunkSelector"))
    enable = PySpin.CBooleanPtr(nodemap.GetNode("ChunkEnable"))
    for name in ("Timestamp", "FrameID"):
        try:
            selector.SetIntValue(selector.GetEntryByName(name).GetValue())
            if PySpin.IsWritable(enable):
                enable.SetValue(True)
        except Exception as exc:
            print(f"    could not enable chunk {name}: {exc}")


def configure(PySpin, cam, target_fps: float) -> CamInfo:
    """Init'd camera -> app-equivalent configuration. Call before BeginAcquisition."""
    tl = cam.GetTLDeviceNodeMap()
    serial = _read_str(PySpin, tl, "DeviceSerialNumber")
    model = _read_str(PySpin, tl, "DeviceModelName")
    speed = _read_str(PySpin, tl, "DeviceCurrentSpeed")

    _enable_chunks(PySpin, cam)
    nodemap = cam.GetNodeMap()
    _set_enum(PySpin, nodemap, "AcquisitionMode", "Continuous")
    # The app locks exposure Off at fps >= 30 (see EXPOSURE_AUTO_LOCK_MIN_FPS).
    _set_enum(PySpin, nodemap, "ExposureAuto", "Off")

    enable = PySpin.CBooleanPtr(nodemap.GetNode("AcquisitionFrameRateEnable"))
    if PySpin.IsWritable(enable):
        enable.SetValue(True)
    rate = PySpin.CFloatPtr(nodemap.GetNode("AcquisitionFrameRate"))

    ceiling = float(rate.GetMax()) if PySpin.IsReadable(rate) else 0.0

    buffer_count = None
    try:
        tl_stream = cam.GetTLStreamNodeMap()
        _set_enum(PySpin, tl_stream, "StreamBufferCountMode", "Manual")
        count = PySpin.CIntegerPtr(tl_stream.GetNode("StreamBufferCountManual"))
        if PySpin.IsWritable(count):
            count.SetValue(min(int(count.GetMax()), round(ceiling * STREAM_BUFFER_SECONDS)))
            buffer_count = int(count.GetValue())
    except Exception as exc:
        print(f"    could not set stream buffers: {exc}")

    if PySpin.IsWritable(rate):
        rate.SetValue(min(float(rate.GetMax()), max(float(rate.GetMin()), target_fps)))
    applied = float(rate.GetValue()) if PySpin.IsReadable(rate) else 0.0

    exposure = PySpin.CFloatPtr(nodemap.GetNode("ExposureTime"))
    if applied > 0 and PySpin.IsWritable(exposure):
        safe_max = (1_000_000.0 / applied) * EXPOSURE_FRAME_PERIOD_HEADROOM
        if float(exposure.GetValue()) > safe_max:
            exposure.SetValue(max(float(exposure.GetMin()), safe_max))

    width = _read_int(PySpin, nodemap, "Width") or 0
    height = _read_int(PySpin, nodemap, "Height") or 0
    pixel_format = _read_str(PySpin, nodemap, "PixelFormat")
    bytes_per_pixel = 2 if "16" in pixel_format else 1  # Mono8/Mono16 are what the app handles
    return CamInfo(
        serial=serial, model=model, speed=speed, width=width, height=height,
        pixel_format=pixel_format, bytes_per_frame=width * height * bytes_per_pixel,
        applied_fps=applied, fps_ceiling=ceiling,
        link_limit=_read_int(PySpin, nodemap, "DeviceLinkThroughputLimit"),
        buffer_count=buffer_count,
    )


def read_stream_counters(PySpin, cam) -> dict:
    out = {}
    try:
        tl_stream = cam.GetTLStreamNodeMap()
    except Exception:
        return out
    for name in TL_STREAM_COUNTERS:
        value = _read_int(PySpin, tl_stream, name)
        if value is not None:
            out[name] = value
    return out


def grab_loop(cam, rec: Recorder, stop: threading.Event, copy_frames: bool, np) -> None:
    while not stop.is_set():
        t0 = time.monotonic()
        try:
            image = cam.GetNextImage(GRAB_TIMEOUT_MS)
        except Exception as exc:
            rec.errors += 1
            rec.last_error = str(exc)
            stop.wait(0.05)  # don't busy-spin on a persistent fault
            continue
        t1 = time.monotonic()
        if image.IsIncomplete():
            rec.incomplete += 1
            image.Release()
            continue
        frame_id = None
        try:
            frame_id = int(image.GetChunkData().GetFrameID())
        except Exception:
            pass
        if copy_frames:
            np.array(image.GetNDArray(), copy=True)
        image.Release()
        rec.arrivals.append(t1)
        rec.grab_ms.append((t1 - t0) * 1000.0)
        rec.frame_ids.append(frame_id)


def run_stage(PySpin, np, system, serials, fps, seconds, copy_frames, mode) -> list[dict]:
    """Acquire from `serials` simultaneously for `seconds`; one result row each."""
    cam_list = system.GetCameras()
    cams, infos, recs, threads = [], [], [], []
    counters_before, counters_after = [], []
    stop = threading.Event()
    rows: list[dict] = []
    try:
        for serial in serials:
            cam = cam_list.GetBySerial(serial)
            cam.Init()
            cams.append(cam)
            infos.append(configure(PySpin, cam, fps))
            recs.append(Recorder(serial))
        for info in infos:
            print(f"    {info.serial} {info.model}: {info.width}x{info.height} {info.pixel_format}, "
                  f"link={info.speed}, applied {info.applied_fps:.2f} fps "
                  f"(ceiling {info.fps_ceiling:.1f}), "
                  f"{info.bytes_per_frame * info.applied_fps / 1e6:.1f} MB/s, "
                  f"buffers={info.buffer_count}, link_limit={info.link_limit}")
            if info.speed not in ("SuperSpeed", "?"):
                print(f"    !! {info.serial} is on {info.speed}, not SuperSpeed -- check cable/port")
        counters_before = [read_stream_counters(PySpin, c) for c in cams]

        cpu0, wall0 = time.process_time(), time.monotonic()
        for cam in cams:
            cam.BeginAcquisition()
        for cam, rec in zip(cams, recs):
            t = threading.Thread(target=grab_loop, args=(cam, rec, stop, copy_frames, np), daemon=True)
            t.start()
            threads.append(t)

        deadline = time.monotonic() + seconds
        next_report = time.monotonic() + 10.0
        try:
            while time.monotonic() < deadline:
                time.sleep(0.25)
                if time.monotonic() >= next_report:
                    next_report += 10.0
                    print("    t+%3.0fs  " % (seconds - (deadline - time.monotonic()))
                          + "  ".join(f"{r.serial}: {len(r.arrivals)} fr, {r.errors} err" for r in recs))
        except KeyboardInterrupt:
            print("    interrupted -- reporting what was captured")

        stop.set()
        for t in threads:
            t.join(timeout=5.0)
        cpu_cores = (time.process_time() - cpu0) / max(1e-9, time.monotonic() - wall0)
        counters_after = [read_stream_counters(PySpin, c) for c in cams]

        for info, rec, before, after in zip(infos, recs, counters_before, counters_after):
            stats = summarize(rec.arrivals, rec.grab_ms, rec.frame_ids, info.bytes_per_frame)
            stream = {k: after[k] - before.get(k, 0) for k in after}
            row = {
                "mode": mode, "target_fps": fps, "serial": info.serial, "model": info.model,
                "link_speed": info.speed, "width": info.width, "height": info.height,
                "pixel_format": info.pixel_format, "applied_fps": info.applied_fps,
                "fps_ceiling": info.fps_ceiling, "seconds": seconds,
                "incomplete": rec.incomplete, "errors": rec.errors,
                "last_error": rec.last_error, "process_cpu_cores": cpu_cores,
                "stream_counters": ";".join(f"{k}={v}" for k, v in stream.items()),
                **stats,
            }
            row["pass"], row["reason"] = verdict(row)
            rows.append(row)
    finally:
        stop.set()
        for cam in cams:
            try:
                cam.EndAcquisition()
            except Exception:
                pass
            try:
                cam.DeInit()
            except Exception:
                pass
        # Native handles must all be dropped before the list is cleared.
        cams.clear()
        cam = None  # noqa: F841
        cam_list.Clear()
    return rows


# ----------------------------------------------------------------------
# Reporting / entry point
# ----------------------------------------------------------------------
CSV_FIELDS = [
    "mode", "target_fps", "serial", "model", "link_speed", "width", "height", "pixel_format",
    "applied_fps", "fps_ceiling", "seconds", "frames", "achieved_fps", "mb_per_s",
    "frame_gaps", "incomplete", "errors", "grab_ms_p95", "grab_ms_max",
    "interval_ms_p99", "interval_ms_max", "process_cpu_cores", "stream_counters",
    "pass", "reason", "last_error",
]


def fmt(value, spec=".2f") -> str:
    return "-" if value is None else format(value, spec)


def print_report(all_rows: list[dict]) -> None:
    print("\n" + "=" * 100)
    header = (f"{'fps':>4} {'mode':<5} {'serial':<9} {'achvd':>7} {'gaps':>5} {'incmp':>5} {'err':>4} "
              f"{'grab95':>7} {'intv99':>7} {'intvMax':>8} {'MB/s':>6} {'cpu':>5}  result")
    print(header)
    for r in all_rows:
        print(f"{r['target_fps']:>4.0f} {r['mode']:<5} {r['serial']:<9} {fmt(r['achieved_fps']):>7} "
              f"{r['frame_gaps']:>5} {r['incomplete']:>5} {r['errors']:>4} "
              f"{fmt(r['grab_ms_p95'], '.1f'):>7} {fmt(r['interval_ms_p99'], '.1f'):>7} "
              f"{fmt(r['interval_ms_max'], '.1f'):>8} {r['mb_per_s']:>6.1f} {r['process_cpu_cores']:>5.2f}  "
              f"{'PASS' if r['pass'] else 'FAIL'}: {r['reason']}")
    print("\nContention check (together vs. the same camera alone):")
    for r in all_rows:
        if r["mode"] != "both":
            continue
        solo = next((s for s in all_rows if s["mode"] == "solo"
                     and s["serial"] == r["serial"] and s["target_fps"] == r["target_fps"]), None)
        print(f"  {r['target_fps']:>4.0f} fps  {r['serial']}: {contention_note(solo, r)}")
    for r in all_rows:
        if r["stream_counters"]:
            print(f"  [{r['mode']} {r['target_fps']:.0f} fps {r['serial']}] stream counters: {r['stream_counters']}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fps", type=float, nargs="+", default=[30.0, 60.0], help="frame rates to test (default: 30 60)")
    ap.add_argument("--seconds", type=float, default=60.0, help="duration of each run (default: 60; use 600 for the real test)")
    ap.add_argument("--serials", nargs="+", help="camera serials (default: every camera found)")
    ap.add_argument("--no-copy", action="store_true", help="skip the per-frame numpy copy (isolates USB from memcpy cost)")
    ap.add_argument("--out-dir", default="probe_output")
    args = ap.parse_args()

    import numpy as np
    import PySpin

    system = PySpin.System.GetInstance()
    try:
        found = system.GetCameras()
        try:
            discovered = []
            for i in range(found.GetSize()):
                cam = found[i]
                discovered.append(_read_str(PySpin, cam.GetTLDeviceNodeMap(), "DeviceSerialNumber"))
                del cam
        finally:
            found.Clear()
        serials = args.serials or discovered
        missing = [s for s in serials if s not in discovered]
        if missing:
            print(f"Serials not found: {missing}. Discovered: {discovered}")
            return 2
        if len(serials) < 2:
            print(f"Need at least 2 cameras for a multi-camera probe; have {serials}")
            return 2

        plan = []
        for fps in args.fps:
            for serial in serials:
                plan.append((fps, "solo", [serial]))
            plan.append((fps, "both", list(serials)))
        total_s = len(plan) * (args.seconds + 3)
        print(f"Cameras: {serials}\nPlan: {len(plan)} runs x {args.seconds:.0f}s (~{total_s / 60:.0f} min)\n")

        all_rows: list[dict] = []
        for n, (fps, mode, group) in enumerate(plan, 1):
            print(f"[{n}/{len(plan)}] {mode} @ {fps:.0f} fps: {group}")
            try:
                all_rows += run_stage(PySpin, np, system, group, fps, args.seconds, not args.no_copy, mode)
            except Exception as exc:
                print(f"    run failed: {exc.__class__.__name__}: {exc}"
                      "\n    (is SpinView or the recorder GUI still holding a camera?)")
                return 1
            time.sleep(2.0)  # let the cameras fully release between runs

        print_report(all_rows)
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / f"multi_camera_probe_{datetime.now():%Y%m%d_%H%M%S}.csv"
        with open(out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"\nWrote {out}")
        return 0 if all(r["pass"] for r in all_rows if r["mode"] == "both") else 1
    finally:
        system.ReleaseInstance()


if __name__ == "__main__":
    sys.exit(main())
