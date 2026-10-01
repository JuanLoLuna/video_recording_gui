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

WRITER STAGE (--record-dir): additionally writes each camera to an AVI through
cv2.VideoWriter on its own writer thread, opened exactly like the app's
_open_segment_writer (grayscale; MJPG with a quality param, or GREY for
uncompressed). Point --record-dir at the drive you will really record to:

    python scripts/multi_camera_probe.py --record-dir E:\\probe --fps 30 --seconds 120 --codec mjpg grey

It then also reports write() time, writer-queue depth (the GUI warns at 5; this
probe FAILs at 30, i.e. ~1 s of backlog), backlog left at stop, file close
time, and the real on-disk rate and GB/hour. Recorded files are deleted after
each run unless --keep is given -- uncompressed is ~184 GB/h for two cameras
at 30 fps, so keep an eye on free space. Rotation is not exercised (one file
per camera per run). Acquisition-only baselines are not repeated in this mode.

Results are also written to probe_output/multi_camera_probe_<timestamp>.csv.
"""
from __future__ import annotations

import argparse
import csv
import sys
import queue
import shutil
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
# Writer backlog, in frames. 5 is the GUI's own compression-warning threshold
# (COMPRESSION_QUEUE_DEPTH_WARNING); ~1 s of 30 fps video means the writer is
# not keeping up and the queue (i.e. RAM) would grow for as long as it runs.
QUEUE_WARN = 5
QUEUE_FAIL = 30
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


def summarize_writer(write_ms, queue_depths, frames_grabbed, frames_written,
                     write_errors, file_bytes, span_s) -> dict:
    """Reduce one camera's writer-thread samples to the reported metrics."""
    rate = file_bytes / span_s if span_s > 0 else 0.0
    return {
        "frames_written": frames_written,
        "frames_lost_in_writer": frames_grabbed - frames_written,
        "write_errors": write_errors,
        "append_ms_mean": (sum(write_ms) / len(write_ms)) if write_ms else None,
        "append_ms_p95": percentile(write_ms, 0.95),
        "append_ms_max": max(write_ms) if write_ms else None,
        "queue_p95": percentile(queue_depths, 0.95),
        "queue_max": max(queue_depths) if queue_depths else 0,
        "file_mb": file_bytes / 1e6,
        "disk_mb_per_s": rate / 1e6,
        "gb_per_hour": rate * 3600 / 1e9,
    }


def verdict(row: dict) -> tuple[bool, str]:
    reasons = []
    notes = []
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
    # Writer fields exist only on recording runs.
    if row.get("write_errors"):
        reasons.append(f"{row['write_errors']} write errors")
    if row.get("frames_lost_in_writer"):
        reasons.append(f"{row['frames_lost_in_writer']} frames never written")
    queue_max = row.get("queue_max") or 0
    if queue_max >= QUEUE_FAIL:
        reasons.append(f"writer queue peaked at {queue_max} (not keeping up)")
    elif queue_max >= QUEUE_WARN:
        notes.append(f"warn: writer queue peaked at {queue_max}")
    if reasons:
        return (False, "; ".join(reasons + notes))
    return (True, "; ".join(notes) or "ok")


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
    # Recording runs only.
    q: "queue.Queue | None" = None
    write_ms: list = field(default_factory=list)
    queue_depths: list = field(default_factory=list)
    frames_written: int = 0
    write_errors: int = 0


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
        # perf_counter, not monotonic: on Windows time.monotonic() ticks at
        # ~15.6 ms, which quantized every interval to 0/15.6/31.2/46.8 ms.
        t0 = time.perf_counter()
        try:
            image = cam.GetNextImage(GRAB_TIMEOUT_MS)
        except Exception as exc:
            rec.errors += 1
            rec.last_error = str(exc)
            stop.wait(0.05)  # don't busy-spin on a persistent fault
            continue
        t1 = time.perf_counter()
        if image.IsIncomplete():
            rec.incomplete += 1
            image.Release()
            continue
        frame_id = None
        try:
            frame_id = int(image.GetChunkData().GetFrameID())
        except Exception:
            pass
        if copy_frames or rec.q is not None:
            arr = np.array(image.GetNDArray(), copy=True)
            if rec.q is not None:
                rec.queue_depths.append(rec.q.qsize())
                rec.q.put(arr)
        image.Release()
        rec.arrivals.append(t1)
        rec.grab_ms.append((t1 - t0) * 1000.0)
        rec.frame_ids.append(frame_id)


def open_writer(cv2, path: Path, codec: str, quality: int, fps: float, width: int, height: int):
    """cv2.VideoWriter opened the way CameraController._open_segment_writer does."""
    if codec == "mjpg":
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        params = [cv2.VIDEOWRITER_PROP_IS_COLOR, 0, cv2.VIDEOWRITER_PROP_QUALITY, int(quality)]
        writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height), params)
        if writer.isOpened():
            return writer
        writer.release()
        print(f"    note: quality={quality} not applied (backend rejected the params overload); "
              "MJPEG will use OpenCV's default quality")
    else:
        fourcc = cv2.VideoWriter_fourcc(*"GREY")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height), isColor=False)
    if not writer.isOpened():
        writer.release()
        raise RuntimeError(f"cv2.VideoWriter could not open {path} ({codec})")
    return writer


def write_loop(rec: Recorder, writer) -> None:
    """Mirrors the app's append thread: one blocking write() per queued frame."""
    while True:
        arr = rec.q.get()
        try:
            if arr is None:
                return
            t0 = time.perf_counter()
            writer.write(arr)
            rec.write_ms.append((time.perf_counter() - t0) * 1000.0)
            rec.frames_written += 1
        except Exception as exc:
            rec.write_errors += 1
            rec.last_error = f"write: {exc}"
        finally:
            rec.q.task_done()


def run_stage(PySpin, np, system, serials, fps, seconds, copy_frames, mode, record=None) -> list[dict]:
    """Acquire from `serials` simultaneously for `seconds`; one result row each.

    `record` (optional): {"dir": Path, "codec": "mjpg"|"grey", "quality": int,
    "keep": bool, "cv2": module}. When set, every camera also writes an AVI
    through a cv2.VideoWriter on its own writer thread, like the app does.
    """
    cam_list = system.GetCameras()
    cams, infos, recs, threads = [], [], [], []
    writers, wthreads, paths = [], [], []
    drain_s = close_s = 0.0
    counters_after = []
    stop = threading.Event()
    rows: list[dict] = []
    try:
        for serial in serials:
            cam = cam_list.GetBySerial(serial)
            cam.Init()
            cams.append(cam)
            infos.append(configure(PySpin, cam, fps))
            recs.append(Recorder(serial))
        if record:
            for info, rec in zip(infos, recs):
                path = record["dir"] / f"{mode}_{info.serial}_{int(fps)}fps.avi"
                writers.append(open_writer(record["cv2"], path, record["codec"], record["quality"],
                                           info.applied_fps, info.width, info.height))
                paths.append(path)
                rec.q = queue.Queue()
                t = threading.Thread(target=write_loop, args=(rec, writers[-1]), daemon=True)
                t.start()
                wthreads.append(t)
        for info in infos:
            print(f"    {info.serial} {info.model}: {info.width}x{info.height} {info.pixel_format}, "
                  f"link={info.speed}, applied {info.applied_fps:.2f} fps "
                  f"(ceiling {info.fps_ceiling:.1f}), "
                  f"{info.bytes_per_frame * info.applied_fps / 1e6:.1f} MB/s, "
                  f"buffers={info.buffer_count}, link_limit={info.link_limit}")
            if info.speed not in ("SuperSpeed", "?"):
                print(f"    !! {info.serial} is on {info.speed}, not SuperSpeed -- check cable/port")

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

        file_sizes = [0] * len(recs)
        if record:
            # Backlog still queued when capture stops = how far behind the writer was.
            t_drain = time.perf_counter()
            for rec in recs:
                rec.q.join()
            drain_s = time.perf_counter() - t_drain
            for rec in recs:
                rec.q.put(None)
            for t in wthreads:
                t.join(timeout=10.0)
            t_close = time.perf_counter()
            for w in writers:
                w.release()
            close_s = time.perf_counter() - t_close
            writers.clear()
            file_sizes = [p.stat().st_size if p.exists() else 0 for p in paths]

        for idx, (info, rec, after) in enumerate(zip(infos, recs, counters_after)):
            stats = summarize(rec.arrivals, rec.grab_ms, rec.frame_ids, info.bytes_per_frame)
            # These reset on BeginAcquisition, so the post-run value is the
            # per-run total; subtracting the pre-run read would be meaningless.
            stream = dict(after)
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
            if record:
                span = (rec.arrivals[-1] - rec.arrivals[0]) if len(rec.arrivals) > 1 else 0.0
                row.update(summarize_writer(
                    rec.write_ms, rec.queue_depths, len(rec.arrivals), rec.frames_written,
                    rec.write_errors, file_sizes[idx], span))
                row.update({"codec": record["codec"], "drain_s": drain_s, "close_s": close_s})
            row["pass"], row["reason"] = verdict(row)
            rows.append(row)
    finally:
        stop.set()
        for w in writers:  # only non-empty if we failed before the clean shutdown above
            try:
                w.release()
            except Exception:
                pass
        if record and not record["keep"]:
            for path in paths:
                path.unlink(missing_ok=True)
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
    "codec", "frames_written", "frames_lost_in_writer", "write_errors", "append_ms_mean",
    "append_ms_p95", "append_ms_max", "queue_p95", "queue_max", "file_mb", "disk_mb_per_s",
    "gb_per_hour", "drain_s", "close_s",
    "pass", "reason", "last_error",
]


def fmt(value, spec=".2f") -> str:
    return "-" if value is None else format(value, spec)


def print_report(all_rows: list[dict]) -> None:
    print("\n" + "=" * 100)
    header = (f"{'fps':>4} {'mode':<9} {'serial':<9} {'achvd':>7} {'gaps':>5} {'incmp':>5} {'err':>4} "
              f"{'grab95':>7} {'intv99':>7} {'intvMax':>8} {'MB/s':>6} {'cpu':>5}  result")
    print(header)
    for r in all_rows:
        print(f"{r['target_fps']:>4.0f} {r['mode']:<9} {r['serial']:<9} {fmt(r['achieved_fps']):>7} "
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
    rec_rows = [r for r in all_rows if r.get("codec")]
    if rec_rows:
        print("\nWriter stage (cv2.VideoWriter per camera, same settings as the app):")
        print(f"{'fps':>4} {'codec':<5} {'serial':<9} {'app_mean':>8} {'app_p95':>7} {'app_max':>7} "
              f"{'q_p95':>5} {'q_max':>5} {'written':>8} {'lost':>5} {'disk MB/s':>9} {'GB/h':>6} "
              f"{'drain_s':>7} {'close_s':>7}")
        for r in rec_rows:
            print(f"{r['target_fps']:>4.0f} {r['codec']:<5} {r['serial']:<9} "
                  f"{fmt(r['append_ms_mean'], '.1f'):>8} {fmt(r['append_ms_p95'], '.1f'):>7} "
                  f"{fmt(r['append_ms_max'], '.1f'):>7} {fmt(r['queue_p95'], '.0f'):>5} {r['queue_max']:>5} "
                  f"{r['frames_written']:>8} {r['frames_lost_in_writer']:>5} {r['disk_mb_per_s']:>9.1f} "
                  f"{r['gb_per_hour']:>6.1f} {r['drain_s']:>7.2f} {r['close_s']:>7.2f}")
        print("  app_* = write() time per frame; q_* = frames waiting in the writer queue "
              "(GUI warns at 5); drain_s = backlog left when capture stopped.")
    for r in all_rows:
        if r["stream_counters"]:
            print(f"  [{r['mode']} {r['target_fps']:.0f} fps {r['serial']}] stream counters: {r['stream_counters']}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fps", type=float, nargs="+", default=[30.0, 60.0], help="frame rates to test (default: 30 60)")
    ap.add_argument("--seconds", type=float, default=60.0, help="duration of each run (default: 60; use 600 for the real test)")
    ap.add_argument("--serials", nargs="+", help="camera serials (default: every camera found)")
    ap.add_argument("--no-solo", action="store_true", help="skip the per-camera solo baselines (no contention comparison)")
    ap.add_argument("--no-copy", action="store_true", help="skip the per-frame numpy copy (isolates USB from memcpy cost)")
    ap.add_argument("--out-dir", default="probe_output")
    ap.add_argument("--record-dir", help="writer stage: also write AVIs to this folder (put it on the drive you will record to)")
    ap.add_argument("--codec", nargs="+", choices=["mjpg", "grey"], default=["mjpg"],
                    help="writer stage codecs (default: mjpg). grey = uncompressed, ~184 GB/h for two cameras at 30 fps")
    ap.add_argument("--quality", type=int, default=75, help="MJPEG quality 0-100 (app default 75)")
    ap.add_argument("--keep", action="store_true", help="keep the recorded AVIs (default: delete after each run)")
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

        record_dir = None
        cv2 = None
        if args.record_dir:
            import cv2
            record_dir = Path(args.record_dir) / f"probe_{datetime.now():%Y%m%d_%H%M%S}"
            record_dir.mkdir(parents=True, exist_ok=True)
            usage = shutil.disk_usage(record_dir)
            print(f"Recording to {record_dir}  (free {usage.free / 1e9:.0f} GB of {usage.total / 1e9:.0f} GB)")

        plan = []
        for fps in args.fps:
            if record_dir is None:
                for serial in ([] if args.no_solo else serials):
                    plan.append((fps, "solo", [serial], None))
                plan.append((fps, "both", list(serials), None))
            else:  # writer stage: acquisition-only baselines already covered by a plain run
                for codec in args.codec:
                    plan.append((fps, f"rec-{codec}", list(serials), codec))
        total_s = len(plan) * (args.seconds + 3)
        print(f"Cameras: {serials}\nPlan: {len(plan)} runs x {args.seconds:.0f}s (~{total_s / 60:.0f} min)\n")

        all_rows: list[dict] = []
        for n, (fps, mode, group, codec) in enumerate(plan, 1):
            print(f"[{n}/{len(plan)}] {mode} @ {fps:.0f} fps: {group}")
            record = None if codec is None else {
                "dir": record_dir, "codec": codec, "quality": args.quality, "keep": args.keep, "cv2": cv2}
            try:
                all_rows += run_stage(PySpin, np, system, group, fps, args.seconds,
                                      not args.no_copy, mode, record)
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
        if record_dir is not None and not args.keep:
            try:
                record_dir.rmdir()
            except OSError:
                pass
        return 0 if all(r["pass"] for r in all_rows if r["mode"] != "solo") else 1
    finally:
        system.ReleaseInstance()


if __name__ == "__main__":
    sys.exit(main())
