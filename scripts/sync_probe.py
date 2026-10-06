#!/usr/bin/env python3
"""Sync probe (plan step 1): what can the cameras do for sync, and how far apart
are their frames today?

Run ON THE RECORDING LAPTOP (needs PySpin), with the recorder GUI and SpinView
CLOSED -- either one holds a camera open and Init() will fail. Step-by-step
instructions: docs/runbooks/2026-10-05-sync-probe-step1.md

    python scripts/sync_probe.py nodes                     # GPIO / trigger / timestamp nodes, ~10 s
    python scripts/sync_probe.py skew                      # 30 and 60 fps, 10 min each
    python scripts/sync_probe.py skew --fps 60 --seconds 60   # quick check
    python scripts/sync_probe.py trigger                   # both cameras on the Raspberry Pi trigger

TRIGGER (plan Part A, option H2): both cameras are put in hardware-trigger
mode on the line wired to the Pi (default: Firefly Line2, Blackfly S Line3;
override with --line SERIAL=LINE), armed, and then the Pi
(pi_trigger/pi_trigger_server.py, found automatically over USB serial) sends
the pulses. Reports, per camera: frames vs pulses sent (missed triggers),
frames that arrived BEFORE the pulses started (stray triggers = wiring noise),
and the frame-interval jitter from the camera's own clock; per pair: the gap
between the two cameras' frames of the same pulse. The cameras are ALWAYS put
back in free-running mode at the end, also when the run fails -- a camera left
in trigger mode would make the recorder wait for pulses that never come.

NODES (read-only except the Line/Trigger *selectors*, which are put back): per
camera, every GPIO line with its allowed LineMode / LineSource / format, the
FrameStart trigger's allowed sources, activations and overlap modes, the
timestamp-latch nodes, and the sensor settings that intrinsics depend on. This
is what decides how the trigger cable can be wired (plan Q1/Q3).

SKEW: both cameras free-run exactly as the app runs them (same configure() as
scripts/multi_camera_probe.py, nothing is written to disk except the CSVs).
Every frame's chunk Timestamp/FrameID and host arrival time is kept, and once
per second each camera's clock is latched (TimestampLatch ->
TimestampLatchValue) between two perf_counter() reads. From that it reports:

  - the camera tick unit (expected 1 ns) and each camera clock's drift vs the
    laptop, in ppm;
  - the gap between the two cameras' exposures (camera B's frame vs the
    nearest camera A frame, on the laptop timeline): at start, at end, how
    fast it drifts and its spread -- this is the number for plan Q4;
  - how jittery host arrival time (the GUI's system_time) is as a clock,
    i.e. how well pairing frames by system_time alone could work;
  - frame gaps / incomplete / errors (should all be 0, as in the
    multi-camera probe).

Everything goes to probe_output/sync_probe_<timestamp>/: report.txt (exactly
what was printed), nodes.json or summary.json, and for skew the raw frames +
latch CSV per camera and fps and a phase CSV per fps. Send that whole folder
back -- no screenshots needed.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from multi_camera_probe import (
    GRAB_TIMEOUT_MS,
    _read_str,
    _set_enum,
    configure,
    count_frame_gaps,
    percentile,
)

LATCH_INTERVAL_S = 1.0
# Latch samples whose perf_counter bracket is in the widest (1 - this) part
# are dropped before fitting: a wide bracket means the USB command was slow,
# so the midpoint is a poor estimate of when the latch happened.
LATCH_KEEP_FRACTION = 0.5

SENSOR_NODES = (
    "Width", "Height", "OffsetX", "OffsetY", "BinningHorizontal", "BinningVertical",
    "DecimationHorizontal", "DecimationVertical", "ReverseX", "ReverseY", "PixelFormat",
    "SensorWidth", "SensorHeight",
)
TIMESTAMP_NODES = (
    "TimestampLatch", "TimestampLatchValue", "TimestampReset", "TimestampIncrement",
    "DeviceClockFrequency", "GevTimestampTickFrequency",
)
CAMERA_NODES = (
    "DeviceFirmwareVersion", "AcquisitionFrameRate", "AcquisitionFrameRateEnable",
    "ExposureAuto", "ExposureTime", "ExposureMode", "TriggerMode", "DeviceLinkThroughputLimit",
)
LINE_NODES = ("LineMode", "LineSource", "LineFormat", "LineInverter", "LineStatus", "V3_3Enable")
# Which camera input the Pi's pulse is wired to, by model (see the wiring runbook):
# Firefly S JST pin 3 (white) = Line2; Blackfly S Hirose pin 1 (green) = Line3.
DEFAULT_TRIGGER_LINES = {"Firefly": "Line2", "Blackfly": "Line3"}
# Exposure is capped to this fraction of the trigger period: the Firefly has no
# TriggerOverlap, so exposure + readout must end before the next pulse.
TRIGGER_EXPOSURE_FRACTION = 0.5
TRIGGER_NODES = ("TriggerMode", "TriggerSource", "TriggerActivation", "TriggerOverlap", "TriggerDelay")


# --------------------------------------------------------------------------
# Pure analysis (no PySpin; tested in tests/test_sync_probe.py)
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class ClockFit:
    """host_s = host0 + slope * (ticks - tick0)."""
    tick0: int
    host0: float
    slope: float          # host seconds per camera tick
    residual_us: float    # RMS residual of the kept samples
    samples_used: int
    samples_total: int
    bracket_us_p50: float

    def to_host(self, ticks: int) -> float:
        return self.host0 + self.slope * (ticks - self.tick0)

    @property
    def tick_ns(self) -> float:
        return self.slope * 1e9

    def drift_ppm(self, nominal_tick_ns: float = 1.0) -> float:
        return (self.tick_ns / nominal_tick_ns - 1.0) * 1e6


def fit_clock(latches: list[tuple[float, float, int]],
              keep_fraction: float = LATCH_KEEP_FRACTION) -> ClockFit | None:
    """Fit camera ticks -> host seconds from (host_before, host_after, ticks) samples."""
    samples = [(b, a, t) for b, a, t in latches if t is not None and a >= b]
    if len(samples) < 3:
        return None
    brackets = sorted(a - b for b, a, _ in samples)
    cutoff = brackets[max(0, min(len(brackets) - 1, math.ceil(len(brackets) * keep_fraction) - 1))]
    kept = [(0.5 * (b + a), t) for b, a, t in samples if a - b <= cutoff]
    if len(kept) < 3:
        kept = [(0.5 * (b + a), t) for b, a, t in samples]
    tick0 = kept[0][1]
    xs = [float(t - tick0) for _, t in kept]
    ys = [h for h, _ in kept]
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return None
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    host0 = my - slope * mx
    resid = [y - (host0 + slope * x) for x, y in zip(xs, ys)]
    rms = math.sqrt(statistics.fmean(r * r for r in resid))
    return ClockFit(
        tick0=tick0, host0=host0, slope=slope, residual_us=rms * 1e6,
        samples_used=len(kept), samples_total=len(samples),
        bracket_us_p50=statistics.median(brackets) * 1e6,
    )


def tick_ns_from_frames(ticks: list[int], fps: float) -> float | None:
    """Camera tick length implied by the median frame-to-frame timestamp step."""
    diffs = [b - a for a, b in zip(ticks, ticks[1:]) if b > a]
    if not diffs or fps <= 0:
        return None
    return (1e9 / fps) / statistics.median(diffs)


def phase_offsets(times_a: list[float], times_b: list[float], period_s: float) -> list[tuple[float, float]]:
    """For each B frame: (time, B minus nearest A frame), wrapped into [-period/2, period/2)."""
    if not times_a or not times_b or period_s <= 0:
        return []
    out = []
    j = 0
    for tb in times_b:
        while j + 1 < len(times_a) and times_a[j + 1] <= tb:
            j += 1
        best = min((tb - times_a[k] for k in (j, j + 1) if k < len(times_a)), key=abs)
        wrapped = (best + period_s / 2) % period_s - period_s / 2
        out.append((tb, wrapped))
    return out


def summarize_phase(offsets: list[tuple[float, float]], period_s: float) -> dict:
    """Gap between the cameras: start/end, drift rate (unwrapped), spread."""
    if len(offsets) < 2:
        return {}
    # Unwrap so a gap drifting through +-period/2 reads as a steady drift.
    unwrapped = [offsets[0][1]]
    for (_, prev), (_, cur) in zip(offsets, offsets[1:]):
        step = cur - prev
        step -= period_s * round(step / period_s)
        unwrapped.append(unwrapped[-1] + step)
    t0 = offsets[0][0]
    xs = [t - t0 for t, _ in offsets]
    mx, my = statistics.fmean(xs), statistics.fmean(unwrapped)
    sxx = sum((x - mx) ** 2 for x in xs)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, unwrapped)) / sxx if sxx > 0 else 0.0
    abs_ms = [abs(d) * 1e3 for _, d in offsets]
    drift_ms_per_min = slope * 60.0 * 1e3
    return {
        "gap_ms_start": offsets[0][1] * 1e3,
        "gap_ms_end": offsets[-1][1] * 1e3,
        "drift_ms_per_min": drift_ms_per_min,
        # Below ~1 us per hour the gap is constant for any practical purpose.
        "full_cycle_min": (period_s * 1e3 / abs(drift_ms_per_min)) if abs(drift_ms_per_min) > 1e-5 else None,
        "abs_gap_ms_p50": percentile(abs_ms, 0.5),
        "abs_gap_ms_p99": percentile(abs_ms, 0.99),
        "abs_gap_ms_max": max(abs_ms),
        "half_period_ms": period_s * 1e3 / 2,
    }


# Frames queued between BeginAcquisition and the grab thread starting arrive
# in a burst; they say nothing about steady-state jitter, so they are reported
# separately instead of setting the max.
START_BURST_S = 2.0


def arrival_jitter_ms(host_arrivals: list[float], mapped: list[float],
                      start_burst_s: float = START_BURST_S) -> dict:
    """Host arrival minus exposure time (mapped camera clock): the pipeline latency."""
    lat = [(h - m) * 1e3 for h, m in zip(host_arrivals, mapped)]
    if not lat:
        return {}
    t0 = host_arrivals[0]
    steady = [x for x, h in zip(lat, host_arrivals) if h - t0 >= start_burst_s] or lat
    base = min(steady)
    burst = [x for x, h in zip(lat, host_arrivals) if h - t0 < start_burst_s]
    return {
        "latency_ms_p50": percentile(steady, 0.5) - base,
        "latency_ms_p99": percentile(steady, 0.99) - base,
        "latency_ms_max": max(steady) - base,
        "start_burst_ms_max": (max(burst) - base) if burst else None,
        "note": f"relative to the fastest frame, after the first {start_burst_s:g} s; "
                "spread = how noisy system_time is as a clock",
    }


def interval_stats_us(ticks: list[int], tick_s: float, period_s: float) -> dict:
    """Frame-to-frame intervals on the camera's own clock vs the trigger period."""
    dt = [(b - a) * tick_s for a, b in zip(ticks, ticks[1:]) if b > a]
    if not dt:
        return {}
    dev = [(d - period_s) * 1e6 for d in dt]
    missed = sum(1 for d in dt if d > 1.5 * period_s)
    return {
        "mean_period_us": statistics.fmean(dt) * 1e6,
        "std_us": statistics.pstdev(dt) * 1e6,
        "max_abs_dev_us": max(abs(x) for x in dev if abs(x) < 0.5 * period_s * 1e6) if any(
            abs(x) < 0.5 * period_s * 1e6 for x in dev) else None,
        "intervals_over_1_5_periods": missed,
    }


def paired_gaps_us(mapped_a: list[float], mapped_b: list[float], period_s: float) -> dict:
    """Same-pulse gap between two triggered cameras: B frame minus the nearest A frame, in us.

    With a shared trigger the gap is a constant offset (the two models start
    exposing at slightly different delays after the edge, and their chunk
    timestamps may mark different moments) plus noise; free-running it is
    anything up to half a period.
    """
    offsets = phase_offsets(mapped_a, mapped_b, period_s)
    if not offsets:
        return {}
    gaps = [d * 1e6 for _, d in offsets]
    mean = statistics.fmean(gaps)
    spread = [abs(g - mean) for g in gaps]
    return {
        "pairs": len(gaps),
        "mean_gap_us": mean,
        "abs_gap_us_p50": percentile([abs(g) for g in gaps], 0.5),
        "abs_gap_us_p99": percentile([abs(g) for g in gaps], 0.99),
        "abs_gap_us_max": max(abs(g) for g in gaps),
        "spread_us_p99": percentile(spread, 0.99),
        "spread_us_max": max(spread),
    }


def trigger_report(result: dict, runs, fps: float, pulses_sent: int | None, before_start: dict) -> dict:
    """Add the trigger-specific numbers to an analyse() result (mutates and returns it)."""
    period = 1.0 / fps
    mapped = {}
    for run in runs:
        cam = result["cameras"][run.serial]
        fit = fit_clock(run.latches) if run.latches else None
        tick_s = fit.slope if fit is not None else 1e-9
        ticks = [t for _, _, t in run.frames if t is not None]
        cam["frames_before_start"] = before_start.get(run.serial, 0)
        cam["pulses_sent"] = pulses_sent
        cam["missed_triggers"] = None if pulses_sent is None else pulses_sent - len(run.frames)
        cam["intervals"] = interval_stats_us(ticks, tick_s, period)
        mapped[run.serial] = ([fit.to_host(t) for t in ticks] if fit is not None
                              else [h for h, _, _ in run.frames])
    if len(runs) >= 2:
        a, b = runs[0].serial, runs[1].serial
        result["trigger_pair"] = {"a": a, "b": b, **paired_gaps_us(mapped[a], mapped[b], period)}
    return result


# --------------------------------------------------------------------------
# Node dump
# --------------------------------------------------------------------------

def _interface(PySpin, node):
    try:
        return node.GetPrincipalInterfaceType()
    except Exception:
        return None


def describe_node(PySpin, nodemap, name: str) -> dict | None:
    """Present/readable/writable, current value and (enums) allowed entries. Never raises."""
    node = nodemap.GetNode(name)
    if node is None:
        return None
    info = {
        "available": bool(PySpin.IsAvailable(node)),
        "readable": bool(PySpin.IsReadable(node)),
        "writable": bool(PySpin.IsWritable(node)),
    }
    kind = _interface(PySpin, node)
    try:
        if kind == PySpin.intfIEnumeration:
            ptr = PySpin.CEnumerationPtr(node)
            info["type"] = "enum"
            entries = []
            for entry in ptr.GetEntries():
                entry = PySpin.CEnumEntryPtr(entry)
                if PySpin.IsAvailable(entry):
                    entries.append(entry.GetSymbolic())
            info["entries"] = entries
            if info["readable"]:
                info["value"] = ptr.GetCurrentEntry().GetSymbolic()
        elif kind == PySpin.intfIBoolean:
            info["type"] = "bool"
            if info["readable"]:
                info["value"] = bool(PySpin.CBooleanPtr(node).GetValue())
        elif kind == PySpin.intfIInteger:
            ptr = PySpin.CIntegerPtr(node)
            info["type"] = "int"
            if info["readable"]:
                info.update(value=int(ptr.GetValue()), min=int(ptr.GetMin()), max=int(ptr.GetMax()))
        elif kind == PySpin.intfIFloat:
            ptr = PySpin.CFloatPtr(node)
            info["type"] = "float"
            if info["readable"]:
                info.update(value=float(ptr.GetValue()), min=float(ptr.GetMin()), max=float(ptr.GetMax()))
        elif kind == PySpin.intfIString:
            info["type"] = "string"
            if info["readable"]:
                info["value"] = PySpin.CStringPtr(node).GetValue()
        elif kind == PySpin.intfICommand:
            info["type"] = "command"
        else:
            info["type"] = str(kind)
    except Exception as exc:
        info["error"] = str(exc)
    return info


def _select(PySpin, nodemap, selector: str, entry: str) -> bool:
    try:
        ptr = PySpin.CEnumerationPtr(nodemap.GetNode(selector))
        if not PySpin.IsWritable(ptr):
            return False
        ptr.SetIntValue(ptr.GetEntryByName(entry).GetValue())
        return True
    except Exception:
        return False


def dump_camera_nodes(PySpin, cam) -> dict:
    tl = cam.GetTLDeviceNodeMap()
    nm = cam.GetNodeMap()
    out = {
        "serial": _read_str(PySpin, tl, "DeviceSerialNumber"),
        "model": _read_str(PySpin, tl, "DeviceModelName"),
        "link_speed": _read_str(PySpin, tl, "DeviceCurrentSpeed"),
        "camera": {n: describe_node(PySpin, nm, n) for n in CAMERA_NODES},
        "sensor": {n: describe_node(PySpin, nm, n) for n in SENSOR_NODES},
        "timestamp": {n: describe_node(PySpin, nm, n) for n in TIMESTAMP_NODES},
        "lines": {},
        "trigger": {},
    }

    line_sel = describe_node(PySpin, nm, "LineSelector")
    if line_sel and line_sel.get("entries"):
        original = line_sel.get("value")
        for line in line_sel["entries"]:
            if _select(PySpin, nm, "LineSelector", line):
                out["lines"][line] = {n: describe_node(PySpin, nm, n) for n in LINE_NODES}
            else:
                out["lines"][line] = {"error": "LineSelector not writable"}
        if original:
            _select(PySpin, nm, "LineSelector", original)

    trig_sel = describe_node(PySpin, nm, "TriggerSelector")
    out["trigger"]["TriggerSelector"] = trig_sel
    if trig_sel and "FrameStart" in (trig_sel.get("entries") or []):
        original = trig_sel.get("value")
        if _select(PySpin, nm, "TriggerSelector", "FrameStart"):
            out["trigger"]["FrameStart"] = {n: describe_node(PySpin, nm, n) for n in TRIGGER_NODES}
        if original:
            _select(PySpin, nm, "TriggerSelector", original)
    return out


def print_nodes(cam: dict) -> None:
    def val(d):
        if not d:
            return "-"
        if "value" in d:
            return f"{d['value']}"
        return d.get("type", "?")

    print(f"\n=== {cam['model']} #{cam['serial']} ({cam['link_speed']}) ===")
    fw = cam["camera"].get("DeviceFirmwareVersion")
    print(f"  firmware: {val(fw)}")
    s = cam["sensor"]
    print(f"  sensor: {val(s.get('Width'))}x{val(s.get('Height'))} offset {val(s.get('OffsetX'))},"
          f"{val(s.get('OffsetY'))} binning {val(s.get('BinningHorizontal'))}x{val(s.get('BinningVertical'))}"
          f" {val(s.get('PixelFormat'))}")
    ts = cam["timestamp"]
    latch_ok = bool(ts.get("TimestampLatch")) and bool(ts.get("TimestampLatchValue"))
    print(f"  timestamp latch: {'yes' if latch_ok else 'NO'}"
          f"  increment: {val(ts.get('TimestampIncrement'))}")
    print("  GPIO lines:")
    if not cam["lines"]:
        print("    (no LineSelector)")
    for line, nodes in cam["lines"].items():
        if "error" in nodes:
            print(f"    {line}: {nodes['error']}")
            continue
        mode = nodes.get("LineMode") or {}
        source = nodes.get("LineSource") or {}
        fmt = nodes.get("LineFormat") or {}
        print(f"    {line}: format={val(fmt)} mode={val(mode)} (allowed {mode.get('entries', '-')})")
        if source:
            print(f"      LineSource={val(source)} allowed {source.get('entries', '-')}")
        if nodes.get("V3_3Enable"):
            print(f"      3.3V output enable: {val(nodes['V3_3Enable'])}")
    fs = cam["trigger"].get("FrameStart")
    if fs:
        print("  FrameStart trigger:")
        for name in ("TriggerSource", "TriggerActivation", "TriggerOverlap"):
            node = fs.get(name) or {}
            print(f"    {name}: {val(node)} allowed {node.get('entries', '-')}")
        delay = fs.get("TriggerDelay") or {}
        if "min" in delay:
            print(f"    TriggerDelay: {delay['min']:.0f}..{delay['max']:.0f} us")
    else:
        print("  FrameStart trigger: not available")


# --------------------------------------------------------------------------
# Skew run
# --------------------------------------------------------------------------

@dataclass
class CamRun:
    serial: str
    cam: object
    applied_fps: float = 0.0
    model: str = "?"
    frames: list = field(default_factory=list)    # (host_arrival, frame_id, ticks)
    latches: list = field(default_factory=list)   # (host_before, host_after, ticks)
    incomplete: int = 0
    errors: int = 0
    latch_errors: int = 0
    last_error: str = ""
    latch_supported: bool = False


def grab_loop(run: CamRun, stop: threading.Event) -> None:
    while not stop.is_set():
        try:
            image = run.cam.GetNextImage(GRAB_TIMEOUT_MS)
        except Exception as exc:
            run.errors += 1
            run.last_error = str(exc)
            stop.wait(0.05)
            continue
        arrived = time.perf_counter()
        if image.IsIncomplete():
            run.incomplete += 1
            image.Release()
            continue
        frame_id = ticks = None
        try:
            chunk = image.GetChunkData()
            frame_id = int(chunk.GetFrameID())
            ticks = int(chunk.GetTimestamp())
        except Exception:
            pass
        image.Release()
        run.frames.append((arrived, frame_id, ticks))


def latch_once(PySpin, run: CamRun) -> None:
    nm = run.cam.GetNodeMap()
    cmd = PySpin.CCommandPtr(nm.GetNode("TimestampLatch"))
    value = PySpin.CIntegerPtr(nm.GetNode("TimestampLatchValue"))
    before = time.perf_counter()
    cmd.Execute()
    after = time.perf_counter()
    run.latches.append((before, after, int(value.GetValue())))


def latch_loop(PySpin, runs: list[CamRun], stop: threading.Event, interval_s: float) -> None:
    while not stop.wait(interval_s):
        for run in runs:
            if not run.latch_supported:
                continue
            try:
                latch_once(PySpin, run)
            except Exception as exc:
                run.latch_errors += 1
                run.last_error = f"latch: {exc}"


def latch_supported(PySpin, cam) -> bool:
    nm = cam.GetNodeMap()
    cmd, value = nm.GetNode("TimestampLatch"), nm.GetNode("TimestampLatchValue")
    return cmd is not None and value is not None and PySpin.IsWritable(cmd) and PySpin.IsReadable(value)


def analyse(runs: list[CamRun], fps: float) -> dict:
    result = {"fps_requested": fps, "cameras": {}, "pair": {}}
    mapped_by_serial = {}
    for run in runs:
        ticks = [t for _, _, t in run.frames if t is not None]
        fit = fit_clock(run.latches) if run.latches else None
        cam = {
            "model": run.model,
            "applied_fps": run.applied_fps,
            "frames": len(run.frames),
            "frame_gaps": count_frame_gaps([f for _, f, _ in run.frames]),
            "incomplete": run.incomplete,
            "errors": run.errors,
            "latch_samples": len(run.latches),
            "latch_errors": run.latch_errors,
            "tick_ns_from_frames": tick_ns_from_frames(ticks, run.applied_fps),
            "last_error": run.last_error,
        }
        # Camera timestamps, not host arrival: the start burst shortens the
        # host span and overstated the rate (Firefly read 30.003 at a real 29.989).
        if len(ticks) >= 2 and ticks[-1] > ticks[0]:
            tick_s = fit.slope if fit is not None else 1e-9
            cam["achieved_fps"] = (len(ticks) - 1) / ((ticks[-1] - ticks[0]) * tick_s)
        elif run.frames:
            span = run.frames[-1][0] - run.frames[0][0]
            cam["achieved_fps"] = (len(run.frames) - 1) / span if span > 0 else None
        if fit is not None:
            cam.update(
                tick_ns_from_latch=fit.tick_ns,
                clock_drift_ppm_vs_laptop=fit.drift_ppm(),
                latch_fit_residual_us=fit.residual_us,
                latch_bracket_us_p50=fit.bracket_us_p50,
                timeline="camera clock mapped to laptop (latch fit)",
            )
            usable = [(h, t) for h, _, t in run.frames if t is not None]
            mapped = [fit.to_host(t) for _, t in usable]
            cam["arrival"] = arrival_jitter_ms([h for h, _ in usable], mapped)
        else:
            mapped = [h for h, _, _ in run.frames]
            cam["timeline"] = "host arrival only (no latch) -- gap numbers include arrival jitter"
        mapped_by_serial[run.serial] = mapped
        result["cameras"][run.serial] = cam

    if len(runs) >= 2:
        a, b = runs[0], runs[1]
        period = 1.0 / a.applied_fps if a.applied_fps > 0 else 1.0 / fps
        offsets = phase_offsets(mapped_by_serial[a.serial], mapped_by_serial[b.serial], period)
        result["pair"] = {"a": a.serial, "b": b.serial, **summarize_phase(offsets, period)}
        result["_offsets"] = offsets
    return result


def write_raw(out_dir: Path, runs: list[CamRun], fps: float, offsets, tag_prefix: str = "") -> None:
    tag = f"{tag_prefix}{fps:g}fps"
    for run in runs:
        with open(out_dir / f"{tag}_{run.serial}_frames.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["host_arrival_s", "camera_frame_id", "camera_timestamp_ticks"])
            w.writerows(run.frames)
        with open(out_dir / f"{tag}_{run.serial}_latch.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["host_before_s", "host_after_s", "camera_timestamp_ticks"])
            w.writerows(run.latches)
    if offsets:
        with open(out_dir / f"{tag}_phase.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["host_time_s", "gap_b_minus_a_ms"])
            w.writerows((t, d * 1e3) for t, d in offsets)


def run_skew(PySpin, system, serials: list[str], fps: float, seconds: float, out_dir: Path) -> dict:
    cam_list = system.GetCameras()
    runs: list[CamRun] = []
    stop = threading.Event()
    threads: list[threading.Thread] = []
    try:
        for serial in serials:
            cam = cam_list.GetBySerial(serial)
            cam.Init()
            run = CamRun(serial=serial, cam=cam)
            info = configure(PySpin, cam, fps)
            run.applied_fps, run.model = info.applied_fps, info.model
            run.latch_supported = latch_supported(PySpin, cam)
            print(f"    {serial} {info.model}: {info.width}x{info.height}, applied {info.applied_fps:.3f} fps, "
                  f"link={info.speed}, timestamp latch={'yes' if run.latch_supported else 'NO'}")
            runs.append(run)

        for run in runs:
            run.cam.BeginAcquisition()
        for run in runs:
            t = threading.Thread(target=grab_loop, args=(run, stop), daemon=True)
            t.start()
            threads.append(t)
        # A few latches right away so short runs still get a fit.
        for _ in range(3):
            for run in runs:
                if run.latch_supported:
                    try:
                        latch_once(PySpin, run)
                    except Exception as exc:
                        run.latch_errors += 1
                        run.last_error = f"latch: {exc}"
            time.sleep(0.05)
        lt = threading.Thread(target=latch_loop, args=(PySpin, runs, stop, LATCH_INTERVAL_S), daemon=True)
        lt.start()
        threads.append(lt)

        deadline = time.monotonic() + seconds
        next_report = time.monotonic() + 30.0
        while time.monotonic() < deadline:
            time.sleep(0.5)
            if time.monotonic() >= next_report:
                left = deadline - time.monotonic()
                counts = ", ".join(f"{r.serial}: {len(r.frames)} frames" for r in runs)
                print(f"    {left / 60:.1f} min left -- {counts}")
                next_report += 30.0
    finally:
        stop.set()
        for t in threads:
            t.join(timeout=5.0)
        for run in runs:
            try:
                run.cam.EndAcquisition()
            except Exception:
                pass
            try:
                run.cam.DeInit()
            except Exception:
                pass
            run.cam = None
        cam_list.Clear()

    result = analyse(runs, fps)
    write_raw(out_dir, runs, fps, result.pop("_offsets", None))
    return result


def set_trigger(PySpin, cam, line: str, fps: float) -> dict:
    """Hardware trigger on `line` (rising edge, FrameStart); exposure capped for the trigger period."""
    nm = cam.GetNodeMap()
    _set_enum(PySpin, nm, "TriggerMode", "Off")  # trigger settings are only writable with the mode off
    enable = PySpin.CBooleanPtr(nm.GetNode("AcquisitionFrameRateEnable"))
    if PySpin.IsWritable(enable):
        enable.SetValue(False)  # the pulses set the rate now
    _set_enum(PySpin, nm, "LineSelector", line)
    _set_enum(PySpin, nm, "LineMode", "Input")  # Firefly lines are bidirectional; input-only lines ignore this
    _set_enum(PySpin, nm, "TriggerSelector", "FrameStart")
    if not _set_enum(PySpin, nm, "TriggerSource", line):
        raise RuntimeError(f"TriggerSource {line} not accepted")
    _set_enum(PySpin, nm, "TriggerActivation", "RisingEdge")
    overlap = _set_enum(PySpin, nm, "TriggerOverlap", "ReadOut")  # Blackfly S only
    exposure = PySpin.CFloatPtr(nm.GetNode("ExposureTime"))
    cap_us = TRIGGER_EXPOSURE_FRACTION * 1e6 / fps
    if PySpin.IsWritable(exposure) and float(exposure.GetValue()) > cap_us:
        exposure.SetValue(max(float(exposure.GetMin()), cap_us))
    if not _set_enum(PySpin, nm, "TriggerMode", "On"):
        raise RuntimeError("TriggerMode On not accepted")
    applied = {name: _read_str(PySpin, nm, name)
               for name in ("TriggerMode", "TriggerSelector", "TriggerSource", "TriggerActivation", "TriggerOverlap")}
    applied["exposure_us"] = float(exposure.GetValue()) if PySpin.IsReadable(exposure) else None
    applied["overlap_set"] = overlap
    if applied["TriggerMode"] != "On" or applied["TriggerSource"] != line:
        raise RuntimeError(f"trigger not applied: {applied}")
    return applied


def clear_trigger(PySpin, cam) -> None:
    """Back to free-running, as the recorder expects. Never raises."""
    try:
        nm = cam.GetNodeMap()
        _set_enum(PySpin, nm, "TriggerMode", "Off")
        enable = PySpin.CBooleanPtr(nm.GetNode("AcquisitionFrameRateEnable"))
        if PySpin.IsWritable(enable):
            enable.SetValue(True)
    except Exception as exc:
        print(f"    !! could not put a camera back to free-running: {exc} -- power-cycle it before recording")


def trigger_line_for(model: str, overrides: dict, serial: str) -> str:
    if serial in overrides:
        return overrides[serial]
    for key, line in DEFAULT_TRIGGER_LINES.items():
        if key.lower() in model.lower():
            return line
    raise RuntimeError(f"no trigger line known for {model} #{serial}: pass --line {serial}=LineN")


def run_trigger(PySpin, system, serials: list[str], fps: float, seconds: float, out_dir: Path,
                pi, line_overrides: dict, pulse_ms: float) -> dict:
    cam_list = system.GetCameras()
    runs: list[CamRun] = []
    stop = threading.Event()
    threads: list[threading.Thread] = []
    pulses_sent = None
    before_start: dict = {}
    applied_by_serial: dict = {}
    pi.stop()  # no pulses while the cameras are being configured
    try:
        for serial in serials:
            cam = cam_list.GetBySerial(serial)
            cam.Init()
            run = CamRun(serial=serial, cam=cam)
            info = configure(PySpin, cam, fps)
            run.model = info.model
            line = trigger_line_for(info.model, line_overrides, serial)
            applied = set_trigger(PySpin, cam, line, fps)
            applied_by_serial[serial] = {"line": line, **applied}
            run.applied_fps = fps  # the trigger sets the rate
            run.latch_supported = latch_supported(PySpin, cam)
            print(f"    {serial} {info.model}: trigger {applied['TriggerSource']} {applied['TriggerActivation']}, "
                  f"overlap {applied['TriggerOverlap']}, exposure {fmt(applied['exposure_us'], '.0f')} us, "
                  f"link={info.speed}")
            runs.append(run)

        for run in runs:
            run.cam.BeginAcquisition()
        for run in runs:
            t = threading.Thread(target=grab_loop, args=(run, stop), daemon=True)
            t.start()
            threads.append(t)
        for _ in range(3):
            for run in runs:
                if run.latch_supported:
                    try:
                        latch_once(PySpin, run)
                    except Exception as exc:
                        run.latch_errors += 1
                        run.last_error = f"latch: {exc}"
            time.sleep(0.05)
        lt = threading.Thread(target=latch_loop, args=(PySpin, runs, stop, LATCH_INTERVAL_S), daemon=True)
        lt.start()
        threads.append(lt)

        time.sleep(1.0)  # armed, no pulses: any frame now is a stray trigger
        for run in runs:
            before_start[run.serial] = len(run.frames)
            if run.frames:
                print(f"    !! {run.serial}: {len(run.frames)} frame(s) before the pulses started "
                      "(stray triggers: check ground and wiring)")
            run.frames.clear()

        started = pi.start(fps, pulse_ms)
        print(f"    Pi: {started.get('actual_hz', 0):.6f} Hz, pulse {pulse_ms:g} ms")
        deadline = time.monotonic() + seconds
        next_report = time.monotonic() + 30.0
        while time.monotonic() < deadline:
            time.sleep(0.5)
            if time.monotonic() >= next_report:
                left = deadline - time.monotonic()
                counts = ", ".join(f"{r.serial}: {len(r.frames)} frames" for r in runs)
                print(f"    {left / 60:.1f} min left -- {counts}")
                next_report += 30.0
        stopped = pi.stop()
        pulses_sent = stopped.get("pulses_sent")
        time.sleep(0.5)  # the last frames in flight
    finally:
        try:
            pi.stop()
        except Exception as exc:
            print(f"    !! could not stop the Pi: {exc}")
        stop.set()
        for t in threads:
            t.join(timeout=5.0)
        for run in runs:
            try:
                run.cam.EndAcquisition()
            except Exception:
                pass
            clear_trigger(PySpin, run.cam)
            try:
                run.cam.DeInit()
            except Exception:
                pass
            run.cam = None
        cam_list.Clear()

    result = analyse(runs, fps)
    result["trigger_settings"] = applied_by_serial
    trigger_report(result, runs, fps, pulses_sent, before_start)
    write_raw(out_dir, runs, fps, result.pop("_offsets", None), tag_prefix="trigger_")
    return result


def print_trigger(result: dict) -> None:
    print(f"\n--- triggered @ {result['fps_requested']:g} Hz ---")
    for serial, cam in result["cameras"].items():
        iv = cam.get("intervals") or {}
        missed = cam.get("missed_triggers")
        ok = (cam["frame_gaps"] == 0 and cam["incomplete"] == 0 and cam["errors"] == 0
              and missed in (0, None) and cam.get("frames_before_start", 0) == 0)
        print(f"  {cam['model']} #{serial}: {cam['frames']} frames for {cam.get('pulses_sent')} pulses "
              f"(missed {fmt(missed, 'd') if missed is not None else '-'}), "
              f"before start {cam.get('frames_before_start', 0)}, gaps {cam['frame_gaps']}, "
              f"incomplete {cam['incomplete']}, errors {cam['errors']}  -> {'OK' if ok else 'PROBLEM'}")
        if iv:
            print(f"    frame interval (camera clock): mean {iv['mean_period_us']:.2f} us, std {iv['std_us']:.2f} us, "
                  f"max deviation {fmt(iv['max_abs_dev_us'], '.2f')} us, intervals > 1.5 periods: "
                  f"{iv['intervals_over_1_5_periods']}")
    tp = result.get("trigger_pair") or {}
    if tp.get("pairs"):
        print(f"  Same-pulse gap (#{tp['b']} minus #{tp['a']}): mean {tp['mean_gap_us']:+.1f} us (constant offset), "
              f"spread p99 {tp['spread_us_p99']:.1f} us, max {tp['spread_us_max']:.1f} us; "
              f"|gap| p99 {tp['abs_gap_us_p99']:.1f} us over {tp['pairs']} pairs")


def fmt(value, spec=".2f") -> str:
    return "-" if value is None else format(value, spec)


def print_skew(result: dict) -> None:
    print(f"\n--- {result['fps_requested']:g} fps ---")
    for serial, cam in result["cameras"].items():
        ok = cam["frame_gaps"] == 0 and cam["incomplete"] == 0 and cam["errors"] == 0
        print(f"  {cam['model']} #{serial}: {cam['frames']} frames, achieved {fmt(cam.get('achieved_fps'), '.3f')} fps, "
              f"gaps {cam['frame_gaps']}, incomplete {cam['incomplete']}, errors {cam['errors']}"
              f"  -> {'OK' if ok else 'PROBLEM'}")
        print(f"    tick = {fmt(cam.get('tick_ns_from_latch'), '.6f')} ns (latch), "
              f"{fmt(cam.get('tick_ns_from_frames'), '.4f')} ns (frames); "
              f"clock drift vs laptop {fmt(cam.get('clock_drift_ppm_vs_laptop'), '+.1f')} ppm; "
              f"latch fit residual {fmt(cam.get('latch_fit_residual_us'), '.0f')} us")
        arr = cam.get("arrival") or {}
        if arr:
            print(f"    host arrival jitter (system_time as a clock): p50 {fmt(arr['latency_ms_p50'])} ms, "
                  f"p99 {fmt(arr['latency_ms_p99'])} ms, max {fmt(arr['latency_ms_max'])} ms; "
                  f"start burst up to {fmt(arr.get('start_burst_ms_max'))} ms")
        if cam.get("latch_errors"):
            print(f"    !! {cam['latch_errors']} latch errors: {cam['last_error']}")
    pair = result.get("pair") or {}
    if "abs_gap_ms_p50" in pair:
        print(f"  Gap between cameras (#{pair['b']} frame minus nearest #{pair['a']} frame):")
        print(f"    start {pair['gap_ms_start']:+.2f} ms, end {pair['gap_ms_end']:+.2f} ms, "
              f"drifting {pair['drift_ms_per_min']:+.3f} ms/min"
              + (f" (passes through every phase every {pair['full_cycle_min']:.0f} min)"
                 if pair.get("full_cycle_min") else ""))
        print(f"    |gap| p50 {pair['abs_gap_ms_p50']:.2f} ms, p99 {pair['abs_gap_ms_p99']:.2f} ms, "
              f"max {pair['abs_gap_ms_max']:.2f} ms (worst possible: {pair['half_period_ms']:.2f} ms)")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

class Tee:
    """Copy everything printed to a file as well, so the report can be shared as text."""

    def __init__(self, path: Path, stream):
        self._file = open(path, "w", encoding="utf-8", buffering=1)
        self._stream = stream

    def write(self, text: str) -> int:
        self._stream.write(text)
        self._file.write(text)
        return len(text)

    def flush(self) -> None:
        self._stream.flush()
        self._file.flush()

    def close(self) -> None:
        self._file.close()


def discover(PySpin, system) -> list[str]:
    found = system.GetCameras()
    try:
        serials = []
        for i in range(found.GetSize()):
            cam = found[i]
            serials.append(_read_str(PySpin, cam.GetTLDeviceNodeMap(), "DeviceSerialNumber"))
            del cam
        return serials
    finally:
        found.Clear()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)
    for name in ("nodes", "skew", "trigger"):
        p = sub.add_parser(name)
        p.add_argument("--serials", nargs="+", help="camera serials, camera A first (default: every camera found)")
        p.add_argument("--out-dir", default="probe_output")
    for name in ("skew", "trigger"):
        sub.choices[name].add_argument("--fps", type=float, nargs="+", default=[30.0, 60.0])
        sub.choices[name].add_argument("--seconds", type=float, default=600.0, help="per fps (default 600 = 10 min)")
    trig = sub.choices["trigger"]
    trig.add_argument("--trigger-port", help="the Pi's trigger COM port (default: found automatically)")
    trig.add_argument("--line", nargs="+", default=[], metavar="SERIAL=LINE",
                      help="trigger input per camera (default: Firefly Line2, Blackfly Line3)")
    trig.add_argument("--pulse-ms", type=float, default=1.0)
    args = ap.parse_args(argv)

    import PySpin

    out_dir = Path(args.out_dir) / f"sync_probe_{datetime.now():%Y%m%d_%H%M%S}"
    out_dir.mkdir(parents=True, exist_ok=True)
    tee = Tee(out_dir / "report.txt", sys.stdout)
    sys.stdout = tee
    try:
        return _run(PySpin, args, out_dir)
    finally:
        sys.stdout = tee._stream
        tee.close()
        print(f"Report saved to {out_dir / 'report.txt'}")


def _run(PySpin, args, out_dir: Path) -> int:
    print(f"sync_probe {args.command}  {datetime.now():%Y-%m-%d %H:%M:%S}  args: {vars(args)}")
    system = PySpin.System.GetInstance()
    try:
        discovered = discover(PySpin, system)
        serials = list(dict.fromkeys(args.serials or sorted(discovered)))
        missing = [s for s in serials if s not in discovered]
        if missing:
            print(f"Serials not found: {missing}. Discovered: {discovered}")
            return 2
        if not serials:
            print("No cameras found. Is the recorder GUI or SpinView still open?")
            return 2
        print(f"Cameras: {serials}   output: {out_dir}")

        summary = {"command": args.command, "serials": serials, "started": datetime.now().isoformat()}
        if args.command == "nodes":
            cam_list = system.GetCameras()
            try:
                summary["cameras"] = []
                for serial in serials:
                    cam = cam_list.GetBySerial(serial)
                    cam.Init()
                    try:
                        summary["cameras"].append(dump_camera_nodes(PySpin, cam))
                    finally:
                        cam.DeInit()
                        del cam
            finally:
                cam_list.Clear()
            for cam in summary["cameras"]:
                print_nodes(cam)
            out = out_dir / "nodes.json"
        elif args.command == "trigger":
            return _run_trigger(PySpin, system, serials, args, out_dir, summary)
        else:
            if len(serials) < 2:
                print(f"skew needs two cameras; have {serials}")
                return 2
            summary["runs"] = []
            for n, fps in enumerate(args.fps, 1):
                print(f"\n[{n}/{len(args.fps)}] both cameras free-running @ {fps:g} fps for {args.seconds / 60:.1f} min")
                try:
                    result = run_skew(PySpin, system, serials[:2], fps, args.seconds, out_dir)
                except Exception as exc:
                    print(f"    run failed: {exc.__class__.__name__}: {exc}"
                          "\n    (is SpinView or the recorder GUI still holding a camera?)")
                    return 1
                summary["runs"].append(result)
                time.sleep(2.0)
            for result in summary["runs"]:
                print_skew(result)
            out = out_dir / "summary.json"

        with open(out, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"\nWrote {out_dir}  -- bring this whole folder back.")
        return 0
    finally:
        system.ReleaseInstance()


def _run_trigger(PySpin, system, serials, args, out_dir: Path, summary: dict) -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from backend.pi_trigger_client import PiTrigger, find_port

    try:
        overrides = dict(item.split("=", 1) for item in args.line)
    except ValueError:
        print(f"--line expects SERIAL=LINE, got {args.line}")
        return 2
    try:
        pi = PiTrigger.open(args.trigger_port) if args.trigger_port else find_port()
    except Exception as exc:
        print(f"Could not open {args.trigger_port}: {exc}")
        return 2
    if pi is None:
        print("No Raspberry Pi trigger found on any USB serial port. Is the Pi plugged into this laptop and booted "
              "(about 30 s)? Is pyserial installed (pip install pyserial)?")
        return 2
    try:
        hello = pi.ping()
        print(f"Pi trigger on {pi.port}: version {hello.get('version')}")
        summary["pi"] = {"port": pi.port, "ping": hello}
        summary["runs"] = []
        for n, fps in enumerate(args.fps, 1):
            print(f"\n[{n}/{len(args.fps)}] both cameras on the Pi trigger @ {fps:g} Hz for {args.seconds / 60:.1f} min")
            try:
                result = run_trigger(PySpin, system, serials[:2], fps, args.seconds, out_dir, pi, overrides,
                                     args.pulse_ms)
            except Exception as exc:
                print(f"    run failed: {exc.__class__.__name__}: {exc}")
                return 1
            summary["runs"].append(result)
            time.sleep(2.0)
        for result in summary["runs"]:
            print_trigger(result)
    finally:
        pi.close()
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nWrote {out_dir}  -- bring this whole folder back.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
