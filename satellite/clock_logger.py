#!/usr/bin/env python3
"""Log and compare two ways of mapping the satellite steady clock onto the laptop steady clock.

Runs on the satellite (Linux).  Needs `pip install thalamus-neuro grpcio numpy`.  The laptop runs
Jarl's server.exe (Thalamus protocol, port 50050, the same one the phones use) or, as a test tool,
satellite/clock_server.py.  Both work: server.exe's ping replies have no CLK1 trailer (the logger then
writes ping rows with t1/t4 only and the SERVER-EXE method is used); a server with the trailer gives
the NEW method as well.

Clocks: satellite = time.monotonic_ns() (Linux CLOCK_MONOTONIC); laptop = time.perf_counter_ns()
(Windows QueryPerformanceCounter).  Offsets are "server steady minus client steady".

Log (one process measures several server addresses at the same time, e.g. cable and Wi-Fi):
  python satellite/clock_logger.py --server laptop=192.168.50.20:50050 --out ~/clock_log \\
      --duration-s 0 --serve-port 50061            (several --server LABEL=HOST:PORT are allowed)
  Options: --duration-s N (default 0 = until Ctrl+C; then one subfolder DIR/YYYYMMDD per day),
  --pings-per-s 5, --serve-port PORT / --serve-host 0.0.0.0 (read-only HTTP server for the CSVs and
  events.log, so the laptop's receiver_service can fetch the clock log at stop; GET / lists the files;
  a file name at the root returns the same-named files of DIR and its last 3 day folders joined).
  Per label it writes in DIR (flushed after every row):
    <label>_analog.csv  seq, client_mono_ns, server_wall_ns, server_steady_ns   (CURRENT method data)
    <label>_ping.csv    seq, t1, t2, t3, t4, rtt_ns, offset_ns, instance_id     (NEW method data;
                        t2/t3/rtt/offset/instance_id are blank when the reply has no trailer)
    events.log          stream drops, reconnects, server instance-ID changes (also printed)
  NEW method per ping: t1 = client mono just before send, t4 = client mono on receipt, t2/t3 =
  server steady at receive/send (carried in the Pong payload trailer, layout in clock_server.py).
    offset = ((t2 - t1) + (t3 - t4)) / 2        rtt = (t4 - t1) - (t3 - t2)

Analyze (needs only numpy):
  python satellite/clock_logger.py analyze DIR [--reference LABEL] [--truth-zero]
  Per label it fits CURRENT (server Wall vs client mono, fit_affine copied from
  smart_sleeve_data_processing), CURRENT-STEADY (server steady vs client mono, same fit) and NEW
  (min-RTT ping per 10 s window, Theil-Sen line of offset vs t1), flags Wall jumps and writes
  report.txt and report.json into DIR.  Wall-based predictions are turned into steady time by
  subtracting the median server (Wall - steady) so all methods are comparable.  It also fits
  SERVER-EXE (needs no trailer): y = server steady - client arrival over the analog messages, the
  largest y (least delayed message) per 10 s window, Theil-Sen line, plus half of the 1st percentile
  of the ping round trips t4 - t1 as one-way delay.
  fit_mapping() below is what receiver_service.py uses at stop; it picks NEW when there are enough
  trailer pings, else SERVER-EXE.
  --reference LABEL: compare each method with that label's NEW mapping over the run.
  --truth-zero: compare with the identity mapping (server steady == client steady), for loopback
  tests on one machine where the true offset and drift are exactly 0.
"""
import argparse
import asyncio
import csv
import json
import logging
import math
import re
import signal
import struct
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

logger = logging.getLogger("clock_logger")

TRAILER = struct.Struct("<4sQqqq")  # magic, instance_id, t2, t3, wall  (see clock_server.py)
MAGIC = b"CLK1"
WALL_JUMP_NS = 5e6          # a change of (Wall - steady) between samples larger than this is a jump
WINDOW_NS = 10e9            # min-RTT selection window
ANALOG_COLS = ["seq", "client_mono_ns", "server_wall_ns", "server_steady_ns"]
PING_COLS = ["seq", "t1", "t2", "t3", "t4", "rtt_ns", "offset_ns", "instance_id"]


# ------------------------------------------------------------------ math (numpy only)
def parse_trailer(payload):
    """Return (instance_id, t2, t3, wall_ns) from the last 36 bytes of a Pong payload, or None."""
    if len(payload) < TRAILER.size:
        return None
    magic, inst, t2, t3, wall = TRAILER.unpack(payload[-TRAILER.size:])
    if magic != MAGIC:
        return None
    return inst, t2, t3, wall


def offset_rtt(t1, t2, t3, t4):
    """NTP formulas. offset = server steady minus client steady; rtt excludes server hold time."""
    return ((t2 - t1) + (t3 - t4)) / 2, (t4 - t1) - (t3 - t2)


def select_min_rtt(t1, rtt, window_ns=WINDOW_NS):
    """Indices of the lowest-RTT ping in each window of window_ns (windows start at min t1)."""
    import numpy as np
    t1 = np.asarray(t1, dtype=np.float64)
    rtt = np.asarray(rtt, dtype=np.float64)
    if t1.size == 0:
        return np.array([], dtype=int)
    win = np.floor((t1 - t1.min()) / window_ns).astype(np.int64)
    keep = []
    for w in np.unique(win):
        idx = np.nonzero(win == w)[0]
        keep.append(idx[np.argmin(rtt[idx])])
    return np.array(sorted(keep), dtype=int)


def theil_sen(x, y, max_pairs=300000, seed=0):
    """Robust line y = slope * x + icpt (median of pairwise slopes, median intercept)."""
    import numpy as np
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = x.size
    if n < 2:
        return 0.0, float(np.median(y)) if n else 0.0
    if n * (n - 1) // 2 <= max_pairs:
        i, j = np.triu_indices(n, k=1)
    else:
        rng = np.random.default_rng(seed)
        i = rng.integers(0, n, max_pairs)
        j = rng.integers(0, n, max_pairs)
    dx = x[j] - x[i]
    ok = dx != 0
    slope = float(np.median((y[j][ok] - y[i][ok]) / dx[ok])) if ok.any() else 0.0
    return slope, float(np.median(y - slope * x))


def fit_affine(sts):
    """Fit system_ns ~ a * steady_ns + b from steady->system pairs.

    Copied unchanged (except the doc line) from smart_sleeve_data_processing,
    pipelines/utils/tha_timeseries.py::fit_affine (median slope, MAD trim with 5 ms floor, least
    squares); not imported because that module imports thalamus.record_reader.  sts is an (N, 2)
    array of [steady_ns, system_ns]; returns (a, b).
    """
    import numpy as np
    AFFINE_FIT_RMS_WARN_NS = 5e8  # 500 ms
    sts = sts.astype(np.float64)
    order = np.argsort(sts[:, 0])  # sort by steady time
    steady = sts[order, 0]
    system_ns = sts[order, 1]

    d_steady = np.diff(steady)
    d_system = np.diff(system_ns)
    valid = d_steady > 0
    slopes = d_system[valid] / d_steady[valid] if valid.any() else np.array([])
    slopes = slopes[np.isfinite(slopes)]

    if slopes.size == 0:
        logger.warning(
            "Insufficient or invalid Laptop/Wall timing for affine fit; "
            "defaulting to a=0.0 and b=median(system_ns)."
        )
        a = 0.0
        b = np.median(system_ns) if system_ns.size else 0.0
    else:
        a0 = np.median(slopes)
        b0 = np.median(system_ns - a0 * steady)
        residuals = system_ns - (a0 * steady + b0)
        mad = np.median(np.abs(residuals - np.median(residuals)))
        tol = max(5e6, 5.0 * mad) if mad > 0 else 5e6  # ~5 ms floor
        keep = np.abs(residuals) <= tol

        if keep.sum() >= 2:
            a, b = np.polyfit(steady[keep], system_ns[keep], 1)
        else:
            a, b = a0, b0

    if slopes.size:
        if not (0.5 <= a <= 1.5):
            logger.warning(f"Unexpected steady->system slope a={a:.6f}; check timing data.")

        residuals = system_ns - (a * steady + b)
        rms = float(np.sqrt(np.mean(residuals ** 2))) if residuals.size else 0.0
        if rms > AFFINE_FIT_RMS_WARN_NS:
            logger.warning(
                f"High affine fit residual RMS={rms:.0f} ns; timing may be noisy or discontinuous."
            )
    return a, b


def find_wall_jumps(wall, steady, thr_ns=WALL_JUMP_NS):
    """Indices i where (wall - steady) changed by more than thr_ns between sample i-1 and i."""
    import numpy as np
    d = np.asarray(wall, dtype=np.float64) - np.asarray(steady, dtype=np.float64)
    if d.size < 2:
        return np.array([], dtype=int)
    return np.nonzero(np.abs(np.diff(d)) > thr_ns)[0] + 1


def _resid_stats_ms(res_ns):
    import numpy as np
    res_ns = np.asarray(res_ns, dtype=np.float64)
    if res_ns.size == 0:
        return {"rms_ms": None, "max_ms": None}
    return {"rms_ms": float(np.sqrt(np.mean(res_ns ** 2))) / 1e6,
            "max_ms": float(np.max(np.abs(res_ns))) / 1e6}


def fit_current(client, server, wall_to_steady=0.0):
    """CURRENT-style fit (fit_affine) of server vs client; returns summary dict + predictor."""
    import numpy as np
    client = np.asarray(client, dtype=np.float64)
    server = np.asarray(server, dtype=np.float64)
    if client.size < 2:
        return None
    a, b = fit_affine(np.column_stack([client, server]))
    res = server - (a * client + b)
    out = {"n": int(client.size), "slope_ppm": float((a - 1.0) * 1e6), "a": float(a), "b": float(b)}
    out.update(_resid_stats_ms(res))
    return out, (lambda x: a * np.asarray(x, dtype=np.float64) + b - wall_to_steady)


def fit_new(t1, rtt, offset, window_ns=WINDOW_NS):
    """NEW method: min-RTT ping per window, Theil-Sen line of offset vs t1."""
    import numpy as np
    t1 = np.asarray(t1, dtype=np.float64)
    rtt = np.asarray(rtt, dtype=np.float64)
    offset = np.asarray(offset, dtype=np.float64)
    if t1.size < 2:
        return None
    k = select_min_rtt(t1, rtt, window_ns)
    x0 = float(t1.min())
    x = t1[k] - x0
    slope, icpt = theil_sen(x, offset[k])
    res = offset[k] - (slope * x + icpt)
    out = {"n_valid": int(t1.size), "n_kept": int(k.size),
           "slope_ppm": slope * 1e6, "offset_at_start_ms": icpt / 1e6,
           "uncertainty_ms_median": float(np.median(rtt[k])) / 2e6,
           "uncertainty_ms_max": float(np.max(rtt[k])) / 2e6,
           "rtt_ms": {"min": float(rtt.min()) / 1e6, "p50": float(np.percentile(rtt, 50)) / 1e6,
                      "p95": float(np.percentile(rtt, 95)) / 1e6, "max": float(rtt.max()) / 1e6}}
    out.update(_resid_stats_ms(res))
    out["x0_ns"] = x0
    return out, (lambda t: np.asarray(t, dtype=np.float64) + icpt + slope * (np.asarray(t, dtype=np.float64) - x0))


def select_max_per_window(x, y, window_ns=WINDOW_NS):
    """Indices of the largest y in each window of window_ns of x (windows start at min x)."""
    import numpy as np
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size == 0:
        return np.array([], dtype=int)
    win = np.floor((x - x.min()) / window_ns).astype(np.int64)
    keep = []
    for w in np.unique(win):
        idx = np.nonzero(win == w)[0]
        keep.append(idx[np.argmax(y[idx])])
    return np.array(sorted(keep), dtype=int)


def fit_server_exe(client, steady, ping_rtt=None, window_ns=WINDOW_NS):
    """SERVER-EXE method (no ping trailer needed).

    y = server_steady - client_arrival of each analog message = offset - one-way delay, so the
    largest y per window is the least delayed message.  Theil-Sen line through those (x and y are
    centred on the first sample: ns values are ~1e14, float64 would lose digits otherwise), then
    add the one-way delay estimate = half of the 1st percentile of the ping round trips t4 - t1
    (client side only; ping_rtt in ns, any rows with t1 and t4).  Same result form as fit_new.
    """
    import numpy as np
    client = np.asarray(client, dtype=np.float64)
    steady = np.asarray(steady, dtype=np.float64)
    if client.size < 2:
        return None
    order = np.argsort(client)
    client, steady = client[order], steady[order]
    x0 = float(client[0])
    y0 = float(steady[0] - client[0])
    x = client - x0
    y = (steady - client) - y0
    k = select_max_per_window(x, y, window_ns)
    slope, icpt = theil_sen(x[k], y[k])
    res = y[k] - (slope * x[k] + icpt)
    if ping_rtt is not None and len(ping_rtt):
        half_rtt = float(np.percentile(np.asarray(ping_rtt, dtype=np.float64), 1)) / 2
    else:
        half_rtt = 0.0
    offset0 = y0 + icpt + half_rtt  # offset at x0
    out = {"n_valid": int(client.size), "n_kept": int(k.size),
           "slope_ppm": slope * 1e6, "offset_at_start_ms": offset0 / 1e6,
           "half_rtt_ms": half_rtt / 1e6, "uncertainty_ms_median": half_rtt / 1e6,
           "uncertainty_ms_max": half_rtt / 1e6}
    out.update(_resid_stats_ms(res))
    out["x0_ns"] = x0
    out["offset0_ns"] = offset0
    out["slope"] = slope
    return out, (lambda t: np.asarray(t, dtype=np.float64) + offset0 + slope * (np.asarray(t, dtype=np.float64) - x0))


MIN_PING_WINDOWS = 6   # at least 60 s of trailer pings, else fall back to the analog method


def fit_mapping(analog_rows, ping_rows):
    """Mapping satellite steady clock -> laptop steady clock from clock-log rows.

    analog_rows / ping_rows: lists of dicts as csv.DictReader reads the logger's CSVs (values may
    be strings or numbers).  Uses "pings" (NEW method) when there are at least MIN_PING_WINDOWS
    windows of trailer pings from one server instance, else "analog+half-rtt" (SERVER-EXE).
    Returns a dict:
      method, slope, drift_ppm, uncertainty_ms, n_samples (kept windows), n_rows (rows used),
      span_s, client_first_ns, client_last_ns, x0_ns, offset_at_x0_ns,
      wall_minus_steady_median_ns (None without analog rows), instance_ids, wall_jumps, warnings.
    The mapping is  laptop_steady(c) = c + offset_at_x0_ns + slope * (c - x0_ns)  for a satellite
    steady time c (use map_client_to_server).  Raises ValueError when there is too little data.
    """
    import numpy as np
    d = _arrays_from_rows(analog_rows, ping_rows)
    warnings, insts = [], {}
    fit = None
    if d["t1"].size:
        insts = {i: d["inst"].count(i) for i in sorted(set(d["inst"]), key=d["inst"].count, reverse=True)}
        top = max(insts, key=insts.get)
        sel = np.array([i == top for i in d["inst"]])
        if len(insts) > 1:
            warnings.append("clock server restarted during the log (instance IDs %s); used the most "
                            "common one" % ", ".join(insts))
        r = fit_new(d["t1"][sel], d["rtt"][sel], d["offset"][sel])
        if r and r[0]["n_kept"] >= MIN_PING_WINDOWS:
            fit = ("pings", r[0], d["t1"][sel])
    if fit is None:
        if d["client"].size < 2:
            raise ValueError("not enough clock samples (%d analog rows, %d trailer pings)"
                             % (d["client"].size, d["t1"].size))
        r = fit_server_exe(d["client"], d["steady"], d["rtt_client"])
        fit = ("analog+half-rtt", r[0], d["client"])
        if d["rtt_client"].size == 0:
            warnings.append("no ping round trips in the log: one-way delay not added (error up to a few ms)")
    method, out, xs = fit
    if out["n_kept"] < 2:
        raise ValueError("not enough clock samples: only %d window(s) of data" % out["n_kept"])
    if method == "pings":
        slope, off0, x0 = out["slope_ppm"] / 1e6, out["offset_at_start_ms"] * 1e6, out["x0_ns"]
    else:
        slope, off0, x0 = out["slope"], out["offset0_ns"], out["x0_ns"]
    wall_med, jumps = None, 0
    if d["client"].size >= 2:
        wall_med = float(np.median(d["wall"].astype(np.float64) - d["steady"].astype(np.float64)))
        jumps = int(find_wall_jumps(d["wall"], d["steady"]).size)
        if jumps:
            warnings.append("server Wall clock jumped %d time(s) during the log; wall dates may be off" % jumps)
    return {"method": method, "slope": float(slope), "drift_ppm": float(slope * 1e6),
            "uncertainty_ms": float(out["uncertainty_ms_median"]), "n_samples": int(out["n_kept"]),
            "n_rows": int(len(xs)), "span_s": float(xs.max() - xs.min()) / 1e9,
            "client_first_ns": int(xs.min()), "client_last_ns": int(xs.max()),
            "x0_ns": int(round(x0)), "offset_at_x0_ns": float(off0),
            "wall_minus_steady_median_ns": wall_med, "instance_ids": insts,
            "wall_jumps": jumps, "warnings": warnings}


def map_client_to_server(fit, client_ns):
    """Satellite steady ns (ints) -> laptop steady ns (ints), from a fit_mapping result."""
    x0, off0, slope = fit["x0_ns"], fit["offset_at_x0_ns"], fit["slope"]
    return [int(c) + int(round(off0 + slope * (int(c) - x0))) for c in client_ns]


def compare_maps(pred, ref, grid):
    """Stats (ms) of pred(grid) - ref(grid)."""
    import numpy as np
    d = (pred(grid) - ref(grid)) / 1e6
    return {"median_ms": float(np.median(d)), "rms_ms": float(np.sqrt(np.mean(d ** 2))),
            "max_ms": float(np.max(np.abs(d)))}


# ------------------------------------------------------------------ analysis
def _read_csv(path):
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def _arrays_from_rows(analog_rows, ping_rows):
    """Parse csv rows (dicts) into arrays. Ping rows without trailer (t2 blank) count only for
    rtt_client (t4 - t1 of every ping with a reply)."""
    import numpy as np
    d = {}
    d["client"] = np.array([int(r["client_mono_ns"]) for r in analog_rows], dtype=np.int64)
    d["wall"] = np.array([int(r["server_wall_ns"]) for r in analog_rows], dtype=np.int64)
    d["steady"] = np.array([int(r["server_steady_ns"]) for r in analog_rows], dtype=np.int64)
    d["ping_total"] = len(ping_rows)
    d["rtt_client"] = np.array([int(r["t4"]) - int(r["t1"]) for r in ping_rows
                                if str(r["t1"]) != "" and str(r["t4"]) != ""], dtype=np.int64)
    rows = [r for r in ping_rows if str(r["t2"]) != "" and str(r["instance_id"]) != ""]
    d["t1"] = np.array([int(r["t1"]) for r in rows], dtype=np.int64)
    d["rtt"] = np.array([int(r["rtt_ns"]) for r in rows], dtype=np.int64)
    d["offset"] = np.array([float(r["offset_ns"]) for r in rows], dtype=np.float64)
    d["inst"] = [str(r["instance_id"]) for r in rows]
    return d


def load_label(out_dir, label):
    ap = Path(out_dir) / f"{label}_analog.csv"
    pp = Path(out_dir) / f"{label}_ping.csv"
    d = _arrays_from_rows(_read_csv(ap) if ap.exists() else [], _read_csv(pp) if pp.exists() else [])
    d["label"] = label
    return d


def discover_labels(out_dir):
    names = {p.name[:-len("_analog.csv")] for p in Path(out_dir).glob("*_analog.csv")}
    names |= {p.name[:-len("_ping.csv")] for p in Path(out_dir).glob("*_ping.csv")}
    return sorted(names)


def analyze_label(d):
    """Per-label fits (no reference comparison yet). Returns (result dict, predictors dict)."""
    import numpy as np
    res, preds = {}, {}
    res["ping_rows"] = d["ping_total"]
    # NEW: keep only the most common server instance (a restart changes the server steady clock).
    if d["t1"].size:
        insts = sorted(set(d["inst"]), key=d["inst"].count, reverse=True)
        sel = np.array([i == insts[0] for i in d["inst"]])
        r = fit_new(d["t1"][sel], d["rtt"][sel], d["offset"][sel])
        res["ping_with_reply"] = int(d["t1"].size)
        res["instance_ids"] = {i: d["inst"].count(i) for i in insts}
        if r:
            res["new"], preds["new"] = r
    # CURRENT and CURRENT-STEADY
    if d["client"].size >= 2:
        wall_minus_steady = float(np.median(d["wall"].astype(np.float64) - d["steady"].astype(np.float64)))
        res["analog_rows"] = int(d["client"].size)
        res["wall_minus_steady_median_ns"] = wall_minus_steady
        r = fit_current(d["client"], d["wall"], wall_minus_steady)
        if r:
            res["current_wall"], preds["current_wall"] = r
        r = fit_current(d["client"], d["steady"])
        if r:
            res["current_steady"], preds["current_steady"] = r
        r = fit_server_exe(d["client"], d["steady"], d["rtt_client"])
        if r:
            res["server_exe"], preds["server_exe"] = r
            res["ping_round_trips_client_side"] = int(d["rtt_client"].size)
        jumps = find_wall_jumps(d["wall"], d["steady"])
        dd = d["wall"].astype(np.float64) - d["steady"].astype(np.float64)
        res["wall_jumps"] = {"count": int(jumps.size), "threshold_ms": WALL_JUMP_NS / 1e6,
                             "first": [{"index": int(i), "jump_ms": float(dd[i] - dd[i - 1]) / 1e6}
                                       for i in jumps[:10]]}
    return res, preds


def analyze_dir(out_dir, reference=None, truth_zero=False):
    import numpy as np
    labels = discover_labels(out_dir)
    if not labels:
        raise SystemExit(f"no *_analog.csv / *_ping.csv in {out_dir}")
    data = {l: load_label(out_dir, l) for l in labels}
    results, preds = {}, {}
    for l in labels:
        results[l], preds[l] = analyze_label(data[l])
    ref_fn, ref_name = None, None
    if truth_zero:
        ref_fn, ref_name = (lambda t: np.asarray(t, dtype=np.float64)), "truth (server steady == client steady)"
    elif reference:
        if reference not in preds or "new" not in preds[reference]:
            raise SystemExit(f"reference label {reference!r} has no NEW fit")
        ref_fn, ref_name = preds[reference]["new"], f"NEW mapping of {reference}"
    if ref_fn is not None:
        for l in labels:
            lo = [float(data[l]["client"].min())] if data[l]["client"].size else []
            hi = [float(data[l]["client"].max())] if data[l]["client"].size else []
            if data[l]["t1"].size:
                lo.append(float(data[l]["t1"].min()))
                hi.append(float(data[l]["t1"].max()))
            if reference and not truth_zero and data[reference]["t1"].size:
                lo.append(float(data[reference]["t1"].min()))
                hi.append(float(data[reference]["t1"].max()))
            if not lo or max(lo) >= min(hi):
                continue
            grid = np.linspace(max(lo), min(hi), 1000)
            results[l]["vs_reference"] = {"reference": ref_name}
            for m, fn in preds[l].items():
                if m == "new" and l == reference and not truth_zero:
                    continue
                results[l]["vs_reference"][m] = compare_maps(fn, ref_fn, grid)
    for l in labels:  # strip internal fields
        for m in ("new", "server_exe"):
            for k in ("x0_ns", "offset0_ns", "slope"):
                results[l].get(m, {}).pop(k, None)
    report = {"dir": str(out_dir), "reference": ref_name, "labels": results}
    Path(out_dir, "report.json").write_text(json.dumps(report, indent=2))
    text = format_report(report)
    Path(out_dir, "report.txt").write_text(text)
    return report, text


def _f(v, fmt="{:.3f}"):
    return "n/a" if v is None else fmt.format(v)


def format_report(rep):
    L = [f"Clock mapping report for {rep['dir']}",
         "Offsets/mappings: server steady (laptop perf_counter) as a function of client steady (monotonic).",
         f"Reference: {rep['reference'] or 'none'}", ""]
    names = {"current_wall": "CURRENT (Wall vs client mono, fit_affine)",
             "current_steady": "CURRENT-STEADY (server steady vs client mono, fit_affine)",
             "new": "NEW (min-RTT pings, Theil-Sen)",
             "server_exe": "SERVER-EXE (analog min-delay + half RTT, no trailer needed)"}
    for label, r in rep["labels"].items():
        L.append(f"=== {label} ===")
        L.append(f"analog rows: {r.get('analog_rows', 0)}   ping rows: {r.get('ping_rows', 0)} "
                 f"(with reply: {r.get('ping_with_reply', 0)})   instance IDs: {r.get('instance_ids', {})}")
        for m in ("current_wall", "current_steady", "new", "server_exe"):
            if m not in r:
                L.append(f"{names[m]}: no data")
                continue
            x = r[m]
            line = (f"{names[m]}: slope {x['slope_ppm']:+.3f} ppm, residual RMS {_f(x['rms_ms'], '{:.4f}')} ms, "
                    f"max {_f(x['max_ms'], '{:.4f}')} ms")
            if m == "server_exe":
                line += (f"; offset at start {x['offset_at_start_ms']:.4f} ms; kept {x['n_kept']}/{x['n_valid']}; "
                         f"half RTT (1st percentile) {x['half_rtt_ms']:.4f} ms")
            if m == "new":
                line += (f"; offset at start {x['offset_at_start_ms']:.4f} ms; kept {x['n_kept']}/{x['n_valid']}; "
                         f"uncertainty rtt/2 median {x['uncertainty_ms_median']:.4f} ms, max {x['uncertainty_ms_max']:.4f} ms")
            L.append(line)
        if "new" in r:
            q = r["new"]["rtt_ms"]
            L.append(f"RTT ms: min {q['min']:.4f}  p50 {q['p50']:.4f}  p95 {q['p95']:.4f}  max {q['max']:.4f}")
        if "wall_minus_steady_median_ns" in r:
            L.append(f"median (Wall - steady) on server: {r['wall_minus_steady_median_ns'] / 1e9:.6f} s")
        wj = r.get("wall_jumps")
        if wj:
            L.append(f"Wall jumps (> {wj['threshold_ms']:g} ms between consecutive samples): {wj['count']}")
            for j in wj["first"]:
                L.append(f"  sample {j['index']}: {j['jump_ms']:+.3f} ms")
        vr = r.get("vs_reference")
        if vr:
            L.append(f"Predicted mapping minus reference ({vr['reference']}), ms "
                     f"(Wall method converted to steady with the median Wall-steady):")
            for m in ("current_wall", "current_steady", "new", "server_exe"):
                if m in vr:
                    s = vr[m]
                    L.append(f"  {names[m].split(' (')[0]:<15} median {s['median_ms']:+.4f}  RMS {s['rms_ms']:.4f}  max |.| {s['max_ms']:.4f}")
        L.append("")
    return "\n".join(L)


# ------------------------------------------------------------------ logging (grpc.aio)
class Rows:
    """CSV writer that flushes after every row."""

    def __init__(self, path, cols):
        self.f = open(path, "w", newline="")
        self.w = csv.writer(self.f)
        self.w.writerow(cols)
        self.f.flush()

    def write(self, row):
        self.w.writerow(row)
        self.f.flush()

    def close(self):
        self.f.close()


class Events:
    def __init__(self, path):
        self.f = open(path, "a")

    def log(self, label, msg):
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} [{label}] {msg}"
        print(line, flush=True)
        self.f.write(line + "\n")
        self.f.flush()


async def _sleep_or_stop(stop, s):
    try:
        await asyncio.wait_for(stop.wait(), s)
    except asyncio.TimeoutError:
        pass


async def analog_loop(label, addr, rows, ev, stop, node_name):
    import grpc
    from thalamus import thalamus_pb2, thalamus_pb2_grpc
    seq, backoff = 0, 1.0
    down_since = None
    while not stop.is_set():
        n = 0
        try:
            async with grpc.aio.insecure_channel(addr) as ch:
                call = thalamus_pb2_grpc.ThalamusStub(ch).analog(
                    thalamus_pb2.AnalogRequest(node=thalamus_pb2.NodeSelector(name=node_name)))
                async for resp in call:
                    arrival = time.monotonic_ns()
                    if down_since is not None:
                        ev.log(label, f"analog stream back after gap of {(time.monotonic() - down_since):.1f}s")
                        down_since = None
                    backoff = 1.0
                    wall = None
                    for sp in resp.spans:
                        if sp.name == "Wall" and sp.begin < len(resp.ulong_data):
                            wall = resp.ulong_data[sp.begin]
                    if wall is None:
                        continue
                    seq += 1
                    n += 1
                    rows.write([seq, arrival, wall, resp.time])
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001  (grpc.aio.AioRpcError and connection errors)
            err = f"{type(e).__name__}: {getattr(e, 'details', lambda: e)()}"
        else:
            err = "stream ended"
        if stop.is_set():
            break
        if down_since is None:
            down_since = time.monotonic()
        ev.log(label, f"analog stream dropped after {n} msgs ({err}); retry in {backoff:.0f}s")
        await _sleep_or_stop(stop, backoff)
        backoff = min(backoff * 2, 10.0)


async def ping_loop(label, addr, rows, ev, stop, pings_per_s):
    import grpc
    from thalamus import thalamus_pb2, thalamus_pb2_grpc
    seq, backoff = 0, 1.0
    last_inst, down_since = None, None
    period = 1.0 / pings_per_s
    while not stop.is_set():
        pending, n, lost_note = {}, 0, ""

        async def requests():
            nonlocal seq
            next_t = time.monotonic()
            while not stop.is_set():
                seq += 1
                msg = thalamus_pb2.Ping(id=seq & 0xFFFFFFFF, payload=b"CLKP" + struct.pack("<I", seq & 0xFFFFFFFF))
                pending[msg.id] = None
                t1 = time.monotonic_ns()  # just before the message is handed to gRPC
                pending[msg.id] = t1
                yield msg
                next_t += period
                await _sleep_or_stop(stop, max(0.0, next_t - time.monotonic()))

        try:
            async with grpc.aio.insecure_channel(addr) as ch:
                call = thalamus_pb2_grpc.ThalamusStub(ch).ping(requests())
                async for pong in call:
                    t4 = time.monotonic_ns()  # first thing on receipt
                    t1 = pending.pop(pong.id, None)
                    if t1 is None:
                        continue
                    if len(pending) > 50:  # replies lost on the way: forget old ones
                        for k in sorted(pending)[:-50]:
                            pending.pop(k)
                    tr = parse_trailer(bytes(pong.payload))
                    if tr is None:
                        if not lost_note:
                            lost_note = "no CLK1 trailer in Pong (server is not clock_server.py?)"
                            ev.log(label, lost_note)
                        rows.write([pong.id, t1, "", "", t4, "", "", ""])
                        continue
                    inst, t2, t3, _wall = tr
                    if down_since is not None:
                        ev.log(label, f"ping stream back after gap of {(time.monotonic() - down_since):.1f}s")
                        down_since = None
                    backoff = 1.0
                    if last_inst is not None and inst != last_inst:
                        ev.log(label, f"server instance ID changed {last_inst:016x} -> {inst:016x} (server restarted)")
                    if last_inst is None:
                        ev.log(label, f"connected to server instance {inst:016x}")
                    last_inst = inst
                    off, rtt = offset_rtt(t1, t2, t3, t4)
                    n += 1
                    rows.write([pong.id, t1, t2, t3, t4, rtt, f"{off:.1f}", f"{inst:016x}"])
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            err = f"{type(e).__name__}: {getattr(e, 'details', lambda: e)()}"
        else:
            err = "stream ended"
        if stop.is_set():
            break
        if down_since is None:
            down_since = time.monotonic()
        ev.log(label, f"ping stream dropped after {n} replies ({err}); retry in {backoff:.0f}s")
        await _sleep_or_stop(stop, backoff)
        backoff = min(backoff * 2, 10.0)


SERVE_NAME_RE = re.compile(r"^(?:[A-Za-z0-9_-]+_(?:analog|ping)\.csv|events\.log)$")
DAY_RE = re.compile(r"^\d{8}$")


MAX_TAIL = 20000


def tail_of(body, n):
    """Header line (CSV) plus the last n data lines of `body` (bytes)."""
    lines = body.splitlines(keepends=True)
    return b"".join(lines[:1] + lines[1:][-n:]) if n > 0 else lines[0] if lines else b""


def parse_tail(query):
    """N from a 'tail=N' query string, capped at MAX_TAIL; None when absent or invalid."""
    for part in query.split("&"):
        if part.startswith("tail="):
            try:
                return max(1, min(int(part[5:]), MAX_TAIL))
            except ValueError:
                return None
    return None


def served_files(base, name, max_days=3):
    """Bytes of `name` for the HTTP server: base/name and the same file in the last max_days day
    folders (YYYYMMDD) of base, oldest first; the header line of a CSV is kept only once."""
    base = Path(base)
    days = sorted(p for p in base.iterdir() if p.is_dir() and DAY_RE.match(p.name))[-max_days:] if base.is_dir() else []
    parts, header_seen = [], False
    for folder in [base] + days:
        f = folder / name
        if not f.is_file():
            continue
        lines = f.read_bytes().splitlines(keepends=True)
        if name.endswith(".csv") and lines:
            if header_seen:
                lines = lines[1:]
            header_seen = True
        parts.append(b"".join(lines if not lines or lines[-1].endswith(b"\n") else lines[:-1]))  # drop a half-written last row
    return b"".join(parts) if parts else None


def make_handler(base):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            name, _, query = self.path.partition("?")
            name = name.lstrip("/")
            tail = parse_tail(query)
            if name == "":
                names = sorted({p.name for d in [Path(base)] + [q for q in Path(base).glob("[0-9]" * 8) if q.is_dir()]
                                for p in d.iterdir() if SERVE_NAME_RE.match(p.name)}) if Path(base).is_dir() else []
                body = ("\n".join(names) + "\n").encode()
            elif SERVE_NAME_RE.match(name):
                body = served_files(base, name)
                if body is not None and tail is not None and name.endswith(".csv"):
                    body = tail_of(body, tail)
            else:
                body = None
            if body is None:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass
    return Handler


def start_http(base, host, port):
    """Read-only HTTP server (stdlib) in a daemon thread; returns the server (call .shutdown())."""
    srv = ThreadingHTTPServer((host, port), make_handler(base))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def _seconds_to_midnight():
    now = time.localtime()
    return 86400 - (now.tm_hour * 3600 + now.tm_min * 60 + now.tm_sec) + 0.5


async def run_segment(servers, folder, seg_s, pings_per_s, node_name, stop):
    """Log into `folder` until stop is set or seg_s seconds (None = no limit) have passed."""
    folder.mkdir(parents=True, exist_ok=True)
    ev = Events(folder / "events.log")
    writers, tasks = [], []
    for label, addr in servers:
        a = Rows(folder / f"{label}_analog.csv", ANALOG_COLS)
        p = Rows(folder / f"{label}_ping.csv", PING_COLS)
        writers += [a, p]
        ev.log(label, f"measuring {addr} (monotonic_ns client clock), pings/s={pings_per_s}")
        tasks.append(asyncio.create_task(analog_loop(label, addr, a, ev, stop, node_name)))
        tasks.append(asyncio.create_task(ping_loop(label, addr, p, ev, stop, pings_per_s)))
    try:
        await _sleep_or_stop(stop, seg_s)
    finally:
        stop.set()
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        for w in writers:
            w.close()
        ev.log("all", "logger stopped")
        ev.f.close()


async def run_logger(servers, out_dir, duration_s, pings_per_s, node_name, serve_port=None,
                     serve_host="0.0.0.0"):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    quit_ev = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, quit_ev.set)
        except NotImplementedError:
            pass
    srv = start_http(out, serve_host, serve_port) if serve_port else None
    if srv:
        print(f"serving clock log of {out} on http://{serve_host}:{serve_port}/", flush=True)
    try:
        while not quit_ev.is_set():
            seg = asyncio.Event()  # ends this segment; quit_ev ends the run
            waiter = asyncio.create_task(_relay(quit_ev, seg))
            if duration_s > 0:   # fixed run: one segment into DIR itself
                await run_segment(servers, out, duration_s, pings_per_s, node_name, seg)
                break
            day = time.strftime("%Y%m%d")  # run until stopped: a new subfolder per local day
            await run_segment(servers, out / day, _seconds_to_midnight(), pings_per_s, node_name, seg)
            waiter.cancel()
    finally:
        if srv:
            srv.shutdown()
    return 0


async def _relay(src, dst):
    await src.wait()
    dst.set()


def parse_server(s):
    if "=" not in s:
        raise argparse.ArgumentTypeError("use LABEL=HOST:PORT")
    label, addr = s.split("=", 1)
    if not label or ":" not in addr or "/" in label:
        raise argparse.ArgumentTypeError("use LABEL=HOST:PORT")
    return label, addr


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "analyze":
        ap = argparse.ArgumentParser(prog="clock_logger.py analyze")
        ap.add_argument("dir")
        ap.add_argument("--reference", help="label whose NEW mapping is the reference")
        ap.add_argument("--truth-zero", action="store_true", help="reference is the identity mapping (loopback test)")
        args = ap.parse_args(argv[1:])
        _rep, text = analyze_dir(args.dir, args.reference, args.truth_zero)
        print(text)
        print(f"written: {Path(args.dir) / 'report.txt'} and report.json")
        return 0
    ap = argparse.ArgumentParser(description="Log clock-mapping data from one or more Thalamus clock servers.")
    ap.add_argument("--server", action="append", type=parse_server, required=True, metavar="LABEL=HOST:PORT",
                    help="repeatable, e.g. laptop=192.168.50.20:50050")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--duration-s", type=float, default=0, help="0 = until Ctrl+C")
    ap.add_argument("--pings-per-s", type=float, default=5)
    ap.add_argument("--analog-node", default="", help="node name in the AnalogRequest (default empty)")
    ap.add_argument("--serve-port", type=int, default=0,
                    help="serve the CSVs and events.log read-only over HTTP on this port (default off; use 50061)")
    ap.add_argument("--serve-host", default="0.0.0.0")
    args = ap.parse_args(argv)
    if len({l for l, _ in args.server}) != len(args.server):
        ap.error("duplicate label")
    try:
        return asyncio.run(run_logger(args.server, args.out, args.duration_s, args.pings_per_s, args.analog_node,
                                      args.serve_port or None, args.serve_host))
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
