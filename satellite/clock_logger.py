#!/usr/bin/env python3
"""Log and compare two ways of mapping the satellite steady clock onto the laptop steady clock.

Runs on the satellite (Linux).  Needs `pip install thalamus-neuro grpcio numpy`; the laptop must
run satellite/clock_server.py (or, for the CURRENT method only, the Sleeve clock service).

Clocks: satellite = time.monotonic_ns() (Linux CLOCK_MONOTONIC); laptop = time.perf_counter_ns()
(Windows QueryPerformanceCounter).  Offsets are "server steady minus client steady".

Log (one process measures several server addresses at the same time, e.g. cable and Wi-Fi):
  python satellite/clock_logger.py --server cable=192.168.50.20:50060 --server wifi=10.1.2.3:50060 \\
      --out DIR [--duration-s 600] [--pings-per-s 5]
  Per label it writes in DIR (flushed after every row):
    <label>_analog.csv  seq, client_mono_ns, server_wall_ns, server_steady_ns   (CURRENT method data)
    <label>_ping.csv    seq, t1, t2, t3, t4, rtt_ns, offset_ns, instance_id     (NEW method data)
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
  subtracting the median server (Wall - steady) so all methods are comparable.
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
import signal
import struct
import sys
import time
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


def load_label(out_dir, label):
    import numpy as np
    d = {"label": label}
    ap = Path(out_dir) / f"{label}_analog.csv"
    rows = _read_csv(ap) if ap.exists() else []
    d["client"] = np.array([int(r["client_mono_ns"]) for r in rows], dtype=np.int64)
    d["wall"] = np.array([int(r["server_wall_ns"]) for r in rows], dtype=np.int64)
    d["steady"] = np.array([int(r["server_steady_ns"]) for r in rows], dtype=np.int64)
    pp = Path(out_dir) / f"{label}_ping.csv"
    rows = _read_csv(pp) if pp.exists() else []
    d["ping_total"] = len(rows)
    rows = [r for r in rows if r["t2"] != "" and r["instance_id"] != ""]
    d["t1"] = np.array([int(r["t1"]) for r in rows], dtype=np.int64)
    d["rtt"] = np.array([int(r["rtt_ns"]) for r in rows], dtype=np.int64)
    d["offset"] = np.array([float(r["offset_ns"]) for r in rows], dtype=np.float64)
    d["inst"] = [r["instance_id"] for r in rows]
    return d


def discover_labels(out_dir):
    names = {p.name[:-len("_analog.csv")] for p in Path(out_dir).glob("*_analog.csv")}
    names |= {p.name[:-len("_ping.csv")] for p in Path(out_dir).glob("*_ping.csv")}
    return sorted(names)


def analyze_label(d):
    """Per-label fits (no reference comparison yet). Returns (result dict, predictors dict)."""
    import numpy as np
    res, preds = {}, {}
    # NEW: keep only the most common server instance (a restart changes the server steady clock).
    if d["t1"].size:
        insts = sorted(set(d["inst"]), key=d["inst"].count, reverse=True)
        sel = np.array([i == insts[0] for i in d["inst"]])
        r = fit_new(d["t1"][sel], d["rtt"][sel], d["offset"][sel])
        res["ping_rows"] = d["ping_total"]
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
        if "new" in results[l]:
            results[l]["new"].pop("x0_ns", None)
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
             "new": "NEW (min-RTT pings, Theil-Sen)"}
    for label, r in rep["labels"].items():
        L.append(f"=== {label} ===")
        L.append(f"analog rows: {r.get('analog_rows', 0)}   ping rows: {r.get('ping_rows', 0)} "
                 f"(with reply: {r.get('ping_with_reply', 0)})   instance IDs: {r.get('instance_ids', {})}")
        for m in ("current_wall", "current_steady", "new"):
            if m not in r:
                L.append(f"{names[m]}: no data")
                continue
            x = r[m]
            line = (f"{names[m]}: slope {x['slope_ppm']:+.3f} ppm, residual RMS {_f(x['rms_ms'], '{:.4f}')} ms, "
                    f"max {_f(x['max_ms'], '{:.4f}')} ms")
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
            for m in ("current_wall", "current_steady", "new"):
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


async def run_logger(servers, out_dir, duration_s, pings_per_s, node_name):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ev = Events(out / "events.log")
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop.set)
        except NotImplementedError:
            pass
    writers, tasks = [], []
    for label, addr in servers:
        a = Rows(out / f"{label}_analog.csv", ANALOG_COLS)
        p = Rows(out / f"{label}_ping.csv", PING_COLS)
        writers += [a, p]
        ev.log(label, f"measuring {addr} (monotonic_ns client clock), pings/s={pings_per_s}")
        tasks.append(asyncio.create_task(analog_loop(label, addr, a, ev, stop, node_name)))
        tasks.append(asyncio.create_task(ping_loop(label, addr, p, ev, stop, pings_per_s)))
    try:
        await _sleep_or_stop(stop, duration_s if duration_s > 0 else None)
    finally:
        stop.set()
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        for w in writers:
            w.close()
        ev.log("all", "logger stopped")
        ev.f.close()
    return 0


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
                    help="repeatable, e.g. cable=192.168.50.20:50060")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--duration-s", type=float, default=0, help="0 = until Ctrl+C")
    ap.add_argument("--pings-per-s", type=float, default=5)
    ap.add_argument("--analog-node", default="", help="node name in the AnalogRequest (default empty)")
    args = ap.parse_args(argv)
    if len({l for l, _ in args.server}) != len(args.server):
        ap.error("duplicate label")
    try:
        return asyncio.run(run_logger(args.server, args.out, args.duration_s, args.pings_per_s, args.analog_node))
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
