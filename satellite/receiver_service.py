#!/usr/bin/env python3
"""Service on the receiving laptop: store the video that arrives from the satellite.

Run inside the laptop's Thalamus conda env, while satellite/launch_receiver.bat is running.

  python satellite/receiver_service.py start  --output-dir D [--serial 4103305000]
  python satellite/receiver_service.py stop   [--output-dir D]
  python satellite/receiver_service.py status [--output-dir D]
  common options: --address localhost:50070  --certs satellite/certs
  clock sync at stop: --clock-log-url http://<satellite>:50061/ [--clock-label LABEL]

`start` points the receiver's STORAGE2 node `storage` at a staging file under
<D>/.incomplete/<camera tag>/ and sets its `Running` true. `stop` sets `Running` false,
waits for the file to stop growing, then (in staging) runs stream_check, converts the H264
stream to mp4 (satellite/h264_to_mp4.py, needs `pip install av`), and only then moves the finished files into D:

  recording_YYYYMMDD_HHMMSS_sat<serial>-0000.mp4         (viewable video)
  recording_YYYYMMDD_HHMMSS_sat<serial>.thalamus         (raw Thalamus recording, STORAGE2 file)
  recording_YYYYMMDD_HHMMSS_sat<serial>.thalamus.json    (its config sidecar, if STORAGE2 wrote one)
  recording_YYYYMMDD_HHMMSS_sat<serial>_stream_check.txt / .json
  recording_YYYYMMDD_HHMMSS_sat<serial>_satellite_sync.csv        (one row per frame, clock mapping)
  recording_YYYYMMDD_HHMMSS_sat<serial>_satellite_manifest.json   (written last)

Clock sync (plan step 3): with --clock-log-url, `stop` fetches the clock log that
satellite/clock_logger.py serves on the satellite (<label>_analog.csv, <label>_ping.csv), fits the
satellite-steady -> laptop-steady mapping (clock_logger.fit_mapping) and writes the sync CSV
(frame_index, satellite_steady_ns, laptop_steady_ns, laptop_wall_ns, arrival_laptop_steady_ns) plus a
`sync` block in the manifest. Without the option, or if the fetch or fit fails, stop still succeeds
and the sync block says "not synced" with the reason; the CSV then holds the fallback estimate
"arrival-min (no clock service)" (min-filtered time - remote_time line, no wall time).

Every command also writes its result to <D>/receiver_service_last.json. Exit code is 0 on
success, 1 on a refused or failed command.

ThalamusControl is built exactly as thalamus/registry.py does it (grpc_tls.configure, then
ThalamusThread(address) and its `config` ObservableDict). It is the only part that needs
Thalamus; everything else takes any object with the same small interface (see
ThalamusControlBase), which is what the tests use.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE.parent), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import clock_logger  # noqa: E402
import stream_check  # noqa: E402
from backend.disk_guard import (  # noqa: E402
    assess_disk,
    resolve_planned_hours,
    sample_disk_usage,
)
from backend.recording_paths import (  # noqa: E402
    CAMERA_TAG_RE,
    SessionPaths,
    ensure_directory,
)

DEFAULT_ADDRESS = "localhost:50070"
DEFAULT_CERTS = _HERE / "certs"
CERT_FILES = {"ca": "ca.pem", "cert": "thalamus.internal.pem", "key": "thalamus.internal-key.pem"}
DEFAULT_SERVER_NAME = "thalamus.internal"
STORAGE_NODE = "storage"
CAMERA_NODE = "sat_cam1"
STREAM_NODE = "sat_cam1"  # node name inside the recording (the Sources entry of `storage`)
DEFAULT_FPS = 15.0
# H264 1280x1024 at 15 fps measured about 2 MB/s (7.2e9 B/h); overridable.
DEFAULT_BYTES_PER_HOUR = 7_200_000_000
CLOCK_PAD_NS = 120 * 10**9      # clock-log rows kept around the recording's remote_time range
CLOCK_MIN_SPAN_NS = 60 * 10**9  # need at least this much clock-log time, else use all rows
CLOCK_TIMEOUT_S = 15.0
FALLBACK_METHOD = "arrival-min (no clock service)"
SYNC_COLS = ["frame_index", "satellite_steady_ns", "laptop_steady_ns", "laptop_wall_ns",
             "arrival_laptop_steady_ns"]
STATE_FILE = "receiver_session.json"
LAST_RESULT_FILE = "receiver_service_last.json"
SCHEMA_VERSION = 1


class ServiceError(RuntimeError):
    """A command was refused or failed; the message says why in plain words."""


# ---------------------------------------------------------------- Thalamus access
class ThalamusControlBase:
    """The small interface the service needs. Tests implement it with a fake."""

    def connect(self) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def close(self) -> None:  # pragma: no cover - interface
        pass

    @property
    def connected(self) -> bool:  # pragma: no cover - interface
        raise NotImplementedError

    def get_node(self, name: str) -> Optional[dict]:  # pragma: no cover - interface
        """Plain-dict snapshot of the node's config/state, or None if there is no such node."""
        raise NotImplementedError

    def set_node_property(self, name: str, key: str, value: Any) -> None:  # pragma: no cover
        """Set one property and return once Thalamus acknowledged it. Raise ServiceError on failure."""
        raise NotImplementedError


class ThalamusControl(ThalamusControlBase):
    """Talks to a running Thalamus over gRPC (TLS client certificates), like thalamus/registry.py.

    UNVERIFIED: written from the installed 1.0.53 source (registry.py, thread.py, grpc_tls.py,
    config.py) but never run against a real Thalamus.
    """

    def __init__(self, address: str = DEFAULT_ADDRESS, certs_dir: str | Path | None = None,
                 server_name: str = DEFAULT_SERVER_NAME, timeout_s: float = 15.0):
        self.address = address
        self.certs_dir = Path(certs_dir) if certs_dir else DEFAULT_CERTS
        self.server_name = server_name
        self.timeout_s = timeout_s
        self._thread = None
        self._qapp = None

    @property
    def connected(self) -> bool:
        t = self._thread
        return t is not None and t.running and t.stub is not None

    def connect(self) -> None:
        import argparse as _ap
        import threading

        paths = {k: self.certs_dir / v for k, v in CERT_FILES.items()}
        missing = [str(p) for p in paths.values() if not p.is_file()]
        if missing:
            raise ServiceError("TLS files not found: " + ", ".join(missing))
        # registry.py creates a QApplication (thalamus.qt); do the same, off screen.
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        try:
            from thalamus.qt import QApplication  # type: ignore
            if QApplication.instance() is None:
                self._qapp = QApplication(sys.argv[:1])
        except Exception:  # Qt is optional for the gRPC thread itself
            self._qapp = None
        try:
            from thalamus import grpc_tls  # type: ignore
            from thalamus.thread import ThalamusThread  # type: ignore
        except ImportError as exc:
            raise ServiceError(
                "the thalamus package is not importable here (%s); run this inside the "
                "laptop's thalamus conda env" % exc) from exc
        grpc_tls.configure(_ap.Namespace(
            cert=str(paths["cert"]), key=str(paths["key"]), ca=str(paths["ca"]),
            server_name=self.server_name))
        thread = ThalamusThread(self.address)
        # ThalamusThread.start() waits forever if the server is unreachable: bound it.
        starter = threading.Thread(target=thread.start, daemon=True)
        starter.start()
        starter.join(self.timeout_s)
        if starter.is_alive() or thread.stub is None:
            thread.running = False
            raise ServiceError("could not connect to Thalamus at %s within %.0f s "
                               "(is launch_receiver.bat running? are the certs right?)"
                               % (self.address, self.timeout_s))
        self._thread = thread

    def close(self) -> None:
        # ThalamusThread.stop() joins a thread blocked on the stream, so only flag it.
        if self._thread is not None:
            self._thread.running = False
        self._thread = None

    def _find(self, name: str):
        for node in self._thread.config.get("nodes", []):
            if node.get("name") == name:
                return node
        return None

    def get_node(self, name: str) -> Optional[dict]:
        if self._thread is None:
            raise ServiceError("not connected to Thalamus")
        node = self._find(name)
        if node is None:
            return None
        return node.unwrap() if hasattr(node, "unwrap") else dict(node)

    def set_node_property(self, name: str, key: str, value: Any) -> None:
        import asyncio

        if self._thread is None or self._thread.loop is None:
            raise ServiceError("not connected to Thalamus")
        thread = self._thread

        async def _set():
            node = self._find(name)
            if node is None:
                raise ServiceError("Thalamus has no node named %r" % name)
            done = asyncio.get_running_loop().create_future()
            node.setitem(key, value, lambda: done.done() or done.set_result(None))
            await asyncio.wait_for(done, self.timeout_s)

        fut = asyncio.run_coroutine_threadsafe(_set(), thread.loop)
        try:
            fut.result(self.timeout_s + 5)
        except ServiceError:
            raise
        except Exception as exc:
            raise ServiceError("setting %s.%s failed: %s: %s" % (name, key, type(exc).__name__, exc)) from exc


# ---------------------------------------------------------------- file helpers
def replace_with_retry(src: Path, dst: Path, attempts: int = 6, delay_s: float = 0.1,
                       sleep: Callable[[float], None] = time.sleep) -> None:
    """Atomic rename (os.replace) that retries briefly: Windows can refuse for a moment
    while an antivirus scanner or the muxer still holds the file."""
    last: Optional[OSError] = None
    for attempt in range(attempts):
        try:
            os.replace(src, dst)
            return
        except OSError as exc:
            last = exc
            if attempt < attempts - 1:
                sleep(delay_s * (attempt + 1))
    assert last is not None
    raise last


def write_json_atomic(path: Path, data: Any) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    replace_with_retry(tmp, path)


def _iso(dt: datetime) -> str:
    return dt.isoformat(timespec="seconds")


def installed_versions() -> dict:
    out = {}
    try:
        from importlib import metadata
        for pkg in ("thalamus", "thalamus-contrib"):
            try:
                out[pkg] = metadata.version(pkg)
            except metadata.PackageNotFoundError:
                out[pkg] = None
    except Exception:
        pass
    return out


def find_recordings(staging: Path, stem: str) -> list[Path]:
    """STORAGE2 files named <stem>.YYYYMMDD.R, oldest to newest by number then mtime."""
    rx = re.compile(r"^%s\.(\d{8})\.(\d+)$" % re.escape(stem))
    found = [p for p in staging.iterdir() if p.is_file() and rx.match(p.name)]
    return sorted(found, key=lambda p: (p.stat().st_mtime, p.name))


def find_sidecar(raw: Path, stem: str) -> Optional[Path]:
    """The .json STORAGE2 writes beside a recording; its exact name is UNVERIFIED, so try both."""
    for cand in (raw.with_name(raw.name + ".json"), raw.with_name(stem + ".json")):
        if cand.is_file():
            return cand
    return None


# ---------------------------------------------------------------- clock sync
def _http_get(url: str, timeout: float) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read().decode("utf-8", "replace")


def fetch_clock_log(url: str, label: Optional[str] = None, timeout: float = CLOCK_TIMEOUT_S):
    """Fetch <label>_analog.csv and <label>_ping.csv from the satellite's clock_logger HTTP server.
    label None: use the only label listed at the root. Returns (label, analog_rows, ping_rows)."""
    base = url if url.endswith("/") else url + "/"
    if not label:
        names = _http_get(base, timeout).split()
        labels = sorted(n[:-len("_analog.csv")] for n in names if n.endswith("_analog.csv"))
        if len(labels) != 1:
            raise ServiceError("clock log lists %s label(s) %s; choose one with --clock-label"
                               % (len(labels), labels))
        label = labels[0]
    rows = {}
    for kind in ("analog", "ping"):
        text = _http_get("%s%s_%s.csv" % (base, label, kind), timeout)
        rows[kind] = list(csv.DictReader(io.StringIO(text)))
    return label, rows["analog"], rows["ping"]


def read_frames(raw: Path, node: str) -> list:
    """[(arrival time, remote_time)] in ns, one per image record of `node`, in file order."""
    return [(itime or rtime, rem)
            for (n, rtime, _w, _h, _f, itime, rem), _st in stream_check.iter_image_records(raw) if n == node]


def window_rows(analog, ping, lo, hi):
    """Clock-log rows covering [lo, hi] (satellite steady ns) plus CLOCK_PAD_NS each side.
    Widens to all rows when that gives under CLOCK_MIN_SPAN_NS of analog time. -> (a, p, widened)"""
    a = [r for r in analog if lo - CLOCK_PAD_NS <= int(r["client_mono_ns"]) <= hi + CLOCK_PAD_NS]
    p = [r for r in ping if str(r["t1"]) != "" and lo - CLOCK_PAD_NS <= int(r["t1"]) <= hi + CLOCK_PAD_NS]
    span = (int(a[-1]["client_mono_ns"]) - int(a[0]["client_mono_ns"])) if len(a) > 1 else 0
    if span >= CLOCK_MIN_SPAN_NS:
        return a, p, False
    return analog, ping, True


def fallback_fit(frames) -> dict:
    """Mapping from the recording alone: y = time - remote_time per frame, the minimum per 10 s
    window (least delayed frame), Theil-Sen line. Includes the smallest transfer delay."""
    import numpy as np
    pts = [(rem, t - rem) for t, rem in frames if rem]
    if len(pts) < 2:
        raise ValueError("fewer than 2 frames with remote_time")
    x = np.array([p[0] for p in pts], dtype=np.float64)
    y = np.array([p[1] for p in pts], dtype=np.float64)
    x0, y0 = float(x[0]), float(y[0])
    k = clock_logger.select_max_per_window(x - x0, -(y - y0))  # max of -y = min of y
    slope, icpt = clock_logger.theil_sen((x - x0)[k], (y - y0)[k])
    return {"method": FALLBACK_METHOD, "slope": float(slope), "drift_ppm": float(slope * 1e6),
            "uncertainty_ms": None, "n_samples": int(k.size), "n_rows": len(pts),
            "span_s": float(x.max() - x.min()) / 1e9, "x0_ns": int(round(x0)),
            "offset_at_x0_ns": y0 + icpt, "wall_minus_steady_median_ns": None,
            "instance_ids": {}, "wall_jumps": 0, "warnings": []}


# ---------------------------------------------------------------- the service
class ReceiverService:
    def __init__(self, control: ThalamusControlBase, *, storage_node: str = STORAGE_NODE,
                 camera_node: str = CAMERA_NODE, stream_node: str = STREAM_NODE,
                 expected_fps: float = DEFAULT_FPS, bytes_per_hour: int = DEFAULT_BYTES_PER_HOUR,
                 camera_serial: str | None = None,
                 sleep: Callable[[float], None] = time.sleep,
                 clock: Callable[[], float] = time.monotonic,
                 now: Callable[[], datetime] = datetime.now,
                 disk_usage: Callable[[str], Any] = shutil.disk_usage,
                 runner: Callable[..., Any] = subprocess.run,
                 log: Callable[[str], None] = print,
                 verify_start_s: float = 10.0,
                 clock_log_url: Optional[str] = None, clock_label: Optional[str] = None,
                 clock_timeout_s: float = CLOCK_TIMEOUT_S,
                 fetch_clock: Callable[..., Any] = fetch_clock_log):
        self.control = control
        self.storage_node = storage_node
        self.camera_node = camera_node
        self.stream_node = stream_node
        self.expected_fps = expected_fps
        self.bytes_per_hour = bytes_per_hour
        self.camera_tag = "sat" + str(camera_serial) if camera_serial else "sat"
        if not CAMERA_TAG_RE.match(self.camera_tag):
            raise ValueError("camera serial %r cannot be used in a filename" % camera_serial)
        self.sleep, self.clock, self.now = sleep, clock, now
        self.disk_usage, self.runner, self.log = disk_usage, runner, log
        # After Running=True, wait up to this long for STORAGE2's file to appear (0 = don't check).
        # Thalamus 1.0.66 once accepted Running=True without ever opening a file (2026-11-24).
        self.verify_start_s = verify_start_s
        self.clock_log_url, self.clock_label = clock_log_url, clock_label
        self.clock_timeout_s, self.fetch_clock = clock_timeout_s, fetch_clock

    # -- paths
    def session_paths(self, output_dir: Path, started_at: datetime) -> SessionPaths:
        return SessionPaths.for_session(output_dir, started_at, self.camera_tag)

    @staticmethod
    def state_path(staging: Path) -> Path:
        return staging / STATE_FILE

    def _write_last(self, output_dir: Optional[Path], result: dict) -> None:
        target = Path(output_dir) if output_dir else Path.cwd()
        try:
            ensure_directory(target)
            write_json_atomic(target / LAST_RESULT_FILE, result)
            result["result_file"] = str(target / LAST_RESULT_FILE)
        except OSError as exc:  # never let reporting hide the real outcome
            result["result_file_error"] = str(exc)

    # -- start
    def start(self, output_dir: str | Path, started_at: Optional[datetime] = None,
              force: bool = False) -> dict:
        output_dir = Path(output_dir)
        started_at = started_at or self.now()
        result: dict = {"command": "start", "ok": False, "output_dir": str(output_dir),
                        "warnings": []}
        try:
            self._start(output_dir, started_at, force, result)
            result["ok"] = True
        except ServiceError as exc:
            result["error"] = str(exc)
        self._write_last(output_dir, result)
        return result

    def _start(self, output_dir: Path, started_at: datetime, force: bool, result: dict) -> None:
        if not self.control.connected:
            self.control.connect()
        storage = self.control.get_node(self.storage_node)
        if storage is None:
            raise ServiceError("Thalamus has no storage node named %r" % self.storage_node)
        if storage.get("Running"):
            raise ServiceError("storage node %r is already Running; stop it first" % self.storage_node)

        ensure_directory(output_dir)
        verdict = assess_disk(
            sample_disk_usage(output_dir, at_s=time.time(), disk_usage=self.disk_usage),
            bytes_per_hour=self.bytes_per_hour, planned_hours=resolve_planned_hours())
        result["disk"] = {"level": verdict.level, "free_bytes": verdict.free_bytes,
                          "hours_remaining": verdict.hours_remaining, "reason": verdict.reason}
        if verdict.level == "danger" and not force:
            raise ServiceError("disk check refused the start: " + verdict.reason)
        if verdict.level != "safe":
            result["warnings"].append("disk: " + verdict.reason)

        cam = self.control.get_node(self.camera_node)
        if cam is None:
            result["warnings"].append("no node %r found; storage will record nothing" % self.camera_node)
        elif not cam.get("Running"):
            result["warnings"].append("%r is not Running; storage will record nothing until it is"
                                      % self.camera_node)

        paths = self.session_paths(output_dir, started_at)
        staging = paths.incomplete_dir
        state_file = self.state_path(staging)
        if state_file.exists():
            raise ServiceError("a session is already active (%s); run stop first" % state_file)
        ensure_directory(staging)
        stem_path = staging / paths.stem
        state = {
            "schema": SCHEMA_VERSION, "phase": "recording",
            "output_dir": str(output_dir), "staging_dir": str(staging),
            "basename": paths.basename, "camera_tag": self.camera_tag, "stem": paths.stem,
            "storage_node": self.storage_node, "stream_node": self.stream_node,
            "expected_fps": self.expected_fps, "bytes_per_hour": self.bytes_per_hour,
            "started_at": _iso(started_at), "warnings": result["warnings"],
        }
        write_json_atomic(state_file, state)
        try:
            self.control.set_node_property(self.storage_node, "Output File", stem_path.as_posix())
            if self.verify_start_s > 0:
                self.sleep(0.5)  # let the new Output File settle before Running flips
            self.control.set_node_property(self.storage_node, "Running", True)
            if self.verify_start_s > 0:
                attempts = []
                for attempt in (1, 2):
                    if self._wait_for_file(staging, paths.stem, self.verify_start_s):
                        break
                    attempts.append(attempt)
                    if attempt == 1:  # one retry: toggle Running off and on
                        self.control.set_node_property(self.storage_node, "Running", False)
                        self.sleep(1.0)
                        self.control.set_node_property(self.storage_node, "Running", True)
                else:
                    self.control.set_node_property(self.storage_node, "Running", False)
                    raise ServiceError(
                        "storage is Running but no %s.YYYYMMDD.R file appeared in %s within %.0f s "
                        "(tried twice); recording NOT started" % (paths.stem, staging, self.verify_start_s))
                if attempts:
                    result["warnings"].append("storage needed a restart before its file appeared")
        except ServiceError:
            with contextlib.suppress(OSError):
                state_file.unlink()
            raise
        result.update({"stem": paths.stem, "staging_dir": str(staging),
                       "output_file_set": stem_path.as_posix(), "state_file": str(state_file),
                       "started_at": state["started_at"]})

    def _wait_for_file(self, staging: Path, stem: str, timeout_s: float) -> bool:
        deadline = self.clock() + timeout_s
        while True:
            if find_recordings(staging, stem):
                return True
            if self.clock() >= deadline:
                return False
            self.sleep(0.25)

    # -- stop
    def locate_state(self, output_dir: Optional[Path]) -> tuple[Path, dict]:
        """Find the active session's state file: from output_dir, else from the storage node's
        current `Output File` (its folder is the staging folder)."""
        candidates: list[Path] = []
        if output_dir:
            base = Path(output_dir) / ".incomplete"
            candidates.append(base)
            if base.is_dir():
                candidates += sorted(p for p in base.iterdir() if p.is_dir())
        else:
            node = self.control.get_node(self.storage_node) or {}
            of = node.get("Output File")
            if of:
                candidates.append(Path(of).parent)
        for d in candidates:
            sf = self.state_path(d)
            if sf.is_file():
                return sf, json.loads(sf.read_text(encoding="utf-8"))
        raise ServiceError("no active session found (looked for %s in: %s)" % (
            STATE_FILE, ", ".join(str(c) for c in candidates) or "nowhere; give --output-dir"))

    def stop(self, output_dir: str | Path | None = None, stable_s: float = 2.0,
             poll_s: float = 0.5, timeout_s: float = 60.0, mux_timeout_s: float = 600.0) -> dict:
        out = Path(output_dir) if output_dir else None
        result: dict = {"command": "stop", "ok": False, "warnings": []}
        try:
            self._stop(out, stable_s, poll_s, timeout_s, mux_timeout_s, result)
            result["ok"] = True
        except ServiceError as exc:
            result["error"] = str(exc)
        self._write_last(Path(result["output_dir"]) if result.get("output_dir") else out, result)
        return result

    def wait_until_stable(self, path: Path, stable_s: float, poll_s: float, timeout_s: float) -> int:
        """Poll the file's size until it is unchanged for stable_s; return the final size."""
        deadline = self.clock() + timeout_s
        last = -1
        since = self.clock()
        while True:
            size = path.stat().st_size
            if size != last:
                last, since = size, self.clock()
            elif self.clock() - since >= stable_s:
                return size
            if self.clock() >= deadline:
                raise ServiceError("%s was still growing after %.0f s" % (path.name, timeout_s))
            self.sleep(poll_s)

    def _stop(self, out: Optional[Path], stable_s, poll_s, timeout_s, mux_timeout_s, result: dict):
        if not self.control.connected:
            self.control.connect()
        state_file, state = self.locate_state(out)
        output_dir, staging, stem = Path(state["output_dir"]), Path(state["staging_dir"]), state["stem"]
        result.update({"output_dir": str(output_dir), "stem": stem})
        stopped_at = self.now()

        self.control.set_node_property(state.get("storage_node", self.storage_node), "Running", False)
        recs = find_recordings(staging, stem)
        if not recs:
            raise ServiceError("storage stopped but no %s.YYYYMMDD.R file exists in %s "
                               "(was the camera Running, so storage had data to write?)" % (stem, staging))
        raw = recs[-1]
        if len(recs) > 1:
            result["warnings"].append("%d recordings for this stem; used the newest (%s)" % (len(recs), raw.name))
        size = self.wait_until_stable(raw, stable_s, poll_s, timeout_s)
        state["phase"] = "stopped"
        state["stopped_at"] = _iso(stopped_at)
        write_json_atomic(state_file, state)

        # (a) stream check, in staging
        check = self._stream_check(raw, staging, state)
        # (a2) clock mapping, in staging (never fails the stop)
        sync = self._clock_sync(raw, staging, stem, state["stream_node"], result)
        # (b) mp4, in staging
        mp4_tmp = staging / (stem + "-0000.mp4.part")
        mux = self._mux(raw, mp4_tmp, mux_timeout_s)
        # (c)/(d) move into output_dir; final names only appear when complete
        files: dict = {}
        sidecar = find_sidecar(raw, stem)
        if mux["ok"]:
            dst = output_dir / (stem + "-0000.mp4")
            replace_with_retry(mp4_tmp, dst)
            files["mp4"] = self._file_info(dst)
        else:
            with contextlib.suppress(OSError):
                mp4_tmp.unlink()
        for key, src, name in (("stream_check_txt", check.get("txt"), stem + "_stream_check.txt"),
                               ("stream_check_json", check.get("json"), stem + "_stream_check.json")):
            if src and Path(src).is_file():
                dst = output_dir / name
                replace_with_retry(Path(src), dst)
                files[key] = self._file_info(dst)
        if sync.get("csv_tmp"):
            dst = output_dir / (stem + "_satellite_sync.csv")
            replace_with_retry(Path(sync.pop("csv_tmp")), dst)
            files["sync_csv"] = self._file_info(dst)
            sync["csv"] = dst.name
        raw_dst = output_dir / (stem + ".thalamus")
        replace_with_retry(raw, raw_dst)  # rename keeps the modification time
        files["raw"] = self._file_info(raw_dst)
        if sidecar is not None:
            sc_dst = output_dir / (stem + ".thalamus.json")
            replace_with_retry(sidecar, sc_dst)
            files["raw_json"] = self._file_info(sc_dst)
        else:
            result["warnings"].append("no .json sidecar found next to the recording")

        if not mux["ok"]:
            result["warnings"].append("mp4 conversion failed; raw recording kept: " + str(mux.get("error")))
        manifest = {
            "schema": SCHEMA_VERSION, "kind": "satellite_recording",
            "basename": state["basename"], "camera_tag": state["camera_tag"], "stem": stem,
            "started_at": state["started_at"], "stopped_at": state["stopped_at"],
            "raw_size_bytes_at_stop": size,
            "frames": check.get("frames"), "fps": check.get("fps"),
            "expected_fps": state["expected_fps"], "stream_node": state["stream_node"],
            "stream_check": {k: check.get(k) for k in ("verdict", "reasons", "dropped_frames", "error")},
            "mp4": {"ok": mux["ok"], "error": mux.get("error"), "command": mux.get("command"),
                    "conversion": mux.get("conversion")},
            "sync": sync,
            "files": files, "versions": installed_versions(),
            "warnings": state.get("warnings", []) + result["warnings"],
        }
        manifest_path = output_dir / (stem + "_satellite_manifest.json")
        write_json_atomic(manifest_path, manifest)
        with contextlib.suppress(OSError):
            state_file.unlink()
        shutil.rmtree(staging / "check", ignore_errors=True)
        with contextlib.suppress(OSError):
            staging.rmdir()  # only succeeds when empty
        result.update({"manifest": str(manifest_path), "files": files, "frames": check.get("frames"),
                       "fps": check.get("fps"), "verdict": check.get("verdict"),
                       "mp4_ok": mux["ok"], "mp4_error": mux.get("error")})

    def _clock_sync(self, raw: Path, staging: Path, stem: str, node: str, result: dict) -> dict:
        """Map every frame's satellite steady time to the laptop clock. Returns the manifest's
        `sync` block (plus csv_tmp, the staged CSV). Any failure is recorded in the block."""
        sync: dict = {"synced": False, "method": FALLBACK_METHOD, "clock_log_url": self.clock_log_url,
                      "warnings": []}
        try:
            frames = read_frames(raw, node)
        except Exception as exc:
            sync["reason"] = "could not read frames: %s: %s" % (type(exc).__name__, exc)
            return sync
        stamped = [r for _t, r in frames if r]
        fit = None
        if not self.clock_log_url:
            sync["reason"] = "no --clock-log-url given"
        elif not stamped:
            sync["reason"] = "frames carry no remote_time"
        else:
            try:
                label, analog, ping = self.fetch_clock(self.clock_log_url, self.clock_label,
                                                       self.clock_timeout_s)
                a, p, widened = window_rows(analog, ping, min(stamped), max(stamped))
                if widened:
                    sync["warnings"].append("clock log has under 60 s of samples around the recording; "
                                            "used all %d rows" % len(analog))
                fit = clock_logger.fit_mapping(a, p)
                fit["warnings"] = sync["warnings"] + fit["warnings"]
                sync.update({"synced": True, "clock_label": label})
            except Exception as exc:  # network, parsing, too little data: the recording is safe anyway
                sync["reason"] = "clock log unusable: %s: %s" % (type(exc).__name__, exc)
        if fit is None:
            try:
                fit = fallback_fit(frames)
            except Exception as exc:
                sync["reason"] += "; fallback failed: %s" % exc
                result["warnings"].append("clock sync: " + sync["reason"])
                return sync
            fit["warnings"] = sync["warnings"]
        mapped = clock_logger.map_client_to_server(fit, [r for _t, r in frames if r])
        it = iter(mapped)
        wall_off = fit["wall_minus_steady_median_ns"]
        csv_path = staging / (stem + "_satellite_sync.csv")
        delays = []
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(SYNC_COLS)
            for i, (t, rem) in enumerate(frames):
                if not rem:
                    w.writerow([i, "", "", "", t])
                    continue
                m = next(it)
                delays.append(t - m)
                w.writerow([i, rem, m, "" if wall_off is None else m + int(round(wall_off)), t])
        sync.update({"method": fit["method"], "drift_ppm": fit["drift_ppm"],
                     "offset_at_start_ns": clock_logger.map_client_to_server(fit, [min(stamped)])[0] - min(stamped),
                     "uncertainty_ms": fit["uncertainty_ms"], "samples": fit["n_samples"],
                     "rows_used": fit["n_rows"], "span_s": fit["span_s"],
                     "clock_instance_ids": list(fit["instance_ids"]),
                     "wall_minus_steady_median_ns": wall_off, "csv_tmp": str(csv_path)})
        if delays:
            tmin = min(delays) / 1e6
            sync["transfer_delay_min_ms"] = tmin
            tol = fit["uncertainty_ms"] if fit["uncertainty_ms"] is not None else 1.0
            if tmin < -tol:
                fit["warnings"].append("transfer_delay_min_ms %.2f is negative beyond the uncertainty "
                                       "(%.2f ms): mapping may be wrong" % (tmin, tol))
            elif tmin > 200 and sync["synced"]:
                fit["warnings"].append("transfer_delay_min_ms %.1f is over 200 ms: mapping may be wrong "
                                       "or the laptop and server.exe clocks differ" % tmin)
        sync["warnings"] = fit["warnings"]
        if not sync["synced"]:
            result["warnings"].append("clock sync: not synced (%s); fallback estimate written" % sync["reason"])
        for wmsg in sync["warnings"]:
            result["warnings"].append("clock sync: " + wmsg)
        return sync

    @staticmethod
    def _file_info(p: Path) -> dict:
        return {"path": str(p), "size_bytes": p.stat().st_size}

    def _stream_check(self, raw: Path, staging: Path, state: dict) -> dict:
        outdir = staging / "check"
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                rc = stream_check.main([str(raw), "--node", state["stream_node"],
                                        "--expected-fps", str(state["expected_fps"]),
                                        "--out", str(outdir)])
            rep = json.loads((outdir / "report.json").read_text(encoding="utf-8"))
        except Exception as exc:
            return {"verdict": "error", "error": "%s: %s" % (type(exc).__name__, exc), "reasons": []}
        rt = rep.get("receiver_time", {})
        dropped = rt.get("estimated_dropped_frames")
        frames = rep.get("frames", 0)
        duration = rt.get("duration_s")
        fps = (frames - 1) / duration if duration and frames > 1 else None
        reasons = list(rep.get("warnings", []))
        if rc != 0 or not frames:
            reasons.append("no frames stored")
        if dropped:
            reasons.append("estimated dropped frames: %s" % dropped)
        return {"verdict": "ok" if not reasons else "problems", "reasons": reasons, "frames": frames,
                "fps": fps, "dropped_frames": dropped,
                "txt": str(outdir / "report.txt"), "json": str(outdir / "report.json")}

    def _mux(self, raw: Path, out_mp4: Path, timeout_s: float) -> dict:
        """H264 to mp4 with satellite/h264_to_mp4.py, in a child process of the same interpreter,
        so a crash inside PyAV (native code) or a corrupt recording cannot take the service down.
        The node is the REMOTE node's name inside the recording (self.stream_node)."""
        cmd = [sys.executable, str(Path(__file__).resolve().with_name("h264_to_mp4.py")),
               str(raw), str(out_mp4), "--node", self.stream_node]
        info: dict = {"ok": False, "command": cmd}
        try:
            proc = self.runner(cmd, capture_output=True, text=True, timeout=timeout_s)
        except (OSError, subprocess.SubprocessError) as exc:
            info["error"] = "%s: %s" % (type(exc).__name__, exc)
            return info
        if proc.returncode != 0:
            detail = ""
            try:  # h264_to_mp4 prints its result as JSON
                detail = "; ".join(json.loads(proc.stdout).get("errors", []))
            except (ValueError, TypeError, AttributeError):
                pass
            tail = detail or ((proc.stderr or "") + (proc.stdout or "")).strip()[-500:]
            info["error"] = "h264_to_mp4 exited with code %s: %s" % (proc.returncode, tail)
        elif not out_mp4.is_file() or out_mp4.stat().st_size == 0:
            info["error"] = "h264_to_mp4 reported success but wrote no mp4"
        else:
            info["ok"] = True
            try:
                info["conversion"] = {k: v for k, v in json.loads(proc.stdout).items()
                                      if k in ("frames_written", "frames_skipped_leading", "duration_s", "fps")}
            except (ValueError, TypeError, AttributeError):
                pass
        return info

    # -- status
    def status(self, output_dir: str | Path | None = None) -> dict:
        out = Path(output_dir) if output_dir else None
        result: dict = {"command": "status", "ok": True, "connected": False}
        try:
            if not self.control.connected:
                self.control.connect()
            result["connected"] = bool(self.control.connected)
            storage = self.control.get_node(self.storage_node)
            cam = self.control.get_node(self.camera_node)
            result["storage"] = None if storage is None else {
                "running": bool(storage.get("Running")), "output_file": storage.get("Output File")}
            result["camera"] = None if cam is None else {
                k: cam.get(k) for k in ("Running", "Connected", "Status", "Bytes Per Second") if k in cam}
            try:
                sf, state = self.locate_state(out)
                result["session"] = {k: state.get(k) for k in ("stem", "started_at", "phase", "staging_dir")}
                result.setdefault("output_dir", state["output_dir"])
                recs = find_recordings(Path(state["staging_dir"]), state["stem"])
                if recs:
                    result["current_file"] = {"path": str(recs[-1]), "size_bytes": recs[-1].stat().st_size}
            except ServiceError:
                result["session"] = None
        except ServiceError as exc:
            result["ok"] = False
            result["error"] = str(exc)
        target = out or (Path(result["output_dir"]) if result.get("output_dir") else None)
        if target is not None and target.exists():
            with contextlib.suppress(OSError):
                result["free_bytes"] = int(self.disk_usage(str(target)).free)
        self._write_last(target, result)
        return result


# ---------------------------------------------------------------- CLI
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--address", default=DEFAULT_ADDRESS, help="receiver Thalamus gRPC address")
    ap.add_argument("--certs", default=str(DEFAULT_CERTS), help="folder with ca.pem, thalamus.internal.pem, thalamus.internal-key.pem")
    ap.add_argument("--serial", help="satellite camera serial, used in the file tag (sat<serial>)")
    ap.add_argument("--expected-fps", type=float, default=DEFAULT_FPS)
    ap.add_argument("--bytes-per-hour", type=float, default=DEFAULT_BYTES_PER_HOUR)
    ap.add_argument("--verify-start-s", type=float, default=10.0,
                    help="seconds to wait for the recording file after start (0 = don't check)")
    ap.add_argument("--clock-log-url", help="satellite clock_logger HTTP server, e.g. http://192.168.50.10:50061/ "
                    "(stop maps the frames' satellite time to laptop time with it)")
    ap.add_argument("--clock-label", help="label in the clock log (default: the only one found)")
    ap.add_argument("--clock-timeout", type=float, default=CLOCK_TIMEOUT_S, help="seconds for each clock-log fetch")
    sub = ap.add_subparsers(dest="command", required=True)
    s = sub.add_parser("start")
    s.add_argument("--output-dir", required=True)
    s.add_argument("--force", action="store_true", help="start even if the disk check says critical")
    t = sub.add_parser("stop")
    t.add_argument("--output-dir")
    t.add_argument("--timeout", type=float, default=60.0, help="seconds to wait for the file to stop growing")
    st = sub.add_parser("status")
    st.add_argument("--output-dir")
    return ap


def main(argv=None, control: Optional[ThalamusControlBase] = None) -> int:
    a = build_parser().parse_args(argv)
    real = control is None
    control = control or ThalamusControl(a.address, a.certs)
    svc = ReceiverService(control, expected_fps=a.expected_fps,
                          bytes_per_hour=int(a.bytes_per_hour), camera_serial=a.serial,
                          verify_start_s=a.verify_start_s, clock_log_url=a.clock_log_url,
                          clock_label=a.clock_label, clock_timeout_s=a.clock_timeout)
    if a.command == "start":
        res = svc.start(a.output_dir, force=a.force)
    elif a.command == "stop":
        res = svc.stop(a.output_dir, timeout_s=a.timeout)
    else:
        res = svc.status(a.output_dir)
    print(json.dumps(res, indent=2, default=str))
    sys.stdout.flush()
    rc = 0 if res.get("ok") else 1
    control.close()
    if real:
        os._exit(rc)  # ThalamusThread's worker thread is not a daemon and blocks exit
    return rc


if __name__ == "__main__":
    sys.exit(main())
