"""Pure logic for satellite_window.py (no Qt, no thalamus): settings, status colour, clock-log tail fit,
label choice, result formatting.  Tests: tests/test_window_logic.py (system python3)."""
from __future__ import annotations

import collections
import csv
import itertools
import json
import os
import sys
import time
import urllib.request
from pathlib import Path
from typing import Callable, Iterable, Optional

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import clock_logger  # noqa: E402

SETTINGS_PATH = _HERE / "configs" / "window.local.json"
DEFAULT_CERTS = _HERE / "certs"
DEFAULTS = {
    "satellite_address": "192.168.50.10",
    "output_dir": "C:/thalamus_recordings/satellite",
    "serial": "4103305000",
    "expected_fps": 15,
    "clock_log_port": 50061,
    "clock_label": "",
    "receiver_address": "localhost:50070",
    "certs_dir": "",
    "start_receiver": True,
    "show_thalamus_window": False,
    "start_clock_server": True,
    "receiver_config_template": "configs/receiver.json",
    "clock_server_port": 50050,
    "preview_max_fps": 1,
    "preview_enabled": True,
}
PREFERRED_LABEL = "router"
CLOCK_TAIL_S = 120          # seconds of clock-log rows kept for the fit
STALE_CLOCK_S = 10.0        # newest clock sample older than this -> orange
STALL_S = 3.0               # recording file not growing for longer than this -> orange
ANALOG_TAIL_LINES = 1000    # requested from the satellite (?tail=N): 120 s at 1/s with margin
PING_TAIL_LINES = 1500      # 120 s at 5/s with margin
GREEN, ORANGE, RED = "green", "orange", "red"


# ---------------------------------------------------------------- settings
def load_settings(path: Path | str = SETTINGS_PATH) -> tuple[dict, Optional[str]]:
    """Settings with defaults for missing keys. Creates the file with defaults when absent.
    Returns (settings, warning); a broken file gives the defaults and a warning (file left alone)."""
    path = Path(path)
    if not path.exists():
        settings = dict(DEFAULTS)
        try:
            save_settings(settings, path)
        except OSError as exc:
            return settings, "could not create %s: %s" % (path, exc)
        return settings, None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("top level is not an object")
    except (OSError, ValueError) as exc:
        return dict(DEFAULTS), "settings file %s unusable (%s); using defaults" % (path, exc)
    return {**DEFAULTS, **data}, None


def save_settings(settings: dict, path: Path | str = SETTINGS_PATH) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def update_setting(key: str, value, path: Path | str = SETTINGS_PATH) -> Optional[str]:
    """Change one key in the settings file, keeping everything else the user wrote (the file is re-read, not
    rebuilt from memory). Returns a warning text when it could not be saved."""
    path = Path(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else dict(DEFAULTS)
        if not isinstance(data, dict):
            raise ValueError("top level is not an object")
        data[key] = value
        save_settings(data, path)
    except (OSError, ValueError) as exc:
        return "could not save %s to %s: %s" % (key, path, exc)
    return None


def clock_log_url(settings: dict) -> str:
    return "http://%s:%d/" % (settings["satellite_address"], int(settings["clock_log_port"]))


def certs_dir(settings: dict) -> Path:
    return Path(settings["certs_dir"]) if settings.get("certs_dir") else DEFAULT_CERTS


# ---------------------------------------------------------------- clock log tail
def choose_label(labels: list[str], wanted: str = "") -> str:
    """Configured label if given; else the only one; else 'router' if present, else the first (sorted)."""
    if wanted:
        return wanted
    labels = sorted(labels)
    if not labels:
        raise ValueError("the clock log lists no labels (no <label>_analog.csv)")
    if len(labels) == 1:
        return labels[0]
    return PREFERRED_LABEL if PREFERRED_LABEL in labels else labels[0]


def labels_from_listing(text: str) -> list[str]:
    suffix = "_analog.csv"
    return sorted(n[:-len(suffix)] for n in text.split() if n.endswith(suffix))


def http_lines(url: str, timeout: float = 2.0) -> Iterable[str]:
    """Stream a text file line by line (nothing is kept in memory)."""
    with urllib.request.urlopen(url, timeout=timeout) as r:
        for raw in r:
            yield raw.decode("utf-8", "replace")


def fetch_tail(get_lines: Callable[[str], Iterable[str]], url: str, max_lines: int) -> list[dict]:
    """CSV rows (dicts) of the last max_lines lines of a served CSV; the header is read once."""
    it = iter(get_lines(url))
    try:
        header = next(it)
    except StopIteration:
        return []
    tail = collections.deque(it, maxlen=max_lines)
    return list(csv.DictReader(itertools.chain([header], tail)))


def _num(row: dict, key: str) -> Optional[int]:
    v = row.get(key)
    if v is None or str(v).strip() == "":
        return None
    try:
        return int(v)
    except ValueError:
        return None


def last_seconds(analog: list[dict], ping: list[dict], seconds: float = CLOCK_TAIL_S):
    """Keep complete rows within `seconds` of the newest sample (satellite steady clock)."""
    a = [r for r in analog if None not in (_num(r, "client_mono_ns"), _num(r, "server_wall_ns"),
                                           _num(r, "server_steady_ns"))]
    p = [r for r in ping if _num(r, "t1") is not None and _num(r, "t4") is not None]
    newest = max([_num(r, "client_mono_ns") for r in a] + [_num(r, "t4") for r in p], default=None)
    if newest is None:
        return [], [], None
    lim = newest - int(seconds * 1e9)
    return ([r for r in a if _num(r, "client_mono_ns") >= lim],
            [r for r in p if _num(r, "t1") >= lim], newest)


class ClockMonitor:
    """Fetches the tail of the satellite clock log and fits the clock mapping. poll() never raises."""

    def __init__(self, base_url: str, label: str = "", *,
                 get_lines: Callable[[str], Iterable[str]] = http_lines,
                 now_ns: Callable[[], int] = time.perf_counter_ns,
                 tail_s: float = CLOCK_TAIL_S):
        self.base = base_url if base_url.endswith("/") else base_url + "/"
        self.wanted = label
        self.get_lines, self.now_ns, self.tail_s = get_lines, now_ns, tail_s

    def poll(self) -> dict:
        label = self.wanted
        try:
            if not label:
                label = choose_label(labels_from_listing("".join(self.get_lines(self.base))))
            analog = fetch_tail(self.get_lines, "%s%s_analog.csv?tail=%d" % (self.base, label, ANALOG_TAIL_LINES), ANALOG_TAIL_LINES)
            ping = fetch_tail(self.get_lines, "%s%s_ping.csv?tail=%d" % (self.base, label, PING_TAIL_LINES), PING_TAIL_LINES)
        except Exception as exc:  # network, 404, parsing
            return {"ok": False, "label": label or None, "reachable": False,
                    "error": "clock log unreachable: %s" % _short(exc)}
        try:
            a, p, newest = last_seconds(analog, ping, self.tail_s)
            fit = clock_logger.fit_mapping(a, p)
        except Exception as exc:
            return {"ok": False, "label": label, "reachable": True, "error": _short(exc)}
        mapped_newest = clock_logger.map_client_to_server(fit, [newest])[0]
        return {"ok": True, "label": label, "reachable": True, "error": None,
                "uncertainty_ms": fit["uncertainty_ms"], "method": fit["method"],
                "drift_ppm": fit["drift_ppm"], "age_s": (self.now_ns() - mapped_newest) / 1e9,
                "n_rows": fit["n_rows"]}


def _short(exc: Exception, n: int = 120) -> str:
    s = "%s: %s" % (type(exc).__name__, exc) if str(exc) else type(exc).__name__
    return s if len(s) <= n else s[:n - 1] + "…"


# ---------------------------------------------------------------- recording growth
class GrowthTracker:
    """Rate (bytes/s over ~5 s) and time since the file last grew. Feed it (size, now) per poll."""

    def __init__(self, window_s: float = 5.0):
        self.window_s = window_s
        self.reset()

    def reset(self) -> None:
        self.samples: collections.deque = collections.deque()
        self.last_size: Optional[int] = None
        self.last_change_t: Optional[float] = None
        self.first_t: Optional[float] = None

    def update(self, size: Optional[int], now: float) -> None:
        if self.first_t is None:
            self.first_t = now
        if size is not None:
            if self.last_size is None or size != self.last_size:
                self.last_change_t = now
            self.last_size = size
            self.samples.append((now, size))
        while self.samples and now - self.samples[0][0] > self.window_s:
            self.samples.popleft()

    @property
    def rate_bps(self) -> Optional[float]:
        if len(self.samples) < 2:
            return None
        (t0, s0), (t1, s1) = self.samples[0], self.samples[-1]
        return max(0.0, (s1 - s0) / (t1 - t0)) if t1 > t0 else None

    def stalled_s(self, now: float) -> float:
        ref = self.last_change_t if self.last_change_t is not None else self.first_t
        return 0.0 if ref is None else now - ref


# ---------------------------------------------------------------- status line
def fmt_rate(bps: Optional[float]) -> str:
    if bps is None:
        return "– MB/s"
    return "%.1f MB/s" % (bps / 1e6) if bps >= 1e6 else "%.0f kB/s" % (bps / 1e3)


def fmt_elapsed(s: float) -> str:
    s = int(max(0, s))
    return "%02d:%02d:%02d" % (s // 3600, s % 3600 // 60, s % 60)


def decide_status(s: dict) -> tuple[str, str]:
    """(level, text) from a status dict. Keys (all optional):
    busy None|'starting'|'stopping'; receiver_connected; recording; elapsed_s; rate_bps; stalled_s;
    clock None (not fetched yet) or the ClockMonitor.poll() dict;
    stream (preview worker: {running, fps, age_s, arriving}; replaces camera_on);
    step (what the setup worker is doing, e.g. "starting receiver…": orange); camera_on True/False/None;
    clock_server_note ("clock server: started here"); service_errors (list of short texts: red).
    red = not synced / receiver not reachable / clock log unreachable;
    orange = busy, clock not checked yet, clock sample older than 10 s, or recording file not growing > 3 s."""
    level = GREEN
    parts = []

    def worse(new):
        nonlocal level
        if new == RED or (new == ORANGE and level == GREEN):
            level = new

    clock = s.get("clock")
    if clock is None:
        worse(ORANGE)
        parts.append("checking clock…")
    elif not clock.get("ok"):
        worse(RED)
        parts.append("NOT synced" if clock.get("reachable") else "clock log unreachable")
    else:
        txt = "Synced ±%.1f ms" % clock["uncertainty_ms"]
        if clock.get("label"):
            txt += " (%s)" % clock["label"]
        if clock["age_s"] > STALE_CLOCK_S:
            worse(ORANGE)
            txt += " – clock log %d s old" % round(clock["age_s"])
        parts.append(txt)

    if s.get("receiver_connected"):
        parts.append("receiver connected")
    elif s.get("step"):  # the setup worker is bringing it up
        worse(ORANGE)
        parts.append("receiver not connected yet")
    else:
        worse(RED)
        parts.append("receiver not reachable")

    st = s.get("stream")
    if s.get("receiver_connected") and st is not None:      # frames really arriving (preview worker)
        if st["arriving"]:
            parts.append("frames %.0f/s" % st["fps"])
        else:
            worse(ORANGE)
            parts.append("camera on, no frames" if st.get("running") else "camera off, no frames")
    elif s.get("receiver_connected") and s.get("camera_on") is not None:
        if s["camera_on"]:
            parts.append("camera on")
        else:
            worse(ORANGE)
            parts.append("camera off")
    if s.get("clock_server_note"):
        parts.append(s["clock_server_note"])
    for err in s.get("service_errors") or []:
        worse(RED)
        parts.append(err)
    if s.get("step"):
        worse(ORANGE)
        parts.append(s["step"])

    busy = s.get("busy")
    if busy == "starting":
        worse(ORANGE)
        parts.append("starting…")
    elif busy == "stopping":
        worse(ORANGE)
        parts.append("stopping… (converting video)")
    elif s.get("recording"):
        txt = "recording %s · %s" % (fmt_elapsed(s.get("elapsed_s") or 0), fmt_rate(s.get("rate_bps")))
        if (s.get("stalled_s") or 0) > STALL_S:
            worse(ORANGE)
            txt += " · file not growing"
        parts.append(txt)
    else:
        parts.append("idle")
    return level, " · ".join(parts)


# ---------------------------------------------------------------- results
def read_json_file(path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def stop_result_line(result: dict, read_json: Callable[[str], dict] = read_json_file) -> tuple[str, str]:
    """One line for a ReceiverService.stop result: (level, text). The sync verdict is in the manifest."""
    if not result.get("ok"):
        return RED, "Stop failed: %s" % (result.get("error") or "unknown error")
    sync = {}
    if result.get("manifest"):
        try:
            sync = read_json(result["manifest"]).get("sync") or {}
        except Exception as exc:
            sync = {"synced": False, "reason": "manifest unreadable (%s)" % _short(exc, 60)}
    level = GREEN
    parts = ["%s frames" % result["frames"] if result.get("frames") is not None else "frames unknown"]
    if sync.get("synced"):
        u = sync.get("uncertainty_ms")
        parts.append("synced ±%.1f ms" % u if u is not None else "synced")
    else:
        level = ORANGE
        parts.append("NOT synced: %s" % (sync.get("reason") or "no sync information"))
    if result.get("mp4_ok") is False:
        level = ORANGE
        parts.append("mp4 failed: %s" % (result.get("mp4_error") or "unknown"))
    if result.get("verdict") not in (None, "ok", "OK", "pass", "PASS"):
        level = ORANGE if level == GREEN else level
        parts.append("stream check: %s" % result["verdict"])
    parts.append(str(result.get("output_dir") or "?"))
    return level, "Saved: " + " · ".join(parts)


def start_result_line(result: dict) -> tuple[str, str]:
    if not result.get("ok"):
        return RED, "Start failed: %s" % (result.get("error") or "unknown error")
    warn = result.get("warnings") or []
    return (ORANGE, "Started with warning: %s" % warn[0]) if warn else (GREEN, "Started: %s" % result.get("stem", ""))
