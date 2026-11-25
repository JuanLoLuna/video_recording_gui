"""Child processes of the satellite window (no Qt, no thalamus; tests: tests/test_window_processes.py).

The window starts the receiver Thalamus (like launch_receiver.bat) and the stand-in clock server when
nothing answers on their ports, and on close stops ONLY the children it started itself."""
from __future__ import annotations

import collections
import copy
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Optional

_HERE = Path(__file__).resolve().parent
SATELLITE_THALAMUS_PORT = 50050     # the camera Thalamus on the satellite (REMOTE nodes connect here)
LOCAL_HOSTS = ("localhost", "127.0.0.1", "::1")
RESTART_EVERY_S = 30.0
STARTUP_TIMEOUT_S = 30.0
LOG_TAIL_LINES = 5


def resolve(path: str | Path, base: Path = _HERE) -> Path:
    p = Path(path)
    return p if p.is_absolute() else base / p


# ---------------------------------------------------------------- config and command lines
def generate_receiver_config(template: dict, satellite_address: str,
                             port: int = SATELLITE_THALAMUS_PORT) -> dict:
    """Copy of the template with "<satellite_address>:<port>" as the Address of every REMOTE node."""
    cfg = copy.deepcopy(template)
    for node in cfg.get("nodes", []):
        if node.get("type") == "REMOTE":
            node["Address"] = "%s:%d" % (satellite_address, port)
    return cfg


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def split_address(address: str, default_port: int = 50070) -> tuple[str, int]:
    host, _, port = address.rpartition(":")
    if not host:
        return address, default_port
    return host, int(port)


def receiver_command(python: str, config: Path, certs: Path, port: int = 50070) -> list[str]:
    """Same arguments as launch_receiver.bat / .sh (ports port, port+1, port+2)."""
    return [str(python), "-m", "thalamus.pipeline", "-C", "-c", str(config),
            "-p", str(port), "-u", str(port + 1), "-d", str(port + 2),
            "--ca", str(certs / "ca.pem"), "--cert", str(certs / "thalamus.internal.pem"),
            "--key", str(certs / "thalamus.internal-key.pem")]


def receiver_env(show_window: bool, base: Optional[dict] = None) -> dict:
    """Environment for the receiver Thalamus. Hidden (default): Qt's offscreen platform, so Thalamus runs with
    no window (its dialogs are non-modal, so nothing can wait on an invisible click; see logs/receiver.log).
    Shown: QT_QPA_PLATFORM removed, because this process may have set it to offscreen for its own gRPC client."""
    env = dict(os.environ if base is None else base)
    if show_window:
        env.pop("QT_QPA_PLATFORM", None)
    else:
        env["QT_QPA_PLATFORM"] = "offscreen"
    return env


def clock_server_command(python: str, script: Path, port: int) -> list[str]:
    return [str(python), str(script), "--port", str(port)]


def port_open(host: str, port: int, timeout: float = 0.3) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


# ---------------------------------------------------------------- one child process
class ManagedProcess:
    """A child this window started. popen/clock are injectable for tests."""

    def __init__(self, name: str, argv: list[str], cwd: Path, log_path: Path, *,
                 popen: Callable = subprocess.Popen, creationflags: int = 0,
                 before_start: Optional[Callable[[], None]] = None,
                 restart_every_s: float = RESTART_EVERY_S, env: Optional[dict] = None):
        self.name, self.argv, self.cwd, self.log_path = name, argv, Path(cwd), Path(log_path)
        self.popen, self.creationflags, self.before_start = popen, creationflags, before_start
        self.restart_every_s = restart_every_s
        self.env = env
        self.proc = None
        self.t_start: Optional[float] = None
        self.starts = 0

    def start(self, now: float) -> None:
        if self.before_start:
            self.before_start()
        self.cwd.mkdir(parents=True, exist_ok=True)
        kw = {"creationflags": self.creationflags} if self.creationflags else {}
        if self.env is not None:
            kw["env"] = self.env
        with open(self.log_path, "ab") as log:
            log.write(("\n=== %s start: %s\n" % (time.strftime("%Y-%m-%d %H:%M:%S"), " ".join(self.argv))).encode())
            log.flush()
            self.proc = self.popen(self.argv, cwd=str(self.cwd), stdout=log, stderr=subprocess.STDOUT,
                                   stdin=subprocess.DEVNULL, **kw)
        self.t_start = now
        self.starts += 1

    @property
    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    @property
    def died(self) -> bool:
        return self.proc is not None and self.proc.poll() is not None

    def can_restart(self, now: float) -> bool:
        return self.t_start is None or now - self.t_start >= self.restart_every_s

    def log_tail(self, n: int = LOG_TAIL_LINES) -> str:
        try:
            with open(self.log_path, "rb") as f:
                lines = collections.deque((l for l in f if not l.startswith(b"=== ")), maxlen=n)
            return " / ".join(l.decode("utf-8", "replace").strip() for l in lines if l.strip())
        except OSError:
            return ""

    def exit_text(self) -> str:
        tail = self.log_tail()
        return "%s exited with code %s%s (log: %s)" % (
            self.name, self.proc.poll(), ": " + tail if tail else "", self.log_path)

    def stop(self, grace_s: float = 5.0) -> None:
        p = self.proc
        if p is None or p.poll() is not None:
            return
        try:
            p.terminate()
            p.wait(grace_s)
        except Exception:
            try:
                p.kill()
                p.wait(grace_s)
            except Exception:
                pass


# ---------------------------------------------------------------- reuse vs start
class Service:
    """Something that must answer on a port: reuse it if it does, else start our own child.
    ensure(now) never blocks (apart from the port probe) and returns a state dict:
      state: found | started | starting | off | error ; text ; error (None unless state == error)."""

    def __init__(self, name: str, probe: Callable[[], bool], proc: Optional[ManagedProcess], enabled: bool = True,
                 startup_timeout_s: float = STARTUP_TIMEOUT_S):
        self.name, self.probe, self.proc, self.enabled = name, probe, proc, enabled
        self.startup_timeout_s = startup_timeout_s
        self.closing = False
        self.last_error: Optional[str] = None

    def _state(self, state, text, error=None):
        return {"state": state, "text": text, "error": error}

    def ensure(self, now: float) -> dict:
        p = self.proc
        if self.probe():
            self.last_error = None
            if p is not None and p.alive:
                return self._state("started", "%s: started here" % self.name)
            return self._state("found", "%s: found running" % self.name)
        if p is not None and p.alive:
            if now - (now if p.t_start is None else p.t_start) > self.startup_timeout_s:
                self.last_error = "%s did not answer within %.0f s (log: %s)" % (
                    self.name, self.startup_timeout_s, p.log_path)
                return self._state("error", "%s: not answering" % self.name, self.last_error)
            return self._state("starting", "starting %s…" % self.name)
        if p is not None and p.died:
            self.last_error = p.exit_text()
        if not self.enabled or p is None:
            if self.last_error:
                return self._state("error", "%s: stopped" % self.name, self.last_error)
            return self._state("off", "%s: not running" % self.name)
        if self.closing or not p.can_restart(now):
            return self._state("error", "%s: stopped" % self.name, self.last_error or "not running")
        try:
            p.start(now)
        except Exception as exc:
            p.t_start = now  # failed attempts are rate-limited like crashes
            self.last_error = "could not start %s: %s: %s" % (self.name, type(exc).__name__, exc)
            return self._state("error", "%s: not started" % self.name, self.last_error)
        return self._state("starting", "starting %s…" % self.name)

    def shutdown(self) -> None:
        """Stop only a child we started; something found running is left alone."""
        self.closing = True
        if self.proc is not None:
            self.proc.stop()
