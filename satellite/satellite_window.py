#!/usr/bin/env python3
"""Satellite cameras window (phase 1): one status line, one Start/Stop button.

Run on the receiving Windows laptop, inside the Thalamus conda env, while launch_receiver.bat (receiver
Thalamus, node sat_cam1 Running) and the clock service (Sleeve server.exe or satellite\\clock_server.py)
are running:

    python satellite\\satellite_window.py [--screenshot PATH --quit-after SECONDS]

It does not launch those programs yet.  Start/Stop call receiver_service.ReceiverService in a worker
thread; the status line is polled every 2 s (receiver) and 10 s (satellite clock log).

Settings: satellite/configs/window.local.json (git-ignored; created with these defaults on first run;
the "Settings..." button opens it in the OS editor; changes apply after restarting the window).
  satellite_address  IP of the satellite, where clock_logger serves its clock log      "192.168.50.10"
  clock_log_port     port of that clock log server                                      50061
  clock_label        clock-log label to use; "" = the only one, and if several are listed
                     "router" when present else the first (the label is shown in the status line)
  output_dir         folder that receives the finished recordings                        "C:/thalamus_recordings/satellite"
  serial             satellite camera serial, used in file names (sat<serial>)           "4103305000"
  expected_fps       frame rate the stream check compares against                        15
  receiver_address   gRPC address of the receiver Thalamus on this laptop                "localhost:50070"
  certs_dir          folder with ca.pem, thalamus.internal.pem, thalamus.internal-key.pem;
                     "" = satellite/certs

Colours: green = synced, receiver connected, (recording and file growing); orange = busy, or clock log's
newest sample older than 10 s, or the recording file has not grown for 3 s; red = not synced, clock log
unreachable, or receiver not reachable.

Threads: the GUI thread only reads in-memory state, stats one file and drains a queue (never blocks more
than a few ms).  Connecting to Thalamus, Start, Stop and the clock-log fetch each run in a plain
threading.Thread and put their result in a queue that a 0.5 s QTimer drains (no Qt signals needed).
"""
from __future__ import annotations

import argparse
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE.parent), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import window_logic as wl  # noqa: E402

TICK_MS = 500
RECEIVER_POLL_S = 2.0
CLOCK_POLL_S = 10.0
CONNECT_RETRY_S = 5.0
COLORS = {wl.GREEN: "#2e9e44", wl.ORANGE: "#e08a00", wl.RED: "#d32f2f"}


def open_in_editor(path: Path) -> None:
    if sys.platform.startswith("win"):
        os.startfile(str(path))  # noqa: S606
    else:
        subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", str(path)])


class Controller:
    """All state and background work; the Qt window only displays `view()` and calls the actions."""

    def __init__(self, settings: dict, make_control=None, make_service=None):
        import receiver_service as rs
        self.rs = rs
        self.settings = settings
        self.control = (make_control or rs.ThalamusControl)(settings["receiver_address"], wl.certs_dir(settings))
        self.service = (make_service or self._default_service)()
        self.clock = wl.ClockMonitor(wl.clock_log_url(settings), settings["clock_label"])
        self.q: queue.Queue = queue.Queue()
        self.busy = None             # None | "starting" | "stopping"
        self.connecting = False
        self.fetching = False
        self.last_connect_try = -1e9
        self.last_receiver_poll = -1e9
        self.last_clock_poll = -1e9
        self.receiver_error = None
        self.connected = False
        self.recording = False
        self.rec_t0 = None           # monotonic time recording began (ours, or first seen)
        self.out_file = None         # storage "Output File" stem while recording
        self.rec_file = None         # cached Path of the growing file
        self.growth = wl.GrowthTracker()
        self.clock_result = None
        self.result_line = None      # (level, text) of the last start/stop
        self.close_after_stop = False
        self.settings_warning = None

    def _default_service(self):
        s = self.settings
        return self.rs.ReceiverService(
            self.control, expected_fps=float(s["expected_fps"]), camera_serial=str(s["serial"]),
            clock_log_url=wl.clock_log_url(s), clock_label=s["clock_label"] or None,
            log=lambda m: None)

    # -- background jobs
    def _spawn(self, name, fn):
        def run():
            try:
                self.q.put((name, fn()))
            except Exception as exc:  # never lose a result, never kill the thread silently
                self.q.put((name, {"ok": False, "error": wl._short(exc)}))
        threading.Thread(target=run, daemon=True, name="satwin-" + name).start()

    def start_recording(self):
        if self.busy:
            return
        self.busy = "starting"
        out = self.settings["output_dir"]
        self._spawn("start", lambda: self.service.start(out))

    def stop_recording(self):
        if self.busy == "stopping":
            return
        self.busy = "stopping"
        out = self.settings["output_dir"]
        self._spawn("stop", lambda: self.service.stop(out))

    def toggle(self):
        if self.recording:
            self.stop_recording()
        else:
            self.start_recording()

    # -- periodic work (called from the GUI tick; all cheap)
    def tick(self, now=None):
        now = time.monotonic() if now is None else now
        self._drain(now)
        if now - self.last_receiver_poll >= RECEIVER_POLL_S:
            self.last_receiver_poll = now
            self._poll_receiver(now)
        if now - self.last_clock_poll >= CLOCK_POLL_S and not self.fetching:
            self.last_clock_poll = now
            self.fetching = True
            self._spawn("clock", self.clock.poll)

    def _drain(self, now):
        while True:
            try:
                name, res = self.q.get_nowait()
            except queue.Empty:
                return
            if name == "clock":
                self.fetching = False
                self.clock_result = res
            elif name == "connect":
                self.connecting = False
                if res.get("ok"):
                    self.receiver_error = None
                else:
                    self.receiver_error = res.get("error")
            elif name == "start":
                self.busy = None
                self.result_line = wl.start_result_line(res)
                if res.get("ok"):
                    self.rec_t0 = now
                    self.growth.reset()
                    self.rec_file = None
            elif name == "stop":
                self.busy = None
                self.result_line = wl.stop_result_line(res)
                if not res.get("ok"):
                    self.close_after_stop = False  # stop failed: keep the window open
                self.rec_t0 = None
                self.rec_file = None
                self.growth.reset()
                self.recording = False

    def _poll_receiver(self, now):
        c = self.control
        try:
            self.connected = bool(c.connected)
            storage = c.get_node(self.rs.STORAGE_NODE) if self.connected else None
        except Exception as exc:  # Thalamus went away: drop the connection, reconnect below
            self.receiver_error = wl._short(exc)
            self.connected = False
            storage = None
            try:
                c.close()
            except Exception:
                pass
        if self.connected and storage is None:
            self.receiver_error = "Thalamus has no node %r" % self.rs.STORAGE_NODE
        elif self.connected:
            self.receiver_error = None
        running = bool(storage and storage.get("Running"))
        if self.busy is None:
            self.recording = running
        if running:
            if self.rec_t0 is None:
                self.rec_t0 = now
            self.out_file = storage.get("Output File")
            self.growth.update(self._file_size(), now)
        else:
            self.rec_t0 = None if self.busy is None else self.rec_t0
            if self.busy is None:
                self.growth.reset()
                self.rec_file = None
        if not self.connected and not self.connecting and self.busy is None \
                and now - self.last_connect_try >= CONNECT_RETRY_S:
            self.last_connect_try = now
            self.connecting = True
            self._spawn("connect", self._connect)

    def _connect(self):
        self.control.connect()   # bounded by its own 15 s timeout; raises ServiceError
        return {"ok": True}

    def _file_size(self):
        try:
            if self.rec_file is None and self.out_file:
                p = Path(self.out_file)
                recs = self.rs.find_recordings(p.parent, p.name)
                self.rec_file = recs[-1] if recs else None
            return self.rec_file.stat().st_size if self.rec_file else None
        except OSError:
            self.rec_file = None
            return None

    # -- what to show
    def view(self, now=None):
        now = time.monotonic() if now is None else now
        recording = self.recording or self.busy == "stopping"
        level, text = wl.decide_status({
            "busy": self.busy, "receiver_connected": self.connected, "recording": recording,
            "elapsed_s": (now - self.rec_t0) if self.rec_t0 is not None else 0,
            "rate_bps": self.growth.rate_bps, "stalled_s": self.growth.stalled_s(now),
            "clock": self.clock_result})
        detail = [e for e in (self.settings_warning,
                              None if self.connected else self.receiver_error,
                              (self.clock_result or {}).get("error")) if e]
        return {"level": level, "text": text, "detail": " | ".join(detail),
                "button": "Stop recording" if recording else "Start recording",
                "button_enabled": self.busy is None, "result": self.result_line}

    def shutdown(self):
        try:
            self.control.close()
        except Exception:
            pass


def build_window(ctrl: Controller, settings_path: Path):
    from thalamus.qt import (QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QMessageBox,
                             QTimer, Qt, QSizePolicy)

    class Window(QWidget):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("Satellite cameras")
            lay = QVBoxLayout(self)
            self.status = QLabel()
            self.status.setWordWrap(True)
            self.status.setTextFormat(Qt.TextFormat.RichText)
            f = self.status.font()
            f.setPointSize(15)
            self.status.setFont(f)
            self.detail = QLabel()
            self.detail.setWordWrap(True)
            self.detail.setStyleSheet("color: #777;")
            self.result = QLabel("")
            self.result.setWordWrap(True)
            self.preview = QLabel("preview — phase 2")
            self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.preview.setFixedSize(640, 400)
            self.preview.setStyleSheet("background: #bdbdbd; color: #555; font-size: 18px;")
            self.button = QPushButton()
            self.button.setMinimumHeight(40)
            self.button.clicked.connect(ctrl.toggle)
            self.settings_btn = QPushButton("Settings…")
            self.settings_btn.clicked.connect(self.open_settings)
            row = QHBoxLayout()
            row.addWidget(self.button, 1)
            row.addWidget(self.settings_btn)
            self.path_label = QLabel("Settings file: %s" % settings_path)
            self.path_label.setStyleSheet("color: #777; font-size: 11px;")
            self.path_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            for w in (self.status, self.detail, row, self.result, self.preview, self.path_label):
                lay.addLayout(w) if isinstance(w, QHBoxLayout) else lay.addWidget(w)
            self.timer = QTimer(self)
            self.timer.timeout.connect(self.refresh)
            self.timer.start(TICK_MS)
            self.refresh()

        def refresh(self):
            try:
                ctrl.tick()
                v = ctrl.view()
            except Exception as exc:  # the window must never crash
                v = {"level": wl.RED, "text": "internal error: %s" % wl._short(exc), "detail": "",
                     "button": self.button.text() or "Start recording", "button_enabled": True, "result": None}
            self.status.setText('<span style="color:%s">●</span> %s' % (COLORS[v["level"]], v["text"]))
            self.detail.setText(v["detail"])
            self.detail.setVisible(bool(v["detail"]))
            self.button.setText(v["button"])
            self.button.setEnabled(v["button_enabled"])
            if v["result"]:
                lvl, txt = v["result"]
                self.result.setText(txt)
                self.result.setStyleSheet("color: %s;" % COLORS[lvl])
            self.maybe_close()

        def open_settings(self):
            try:
                wl.load_settings(settings_path)  # creates the file if missing
                open_in_editor(settings_path)
            except Exception as exc:
                self.result.setText("Could not open %s: %s" % (settings_path, exc))
                self.result.setStyleSheet("color: %s;" % COLORS[wl.RED])

        def maybe_close(self):
            if ctrl.close_after_stop and ctrl.busy is None and not ctrl.recording:
                ctrl.close_after_stop = False
                self.close()

        def closeEvent(self, event):
            if ctrl.busy == "stopping":
                ctrl.close_after_stop = True
                event.ignore()
                return
            if ctrl.recording or ctrl.busy == "starting":
                btn = QMessageBox.question(
                    self, "Satellite cameras", "Stop recording?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No)
                if btn != QMessageBox.StandardButton.Yes:
                    event.ignore()
                    return
                ctrl.close_after_stop = True
                if ctrl.busy is None:
                    ctrl.stop_recording()
                event.ignore()
                return
            self.timer.stop()
            event.accept()

    return Window()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Satellite cameras window")
    ap.add_argument("--settings", default=str(wl.SETTINGS_PATH))
    ap.add_argument("--screenshot", help="save a screenshot of the window to this PNG after --quit-after s, then quit")
    ap.add_argument("--quit-after", type=float, default=0.0, help="quit after this many seconds (0 = run until closed)")
    a = ap.parse_args(argv)
    settings_path = Path(a.settings)
    settings, warning = wl.load_settings(settings_path)
    from thalamus.qt import QApplication, QTimer
    app = QApplication(sys.argv[:1])
    ctrl = Controller(settings)
    ctrl.settings_warning = warning
    win = build_window(ctrl, settings_path)
    win.show()
    if a.screenshot or a.quit_after:
        def finish():
            if a.screenshot:
                win.grab().save(a.screenshot)
                print("screenshot:", a.screenshot)
                print("status:", ctrl.view()["text"])
            app.quit()
        QTimer.singleShot(int((a.quit_after or 5.0) * 1000), finish)
    rc = app.exec()
    ctrl.shutdown()
    sys.stdout.flush()
    os._exit(rc)  # ThalamusThread's worker thread is not a daemon and blocks a normal exit


if __name__ == "__main__":
    sys.exit(main())
