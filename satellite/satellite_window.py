#!/usr/bin/env python3
"""Satellite cameras window (phase 1): one status line, one Start/Stop button.

Run on the receiving Windows laptop, inside the Thalamus conda env; nothing else needs starting by hand:

    python satellite\\satellite_window.py [--screenshot PATH --quit-after SECONDS]

On opening, a worker (a) starts the receiver Thalamus like launch_receiver.bat unless something already
answers on receiver_address (its config is generated from receiver_config_template with satellite_address
as the Address of every REMOTE node, written to configs/receiver.window.local.json; cwd and logs in
satellite/logs/), (b) switches the camera node sat_cam1 on, (c) starts satellite\\clock_server.py unless
something already answers on clock_server_port (e.g. Sleeve server.exe).  A child that dies shows red with
the last log lines and is restarted by a later poll (at most once per 30 s).  On close only the children
this window started are stopped.  Start/Stop call receiver_service.ReceiverService in a worker
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
  start_receiver     start the receiver Thalamus here when nothing answers on receiver_address   true
  show_thalamus_window  show the receiver Thalamus's own window (troubleshooting); false = it runs
                     off screen (Qt offscreen platform), messages only in satellite/logs/receiver.log  false
  start_clock_server start satellite/clock_server.py when nothing answers on clock_server_port  true
  receiver_config_template  receiver Thalamus config to copy (relative to satellite/)            "configs/receiver.json"
  preview_enabled    picture on/off (the "Preview" button changes it; off = image stream not subscribed)   true
  preview_max_fps    keyframes decoded per second for the picture (0 = no preview); the picture comes
                     from the H264 node sat_cam1 over the receiver connection, 640 px wide, grey   1
  clock_server_port  laptop port of the clock service (Sleeve server.exe uses 50050)            50050

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
import window_processes as wp  # noqa: E402
import preview as pv  # noqa: E402
import preview_logic as pl  # noqa: E402

TICK_MS = 500
RECEIVER_POLL_S = 2.0
CLOCK_POLL_S = 10.0
COLORS = {wl.GREEN: "#2e9e44", wl.ORANGE: "#e08a00", wl.RED: "#d32f2f"}


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
        self.supervising = False
        self.fetching = False
        self.step = None             # what the setup worker is doing right now
        self.camera_on = None
        self.last_camera_try = -1e9
        self.service_states: dict = {}
        self.make_services()
        fps = float(settings.get("preview_max_fps", 1) or 0)
        self.settings_path = wl.SETTINGS_PATH
        self.preview = pv.PreviewWorker(self.control, rs.CAMERA_NODE, fps,
                                        enabled=bool(settings.get("preview_enabled", True))) if fps > 0 else None
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

    def set_preview(self, on: bool):
        """Preview button: off unsubscribes the image stream completely; the choice is remembered."""
        if not self.preview:
            return
        self.preview.set_enabled(on)
        self.settings["preview_enabled"] = bool(on)
        warn = wl.update_setting("preview_enabled", bool(on), self.settings_path)
        if warn:
            self.settings_warning = warn

    def start(self):
        if self.preview:
            self.preview.start()

    def make_services(self):
        s = self.settings
        py = sys.executable
        logs = _HERE / "logs"
        no_window = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        host, port = wp.split_address(s["receiver_address"])
        certs = wl.certs_dir(s)
        template, config_out = wp.resolve(s["receiver_config_template"]), _HERE / "configs" / "receiver.window.local.json"

        def write_config():
            import json
            wp.write_json(config_out, wp.generate_receiver_config(
                json.loads(template.read_text(encoding="utf-8")), s["satellite_address"]))

        local = host in wp.LOCAL_HOSTS
        recv_proc = wp.ManagedProcess(
            "receiver", wp.receiver_command(py, config_out, certs, port), logs, logs / "receiver.log",
            before_start=write_config, env=wp.receiver_env(bool(s["show_thalamus_window"])),
            creationflags=0 if s["show_thalamus_window"] else no_window) if (s["start_receiver"] and local) else None
        self.recv_svc = wp.Service("receiver", lambda: wp.port_open(host, port), recv_proc,
                                   enabled=recv_proc is not None)
        cport = int(s["clock_server_port"])
        clock_proc = wp.ManagedProcess(
            "clock server", wp.clock_server_command(py, _HERE / "clock_server.py", cport), logs,
            logs / "clock_server.log", creationflags=no_window) if s["start_clock_server"] else None
        self.clock_svc = wp.Service("clock server", lambda: wp.port_open("127.0.0.1", cport), clock_proc,
                                    enabled=clock_proc is not None)

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
            if not self.supervising and self.busy is None:
                self.supervising = True
                self._spawn("supervise", self._supervise)
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
            elif name == "supervise":
                self.supervising = False
                self.step = None
                if res.get("error"):
                    self.receiver_error = res["error"]
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
        try:
            cam = c.get_node(self.rs.CAMERA_NODE) if self.connected else None
            self.camera_on = None if cam is None else bool(cam.get("Running"))
        except Exception:
            self.camera_on = None

    def _supervise(self):
        """Worker: keep the clock server and receiver Thalamus up, connect, switch the camera on."""
        now = time.monotonic()
        step = None
        states = {"clock": self.clock_svc.ensure(now), "receiver": self.recv_svc.ensure(now)}
        self.service_states = states
        if any(st["state"] == "starting" for st in states.values()):
            step = " · ".join(st["text"] for st in states.values() if st["state"] == "starting")
        self.step = step
        err = None
        if states["receiver"]["state"] in ("found", "started"):
            try:
                if not self.control.connected:
                    self.step = "connecting to receiver…"
                    self.control.connect()   # bounded by its own 15 s timeout; raises ServiceError
                cam = self.control.get_node(self.rs.CAMERA_NODE)
                if cam is None:
                    err = "Thalamus has no node %r" % self.rs.CAMERA_NODE
                elif not cam.get("Running") and now - self.last_camera_try >= 30.0:
                    self.last_camera_try = now
                    self.step = "switching camera on…"
                    self.control.set_node_property(self.rs.CAMERA_NODE, "Running", True)
            except Exception as exc:
                err = wl._short(exc, 200)
        return {"error": err}

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
        states = dict(self.service_states)
        service_errors = [st["text"] for st in states.values() if st["state"] == "error"]
        snap = self.preview.snapshot() if self.preview else None
        stream = (pl.stream_status(self.camera_on, snap["fps"], snap["frame_age_s"])
                  if snap and snap["enabled"] else None)
        level, text = wl.decide_status({
            "busy": self.busy, "receiver_connected": self.connected, "recording": recording,
            "elapsed_s": (now - self.rec_t0) if self.rec_t0 is not None else 0,
            "rate_bps": self.growth.rate_bps, "stalled_s": self.growth.stalled_s(now),
            "clock": self.clock_result, "step": self.step, "camera_on": self.camera_on, "stream": stream,
            "clock_server_note": next((st["text"] for k, st in states.items()
                                       if k == "clock" and st["state"] in ("found", "started")), None),
            "service_errors": service_errors})
        detail = [e for e in (self.settings_warning,
                              *[st["error"] for st in states.values() if st["state"] == "error"],
                              None if self.connected else self.receiver_error,
                              (self.clock_result or {}).get("error")) if e]
        return {"level": level, "text": text, "detail": " | ".join(detail),
                "button": "Stop recording" if recording else "Start recording",
                "button_enabled": self.busy is None, "result": self.result_line, "preview": snap}

    def shutdown(self):
        """Stop only the children this window started (receiver Thalamus, clock server)."""
        if self.preview:
            self.preview.stop()
        try:
            self.control.close()
        except Exception:
            pass
        for svc in (self.recv_svc, self.clock_svc):
            try:
                svc.shutdown()
            except Exception:
                pass


def build_window(ctrl: Controller, settings_path: Path):
    from thalamus.qt import (QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QMessageBox,
                             QTimer, Qt, QSizePolicy, QImage, QPixmap, QPainter, QColor)

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
            self.preview = QLabel("preview off")
            self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.preview.setFixedSize(640, 400)
            self.preview.setStyleSheet("background: #bdbdbd; color: #555; font-size: 18px;")
            self.caption = QLabel("")
            self.caption.setStyleSheet("color: #777;")
            self.shown_seq = -1
            self.button = QPushButton()
            self.button.setMinimumHeight(40)
            self.button.clicked.connect(ctrl.toggle)
            self.preview_btn = QPushButton()
            self.preview_btn.setEnabled(ctrl.preview is not None)
            self.preview_btn.clicked.connect(self.toggle_preview)
            row = QHBoxLayout()
            row.addWidget(self.button, 1)
            row.addWidget(self.preview_btn)
            for w in (self.status, self.detail, row, self.result, self.preview, self.caption):
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
                     "button": self.button.text() or "Start recording", "button_enabled": True, "result": None,
                     "preview": None}
            self.status.setText('<span style="color:%s">●</span> %s' % (COLORS[v["level"]], v["text"]))
            self.detail.setText(v["detail"])
            self.detail.setVisible(bool(v["detail"]))
            self.button.setText(v["button"])
            self.button.setEnabled(v["button_enabled"])
            if v["result"]:
                lvl, txt = v["result"]
                self.result.setText(txt)
                self.result.setStyleSheet("color: %s;" % COLORS[lvl])
            self.show_preview(v.get("preview"))
            self.maybe_close()

        def toggle_preview(self):
            if ctrl.preview:
                ctrl.set_preview(not ctrl.preview.enabled)
                self.shown_seq = -1
                self.refresh()

        def show_preview(self, snap):
            on = bool(snap and snap["enabled"])
            self.preview_btn.setText("Preview: on" if on else "Preview: off")
            if snap is None or not on:
                self.caption.setText("")
                self.shown_seq = -1
                self.preview.setPixmap(QPixmap())
                self.preview.setText("preview off")
                return
            age = snap["picture_age_s"]
            self.caption.setText(pl.picture_caption(age, snap["error"]))
            if snap["picture"] is None:
                self.preview.setText(self.caption.text())
                return
            stale = pl.picture_state(age) == "stale"
            key = (snap["picture_seq"], stale)
            if key == self.shown_seq:
                return
            self.shown_seq = key
            w, h, stride, data = snap["picture"]
            img = QImage(data, w, h, stride, QImage.Format.Format_Grayscale8).copy()
            pm = QPixmap.fromImage(img).scaled(self.preview.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                               Qt.TransformationMode.SmoothTransformation)
            if stale:  # grey it out
                faded = QPixmap(pm.size())
                faded.fill(QColor("#bdbdbd"))
                p = QPainter(faded)
                p.setOpacity(0.3)
                p.drawPixmap(0, 0, pm)
                p.end()
                pm = faded
            self.preview.setPixmap(pm)

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
    ctrl.settings_path = settings_path
    ctrl.settings_warning = warning
    ctrl.start()
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
