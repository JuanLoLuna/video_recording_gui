"""satellite/receiver_service.py with a fake Thalamus control (real Thalamus is not needed)."""
import json
import subprocess
import sys
import tempfile
import unittest
from collections import namedtuple
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "satellite"))
sys.path.insert(0, str(ROOT / "tests"))
import receiver_service as rs  # noqa: E402
from test_satellite_stream_check import _record  # noqa: E402
from test_clock_sync import synth_clock_log  # noqa: E402

Usage = namedtuple("Usage", "total used free")
T0 = datetime(2026, 11, 24, 10, 30, 12)



def _frames(n, start=0):
    return b"".join(_record("sat_cam1", int((start + i) * 1e9 / 15), int(5e9 + (start + i) * 1e9 / 15),
                            w=1280, h=1024, fmt=12) for i in range(n))


class FakeControl(rs.ThalamusControlBase):
    def __init__(self, camera_running=True):
        self.is_connected = False
        self.nodes = {"storage": {"name": "storage", "Running": False, "Output File": "C:/x/sat_test"},
                      "sat_cam1": {"name": "sat_cam1", "Running": camera_running}}
        self.calls = []
        self.on_set = None

    @property
    def connected(self):
        return self.is_connected

    def connect(self):
        self.is_connected = True

    def get_node(self, name):
        n = self.nodes.get(name)
        return dict(n) if n else None

    def set_node_property(self, name, key, value):
        self.calls.append((name, key, value))
        self.nodes[name][key] = value
        if self.on_set:
            self.on_set(name, key, value)


def ok_runner(cmd, **kw):
    Path(cmd[3]).write_bytes(b"mp4data")  # cmd = [python, h264_to_mp4.py, RAW, OUT, --node, NODE]
    assert cmd[1].endswith("h264_to_mp4.py") and cmd[-2:] == ["--node", "sat_cam1"], cmd
    return subprocess.CompletedProcess(cmd, 0, json.dumps({"frames_written": 3}), "")


def fail_runner(cmd, **kw):
    return subprocess.CompletedProcess(cmd, 1, json.dumps({"errors": ["no such node"]}), "")


def make_service(control, runner=ok_runner, free=10**12, sleep=lambda s: None, **kw):
    t = [0.0]

    def clock():
        t[0] += 0.25
        return t[0]
    return rs.ReceiverService(
        control, camera_serial="4103305000", runner=runner, sleep=sleep, clock=clock,
        disk_usage=lambda p: Usage(free * 2, free, free), log=lambda m: None,
        **{"verify_start_s": 0, **kw})


class StartTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out = Path(self.tmp.name) / "rec"
        self.addCleanup(self.tmp.cleanup)

    def test_naming_and_staging(self):
        c = FakeControl()
        res = make_service(c).start(self.out, started_at=T0)
        self.assertTrue(res["ok"], res)
        stem = "recording_20261124_103012_sat4103305000"
        staging = self.out / ".incomplete" / "sat4103305000"
        self.assertEqual(res["stem"], stem)
        self.assertTrue(staging.is_dir())
        self.assertEqual(c.calls[0], ("storage", "Output File", (staging / stem).as_posix()))
        self.assertEqual(c.calls[1], ("storage", "Running", True))
        self.assertTrue((self.out / "receiver_service_last.json").is_file())
        self.assertTrue((staging / rs.STATE_FILE).is_file())

    def test_disk_guard_refuses_critical(self):
        c = FakeControl()
        res = make_service(c, free=5 * 1024**3).start(self.out, started_at=T0)
        self.assertFalse(res["ok"])
        self.assertIn("disk check refused", res["error"])
        self.assertEqual(c.calls, [])
        self.assertFalse((self.out / ".incomplete" / "sat4103305000" / rs.STATE_FILE).exists())
        self.assertEqual(json.loads((self.out / "receiver_service_last.json").read_text())["ok"], False)

    def test_disk_warning_still_starts(self):
        # 7.2e9 B/h, 50 GB free -> ~7 h: warning, not danger
        res = make_service(FakeControl(), free=50 * 10**9).start(self.out, started_at=T0)
        self.assertTrue(res["ok"], res)
        self.assertTrue(any(w.startswith("disk:") for w in res["warnings"]))

    def test_camera_not_running_warns(self):
        res = make_service(FakeControl(camera_running=False)).start(self.out, started_at=T0)
        self.assertTrue(res["ok"])
        self.assertTrue(any("not Running" in w for w in res["warnings"]))

    def test_refuses_second_session_and_running_storage(self):
        c = FakeControl()
        svc = make_service(c)
        self.assertTrue(svc.start(self.out, started_at=T0)["ok"])
        res = svc.start(self.out, started_at=T0)  # storage is Running now
        self.assertFalse(res["ok"])
        c.nodes["storage"]["Running"] = False
        res = svc.start(self.out, started_at=T0)  # state file still there
        self.assertIn("already active", res["error"])

    def test_bad_serial_rejected(self):
        with self.assertRaises(ValueError):
            rs.ReceiverService(FakeControl(), camera_serial="41_03")

    def test_state_file_round_trip_across_processes(self):
        c = FakeControl()
        self.assertTrue(make_service(c).start(self.out, started_at=T0)["ok"])
        code = ("import sys, json; sys.path.insert(0, %r); import receiver_service as rs; "
                "from pathlib import Path; "
                "svc = rs.ReceiverService(None); sf, st = svc.locate_state(Path(%r)); "
                "print(json.dumps(st))" % (str(ROOT / "satellite"), str(self.out)))
        proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        st = json.loads(proc.stdout)
        self.assertEqual(st["stem"], "recording_20261124_103012_sat4103305000")
        self.assertEqual(st["started_at"], "2026-11-24T10:30:12")

    def test_locate_state_from_storage_output_file(self):
        c = FakeControl()
        make_service(c).start(self.out, started_at=T0)
        sf, st = make_service(c).locate_state(None)
        self.assertEqual(st["output_dir"], str(self.out))


class StopTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out = Path(self.tmp.name) / "rec"
        self.addCleanup(self.tmp.cleanup)
        self.c = FakeControl()
        self.staging = self.out / ".incomplete" / "sat4103305000"
        self.stem = "recording_20261124_103012_sat4103305000"
        self.raw = self.staging / (self.stem + ".20261124.1")

    def begin(self, svc):
        self.assertTrue(svc.start(self.out, started_at=T0)["ok"])
        self.raw.write_bytes(_frames(30))
        (self.staging / (self.stem + ".20261124.1.json")).write_text("{}")

    def test_stop_with_growing_file_and_manifest(self):
        grow = {"n": 3}

        def sleep(_):
            if grow["n"] > 0:  # STORAGE2 still flushing after Running went false
                grow["n"] -= 1
                with open(self.raw, "ab") as f:
                    f.write(_frames(10, start=100 + grow["n"] * 10))

        svc = make_service(self.c, sleep=sleep)
        self.begin(svc)
        res = svc.stop(self.out, stable_s=1.0, poll_s=0.1, timeout_s=60)
        self.assertTrue(res["ok"], res)
        self.assertIn(("storage", "Running", False), self.c.calls)
        m = json.loads((self.out / (self.stem + "_satellite_manifest.json")).read_text())
        self.assertEqual(m["frames"], 60)  # 30 + 3 * 10: waited for the growth to end
        self.assertEqual(m["stem"], self.stem)
        self.assertTrue(m["mp4"]["ok"])
        for key in ("mp4", "raw", "raw_json", "stream_check_txt", "stream_check_json"):
            self.assertIn(key, m["files"])
            self.assertTrue(Path(m["files"][key]["path"]).is_file())
        self.assertEqual(Path(m["files"]["raw"]["path"]).name, self.stem + ".thalamus")
        self.assertEqual(Path(m["files"]["mp4"]["path"]).name, self.stem + "-0000.mp4")
        self.assertEqual(Path(m["files"]["raw_json"]["path"]).name, self.stem + ".thalamus.json")
        self.assertEqual(m["started_at"], "2026-11-24T10:30:12")
        self.assertIn("thalamus", m["versions"] if m["versions"] else {"thalamus": None})
        self.assertIn(m["stream_check"]["verdict"], ("ok", "problems"))
        # staging holds nothing, state file gone, only final names in output_dir
        self.assertFalse((self.staging / rs.STATE_FILE).exists())
        self.assertFalse(any(self.staging.glob("*")) if self.staging.exists() else False)
        self.assertFalse(any(p.suffix in (".part", ".tmp") for p in self.out.iterdir()))

    def test_still_growing_times_out(self):
        def grow(_):
            with open(self.raw, "ab") as f:
                f.write(b"x")
        svc = make_service(self.c, sleep=grow)
        self.begin(svc)
        res = svc.stop(self.out, stable_s=5.0, poll_s=0.1, timeout_s=3)
        self.assertFalse(res["ok"])
        self.assertIn("still growing", res["error"])
        self.assertTrue(self.raw.exists())
        self.assertTrue((self.staging / rs.STATE_FILE).exists())  # can retry stop

    def test_mp4_failure_keeps_raw_and_records_error(self):
        svc = make_service(self.c, runner=fail_runner)
        self.begin(svc)
        res = svc.stop(self.out, stable_s=0.5, poll_s=0.1)
        self.assertTrue(res["ok"], res)
        self.assertFalse(res["mp4_ok"])
        self.assertTrue((self.out / (self.stem + ".thalamus")).is_file())
        self.assertFalse((self.out / (self.stem + "-0000.mp4")).exists())
        m = json.loads((self.out / (self.stem + "_satellite_manifest.json")).read_text())
        self.assertFalse(m["mp4"]["ok"])
        self.assertIn("exit", m["mp4"]["error"])
        self.assertIn("no such node", m["mp4"]["error"])
        self.assertNotIn("mp4", m["files"])
        self.assertTrue(any("mp4 conversion failed" in w for w in m["warnings"]))

    def test_manifest_numbers_from_stream_check(self):
        svc = make_service(self.c)
        self.begin(svc)
        svc.stop(self.out, stable_s=0.5, poll_s=0.1)
        m = json.loads((self.out / (self.stem + "_satellite_manifest.json")).read_text())
        self.assertEqual(m["frames"], 30)
        self.assertAlmostEqual(m["fps"], 15.0, places=3)
        self.assertEqual(m["stream_check"]["verdict"], "ok")
        self.assertEqual(m["stream_check"]["dropped_frames"], 0)
        self.assertEqual(m["expected_fps"], 15.0)

    def test_no_recording_file_is_an_error(self):
        svc = make_service(self.c)
        self.assertTrue(svc.start(self.out, started_at=T0)["ok"])
        res = svc.stop(self.out, stable_s=0.5, poll_s=0.1)
        self.assertFalse(res["ok"])
        self.assertIn("no", res["error"])

    def test_stop_without_session(self):
        res = make_service(self.c).stop(self.out)
        self.assertFalse(res["ok"])
        self.assertIn("no active session", res["error"])

    def test_status(self):
        svc = make_service(self.c)
        self.begin(svc)
        res = svc.status(self.out)
        self.assertTrue(res["ok"] and res["connected"])
        self.assertTrue(res["storage"]["running"])
        self.assertEqual(res["session"]["stem"], self.stem)
        self.assertEqual(res["current_file"]["size_bytes"], self.raw.stat().st_size)
        self.assertIn("free_bytes", res)

    def test_cli_main_with_fake(self):
        rc = rs.main(["--serial", "4103305000", "--verify-start-s", "0", "start", "--output-dir", str(self.out)], control=self.c)
        self.assertEqual(rc, 0)
        self.assertTrue((self.out / "receiver_service_last.json").is_file())



class SyncStopTests(unittest.TestCase):
    """stop with a clock log: the fetch is replaced by a function returning synthetic rows."""
    N = 30

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out = Path(self.tmp.name) / "rec"
        self.addCleanup(self.tmp.cleanup)
        self.c = FakeControl()
        self.staging = self.out / ".incomplete" / "sat4103305000"
        self.stem = "recording_20261124_103012_sat4103305000"
        self.raw = self.staging / (self.stem + ".20261124.1")
        self.analog, self.ping, self.base, self.off = synth_clock_log(trailer=False)

    def frames(self):
        """Satellite steady c, laptop arrival = true laptop steady + 3 ms + a little delay."""
        out = []
        for i in range(self.N):
            c = self.base + 100_000_000_000 + i * 66_666_667
            out.append(_record("sat_cam1", int(c + self.off(c) + 3_000_000 + (i % 5) * 400_000), int(c),
                               w=1280, h=1024, fmt=12))
        return b"".join(out)

    def run_stop(self, **kw):
        svc = make_service(self.c, **kw)
        self.assertTrue(svc.start(self.out, started_at=T0)["ok"])
        self.raw.write_bytes(self.frames())
        (self.staging / (self.stem + ".20261124.1.json")).write_text("{}")
        res = svc.stop(self.out, stable_s=0.5, poll_s=0.1)
        self.assertTrue(res["ok"], res)
        m = json.loads((self.out / (self.stem + "_satellite_manifest.json")).read_text())
        return res, m

    def read_sync(self):
        import csv
        with open(self.out / (self.stem + "_satellite_sync.csv"), newline="") as f:
            return list(csv.DictReader(f))

    def test_synced_csv_and_manifest(self):
        calls = []

        def fetch(url, label, timeout):
            calls.append((url, label))
            return "laptop", self.analog, self.ping
        res, m = self.run_stop(clock_log_url="http://x:50061/", fetch_clock=fetch)
        self.assertEqual(calls, [("http://x:50061/", None)])
        rows = self.read_sync()
        self.assertEqual(len(rows), self.N)
        self.assertEqual(list(rows[0]), rs.SYNC_COLS)
        sy = m["sync"]
        self.assertTrue(sy["synced"], sy)
        self.assertEqual(sy["method"], "analog+half-rtt")
        self.assertEqual(sy["csv"], self.stem + "_satellite_sync.csv")
        self.assertAlmostEqual(sy["drift_ppm"], 12.0, delta=1.0)
        self.assertEqual(sy["clock_label"], "laptop")
        for k in ("offset_at_start_ns", "uncertainty_ms", "samples", "span_s", "clock_instance_ids",
                  "transfer_delay_min_ms"):
            self.assertIn(k, sy)
        self.assertAlmostEqual(sy["offset_at_start_ns"], self.off(self.base + 100e9), delta=1e6)
        # arrival minus mapped time is about the 3 ms built into the frames
        self.assertAlmostEqual(sy["transfer_delay_min_ms"], 3.0, delta=1.0)
        self.assertEqual(sy["warnings"], [])
        r = rows[0]
        c0 = self.base + 100_000_000_000
        self.assertEqual(int(r["satellite_steady_ns"]), c0)
        self.assertAlmostEqual(int(r["laptop_steady_ns"]) - c0, self.off(c0), delta=1e6)
        self.assertEqual(int(r["laptop_wall_ns"]) - int(r["laptop_steady_ns"]), int(1.8e18))
        self.assertIn("sync_csv", m["files"])

    def test_no_clock_url_falls_back(self):
        res, m = self.run_stop()
        sy = m["sync"]
        self.assertFalse(sy["synced"])
        self.assertEqual(sy["method"], "arrival-min (no clock service)")
        self.assertIn("no --clock-log-url", sy["reason"])
        rows = self.read_sync()
        self.assertEqual(len(rows), self.N)
        self.assertEqual(rows[0]["laptop_wall_ns"], "")
        # fallback line contains the smallest transfer delay, so arrival - mapped is about 0
        self.assertLess(abs(sy["transfer_delay_min_ms"]), 1.0)
        self.assertTrue(any("not synced" in w for w in res["warnings"]))

    def test_fetch_failure_does_not_break_stop(self):
        def fetch(url, label, timeout):
            raise OSError("connection refused")
        res, m = self.run_stop(clock_log_url="http://x:50061/", fetch_clock=fetch)
        self.assertTrue(m["mp4"]["ok"])
        self.assertTrue((self.out / (self.stem + ".thalamus")).is_file())
        self.assertFalse(m["sync"]["synced"])
        self.assertIn("connection refused", m["sync"]["reason"])
        self.assertEqual(len(self.read_sync()), self.N)

    def test_clock_log_too_short_falls_back(self):
        res, m = self.run_stop(clock_log_url="u", fetch_clock=lambda *a: ("l", self.analog[:1], []))
        self.assertFalse(m["sync"]["synced"])
        self.assertIn("not enough clock samples", m["sync"]["reason"])

    def test_wrong_clock_warns(self):
        # clock log 500 ms ahead of the true offset: frames seem to arrive before they were sent
        def shifted(d):
            return [dict(r, server_steady_ns=int(r["server_steady_ns"]) + d) for r in self.analog]
        res, m = self.run_stop(clock_log_url="u", fetch_clock=lambda *a: ("l", shifted(500_000_000), self.ping))
        self.assertTrue(m["sync"]["synced"])
        self.assertLess(m["sync"]["transfer_delay_min_ms"], -400)
        self.assertTrue(any("negative beyond" in w for w in m["sync"]["warnings"]), m["sync"])

    def test_negative_delay_check_skipped_when_not_synced(self):
        # fallback line is fitted to the fastest arrivals, so a negative minimum delay is no evidence
        orig = rs.fallback_fit

        def shifted_fit(frames):
            fit = orig(frames)
            fit["offset_at_x0_ns"] += 2_000_000  # mapped time 2 ms late -> delays about -2 ms
            return fit
        rs.fallback_fit = shifted_fit
        try:
            res, m = self.run_stop(clock_log_url="u", fetch_clock=lambda *a: ("l", self.analog[:1], []))
        finally:
            rs.fallback_fit = orig
        self.assertFalse(m["sync"]["synced"])
        self.assertLess(m["sync"]["transfer_delay_min_ms"], -1.5)
        self.assertFalse(any("negative beyond" in w for w in res["warnings"] + m["sync"]["warnings"]), m["sync"])

    def test_large_transfer_delay_warns(self):
        def shifted(d):
            return [dict(r, server_steady_ns=int(r["server_steady_ns"]) + d) for r in self.analog]
        res, m = self.run_stop(clock_log_url="u", fetch_clock=lambda *a: ("l", shifted(-500_000_000), self.ping))
        self.assertGreater(m["sync"]["transfer_delay_min_ms"], 400)
        self.assertTrue(any("over 200 ms" in w for w in m["sync"]["warnings"]), m["sync"])

    def test_cli_args(self):
        a = rs.build_parser().parse_args(["--clock-log-url", "http://h:50061/", "--clock-label", "laptop", "stop"])
        self.assertEqual((a.clock_log_url, a.clock_label), ("http://h:50061/", "laptop"))

    def test_fetch_clock_log_against_local_http_server(self):
        import threading
        from http.server import ThreadingHTTPServer
        import clock_logger as cl
        d = Path(self.tmp.name) / "log"
        d.mkdir()
        (d / "laptop_analog.csv").write_text(",".join(cl.ANALOG_COLS) + "\n1,2,3,4\n")
        (d / "laptop_ping.csv").write_text(",".join(cl.PING_COLS) + "\n1,2,,,5,,,\n")
        (d / "secret.txt").write_text("no")
        srv = ThreadingHTTPServer(("127.0.0.1", 0), cl.make_handler(d))
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        self.addCleanup(srv.shutdown)
        url = "http://127.0.0.1:%d/" % srv.server_address[1]
        label, a, p = rs.fetch_clock_log(url, None, 5)
        self.assertEqual(label, "laptop")
        self.assertEqual(a[0]["server_steady_ns"], "4")
        self.assertEqual(p[0]["t2"], "")
        with self.assertRaises(Exception):
            rs._http_get(url + "secret.txt", 5)


class HelperTests(unittest.TestCase):
    def test_replace_with_retry(self):
        with tempfile.TemporaryDirectory() as d:
            a, b = Path(d) / "a", Path(d) / "b"
            a.write_text("x")
            calls = []
            import os
            real = os.replace

            def flaky(s, t):
                calls.append(1)
                if len(calls) < 3:
                    raise PermissionError("busy")
                real(s, t)
            rs.os.replace = flaky
            try:
                rs.replace_with_retry(a, b, sleep=lambda s: None)
            finally:
                rs.os.replace = real
            self.assertTrue(b.exists() and len(calls) == 3)



class StartVerifyTests(unittest.TestCase):
    """start() confirms STORAGE2 actually opened its file (Thalamus 1.0.66 once did not)."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out = Path(self.tmp.name) / "rec"
        self.addCleanup(self.tmp.cleanup)
        self.t = [0.0]

    def _svc(self, control):
        return make_service(control, verify_start_s=2.0)

    def _writes_on_running(self, control, after_n_starts):
        orig = control.set_node_property
        starts = {"n": 0}

        def set_prop(name, key, value):
            orig(name, key, value)
            if name == "storage" and key == "Running" and value is True:
                starts["n"] += 1
                if starts["n"] >= after_n_starts:
                    stem = Path(control.nodes["storage"]["Output File"])
                    Path(str(stem) + ".20261124.1").write_bytes(b"")
        control.set_node_property = set_prop

    def test_file_appears(self):
        c = FakeControl()
        self._writes_on_running(c, 1)
        res = self._svc(c).start(self.out, started_at=T0)
        self.assertTrue(res["ok"], res)
        self.assertFalse(any("restart" in w for w in res["warnings"]))

    def test_file_appears_after_one_restart(self):
        c = FakeControl()
        self._writes_on_running(c, 2)
        res = self._svc(c).start(self.out, started_at=T0)
        self.assertTrue(res["ok"], res)
        self.assertTrue(any("restart" in w for w in res["warnings"]))

    def test_no_file_is_an_error_and_storage_off(self):
        c = FakeControl()
        res = self._svc(c).start(self.out, started_at=T0)
        self.assertFalse(res["ok"])
        self.assertIn("NOT started", res["error"])
        self.assertFalse(c.nodes["storage"]["Running"])


if __name__ == "__main__":
    unittest.main()
