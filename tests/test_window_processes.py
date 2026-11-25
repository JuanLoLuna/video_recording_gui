"""satellite/window_processes.py and the phase-2 additions of window_logic (no Qt, no thalamus)."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "satellite"))
import window_logic as wl  # noqa: E402
import window_processes as wp  # noqa: E402

REPO_TEMPLATE = Path(__file__).resolve().parent.parent / "satellite" / "configs" / "receiver.json"


class FakePopen:
    instances = []

    def __init__(self, argv, **kw):
        self.argv, self.kw = argv, kw
        self.returncode = None
        self.terminated = self.killed = False
        self.hang = False
        FakePopen.instances.append(self)

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True
        if not self.hang:
            self.returncode = -15

    def kill(self):
        self.killed = True
        self.returncode = -9

    def wait(self, timeout=None):
        if self.returncode is None:
            raise TimeoutError("still running")
        return self.returncode


class ConfigTests(unittest.TestCase):
    def test_remote_addresses_replaced_rest_untouched(self):
        tpl = json.loads(REPO_TEMPLATE.read_text())
        cfg = wp.generate_receiver_config(tpl, "10.1.2.3")
        remotes = [n for n in cfg["nodes"] if n["type"] == "REMOTE"]
        self.assertGreaterEqual(len(remotes), 2)
        self.assertTrue(all(n["Address"] == "10.1.2.3:50050" for n in remotes))
        self.assertEqual(tpl["nodes"][0]["Address"], "192.168.50.10:50050")  # template object not modified
        back = json.loads(json.dumps(cfg))
        for n in back["nodes"]:
            if n["type"] == "REMOTE":
                n["Address"] = "192.168.50.10:50050"
        self.assertEqual(back, tpl)                                           # nothing else changed
        self.assertEqual([n["Node"] for n in remotes], ["thorcam1_h264", "thorcam1"])
        self.assertIs(remotes[0]["Running"], False)

    def test_write_json_creates_folder(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "a" / "c.json"
            wp.write_json(p, {"x": 1})
            self.assertEqual(json.loads(p.read_text()), {"x": 1})


class CommandTests(unittest.TestCase):
    def test_receiver_command_matches_bat(self):
        cmd = wp.receiver_command("C:\\env\\python.exe", Path("C:/s/configs/r.json"), Path("C:/s/certs"), 50070)
        self.assertEqual(cmd[:6], ["C:\\env\\python.exe", "-m", "thalamus.pipeline", "-C", "-c", str(Path("C:/s/configs/r.json"))])
        self.assertEqual(cmd[6:12], ["-p", "50070", "-u", "50071", "-d", "50072"])
        self.assertEqual(cmd[12::2], ["--ca", "--cert", "--key"])
        self.assertTrue(cmd[13].endswith("ca.pem") and cmd[15].endswith("thalamus.internal.pem")
                        and cmd[17].endswith("thalamus.internal-key.pem"))

    def test_other_port_and_linux_paths(self):
        cmd = wp.receiver_command("/opt/py", Path("/tmp/c.json"), Path("/tmp/certs"), 51000)
        self.assertEqual(cmd[6:12], ["-p", "51000", "-u", "51001", "-d", "51002"])
        self.assertEqual(cmd[0], "/opt/py")
        self.assertEqual(wp.clock_server_command("/opt/py", Path("/r/clock_server.py"), 50050),
                         ["/opt/py", "/r/clock_server.py", "--port", "50050"])

    def test_split_address_and_resolve(self):
        self.assertEqual(wp.split_address("localhost:50070"), ("localhost", 50070))
        self.assertEqual(wp.split_address("somehost"), ("somehost", 50070))
        self.assertEqual(wp.resolve("/abs/x.json"), Path("/abs/x.json"))
        self.assertEqual(wp.resolve("configs/receiver.json"), wp._HERE / "configs" / "receiver.json")


class ServiceTests(unittest.TestCase):
    def setUp(self):
        FakePopen.instances = []
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.answers = False

    def make(self, enabled=True, with_proc=True, **kw):
        proc = wp.ManagedProcess("receiver", ["x", "--port", "1"], Path(self.tmp.name) / "logs",
                                 Path(self.tmp.name) / "logs" / "r.log", popen=FakePopen, **kw) if with_proc else None
        return wp.Service("receiver", lambda: self.answers, proc, enabled=enabled), proc

    def test_reuse_when_answering_never_starts_or_stops(self):
        svc, proc = self.make()
        self.answers = True
        st = svc.ensure(0.0)
        self.assertEqual((st["state"], st["text"]), ("found", "receiver: found running"))
        self.assertEqual(FakePopen.instances, [])
        svc.shutdown()
        self.assertEqual(FakePopen.instances, [])      # nothing of ours to stop

    def test_start_then_started_here_then_stop_only_ours(self):
        svc, proc = self.make()
        self.assertEqual(svc.ensure(0.0)["state"], "starting")
        self.assertEqual(len(FakePopen.instances), 1)
        self.assertEqual(svc.ensure(1.0)["state"], "starting")   # alive, not answering yet: no second start
        self.assertEqual(len(FakePopen.instances), 1)
        self.answers = True
        self.assertEqual(svc.ensure(2.0)["text"], "receiver: started here")
        svc.shutdown()
        self.assertTrue(FakePopen.instances[0].terminated)
        self.assertFalse(FakePopen.instances[0].killed)
        self.assertEqual(FakePopen.instances[0].kw["cwd"], str(Path(self.tmp.name) / "logs"))

    def test_kill_when_terminate_is_ignored(self):
        svc, proc = self.make()
        svc.ensure(0.0)
        FakePopen.instances[0].hang = True
        proc.stop(grace_s=0.01)
        self.assertTrue(FakePopen.instances[0].killed)

    def test_disabled_or_no_proc_does_not_start(self):
        svc, _ = self.make(enabled=False)
        self.assertEqual(svc.ensure(0.0)["state"], "off")
        svc, _ = self.make(with_proc=False)
        self.assertEqual(svc.ensure(0.0)["state"], "off")
        self.assertEqual(FakePopen.instances, [])

    def test_crash_goes_red_with_log_tail_and_restart_is_rate_limited(self):
        svc, proc = self.make()
        svc.ensure(0.0)
        proc.log_path.write_text("a\nTLS files not found: ca.pem\n")
        FakePopen.instances[0].returncode = 3
        st = svc.ensure(5.0)
        self.assertEqual(st["state"], "error")
        self.assertIn("exited with code 3", st["error"])
        self.assertIn("TLS files not found: ca.pem", st["error"])
        self.assertEqual(len(FakePopen.instances), 1)           # < 30 s since the start: no restart
        self.assertEqual(svc.ensure(29.0)["state"], "error")
        self.assertEqual(len(FakePopen.instances), 1)
        self.assertEqual(svc.ensure(30.5)["state"], "starting")  # next poll after 30 s restarts it
        self.assertEqual(len(FakePopen.instances), 2)
        self.answers = True
        self.assertEqual(svc.ensure(31.0)["state"], "started")
        self.assertIsNone(svc.last_error)

    def test_never_answers_is_reported_but_not_restarted(self):
        svc, proc = self.make()
        svc.ensure(0.0)
        st = svc.ensure(31.0)
        self.assertEqual(st["state"], "error")
        self.assertIn("did not answer within 30 s", st["error"])
        self.assertEqual(len(FakePopen.instances), 1)

    def test_no_start_after_shutdown(self):
        svc, _ = self.make()
        svc.shutdown()
        svc.ensure(100.0)
        self.assertEqual(FakePopen.instances, [])

    def test_before_start_hook_and_start_failure(self):
        calls = []
        svc, proc = self.make(before_start=lambda: calls.append(1))
        svc.ensure(0.0)
        self.assertEqual(calls, [1])

        def boom(*a, **k):
            raise FileNotFoundError("no python")
        proc2 = wp.ManagedProcess("r", ["x"], Path(self.tmp.name), Path(self.tmp.name) / "l.log", popen=boom)
        svc2 = wp.Service("r", lambda: False, proc2)
        st = svc2.ensure(0.0)
        self.assertEqual(st["state"], "error")
        self.assertIn("could not start", st["error"])
        self.assertEqual(svc2.ensure(5.0)["state"], "error")     # rate-limited


class StatusPartsTests(unittest.TestCase):
    base = {"receiver_connected": True, "clock": {"ok": True, "reachable": True, "label": "r", "uncertainty_ms": 1.0, "age_s": 1}}

    def test_defaults_include_new_keys(self):
        d = wl.DEFAULTS
        self.assertEqual((d["start_receiver"], d["start_clock_server"], d["receiver_config_template"]),
                         (True, True, "configs/receiver.json"))
        self.assertEqual(d["clock_server_port"], 50050)

    def test_phase1_settings_file_still_merges(self):
        with tempfile.TemporaryDirectory() as dd:
            p = Path(dd) / "w.json"
            p.write_text(json.dumps({"serial": "9", "clock_log_port": 50061}))
            s, warn = wl.load_settings(p)
            self.assertIsNone(warn)
            self.assertIs(s["start_receiver"], True)

    def test_camera_note_step_and_errors(self):
        lvl, txt = wl.decide_status({**self.base, "camera_on": True, "clock_server_note": "clock server: started here"})
        self.assertEqual(lvl, "green")
        self.assertIn("camera on", txt)
        self.assertIn("clock server: started here", txt)
        self.assertEqual(wl.decide_status({**self.base, "camera_on": False})[0], "orange")
        lvl, txt = wl.decide_status({**self.base, "receiver_connected": False, "step": "starting receiver…"})
        self.assertEqual(lvl, "orange")
        self.assertIn("starting receiver…", txt)
        lvl, txt = wl.decide_status({**self.base, "service_errors": ["receiver: stopped"]})
        self.assertEqual((lvl, "receiver: stopped" in txt), ("red", True))


if __name__ == "__main__":
    unittest.main()


class HiddenThalamusTests(unittest.TestCase):
    def test_hidden_sets_offscreen(self):
        env = wp.receiver_env(False, base={"PATH": "x"})
        self.assertEqual(env["QT_QPA_PLATFORM"], "offscreen")
        self.assertEqual(env["PATH"], "x")

    def test_shown_removes_offscreen_inherited_from_this_process(self):
        env = wp.receiver_env(True, base={"PATH": "x", "QT_QPA_PLATFORM": "offscreen"})
        self.assertNotIn("QT_QPA_PLATFORM", env)

    def test_base_not_modified(self):
        base = {"QT_QPA_PLATFORM": "offscreen"}
        wp.receiver_env(True, base=base)
        self.assertEqual(base, {"QT_QPA_PLATFORM": "offscreen"})

    def test_managed_process_passes_env(self):
        FakePopen.instances = []
        with tempfile.TemporaryDirectory() as tmp:
            p = wp.ManagedProcess("receiver", ["py"], Path(tmp), Path(tmp) / "r.log", popen=FakePopen,
                                  env={"QT_QPA_PLATFORM": "offscreen"})
            p.start(0.0)
            self.assertEqual(FakePopen.instances[-1].kw["env"], {"QT_QPA_PLATFORM": "offscreen"})
            q = wp.ManagedProcess("clock", ["py"], Path(tmp), Path(tmp) / "c.log", popen=FakePopen)
            q.start(0.0)
            self.assertNotIn("env", FakePopen.instances[-1].kw)
