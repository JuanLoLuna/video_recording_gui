"""satellite/window_logic.py: settings, status colour, label choice, clock tail fit, result lines."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "satellite"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import window_logic as wl  # noqa: E402
from test_clock_sync import synth_clock_log  # noqa: E402

OK_CLOCK = {"ok": True, "reachable": True, "label": "router", "uncertainty_ms": 1.3, "age_s": 1.0}


def status(**kw):
    base = {"busy": None, "receiver_connected": True, "recording": False, "clock": OK_CLOCK}
    base.update(kw)
    return wl.decide_status(base)


class ColourTests(unittest.TestCase):
    def test_green_idle_and_recording(self):
        lvl, txt = status()
        self.assertEqual(lvl, "green")
        self.assertIn("Synced ±1.3 ms (router)", txt)
        self.assertIn("receiver connected", txt)
        self.assertIn("idle", txt)
        lvl, txt = status(recording=True, elapsed_s=155, rate_bps=1.7e6, stalled_s=0.5)
        self.assertEqual(lvl, "green")
        self.assertIn("recording 00:02:35 · 1.7 MB/s", txt)

    def test_red_cases(self):
        self.assertEqual(status(receiver_connected=False)[0], "red")
        self.assertIn("receiver not reachable", status(receiver_connected=False)[1])
        lvl, txt = status(clock={"ok": False, "reachable": False, "error": "x"})
        self.assertEqual((lvl, "clock log unreachable" in txt), ("red", True))
        lvl, txt = status(clock={"ok": False, "reachable": True, "error": "not enough"})
        self.assertEqual((lvl, "NOT synced" in txt), ("red", True))

    def test_orange_cases(self):
        self.assertEqual(status(clock=None)[0], "orange")
        self.assertEqual(status(clock={**OK_CLOCK, "age_s": 10.5})[0], "orange")
        self.assertEqual(status(clock={**OK_CLOCK, "age_s": 9.9})[0], "green")
        lvl, txt = status(recording=True, elapsed_s=5, rate_bps=0, stalled_s=3.5)
        self.assertEqual(lvl, "orange")
        self.assertIn("file not growing", txt)
        self.assertEqual(status(recording=True, stalled_s=3.0)[0], "green")

    def test_busy_text_and_red_wins(self):
        lvl, txt = status(busy="starting")
        self.assertEqual((lvl, "starting…" in txt), ("orange", True))
        lvl, txt = status(busy="stopping", recording=True)
        self.assertEqual((lvl, "stopping… (converting video)" in txt), ("orange", True))
        self.assertEqual(status(busy="stopping", receiver_connected=False)[0], "red")

    def test_rate_and_elapsed_format(self):
        self.assertEqual(wl.fmt_rate(None), "– MB/s")
        self.assertEqual(wl.fmt_rate(250_000), "250 kB/s")
        self.assertEqual(wl.fmt_elapsed(3725), "01:02:05")


class GrowthTests(unittest.TestCase):
    def test_rate_and_stall(self):
        g = wl.GrowthTracker()
        for t in range(0, 6):
            g.update(t * 2_000_000, float(t))
        self.assertAlmostEqual(g.rate_bps, 2e6)
        self.assertLess(g.stalled_s(5.0), 0.1)
        for t in (6, 7, 8, 9, 10):
            g.update(10_000_000, float(t))
        self.assertEqual(g.stalled_s(10.0), 5.0)
        self.assertEqual(g.rate_bps, 0.0)

    def test_missing_file_counts_as_stalled_from_first_poll(self):
        g = wl.GrowthTracker()
        g.update(None, 100.0)
        g.update(None, 104.0)
        self.assertEqual(g.stalled_s(104.0), 4.0)
        self.assertIsNone(g.rate_bps)


class SettingsTests(unittest.TestCase):
    def test_created_with_defaults_then_merged(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "configs" / "window.local.json"
            s, warn = wl.load_settings(p)
            self.assertIsNone(warn)
            self.assertEqual(s, wl.DEFAULTS)
            self.assertEqual(json.loads(p.read_text()), wl.DEFAULTS)
            p.write_text(json.dumps({"serial": "123", "extra": 1}))
            s, warn = wl.load_settings(p)
            self.assertEqual(s["serial"], "123")
            self.assertEqual(s["extra"], 1)
            self.assertEqual(s["clock_log_port"], 50061)  # missing key -> default

    def test_broken_file_gives_defaults_and_warning_and_is_kept(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "w.json"
            p.write_text("{nope")
            s, warn = wl.load_settings(p)
            self.assertEqual(s, wl.DEFAULTS)
            self.assertIn("unusable", warn)
            self.assertEqual(p.read_text(), "{nope")

    def test_default_values_and_helpers(self):
        d = wl.DEFAULTS
        self.assertEqual((d["satellite_address"], d["serial"], d["expected_fps"], d["clock_log_port"]),
                         ("192.168.50.10", "4103305000", 15, 50061))
        self.assertEqual(d["receiver_address"], "localhost:50070")
        self.assertEqual(wl.clock_log_url(d), "http://192.168.50.10:50061/")
        self.assertEqual(wl.certs_dir(d), wl.DEFAULT_CERTS)
        self.assertEqual(wl.certs_dir({**d, "certs_dir": "/x"}), Path("/x"))


class LabelTests(unittest.TestCase):
    def test_choice(self):
        self.assertEqual(wl.choose_label(["a_b"]), "a_b")
        self.assertEqual(wl.choose_label(["cable", "router"]), "router")
        self.assertEqual(wl.choose_label(["wifi", "cable"]), "cable")      # no router: first (sorted)
        self.assertEqual(wl.choose_label(["cable", "router"], "cable"), "cable")
        with self.assertRaises(ValueError):
            wl.choose_label([])

    def test_listing(self):
        self.assertEqual(wl.labels_from_listing("router_analog.csv\nrouter_ping.csv\nevents.log\ncable_analog.csv\n"),
                         ["cable", "router"])


def to_csv_lines(rows, cols):
    return [",".join(cols) + "\n"] + [",".join(str(r[c]) for c in cols) + "\n" for r in rows]


def fake_server(analog, ping, labels=("router",)):
    files = {"": "".join("%s_%s.csv\n" % (l, k) for l in labels for k in ("analog", "ping")) + "events.log\n"}
    for l in labels:
        files[l + "_analog.csv"] = "".join(to_csv_lines(analog, wl.clock_logger.ANALOG_COLS))
        files[l + "_ping.csv"] = "".join(to_csv_lines(ping, wl.clock_logger.PING_COLS))
    asked = []

    def get_lines(url):
        asked.append(url)
        name = url.split("/", 3)[-1].split("?")[0]
        if name not in files:
            raise OSError("404 " + name)
        return files[name].splitlines(keepends=True)
    return get_lines, asked


class ClockMonitorTests(unittest.TestCase):
    def test_fit_on_synthetic_tail(self):
        analog, ping, base, off = synth_clock_log(True, n_s=300)
        get_lines, asked = fake_server(analog, ping, labels=("cable", "router"))
        newest = max(int(analog[-1]["client_mono_ns"]), int(ping[-1]["t4"]))  # newest sample of either file
        now_ns = newest + int(off(newest)) + 4_000_000_000     # laptop steady "now" = 4 s after newest sample
        m = wl.ClockMonitor("http://x:1/", "", get_lines=get_lines, now_ns=lambda: now_ns)
        r = m.poll()
        self.assertTrue(r["ok"], r)
        self.assertEqual(r["label"], "router")
        self.assertEqual(r["method"], "pings")
        self.assertLess(r["uncertainty_ms"], 2.0)
        self.assertAlmostEqual(r["age_s"], 4.0, delta=0.05)
        self.assertTrue(any("router_analog.csv" in u for u in asked))
        self.assertLessEqual(r["n_rows"], 5 * 125)               # only ~120 s of pings were used

    def test_requests_tail_urls(self):
        analog, ping, *_ = synth_clock_log(True, n_s=300)
        get_lines, asked = fake_server(analog, ping)
        orig = get_lines
        urls = []

        def strip(url):  # the fake server knows plain names; record the full URL first
            urls.append(url)
            return orig(url.split("?")[0])
        wl.ClockMonitor("http://x:1/", "router", get_lines=strip).poll()
        self.assertIn("http://x:1/router_analog.csv?tail=%d" % wl.ANALOG_TAIL_LINES, urls)
        self.assertIn("http://x:1/router_ping.csv?tail=%d" % wl.PING_TAIL_LINES, urls)

    def test_tail_keeps_only_last_lines(self):
        analog, ping, *_ = synth_clock_log(True, n_s=40)
        get_lines, _ = fake_server(analog, ping)
        rows = wl.fetch_tail(get_lines, "http://x/router_analog.csv", 10)
        self.assertEqual(len(rows), 10)
        self.assertEqual(int(rows[-1]["seq"]), 39)

    def test_unreachable_and_too_little_data(self):
        def boom(url):
            raise OSError("timed out")
        r = wl.ClockMonitor("http://x/", "router", get_lines=boom).poll()
        self.assertFalse(r["ok"])
        self.assertFalse(r["reachable"])
        self.assertIn("unreachable", r["error"])
        analog, ping, *_ = synth_clock_log(True, n_s=5)
        get_lines, _ = fake_server(analog[:1], [])
        r = wl.ClockMonitor("http://x/", "", get_lines=get_lines).poll()
        self.assertFalse(r["ok"])
        self.assertTrue(r["reachable"])
        self.assertEqual(wl.decide_status({"receiver_connected": True, "clock": r})[0], "red")

    def test_analog_only_server_exe_log(self):
        analog, ping, base, off = synth_clock_log(False, n_s=200)
        get_lines, _ = fake_server(analog, ping)
        r = wl.ClockMonitor("http://x/", "router", get_lines=get_lines).poll()
        self.assertTrue(r["ok"], r)
        self.assertEqual(r["method"], "analog+half-rtt")


class ResultLineTests(unittest.TestCase):
    def manifest(self, sync):
        return lambda path: {"sync": sync}

    def test_synced(self):
        res = {"ok": True, "frames": 4510, "mp4_ok": True, "verdict": "ok", "output_dir": "D:/rec", "manifest": "m.json"}
        lvl, txt = wl.stop_result_line(res, self.manifest({"synced": True, "uncertainty_ms": 0.84}))
        self.assertEqual(lvl, "green")
        self.assertEqual(txt, "Saved: 4510 frames · synced ±0.8 ms · D:/rec")

    def test_not_synced_reason(self):
        res = {"ok": True, "frames": 10, "mp4_ok": True, "verdict": "ok", "output_dir": "D:/rec", "manifest": "m.json"}
        lvl, txt = wl.stop_result_line(res, self.manifest({"synced": False, "reason": "clock log unusable: x"}))
        self.assertEqual(lvl, "orange")
        self.assertIn("NOT synced: clock log unusable: x", txt)

    def test_manifest_unreadable_and_mp4_failed(self):
        def bad(path):
            raise OSError("gone")
        res = {"ok": True, "frames": 10, "mp4_ok": False, "mp4_error": "av missing", "output_dir": "D:/r", "manifest": "m"}
        lvl, txt = wl.stop_result_line(res, bad)
        self.assertEqual(lvl, "orange")
        self.assertIn("NOT synced: manifest unreadable", txt)
        self.assertIn("mp4 failed: av missing", txt)

    def test_failures(self):
        lvl, txt = wl.stop_result_line({"ok": False, "error": "no active session"})
        self.assertEqual((lvl, txt), ("red", "Stop failed: no active session"))
        self.assertEqual(wl.start_result_line({"ok": False, "error": "disk"})[0], "red")
        self.assertEqual(wl.start_result_line({"ok": True, "stem": "s", "warnings": []}), ("green", "Started: s"))
        self.assertEqual(wl.start_result_line({"ok": True, "warnings": ["w"]})[0], "orange")


if __name__ == "__main__":
    unittest.main()
