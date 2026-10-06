"""scripts/validate_3d.py end to end on a synthetic two-camera session.

Two cameras with known lenses and a known setup "record" board A-big held still
at five places (with the board out of view in between), written exactly as the
recorder writes a session: AVI per camera, metadata CSV with the host clock,
events header naming the serial, and the _calibration.json snapshot. The script
must find the five placements and measure them at the board's true size.
"""
import csv
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

import test_calibration as synth
from backend import calibration as cal

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
FPS = 10.0
K2 = np.array([[1150.0, 0, 640], [0, 1150.0, 512], [0, 0, 1]])
D2 = np.array([-0.05, 0.01, 0, 0, 0])


def load_script():
    spec = importlib.util.spec_from_file_location("validate_3d", SCRIPTS / "validate_3d.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["validate_3d"] = module
    spec.loader.exec_module(module)
    return module


def write_camera(folder, stem, serial, frames, start_s, codec="GREY"):
    h, w = frames[0].shape
    writer = cv2.VideoWriter(str(folder / f"{stem}-0000.avi"), cv2.VideoWriter_fourcc(*codec), FPS, (w, h), False)
    for f in frames:
        writer.write(f)
    writer.release()
    with open(folder / f"{stem}_metadata.csv", "w", newline="") as f:
        out = csv.writer(f)
        out.writerow(["record_frame_index", "camera_frame_id", "timestamp_us", "system_time", "monotonic_s"])
        for i in range(len(frames)):
            out.writerow([i + 1, i, int(i * 1e9 / FPS), 1.79e9 + start_s + i / FPS, start_s + i / FPS])
    (folder / f"{stem}_events.jsonl").write_text(json.dumps(
        {"rec": "header", "camera_serial": serial, "recording": stem}) + "\n")


class Validate3DTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.vd = load_script()
        cls.cfg = cal.BOARD_PRESETS[cal.A_BIG_PRESET]
        A = cls.cfg
        I = np.eye(3)
        cls.R1, cls.t1 = synth.pose_looking_at_board(1.0, (-20, 10, 0), (0.06, 0.0), A)
        cls.R2, cls.t2 = synth.pose_looking_at_board(1.1, (15, -25, 30), (-0.08, 0.0), A)
        tilt, _ = cv2.Rodrigues(np.radians([15, -10, 5]))
        cls.placements = [(I, np.zeros(3)), (I, np.array([0.05, 0.03, 0.0])), (tilt, np.array([0.0, 0.0, -0.18])),
                          (I, np.array([-0.06, -0.02, -0.08])), (tilt, np.array([0.04, -0.04, -0.12]))]
        blank1 = np.full((540, 720), 90, np.uint8)
        blank2 = np.full((1024, 1280), 90, np.uint8)
        f1, f2 = [], []
        for M, m in cls.placements:
            img1 = synth.render_scene([(A, *synth.in_camera(cls.R1, cls.t1, M, m))], K=synth.K_TRUE, D=synth.D_TRUE,
                                      size=(720, 540))
            img2 = synth.render_scene([(A, *synth.in_camera(cls.R2, cls.t2, M, m))], K=K2, D=D2, size=(1280, 1024))
            f1 += [img1] * int(4 * FPS) + [blank1] * int(1.5 * FPS)   # 4 s still, 1.5 s board away
            f2 += [img2] * int(4 * FPS) + [blank2] * int(1.5 * FPS)
        cls.frames = (f1, f2)

    def make_session(self, folder: Path, codec="GREY"):
        base = "recording_20261006_140000"
        write_camera(folder, base, "23227865", self.frames[0], 100.0, codec)
        write_camera(folder, f"{base}_cam26134271", "26134271", self.frames[1], 100.03, codec)
        snap = {
            "status": {"ready": True, "headline": "3D pose: ready"},
            "setup": {"id": "s1", "board": self.cfg.to_dict(), "baseline_mm": 500.0, "verify": {"scale_error_pct": 0.1},
                      "cameras": [{"serial": "23227865", "R": self.R1.tolist(), "t": self.t1.tolist(), "rms_px": 0.2},
                                  {"serial": "26134271", "R": self.R2.tolist(), "t": self.t2.tolist(), "rms_px": 0.2}]},
            "intrinsics": {"23227865": {"K": synth.K_TRUE.tolist(), "D": list(synth.D_TRUE)},
                           "26134271": {"K": K2.tolist(), "D": list(D2)}},
        }
        (folder / f"{base}_calibration.json").write_text(json.dumps(snap))
        return base

    def test_finds_every_placement_and_measures_the_true_size(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            self.make_session(folder)
            out = folder / "out"
            code = self.vd.main([str(folder), "--out", str(out)])
            self.assertEqual(code, 0, (out / "report.txt").read_text())
            report = json.loads((out / "report.json").read_text())
            usable = [p for p in report["placements"] if p["usable"]]
            self.assertEqual(len(usable), len(self.placements), (out / "report.txt").read_text())
            for p in usable:
                self.assertLess(abs(p["size_error_pct"]), 0.5)
                self.assertLess(p["shape_rms_mm"], 1.0)
            # The first placement is the setup spot; the raised, tilted one is >10 cm from it.
            self.assertLess(usable[0]["from_setup_spot_cm"], 1.0)
            self.assertGreater(usable[2]["from_setup_spot_cm"], 10.0)
            self.assertEqual(len(list(out.glob("placement_*_*.png"))), 2 * len(self.placements))
            self.assertIn("Summary over 5 placement(s)", (out / "report.txt").read_text())
            with open(out / "placements.csv") as f:
                self.assertEqual(len(list(csv.DictReader(f))), 5)

    def test_a_wrong_lens_shows_up_as_size_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            base = self.make_session(folder)
            snap_path = folder / f"{base}_calibration.json"
            snap = json.loads(snap_path.read_text())
            snap["intrinsics"]["23227865"]["K"] = (synth.K_TRUE * np.array([[1.15], [1.15], [1]])).tolist()
            snap_path.write_text(json.dumps(snap))
            out = folder / "out"
            self.vd.main([str(folder), "--out", str(out)])
            report = json.loads((out / "report.json").read_text())
            raised = [p for p in report["placements"] if p["usable"] and p["from_setup_spot_cm"] > 15]
            self.assertTrue(raised)
            self.assertTrue(any(abs(p["size_error_pct"]) > 2.0 or p["shape_rms_mm"] > 3.0 for p in raised),
                            [(p["size_error_pct"], p["shape_rms_mm"]) for p in raised])

    def test_needs_the_calibration_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SystemExit):
                self.vd.main([tmp])

    def test_still_runs_and_matching(self):
        det = cal.BoardDetector(self.cfg)
        still = det.detect(self.frames[0][0])
        empty = det.detect(np.full((540, 720), 90, np.uint8))
        samples = [self.vd.Sample(i, i * 0.2, still if 2 <= i < 20 else empty) for i in range(30)]
        runs = self.vd.still_runs(samples, min_static_s=2.0)
        self.assertEqual(len(runs), 1)
        self.assertGreaterEqual(runs[0].start, 0.4 + self.vd.TRIM_S - 1e-9)
        other = [self.vd.Sample(i, i * 0.2 + 0.05, still if 3 <= i < 25 else empty) for i in range(30)]
        stretches = self.vd.match_runs(runs, self.vd.still_runs(other, 2.0))
        self.assertEqual(len(stretches), 1)
        self.assertGreater(stretches[0][1] - stretches[0][0], 1.5)  # both ends trimmed by TRIM_S


if __name__ == "__main__":
    unittest.main()
