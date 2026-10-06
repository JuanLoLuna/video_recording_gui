#!/usr/bin/env python3
"""Measure the real 3D accuracy of a session's camera setup (plan step 16).

Record a session (after "3D pose: ready") in which a ChArUco board -- by default
the session's setup board A / A-big -- is held STILL for a few seconds at several
spots across the work area, with hands out of view each time. Then:

    python scripts/validate_3d.py <session folder>
    python scripts/validate_3d.py <session folder> --basename recording_20261006_140000
    python scripts/validate_3d.py <session folder> --board "A4 board A - handheld (5x5 markers)" --square-mm 35.86

Everything comes from the session folder: <basename>_calibration.json (the
setup, both cameras' intrinsics, the setup board), each camera's video
segments, metadata CSV (host clock per frame) and events header (which serial
is which file). Nothing is written next to the recording except the output
folder <session>/validate_3d_<time>/ (or --out):

    report.txt        what was printed
    report.json       every placement, every number
    placements.csv    one row per placement
    placement_NN_<serial>.png   each camera's frame with the triangulated
                                corners reprojected (green) and detected (red)

How: the board is detected ~5 times a second in each video; a "still" run is
>= --min-static-s of detections whose corners move < 1 px; runs of the two
cameras are matched on the host clock (metadata monotonic_s, shared by both
cameras); within each matched stretch the detections are averaged, triangulated
with the SAVED setup poses and compared with the real board exactly like the
setup's Verify step (rigid fit: size error %, shape RMS mm). The cameras are
not synchronised, which does not matter for a still board.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend import calibration as cal  # noqa: E402

SAMPLE_HZ = 5.0
STILL_PX = 1.0
TRIM_S = 0.5           # dropped at both ends of a still run (hand arriving/leaving)
MIN_OVERLAP_S = 1.0


# --------------------------------------------------------------------------
# Session discovery
# --------------------------------------------------------------------------

@dataclass
class CameraFiles:
    serial: str
    stem: str
    segments: list            # video paths in order
    metadata: Path


def find_basename(folder: Path, basename: str | None) -> str:
    if basename:
        return basename
    found = sorted(p.name[: -len("_calibration.json")] for p in folder.glob("*_calibration.json"))
    if not found:
        raise SystemExit(f"No *_calibration.json in {folder}: was the session recorded with 3D pose ready?")
    if len(found) > 1:
        raise SystemExit(f"Several sessions in {folder}: pass --basename, one of {found}")
    return found[0]


def camera_files(folder: Path, basename: str, serials) -> dict[str, CameraFiles]:
    """serial -> its files. Each stem's events header names its camera serial."""
    out = {}
    stems = [basename] + [p.name[: -len("_metadata.csv")] for p in folder.glob(f"{basename}_cam*_metadata.csv")]
    for stem in stems:
        serial = None
        events = folder / f"{stem}_events.jsonl"
        if events.exists():
            try:
                serial = str(json.loads(events.read_text(encoding="utf-8").splitlines()[0]).get("camera_serial"))
            except (ValueError, IndexError):
                serial = None
        if serial in (None, "None"):
            m = re.search(r"_cam(\w+)$", stem)
            serial = m.group(1) if m else None
        manifest = folder / f"{stem}_segments.csv"
        if manifest.exists():
            with open(manifest, newline="", encoding="utf-8") as f:
                rows = sorted(csv.DictReader(f), key=lambda r: int(r["segment_index"]))
            segments = [folder / r["segment_file"] for r in rows]
        else:
            segments = sorted(folder.glob(f"{stem}-[0-9][0-9][0-9][0-9].avi"))
        metadata = folder / f"{stem}_metadata.csv"
        if serial and segments and metadata.exists():
            out[serial] = CameraFiles(serial, stem, segments, metadata)
    missing = [s for s in serials if s not in out]
    if missing:
        raise SystemExit(f"No video/metadata found for camera(s) {missing} under {folder} ({basename})")
    return out


def load_host_times(metadata: Path) -> np.ndarray:
    """Host clock (monotonic_s) per record_frame_index, 0-based position = index - 1."""
    with open(metadata, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    times = np.full(len(rows), np.nan)
    for r in rows:
        i = int(r["record_frame_index"]) - 1
        if 0 <= i < len(times) and r.get("monotonic_s"):
            times[i] = float(r["monotonic_s"])
    return times


# --------------------------------------------------------------------------
# Detection, still runs, matching (pure)
# --------------------------------------------------------------------------

@dataclass
class Sample:
    frame: int                # 0-based position in the recording
    time: float               # host seconds
    det: cal.Detection


@dataclass
class Run:
    samples: list = field(default_factory=list)

    @property
    def start(self) -> float:
        return self.samples[0].time

    @property
    def end(self) -> float:
        return self.samples[-1].time


def detect_samples(cam: CameraFiles, cfg: cal.BoardConfig, sample_hz: float = SAMPLE_HZ,
                   progress=print) -> tuple[list[Sample], int]:
    times = load_host_times(cam.metadata)
    finite = times[np.isfinite(times)]
    fps = (len(finite) - 1) / (finite[-1] - finite[0]) if len(finite) > 2 and finite[-1] > finite[0] else 30.0
    stride = max(1, int(round(fps / sample_hz)))
    detector = cal.BoardDetector(cfg)
    samples, position = [], 0
    for seg in cam.segments:
        cap = cv2.VideoCapture(str(seg))
        while True:
            ok = cap.grab()
            if not ok:
                break
            if position % stride == 0 and position < len(times) and np.isfinite(times[position]):
                ok, frame = cap.retrieve()
                if ok:
                    gray = frame if frame.ndim == 2 else frame[:, :, 0]
                    samples.append(Sample(position, float(times[position]), detector.detect(gray)))
            position += 1
        cap.release()
        progress(f"  {cam.serial}: {seg.name} done ({position} frames so far)")
    if position != len(times):
        progress(f"  !! {cam.serial}: {position} frames decoded but {len(times)} metadata rows")
    return samples, position


def still_runs(samples: list[Sample], min_static_s: float, still_px: float = STILL_PX,
               trim_s: float = TRIM_S) -> list[Run]:
    """Stretches where the board is found and its corners move < still_px between samples."""
    runs, current = [], Run()

    def close():
        if not current.samples:
            return
        kept = [s for s in current.samples if current.start + trim_s <= s.time <= current.end - trim_s]
        if kept and kept[-1].time - kept[0].time >= min_static_s - 2 * trim_s:
            runs.append(Run(kept))

    for s in samples:
        if not s.det.ok:
            close()
            current = Run()
            continue
        if current.samples:
            motion = cal.corner_motion_px(current.samples[-1].det, s.det)
            if motion is None or motion > still_px:
                close()
                current = Run()
        current.samples.append(s)
    close()
    return runs


def match_runs(runs_a: list[Run], runs_b: list[Run], min_overlap_s: float = MIN_OVERLAP_S) -> list[tuple]:
    """(t0, t1) of every stretch where camera A and camera B are both still on the board."""
    out = []
    for ra in runs_a:
        for rb in runs_b:
            t0, t1 = max(ra.start, rb.start), min(ra.end, rb.end)
            if t1 - t0 >= min_overlap_s:
                out.append((t0, t1))
    return sorted(out)


def average_in(samples: list[Sample], t0: float, t1: float):
    inside = [s for s in samples if t0 <= s.time <= t1]
    return cal.average_detections([s.det for s in inside]), inside


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------

def triangulate(det_a, pose_a, K_a, D_a, det_b, pose_b, K_b, D_b):
    common, ia, ib = np.intersect1d(det_a.ids, det_b.ids, return_indices=True)
    na = cv2.undistortPoints(det_a.corners[ia].reshape(-1, 1, 2), K_a, D_a).reshape(-1, 2).T
    nb = cv2.undistortPoints(det_b.corners[ib].reshape(-1, 1, 2), K_b, D_b).reshape(-1, 2).T
    X = cv2.triangulatePoints(np.hstack([pose_a.R, pose_a.t.reshape(3, 1)]),
                              np.hstack([pose_b.R, pose_b.t.reshape(3, 1)]), na, nb)
    return common, (X[:3] / X[3]).T


def corner_errors_mm(points_m: np.ndarray, truth_m: np.ndarray) -> np.ndarray:
    """Per-corner distance (mm) after the best rigid fit (same fit as the Verify step)."""
    p, q = points_m - points_m.mean(axis=0), truth_m - truth_m.mean(axis=0)
    U, _, Vt = np.linalg.svd(p.T @ q)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return np.linalg.norm((R @ p.T).T - q, axis=1) * 1000.0


def evaluate(det_a, det_b, pose_a, pose_b, intr_a, intr_b, board, setup_board_centre) -> dict:
    verify = cal.verify_setup(det_a, pose_a, *intr_a, det_b, pose_b, *intr_b, board)
    if verify.rms_mm is None:
        return {"corners": verify.corners, "usable": False}
    ids, X = triangulate(det_a, pose_a, *intr_a, det_b, pose_b, *intr_b)
    truth = board.getChessboardCorners()[ids].astype(np.float64)
    errs = corner_errors_mm(X, truth)
    centre = X.mean(axis=0)
    cam_a, cam_b = pose_a.camera_centre_m, pose_b.camera_centre_m
    return {
        "usable": True,
        "corners": int(len(ids)),
        "size_error_pct": float(verify.scale_error_pct),
        "shape_rms_mm": float(verify.rms_mm),
        "max_corner_error_mm": float(errs.max()),
        "position_world_cm": [float(v) * 100 for v in centre],
        "from_setup_spot_cm": float(np.linalg.norm(centre - np.asarray(setup_board_centre)) * 100),
        "distance_to_cameras_cm": [float(np.linalg.norm(centre - cam_a) * 100),
                                   float(np.linalg.norm(centre - cam_b) * 100)],
        "_X": X, "_ids": ids,
    }


def overlay(where: tuple, det, X, pose, K, D, out_png: Path) -> None:
    """One frame (where = (segment, index in it)) with detected (red) and reprojected triangulated (green) corners."""
    segment, index = where
    cap = cv2.VideoCapture(str(segment))
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return
    vis = frame if frame.ndim == 3 else cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    for x, y in det.corners:
        cv2.circle(vis, (int(x), int(y)), 4, (0, 0, 255), -1, cv2.LINE_AA)
    rvec, _ = cv2.Rodrigues(pose.R)
    proj, _ = cv2.projectPoints(X, rvec, pose.t, K, D)
    for x, y in proj.reshape(-1, 2):
        cv2.circle(vis, (int(x), int(y)), 6, (0, 200, 0), 2, cv2.LINE_AA)
    cv2.imwrite(str(out_png), vis)


def locate_frame(cam: CameraFiles, position: int):
    """(segment path, index within it) of a 0-based recording position, from the segment manifest."""
    manifest = cam.metadata.with_name(f"{cam.stem}_segments.csv")
    if manifest.exists():
        with open(manifest, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                first, last = int(r["first_record_frame_index"]) - 1, int(r["last_record_frame_index"]) - 1
                if first <= position <= last:
                    return cam.metadata.parent / r["segment_file"], position - first
    return cam.segments[0], position


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

class Tee:
    def __init__(self, path: Path, stream):
        self._file = open(path, "w", encoding="utf-8", buffering=1)
        self._stream = stream

    def write(self, text):
        self._stream.write(text)
        self._file.write(text)
        return len(text)

    def flush(self):
        self._stream.flush()
        self._file.flush()

    def close(self):
        self._file.close()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("session", type=Path, help="folder with the recording and its _calibration.json")
    ap.add_argument("--basename", help="recording_YYYYMMDD_HHMMSS (needed only if the folder has several)")
    ap.add_argument("--board", help="board preset moved around (default: the session's setup board)")
    ap.add_argument("--square-mm", type=float, help="measured square (or grid marker) of --board")
    ap.add_argument("--min-static-s", type=float, default=2.0, help="shortest still placement (s)")
    ap.add_argument("--out", type=Path, help="output folder (default: <session>/validate_3d_<time>)")
    args = ap.parse_args(argv)

    folder = args.session
    basename = find_basename(folder, args.basename)
    out = args.out or folder / f"validate_3d_{datetime.now():%Y%m%d_%H%M%S}"
    out.mkdir(parents=True, exist_ok=True)
    tee = Tee(out / "report.txt", sys.stdout)
    sys.stdout = tee
    try:
        return run(folder, basename, out, args)
    finally:
        sys.stdout = tee._stream
        tee.close()
        print(f"Report saved to {out}")


def run(folder: Path, basename: str, out: Path, args) -> int:
    snap = json.loads((folder / f"{basename}_calibration.json").read_text(encoding="utf-8"))
    setup, status = snap.get("setup"), snap.get("status") or {}
    print(f"validate_3d  {datetime.now():%Y-%m-%d %H:%M:%S}  session {basename}")
    print(f"3D status at recording start: {status.get('headline', '?')}")
    if not setup or not setup.get("cameras") or len(setup["cameras"]) != 2:
        print("This recording has no two-camera setup in its _calibration.json: nothing to validate.")
        return 2
    serials = [c["serial"] for c in setup["cameras"]]
    poses, intr = {}, {}
    for cam in setup["cameras"]:
        rec = (snap.get("intrinsics") or {}).get(cam["serial"])
        if rec is None:
            print(f"No intrinsics for camera {cam['serial']} in the snapshot.")
            return 2
        poses[cam["serial"]] = cal.BoardPose(R=np.asarray(cam["R"], float), t=np.asarray(cam["t"], float),
                                             rms_px=float(cam["rms_px"]), n_corners=0)
        intr[cam["serial"]] = (np.asarray(rec["K"], float), np.asarray(rec["D"], float))
    setup_board = cal.BoardConfig.from_dict(setup["board"])
    if args.board:
        cfg = cal.BOARD_PRESETS[args.board]
        cfg = cal.with_measured_square(cfg, args.square_mm or cal.default_measured_mm(args.board))
    else:
        cfg = setup_board
    if cfg.kind != "charuco":
        print("The validation board must be a ChArUco board (A, A-big or the lab board).")
        return 2
    board = cal.make_board(cfg)
    print(f"Board: {cfg.squares_x}x{cfg.squares_y} {cfg.dictionary}, squares {cfg.square_length_m * 1000:.2f} mm")
    print(f"Setup: baseline {setup.get('baseline_mm') or float('nan'):.0f} mm, verify "
          f"{(setup.get('verify') or {}).get('scale_error_pct', float('nan')):+.2f} %")

    files = camera_files(folder, basename, serials)
    samples, runs = {}, {}
    for serial in serials:
        print(f"Detecting the board in camera {serial} ({len(files[serial].segments)} segment(s))…")
        samples[serial], _ = detect_samples(files[serial], cfg)
        runs[serial] = still_runs(samples[serial], args.min_static_s)
        print(f"  {sum(s.det.ok for s in samples[serial])}/{len(samples[serial])} sampled frames see the board, "
              f"{len(runs[serial])} still stretch(es)")
    a, b = serials
    stretches = match_runs(runs[a], runs[b])
    print(f"Placements seen still by both cameras: {len(stretches)}\n")

    rows, results = [], []
    for n, (t0, t1) in enumerate(stretches, 1):
        det_a, in_a = average_in(samples[a], t0, t1)
        det_b, in_b = average_in(samples[b], t0, t1)
        if det_a is None or det_b is None:
            continue
        res = evaluate(det_a, det_b, poses[a], poses[b], intr[a], intr[b], board, setup_board.centre_m)
        res.update(placement=n, start_s=t0, duration_s=t1 - t0, frames=[len(in_a), len(in_b)])
        if res["usable"]:
            for serial, det, inside in ((a, det_a, in_a), (b, det_b, in_b)):
                mid = inside[len(inside) // 2].frame
                overlay(locate_frame(files[serial], mid), det, res["_X"], poses[serial], *intr[serial],
                        out / f"placement_{n:02d}_{serial}.png")
            p = res["position_world_cm"]
            print(f"#{n:02d} {t1 - t0:4.1f} s  corners {res['corners']:2d}  size {res['size_error_pct']:+5.2f} %  "
                  f"shape {res['shape_rms_mm']:4.2f} mm (max corner {res['max_corner_error_mm']:4.1f} mm)  "
                  f"at ({p[0]:6.1f}, {p[1]:6.1f}, {p[2]:6.1f}) cm, {res['from_setup_spot_cm']:5.1f} cm from setup "
                  f"spot, {res['distance_to_cameras_cm'][0]:.0f}/{res['distance_to_cameras_cm'][1]:.0f} cm "
                  f"from the cameras")
        else:
            print(f"#{n:02d} {t1 - t0:4.1f} s  only {res['corners']} corners seen by both cameras: skipped")
        results.append({k: v for k, v in res.items() if not k.startswith("_")})
        rows.append(results[-1])

    usable = [r for r in results if r["usable"]]
    summary = {}
    if usable:
        size = np.abs([r["size_error_pct"] for r in usable])
        shape = np.array([r["shape_rms_mm"] for r in usable])
        worst = np.array([r["max_corner_error_mm"] for r in usable])
        summary = {
            "placements": len(usable),
            "size_error_pct_abs": {"median": float(np.median(size)), "max": float(size.max())},
            "shape_rms_mm": {"median": float(np.median(shape)), "max": float(shape.max())},
            "max_corner_error_mm": float(worst.max()),
        }
        print(f"\nSummary over {len(usable)} placement(s): |size error| median {summary['size_error_pct_abs']['median']:.2f} %"
              f" (max {summary['size_error_pct_abs']['max']:.2f} %), shape RMS median "
              f"{summary['shape_rms_mm']['median']:.2f} mm (max {summary['shape_rms_mm']['max']:.2f} mm), "
              f"worst single corner {summary['max_corner_error_mm']:.1f} mm")
        far = [r for r in usable if r["from_setup_spot_cm"] >= 15]
        if far:
            print(f"Placements >= 15 cm from the setup spot (the honest test): {len(far)}, |size error| max "
                  f"{max(abs(r['size_error_pct']) for r in far):.2f} %, shape RMS max "
                  f"{max(r['shape_rms_mm'] for r in far):.2f} mm")
    else:
        print("\nNo usable placement: hold the board still for >= 2 s where both cameras see it whole.")

    (out / "report.json").write_text(json.dumps({
        "session": basename, "board": cfg.to_dict(), "setup_id": setup.get("id"),
        "placements": results, "summary": summary}, indent=2), encoding="utf-8")
    with open(out / "placements.csv", "w", newline="", encoding="utf-8") as f:
        cols = ["placement", "start_s", "duration_s", "corners", "size_error_pct", "shape_rms_mm",
                "max_corner_error_mm", "from_setup_spot_cm"]
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    return 0 if usable else 1


if __name__ == "__main__":
    sys.exit(main())
