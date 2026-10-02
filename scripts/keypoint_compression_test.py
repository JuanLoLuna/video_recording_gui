#!/usr/bin/env python3
"""Does MJPEG compression change the keypoints / joint angles the pose model gives?

PSNR says how close the pixels are; what matters for the study is whether the
pose pipeline (RTMDet-m person detector -> RTMW-x, COCO-WholeBody 133 keypoints)
returns different joint angles on the compressed video than on the lossless one.

Workflow -- the same clip in four forms, pose model run on each, then compared:

  1. Record a clip with compression OFF (the app default: lossless GREY AVI).
     This is the reference.
  2. python scripts/keypoint_compression_test.py encode --video clip.avi --out-dir kp_test
     writes, next to each other, from the SAME decoded frames:
       <stem>_mjpeg_app.avi        MJPG via the FFMPEG backend, plain constructor,
                                   grayscale -- exactly what the app writes with
                                   compression ON.
       <stem>_noise_control.avi    lossless, plus +-1-level Gaussian noise. An
                                   imperceptible change (~48 dB). How much the
                                   keypoints move for THIS tells you the pose model's
                                   own sensitivity: compression is harmless if it
                                   moves them about as little.
       <stem>_mjpeg_cvcolor.avi    (--cv-color) OpenCV's own MJPEG encoder on 3-channel
                                   frames: ~45 dB, ~3x larger than the app's. The
                                   fallback if the app's quality turns out too low.
  3. Run your pose pipeline on the reference and each variant and save, per video, an
     .npz with  keypoints (T, 133, 2) pixels  and  scores (T, 133).
     Frames with no detected person: NaN keypoints (scores ignored). If several people
     are detected keep the highest-scoring one, the same way for every video.
  4. python scripts/keypoint_compression_test.py compare \\
         --reference kp_clip_lossless.npz \\
         --variant mjpeg_app=kp_clip_mjpeg_app.npz noise_control=kp_clip_noise_control.npz \\
         --out-dir kp_test

compare reports, per variant: keypoint displacement in pixels (by body part), keypoints
and whole frames lost or gained, and the joint-angle difference in degrees (elbow,
shoulder, wrist and finger joints, computed in the image plane) next to the scale of the
real signal. Decide whether the angle difference matters for your analysis by comparing it
with noise_control and with how much the angles actually move.

Needs numpy and opencv-python only (the pose model is run by you).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

K = 133  # COCO-WholeBody keypoints
GROUPS = {  # name -> slice of the 133 keypoints
    "body": slice(0, 17), "feet": slice(17, 23), "face": slice(23, 91),
    "left_hand": slice(91, 112), "right_hand": slice(112, 133),
}
NOISE_SIGMA = 1.0
DEFAULT_SCORE_THR = 0.3


# ----------------------------------------------------------------------
# Joint angles (COCO-WholeBody indices)
# ----------------------------------------------------------------------
def joint_definitions() -> dict[str, tuple[int, int, int]]:
    """name -> (a, b, c); the angle is at b, between the segments b->a and b->c."""
    defs: dict[str, tuple[int, int, int]] = {}
    for side, (sh, el, wr, hip, hand) in {"L": (5, 7, 9, 11, 91), "R": (6, 8, 10, 12, 112)}.items():
        defs[f"{side}_shoulder"] = (hip, sh, el)
        defs[f"{side}_elbow"] = (sh, el, wr)
        defs[f"{side}_wrist"] = (el, hand, hand + 9)  # elbow - hand wrist - middle-finger MCP
        for finger, start in (("index", 5), ("middle", 9), ("ring", 13), ("pinky", 17)):
            defs[f"{side}_{finger}_mcp"] = (hand, hand + start, hand + start + 1)
            defs[f"{side}_{finger}_pip"] = (hand + start, hand + start + 1, hand + start + 2)
            defs[f"{side}_{finger}_dip"] = (hand + start + 1, hand + start + 2, hand + start + 3)
        defs[f"{side}_thumb_mcp"] = (hand + 1, hand + 2, hand + 3)
        defs[f"{side}_thumb_ip"] = (hand + 2, hand + 3, hand + 4)
    return defs


def valid_mask(keypoints: np.ndarray, scores: np.ndarray, thr: float) -> np.ndarray:
    """(T, K) bool: keypoint present and confident."""
    return np.isfinite(keypoints).all(axis=-1) & np.isfinite(scores) & (scores >= thr)


def angle_series(kp: np.ndarray, valid: np.ndarray, a: int, b: int, c: int) -> np.ndarray:
    """Angle at b in degrees per frame, NaN where any of the three points is invalid."""
    ba, bc = kp[:, a] - kp[:, b], kp[:, c] - kp[:, b]
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = (ba * bc).sum(-1) / (np.linalg.norm(ba, axis=-1) * np.linalg.norm(bc, axis=-1))
    out = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
    out[~(valid[:, a] & valid[:, b] & valid[:, c])] = np.nan
    return out


def all_angles(kp, valid) -> dict[str, np.ndarray]:
    return {name: angle_series(kp, valid, *idx) for name, idx in joint_definitions().items()}


# ----------------------------------------------------------------------
# Comparison
# ----------------------------------------------------------------------
def _stats(values: np.ndarray) -> dict:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"n": 0, "median": None, "p95": None, "p99": None, "max": None}
    return {"n": int(values.size), "median": float(np.median(values)), "p95": float(np.percentile(values, 95)),
            "p99": float(np.percentile(values, 99)), "max": float(values.max())}


def load_keypoints(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    kp, sc = np.asarray(data["keypoints"], dtype=np.float64), np.asarray(data["scores"], dtype=np.float64)
    if kp.ndim != 3 or kp.shape[1:] != (K, 2) or sc.shape != kp.shape[:2]:
        raise ValueError(f"{path}: expected keypoints (T,{K},2) and scores (T,{K}); got {kp.shape} / {sc.shape}")
    return kp, sc


def compare_pair(ref: tuple, var: tuple, thr: float) -> dict:
    (kp_r, sc_r), (kp_v, sc_v) = ref, var
    if len(kp_r) != len(kp_v):
        raise ValueError(f"frame counts differ: reference {len(kp_r)} vs variant {len(kp_v)}")
    vr, vv = valid_mask(kp_r, sc_r, thr), valid_mask(kp_v, sc_v, thr)
    both = vr & vv

    dist = np.linalg.norm(kp_r - kp_v, axis=-1)
    dist[~both] = np.nan
    displacement = {"all": _stats(dist)}
    displacement.update({name: _stats(dist[:, sl]) for name, sl in GROUPS.items()})

    detected_r, detected_v = vr.any(axis=1), vv.any(axis=1)
    ang_r, ang_v = all_angles(kp_r, vr), all_angles(kp_v, vv)
    per_angle, pooled = {}, []
    for name in ang_r:
        diff = np.abs(ang_r[name] - ang_v[name])
        ref_series = ang_r[name][np.isfinite(ang_r[name])]
        step = np.abs(np.diff(ang_r[name]))
        per_angle[name] = {
            "abs_diff_deg": _stats(diff),
            "ref_range_deg": float(np.percentile(ref_series, 95) - np.percentile(ref_series, 5)) if ref_series.size > 2 else None,
            "ref_median_frame_step_deg": float(np.nanmedian(step)) if np.isfinite(step).any() else None,
            "valid_in_ref": int(np.isfinite(ang_r[name]).sum()),
            "valid_in_both": int(np.isfinite(diff).sum()),
        }
        pooled.append(diff[np.isfinite(diff)])
    pooled = np.concatenate(pooled) if pooled else np.array([])
    return {
        "frames": int(len(kp_r)),
        "frames_with_person": {"reference": int(detected_r.sum()), "variant": int(detected_v.sum()),
                               "lost": int((detected_r & ~detected_v).sum()), "gained": int((~detected_r & detected_v).sum())},
        "keypoints_valid": {"reference": int(vr.sum()), "variant": int(vv.sum()),
                            "lost": int((vr & ~vv).sum()), "gained": int((~vr & vv).sum()),
                            "pct_lost": float((vr & ~vv).sum() / max(1, vr.sum()) * 100)},
        "displacement_px": displacement,
        "angle_abs_diff_deg_pooled": _stats(pooled),
        "angles": per_angle,
    }


# ----------------------------------------------------------------------
# Re-encoding
# ----------------------------------------------------------------------
def read_gray_frames(cv2, path: Path):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    cap.release()
    return frames, float(fps)


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return float("inf") if mse == 0 else 10.0 * np.log10(255.0 ** 2 / mse)


def write_video(cv2, path: Path, frames, fps: float, *, fourcc: str, api, color: bool, params=None):
    h, w = frames[0].shape
    code = cv2.VideoWriter_fourcc(*fourcc)
    if params is not None:
        writer = cv2.VideoWriter(str(path), api, code, fps, (w, h), params)
    else:
        writer = cv2.VideoWriter(str(path), api, code, fps, (w, h), color)
    if not writer.isOpened():
        writer.release()
        raise RuntimeError(f"could not open {path} ({fourcc}, api={api})")
    try:
        backend = writer.getBackendName()
    except Exception:
        backend = "?"
    for f in frames:
        writer.write(cv2.cvtColor(f, cv2.COLOR_GRAY2BGR) if color else f)
    writer.release()
    return backend


def cmd_encode(args) -> int:
    import cv2

    src = Path(args.video)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames, fps = read_gray_frames(cv2, src)
    if not frames:
        print(f"Could not read any frames from {src}")
        return 2
    print(f"{src.name}: {len(frames)} frames, {frames[0].shape[1]}x{frames[0].shape[0]}, {fps:.2f} fps")
    stem = src.stem
    rng = np.random.default_rng(0)  # fixed seed: the control is reproducible
    plan = [
        ("mjpeg_app", lambda p: write_video(cv2, p, frames, fps, fourcc="MJPG", api=cv2.CAP_FFMPEG, color=False)),
        ("noise_control", lambda p: write_video(
            cv2, p,
            [np.clip(f.astype(np.float32) + rng.normal(0, NOISE_SIGMA, f.shape), 0, 255).round().astype(np.uint8) for f in frames],
            fps, fourcc="GREY", api=cv2.CAP_FFMPEG, color=False)),
    ]
    if args.cv_color:
        plan.append(("mjpeg_cvcolor", lambda p: write_video(
            cv2, p, frames, fps, fourcc="MJPG", api=cv2.CAP_OPENCV_MJPEG, color=True,
            params=[cv2.VIDEOWRITER_PROP_IS_COLOR, 1, cv2.VIDEOWRITER_PROP_QUALITY, 75])))

    report = {"source": str(src), "frames": len(frames), "fps": fps, "variants": {}}
    raw_bytes = frames[0].size * len(frames)
    for name, make in plan:
        path = out_dir / f"{stem}_{name}.avi"
        backend = make(path)
        decoded, _ = read_gray_frames(cv2, path)
        n = min(len(decoded), len(frames))
        ps = [psnr(a, b) for a, b in zip(frames[:n], decoded[:n])]
        size = path.stat().st_size
        report["variants"][name] = {
            "file": path.name, "backend": backend, "frames_written_readable": len(decoded),
            "psnr_mean_db": float(np.mean([p for p in ps if np.isfinite(p)])) if any(np.isfinite(p) for p in ps) else "inf",
            "size_mb": size / 1e6, "ratio_vs_raw": raw_bytes / size if size else None,
        }
        flag = "" if len(decoded) == len(frames) else f"  !! decoded {len(decoded)} of {len(frames)} frames"
        print(f"  wrote {path.name}  backend={backend}  PSNR {report['variants'][name]['psnr_mean_db']:.1f} dB  "
              f"{size / 1e6:.1f} MB ({report['variants'][name]['ratio_vs_raw']:.1f}x smaller){flag}")
    (out_dir / f"{stem}_encode_report.json").write_text(json.dumps(report, indent=2, default=str))
    print(f"\nNext: run your pose pipeline on {src.name} and the files above, saving keypoints (T,133,2) + scores (T,133) "
          "to .npz, then use the 'compare' subcommand.")
    return 0


# ----------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------
def fmt(x, spec=".2f") -> str:
    return "-" if x is None else format(x, spec)


def cmd_compare(args) -> int:
    ref = load_keypoints(args.reference)
    results = {"reference": args.reference, "score_threshold": args.score_thr, "variants": {}}
    for item in args.variant:
        name, _, path = item.partition("=")
        if not path:
            print(f"--variant expects NAME=path.npz, got {item!r}")
            return 2
        results["variants"][name] = compare_pair(ref, load_keypoints(path), args.score_thr)

    print(f"Reference: {args.reference}  ({next(iter(results['variants'].values()))['frames']} frames, "
          f"score threshold {args.score_thr})\n")
    print("Joint-angle difference vs reference, degrees (pooled over every angle and frame):")
    print(f"  {'variant':<16}{'n':>8}{'median':>8}{'p95':>8}{'p99':>8}{'max':>8}")
    for name, r in results["variants"].items():
        s = r["angle_abs_diff_deg_pooled"]
        print(f"  {name:<16}{s['n']:>8}{fmt(s['median']):>8}{fmt(s['p95']):>8}{fmt(s['p99']):>8}{fmt(s['max']):>8}")

    print("\nKeypoint displacement, pixels (median / p95), by body part:")
    print(f"  {'variant':<16}" + "".join(f"{g:>16}" for g in ["all", *GROUPS]))
    for name, r in results["variants"].items():
        print(f"  {name:<16}" + "".join(
            f"{fmt(r['displacement_px'][g]['median']) + ' / ' + fmt(r['displacement_px'][g]['p95']):>16}"
            for g in ["all", *GROUPS]))

    print("\nDetection (frames with a person / keypoints above the score threshold):")
    for name, r in results["variants"].items():
        f, k = r["frames_with_person"], r["keypoints_valid"]
        print(f"  {name:<16} frames lost {f['lost']} gained {f['gained']} (of {r['frames']}) | "
              f"keypoints lost {k['lost']} ({k['pct_lost']:.2f}%) gained {k['gained']}")

    first = next(iter(results["variants"].values()))
    scale = [(n, a["ref_range_deg"], a["ref_median_frame_step_deg"]) for n, a in first["angles"].items()
             if a["ref_range_deg"] is not None]
    if scale:
        ranges = np.array([s[1] for s in scale])
        steps = np.array([s[2] for s in scale if s[2] is not None])
        print(f"\nSignal scale in the reference: typical angle range (p95-p5) {np.median(ranges):.1f} deg, "
              f"typical frame-to-frame change {np.median(steps):.2f} deg." if steps.size else "")

    print("\nWorst angles for each variant (p95 difference, deg):")
    for name, r in results["variants"].items():
        worst = sorted(((a["abs_diff_deg"]["p95"], n) for n, a in r["angles"].items() if a["abs_diff_deg"]["p95"] is not None),
                       reverse=True)[:5]
        print(f"  {name:<16}" + ", ".join(f"{n} {v:.1f}" for v, n in worst))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "keypoint_compression_results.json"
    path.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {path}")
    print("Reading it: compare each compressed variant's angle difference with noise_control (the pose "
          "model's sensitivity to an imperceptible change) and with the signal scale above.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    enc = sub.add_parser("encode", help="re-encode a lossless clip into the comparison variants")
    enc.add_argument("--video", required=True, help="the lossless reference clip (app default, compression OFF)")
    enc.add_argument("--out-dir", default="kp_test")
    enc.add_argument("--cv-color", action="store_true", help="also write the higher-quality OpenCV color MJPEG variant")
    enc.set_defaults(func=cmd_encode)
    cmp_ = sub.add_parser("compare", help="compare keypoints/angles from the pose model across variants")
    cmp_.add_argument("--reference", required=True, help="npz from the lossless clip")
    cmp_.add_argument("--variant", nargs="+", required=True, help="NAME=path.npz, one or more")
    cmp_.add_argument("--score-thr", type=float, default=DEFAULT_SCORE_THR,
                      help=f"keypoint confidence below this is treated as missing (default {DEFAULT_SCORE_THR})")
    cmp_.add_argument("--out-dir", default="kp_test")
    cmp_.set_defaults(func=cmd_compare)
    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
