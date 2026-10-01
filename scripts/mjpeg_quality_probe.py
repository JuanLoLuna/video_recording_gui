#!/usr/bin/env python3
"""How much image quality does MJPEG cost, and which encoder settings control it?

Takes real frames from a camera, encodes the SAME frames several ways, decodes
them again, and compares against the originals. Reports PSNR, SSIM, worst-case
pixel error, file size and encode time for each, and saves side-by-side crops
so you can judge by eye.

Background (see the multi-camera findings): passing the (IS_COLOR, QUALITY)
params to cv2.VideoWriter makes the FFMPEG backend reject the call, and OpenCV
silently falls back to its built-in CV_MJPEG encoder -- ~80x slower here. The
plain FFMPEG constructor is fast but gives us no quality control. This probe
answers: how good is the fast path's default quality, can quality be set on it,
and what does each CV_MJPEG quality level look like for comparison?

Variants (each writes its own AVI in a temp folder, deleted afterwards):
  control        GREY, uncompressed. Must come back (almost) identical; its score is
                 the measurement floor of this decode path.
  ffmpeg         MJPG via the FFMPEG backend, plain constructor (the proposed fix).
  ffmpeg+q=N     same, then writer.set(VIDEOWRITER_PROP_QUALITY, N) after open.
                 If the file size equals plain ffmpeg's, the call is ignored.
  cv_mjpeg gray  OpenCV's built-in MJPEG encoder with the params overload and
                 isColor=False. q=75 is what the app does today when compression is on.
  cv_mjpeg COLOR the same encoder fed 3-channel frames.
                 Both are slow (~0.8 s/frame on the rig) but encoded offline here, so
                 speed doesn't matter. Expect the full run to take a few minutes.

USAGE (on the recording laptop; needs PySpin only for the capture step)
    # point the camera at the real scene, lit the way you will record, then:
    python scripts/mjpeg_quality_probe.py --serial 26134271
    python scripts/mjpeg_quality_probe.py --serial 23227865

    # re-analyse saved frames anywhere (no camera needed):
    python scripts/mjpeg_quality_probe.py --frames-file quality_output/frames_26134271.npy

Quality depends on the scene -- sensor noise and fine texture are what JPEG
loses first -- so use a representative one, not a blank wall.

PSNR rule of thumb (a guide, not a verdict -- what matters is whether your
downstream analysis is sensitive to the loss): >= 45 dB effectively
indistinguishable, 40-45 excellent, 35-40 good, < 30 visible artifacts.

Frames are decoded with cv2.VideoCapture, which returns 3-channel BGR even for
grayscale files; they are converted back to gray before comparing. The control
variant measures what that round trip costs on its own.
"""
from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

QUALITIES_FFMPEG_SET = (50, 75, 95)
QUALITIES_CV_MJPEG = (50, 75, 95)
CORRUPT_PSNR_DB = 20.0  # below this the decoded frame doesn't resemble the original
CROP_W, CROP_H = 320, 240
DIFF_GAIN = 8  # amplification of |orig - decoded| in the saved montage


# ----------------------------------------------------------------------
# Pure metrics (numpy/cv2 only)
# ----------------------------------------------------------------------
def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return float("inf") if mse == 0 else 10.0 * np.log10(255.0 ** 2 / mse)


def ssim(cv2, a: np.ndarray, b: np.ndarray) -> float:
    """Mean SSIM, 11x11 Gaussian window (sigma 1.5), the standard formulation."""
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2

    def blur(x):
        return cv2.GaussianBlur(x, (11, 11), 1.5)

    mu_a, mu_b = blur(a), blur(b)
    var_a = blur(a * a) - mu_a ** 2
    var_b = blur(b * b) - mu_b ** 2
    cov = blur(a * b) - mu_a * mu_b
    s = ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (var_a + var_b + c2))
    return float(s.mean())


def compare(cv2, originals, decoded) -> dict:
    """Per-variant quality summary over frames that exist on both sides."""
    n = min(len(originals), len(decoded))
    if n == 0:
        return {"frames_compared": 0}
    ps, ss, worst, frac = [], [], [], []
    for orig, dec in zip(originals[:n], decoded[:n]):
        diff = np.abs(orig.astype(np.int16) - dec.astype(np.int16))
        ps.append(psnr(orig, dec))
        ss.append(ssim(cv2, orig, dec))
        worst.append(int(diff.max()))
        frac.append(float((diff > 4).mean()) * 100.0)
    finite = [p for p in ps if np.isfinite(p)]
    return {
        "frames_compared": n,
        "psnr_mean": float(np.mean(finite)) if finite else float("inf"),
        "psnr_min": float(min(ps)),
        "ssim_mean": float(np.mean(ss)),
        "max_abs_err_mean": float(np.mean(worst)),
        "pct_pixels_err_gt4": float(np.mean(frac)),
    }


# ----------------------------------------------------------------------
# Capture (PySpin)
# ----------------------------------------------------------------------
def capture_frames(serial, n: int, every: int, warmup: int, fps: float):
    import PySpin
    import multi_camera_probe as mcp  # reuse the app-equivalent configure()

    system = PySpin.System.GetInstance()
    cam_list = system.GetCameras()
    cam = None
    frames = []
    try:
        if serial:
            cam = cam_list.GetBySerial(serial)
        elif cam_list.GetSize() == 1:
            cam = cam_list[0]
        else:
            raise SystemExit(f"{cam_list.GetSize()} cameras found; pass --serial")
        cam.Init()
        info = mcp.configure(PySpin, cam, fps)
        print(f"Capturing from {info.serial} {info.model}: {info.width}x{info.height} {info.pixel_format}, "
              f"{info.applied_fps:.1f} fps, link={info.speed}")
        if "Mono8" not in info.pixel_format:
            raise SystemExit(f"Pixel format {info.pixel_format}: this probe compares 8-bit grayscale only")
        cam.BeginAcquisition()
        grabbed = 0
        while len(frames) < n:
            image = cam.GetNextImage(2000)
            if image.IsIncomplete():
                image.Release()
                continue
            grabbed += 1
            if grabbed > warmup and (grabbed - warmup - 1) % every == 0:
                frames.append(np.array(image.GetNDArray(), copy=True))
            image.Release()
        return np.stack(frames), info.serial
    finally:
        if cam is not None:
            try:
                cam.EndAcquisition()
            except Exception:
                pass
            try:
                cam.DeInit()
            except Exception:
                pass
        cam = None  # noqa: F841 -- native handle must go before the list is cleared
        cam_list.Clear()
        system.ReleaseInstance()


# ----------------------------------------------------------------------
# Encode / decode variants
# ----------------------------------------------------------------------
def build_variants() -> list[dict]:
    variants = [
        {"name": "control (GREY, lossless)", "kind": "ffmpeg", "fourcc": "GREY", "set_q": None},
        {"name": "ffmpeg (plain, proposed fix)", "kind": "ffmpeg", "fourcc": "MJPG", "set_q": None},
    ]
    variants += [{"name": f"ffmpeg + set(q={q})", "kind": "ffmpeg", "fourcc": "MJPG", "set_q": q}
                 for q in QUALITIES_FFMPEG_SET]
    # isColor=False is what the app passes today (grayscale), so the (IS_COLOR=0, QUALITY)
    # fallback lands here. The COLOR rows write 3-channel BGR frames instead.
    variants += [{"name": f"cv_mjpeg gray q={q}" + (" (app today)" if q == 75 else ""), "kind": "cv_mjpeg",
                  "fourcc": "MJPG", "q": q, "color": False} for q in QUALITIES_CV_MJPEG]
    variants += [{"name": f"cv_mjpeg COLOR q={q}", "kind": "cv_mjpeg",
                  "fourcc": "MJPG", "q": q, "color": True} for q in QUALITIES_CV_MJPEG]
    return variants


def open_variant(cv2, v: dict, path: Path, fps: float, w: int, h: int):
    fourcc = cv2.VideoWriter_fourcc(*v["fourcc"])
    if v["kind"] == "ffmpeg":
        writer = cv2.VideoWriter(str(path), cv2.CAP_FFMPEG, fourcc, fps, (w, h), False)
        if writer.isOpened() and v.get("set_q") is not None:
            v["set_returned"] = bool(writer.set(cv2.VIDEOWRITER_PROP_QUALITY, float(v["set_q"])))
    else:
        params = [cv2.VIDEOWRITER_PROP_IS_COLOR, 1 if v.get("color") else 0,
                  cv2.VIDEOWRITER_PROP_QUALITY, int(v["q"])]
        writer = cv2.VideoWriter(str(path), cv2.CAP_OPENCV_MJPEG, fourcc, fps, (w, h), params)
    return writer


def decode_all(cv2, path: Path) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    out = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            out.append(frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    finally:
        cap.release()
    return out


def montage(cv2, orig: np.ndarray, dec: np.ndarray) -> np.ndarray:
    """orig | decoded | amplified |diff|, centre crop."""
    h, w = orig.shape
    ch, cw = min(CROP_H, h), min(CROP_W, w)
    y0, x0 = (h - ch) // 2, (w - cw) // 2
    o = orig[y0:y0 + ch, x0:x0 + cw]
    d = dec[y0:y0 + ch, x0:x0 + cw]
    diff = np.clip(np.abs(o.astype(np.int16) - d.astype(np.int16)) * DIFF_GAIN, 0, 255).astype(np.uint8)
    return np.hstack([o, d, diff])


def run_variants(cv2, frames: np.ndarray, fps: float, tmp: Path, out_dir: Path, ffmpeg_options: str | None):
    n, h, w = frames.shape
    raw_bytes = h * w
    originals = list(frames)
    rows = []
    baseline_size = None
    first_size = {}  # (kind, color) -> size of the first quality level, to spot an ignored QUALITY
    if ffmpeg_options:
        os.environ["OPENCV_FFMPEG_WRITER_OPTIONS"] = ffmpeg_options
        print(f"OPENCV_FFMPEG_WRITER_OPTIONS={ffmpeg_options!r} (applies to the FFMPEG variants; "
              "check that the plain-ffmpeg file size actually changes)")
    for v in build_variants():
        path = tmp / (v["name"].split(" ")[0].replace("(", "") + f"_{len(rows)}.avi")
        row = {"variant": v["name"]}
        print(f"  encoding {v['name']} ...", flush=True)
        writer = open_variant(cv2, v, path, fps, w, h)
        if not writer.isOpened():
            writer.release()
            row["note"] = "could not open"
            rows.append(row)
            continue
        try:
            row["backend"] = writer.getBackendName()
        except Exception:
            row["backend"] = "?"
        times = []
        for frame in originals:
            to_write = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR) if v.get("color") else frame
            t0 = time.perf_counter()
            writer.write(to_write)
            times.append((time.perf_counter() - t0) * 1000.0)
        writer.release()
        size = path.stat().st_size
        decoded = decode_all(cv2, path)
        row.update(compare(cv2, originals, decoded))
        row.update({
            "frames_decoded": len(decoded),
            "kb_per_frame": size / n / 1024.0,
            "ratio_vs_raw": raw_bytes * n / size if size else 0.0,
            "mb_per_s_at_fps": size / n * fps / 1e6,
            "encode_ms_mean": float(np.mean(times)),
        })
        if v["name"].startswith("ffmpeg (plain"):
            baseline_size = size
        if v.get("set_q") is not None:
            row["note"] = f"set() returned {v.get('set_returned')}"
            if baseline_size:
                changed = abs(size - baseline_size) / baseline_size > 0.03
                row["note"] += "; size " + ("CHANGED -> quality is controllable" if changed
                                            else "unchanged -> set() is ignored")
        if v["kind"] == "cv_mjpeg":
            key = (v["kind"], bool(v.get("color")))
            first = first_size.setdefault(key, size)
            if v["q"] != QUALITIES_CV_MJPEG[0] and abs(size - first) / first <= 0.03:
                row["note"] = "size identical to the first quality level -> QUALITY param has no effect"
        if row.get("psnr_mean") is not None and row["psnr_mean"] < CORRUPT_PSNR_DB:
            row["note"] = (row.get("note", "") + "; " if row.get("note") else "") + \
                "!! decoded frames do not resemble the originals (corrupt stream?)"
        if decoded and row.get("frames_compared"):
            idx = len(originals) // 2
            safe = "".join(c if c.isalnum() else "_" for c in v["name"]).strip("_")
            cv2.imwrite(str(out_dir / f"{safe}.png"), montage(cv2, originals[idx], decoded[idx]))
        rows.append(row)
        path.unlink(missing_ok=True)
    return rows


def fmt(value, spec) -> str:
    if value is None:
        return "-"
    if isinstance(value, float) and value == float("inf"):
        return "inf"
    return format(value, spec)


def print_table(rows, fps: float) -> None:
    print("\n" + "=" * 118)
    print(f"{'variant':<32} {'backend':<11} {'KB/fr':>7} {'ratio':>6} {'MB/s@' + format(fps, '.0f'):>8} "
          f"{'enc ms':>7} {'PSNR dB':>8} {'min':>6} {'SSIM':>7} {'maxerr':>6} {'%>4':>5}")
    for r in rows:
        if "kb_per_frame" not in r:
            print(f"{r['variant']:<32} {r.get('note', '')}")
            continue
        print(f"{r['variant']:<32} {r.get('backend', '?'):<11} {fmt(r['kb_per_frame'], '.1f'):>7} "
              f"{fmt(r['ratio_vs_raw'], '.1f'):>6} {fmt(r['mb_per_s_at_fps'], '.2f'):>8} "
              f"{fmt(r['encode_ms_mean'], '.1f'):>7} {fmt(r.get('psnr_mean'), '.1f'):>8} "
              f"{fmt(r.get('psnr_min'), '.1f'):>6} {fmt(r.get('ssim_mean'), '.4f'):>7} "
              f"{fmt(r.get('max_abs_err_mean'), '.0f'):>6} {fmt(r.get('pct_pixels_err_gt4'), '.1f'):>5}"
              + (f"   {r['note']}" if r.get("note") else ""))
    print("\n  PSNR/SSIM: higher is better (PSNR inf = identical). maxerr = worst single-pixel error (0-255), "
          "averaged over frames; %>4 = share of pixels off by more than 4 levels.")
    print("  The control row is the floor of this measurement; cv_mjpeg q=75 is what the app does today.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--serial", help="camera to capture from (default: the only camera)")
    ap.add_argument("--frames-file", help="skip capture; analyse frames saved earlier (.npy, shape N,H,W uint8)")
    ap.add_argument("--frames", type=int, default=30, help="frames to capture (default 30)")
    ap.add_argument("--every", type=int, default=5, help="keep every Nth frame, so frames differ (default 5)")
    ap.add_argument("--warmup", type=int, default=15, help="frames to discard first (default 15)")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--max-frames", type=int, help="analyse only the first N frames (cv_mjpeg can be slow)")
    ap.add_argument("--ffmpeg-options", help="experimental: OPENCV_FFMPEG_WRITER_OPTIONS, e.g. 'qmin;2|qmax;2'")
    ap.add_argument("--out-dir", default="quality_output")
    args = ap.parse_args()

    import cv2

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.frames_file:
        frames = np.load(args.frames_file)
        label = Path(args.frames_file).stem.replace("frames_", "")
    else:
        frames, label = capture_frames(args.serial, args.frames, args.every, args.warmup, args.fps)
        saved = out_dir / f"frames_{label}.npy"
        np.save(saved, frames)
        print(f"Saved {len(frames)} frames to {saved} (re-run with --frames-file to re-analyse)")
    if frames.ndim != 3 or frames.dtype != np.uint8:
        print(f"Expected (N,H,W) uint8 frames, got {frames.shape} {frames.dtype}")
        return 2
    if args.max_frames:
        frames = frames[: args.max_frames]
    print(f"\nAnalysing {len(frames)} frames of {frames.shape[2]}x{frames.shape[1]} (OpenCV {cv2.__version__})")

    img_dir = out_dir / f"montages_{label}"
    img_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(img_dir / "original_frame.png"), frames[len(frames) // 2])
    tmp = Path(tempfile.mkdtemp(prefix="mjpeg_quality_"))
    try:
        rows = run_variants(cv2, frames, args.fps, tmp, img_dir, args.ffmpeg_options)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print_table(rows, args.fps)
    csv_path = out_dir / f"mjpeg_quality_{label}.csv"
    keys = ["variant", "backend", "frames_compared", "frames_decoded", "kb_per_frame", "ratio_vs_raw",
            "mb_per_s_at_fps", "encode_ms_mean", "psnr_mean", "psnr_min", "ssim_mean",
            "max_abs_err_mean", "pct_pixels_err_gt4", "note"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {csv_path}\nSide-by-side crops (original | decoded | difference x{DIFF_GAIN}) in {img_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
