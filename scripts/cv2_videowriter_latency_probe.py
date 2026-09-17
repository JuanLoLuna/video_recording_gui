#!/usr/bin/env python3
"""SDK-independent probe for cv2.VideoWriter's per-frame write cost.

Run this ON THE RECORDING LAPTOP, pointed at the SAME output disk you
record to. No camera or PySpin needed: uses synthetic frames sized to
match our measured real frame size (~1.97 MB/frame). This is the same
kind of test as disk_write_latency_probe.py, but through
cv2.VideoWriter instead of a raw file write -- the question is whether
OpenCV's AVI writer has its own fixed per-call overhead (like SpinVideo's
~18ms/frame Append(), already shown NOT to be explained by raw disk
latency), or whether it's actually fast, making it a viable SpinVideo
replacement.

v2: the first version used pure random per-pixel noise, which is the
worst possible case for any lossless/compressed codec (incompressible,
so e.g. Huffyuv's "compressed" output came out LARGER than the raw
input) -- real camera footage has spatial structure noise doesn't, so
those numbers were likely pessimistic. This version uses a smooth
gradient plus modest noise, closer to real footage's compressibility.
Also tries more raw/uncompressed fourcc candidates in both grayscale
and BGR modes, since the previous 'DIB ' grayscale attempt failed to
open on one system (some FFmpeg-backed builds only accept certain raw
tags in color mode).

Usage:
    python3 scripts/cv2_videowriter_latency_probe.py /path/to/output/dir

Interpreting the results, compared against:
  - disk_write_latency_probe.py's raw-write numbers (the disk's actual
    ceiling), and
  - SpinVideo's observed ~18ms/frame Append() cost:

  - A codec here landing close to the raw-disk numbers (a few ms or
    less): cv2.VideoWriter is a viable SpinVideo replacement for that
    codec -- worth wiring into the real pipeline and re-profiling with
    an actual camera.
  - Every codec here also ~15-20ms+ regardless of choice: the cost is
    likely inherent to AVI-container writing in general (per-frame
    header/index bookkeeping), not specific to SpinVideo -- pointing
    instead toward a raw-frame-dump-then-convert approach rather than
    just swapping which library writes the AVI.
"""
import os
import sys
import time
import statistics

import numpy as np
import cv2

WIDTH = 1928
HEIGHT = 1020
N_FRAMES = 200
FPS = 100.0

# (fourcc, description, isColor)
CANDIDATES = [
    ("DIB ", "uncompressed grayscale (DIB)", False),
    ("DIB ", "uncompressed color (DIB)", True),
    ("Y800", "uncompressed grayscale (Y800)", False),
    ("GREY", "uncompressed grayscale (GREY)", False),
    ("I420", "uncompressed color YUV420 (I420)", True),
    ("HFYU", "Huffyuv color (lossless)", True),
    ("FFV1", "FFV1 color (lossless)", True),
    ("MJPG", "MJPEG color (for comparison vs SpinVideo's MJPGOption)", True),
]


def make_frame(is_color: bool) -> np.ndarray:
    """A smooth gradient + modest noise -- compressible like real footage,
    unlike pure random noise (which is worst-case for any codec and
    misrepresents how fast real recording would actually be).
    """
    x = np.linspace(0, 255, WIDTH, dtype=np.float32)
    y = np.linspace(0, 255, HEIGHT, dtype=np.float32)
    gradient = (x[None, :] + y[:, None]) / 2.0
    noise = np.random.randint(-8, 8, (HEIGHT, WIDTH)).astype(np.float32)
    gray = np.clip(gradient + noise, 0, 255).astype(np.uint8)
    if not is_color:
        return gray
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def probe_codec(out_dir: str, fourcc_str: str, label: str, is_color: bool) -> None:
    frame = make_frame(is_color)
    path = os.path.join(out_dir, f"_cv2_probe_{fourcc_str.strip()}_{'c' if is_color else 'g'}.avi")
    fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
    writer = cv2.VideoWriter(path, fourcc, FPS, (WIDTH, HEIGHT), isColor=is_color)
    if not writer.isOpened():
        print(f"{label:55s} -- could not open (codec unavailable on this system)")
        writer.release()
        # cv2 can leave a 0-byte file behind even when isOpened() is
        # False -- don't leave that sitting in the (real) output dir.
        try:
            os.remove(path)
        except OSError:
            pass
        return

    latencies_ms = []
    for _ in range(N_FRAMES):
        t0 = time.monotonic()
        writer.write(frame)
        latencies_ms.append((time.monotonic() - t0) * 1000.0)
    writer.release()

    try:
        size = os.path.getsize(path)
    except OSError:
        size = None
    try:
        os.remove(path)
    except OSError:
        pass

    ordered = sorted(latencies_ms)
    p95 = ordered[int(len(ordered) * 0.95)]
    size_str = f", file={size / 1e6:.1f}MB" if size is not None else ""
    print(
        f"{label:55s} mean={statistics.mean(latencies_ms):7.2f}ms  "
        f"p95={p95:7.2f}ms  max={max(latencies_ms):7.2f}ms{size_str}"
    )


def main() -> None:
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} /path/to/output/dir")
        sys.exit(1)
    out_dir = sys.argv[1]
    os.makedirs(out_dir, exist_ok=True)

    print(f"cv2 version: {cv2.__version__}")
    print(
        f"Writing {N_FRAMES} frames (~{WIDTH * HEIGHT / 1e6:.2f} MB/frame gray, "
        f"~{WIDTH * HEIGHT * 3 / 1e6:.2f} MB/frame color) to {out_dir}"
    )
    print()
    for fourcc_str, desc, is_color in CANDIDATES:
        probe_codec(out_dir, fourcc_str, desc, is_color)


if __name__ == "__main__":
    main()
