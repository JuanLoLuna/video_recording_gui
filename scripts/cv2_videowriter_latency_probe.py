#!/usr/bin/env python3
"""SDK-independent probe for cv2.VideoWriter's per-frame write cost.

Run this ON THE RECORDING LAPTOP, pointed at the SAME output disk you
record to. No camera or PySpin needed: uses synthetic grayscale frames
sized to match our measured real frame size (~1.97 MB/frame). This is
the same kind of test as disk_write_latency_probe.py, but through
cv2.VideoWriter instead of a raw file write -- the question is whether
OpenCV's AVI writer has its own fixed per-call overhead (like SpinVideo's
~18ms/frame Append(), already shown NOT to be explained by raw disk
latency), or whether it's actually fast, making it a viable SpinVideo
replacement.

Usage:
    python3 scripts/cv2_videowriter_latency_probe.py /path/to/output/dir

Tries a few common fourcc codes and reports per-write timing for each
one OpenCV can actually open on this machine (codec availability is
platform/build-dependent, so some may report "could not open" -- that's
expected, not a bug).

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

# (fourcc, description)
CANDIDATES = [
    ("DIB ", "uncompressed (DIB)"),
    ("HFYU", "Huffyuv (lossless)"),
    ("FFV1", "FFV1 (lossless)"),
    ("MJPG", "MJPEG (for comparison against SpinVideo's MJPGOption)"),
]


def probe_codec(out_dir: str, fourcc_str: str, label: str) -> None:
    frame = np.random.randint(0, 256, (HEIGHT, WIDTH), dtype=np.uint8)
    path = os.path.join(out_dir, f"_cv2_probe_{fourcc_str.strip()}.avi")
    fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
    writer = cv2.VideoWriter(path, fourcc, FPS, (WIDTH, HEIGHT), isColor=False)
    if not writer.isOpened():
        print(f"{label:55s} -- could not open (codec unavailable on this system)")
        writer.release()
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
        f"Writing {N_FRAMES} grayscale frames "
        f"({WIDTH}x{HEIGHT} = {WIDTH * HEIGHT / 1e6:.2f} MB/frame) to {out_dir}"
    )
    print()
    for fourcc_str, desc in CANDIDATES:
        probe_codec(out_dir, fourcc_str, desc)


if __name__ == "__main__":
    main()
