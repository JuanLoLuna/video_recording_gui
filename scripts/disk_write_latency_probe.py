#!/usr/bin/env python3
"""Standalone, SDK-independent probe for per-frame disk write latency.

Run this ON THE RECORDING LAPTOP, pointed at the SAME directory you record
into (not this dev machine -- disk characteristics are host/drive-specific).

FLIR's own "Saving Images at High Bandwidth" application note warns that
advertised SSD sequential-write speeds don't apply to machine-vision
recording: benchmarks use large (1GB+) sequential files, while camera
frames are many small (1-10MB) writes, which can have much worse real-world
throughput than the drive's spec-sheet number. This script writes
realistically-sized chunks (matching our measured ~1.97MB/frame) and times
each write individually, with and without a forced flush -- to see whether
our observed ~18ms/append cost is a genuine disk characteristic under this
exact write pattern, or something specific to SpinVideo.

Usage:
    python3 scripts/disk_write_latency_probe.py /path/to/your/recording/output/dir

Interpreting the results:
  - If "no fsync" writes are fast (order of 1ms or less) but "fsync" writes
    are ~15-20ms: the disk's fsync/flush round-trip latency IS the ~18ms
    cost, matching what a naive per-frame-sync writer (like SpinVideo may
    be) would pay -- a genuine, largely irreducible disk characteristic
    for this write pattern, not a SpinVideo-specific bug.
  - If even "no fsync" writes are ~15-20ms: something else is throttling
    writes at the OS/filesystem level (e.g. sustained write-back pressure),
    independent of any explicit sync -- worth digging into separately
    (Spotlight indexing, antivirus scanning, or the destination filesystem
    itself).
  - If both are fast (sub-ms): the disk itself is not the bottleneck, and
    the ~18ms cost is specific to SpinVideo's internal behavior (its own
    bookkeeping, mutex, or an SDK-side inefficiency) -- a very different,
    and less fixable-by-us, conclusion.
"""
import os
import sys
import time
import statistics

FRAME_BYTES = 1_966_104  # matches segment bytes/frame_count from a real recording
N_FRAMES = 200           # ~2s worth of "frames" per pass -- enough for stable stats


def run_pass(path: str, *, use_fsync: bool) -> list[float]:
    chunk = os.urandom(FRAME_BYTES)  # random data: some filesystems can compress zeros
    latencies_ms = []
    with open(path, "wb") as f:
        for _ in range(N_FRAMES):
            t0 = time.monotonic()
            f.write(chunk)
            if use_fsync:
                f.flush()
                os.fsync(f.fileno())
            latencies_ms.append((time.monotonic() - t0) * 1000.0)
    os.remove(path)
    return latencies_ms


def summarize(label: str, samples: list[float]) -> None:
    samples_sorted = sorted(samples)
    p95 = samples_sorted[int(len(samples_sorted) * 0.95)]
    print(
        f"{label:22s} mean={statistics.mean(samples):7.2f}ms  "
        f"p95={p95:7.2f}ms  max={max(samples):7.2f}ms  "
        f"min={min(samples):7.2f}ms  n={len(samples)}"
    )


def main() -> None:
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} /path/to/your/recording/output/dir")
        sys.exit(1)
    out_dir = sys.argv[1]
    os.makedirs(out_dir, exist_ok=True)

    print(f"Writing {N_FRAMES} chunks of {FRAME_BYTES/1e6:.2f} MB to {out_dir}")
    print()

    no_sync_path = os.path.join(out_dir, "_probe_no_fsync.bin")
    no_sync = run_pass(no_sync_path, use_fsync=False)
    summarize("no fsync (buffered)", no_sync)

    fsync_path = os.path.join(out_dir, "_probe_fsync.bin")
    fsync_samples = run_pass(fsync_path, use_fsync=True)
    summarize("with fsync (synced)", fsync_samples)

    print()
    print(
        f"Effective throughput -- no fsync: "
        f"{FRAME_BYTES / (statistics.mean(no_sync) / 1000) / 1e6:.1f} MB/s, "
        f"with fsync: {FRAME_BYTES / (statistics.mean(fsync_samples) / 1000) / 1e6:.1f} MB/s"
    )


if __name__ == "__main__":
    main()
