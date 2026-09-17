#!/usr/bin/env python3
"""Independently verify a recorded AVI's integrity, frame count, and fps.

Run this AFTER a recording (and its AVI) has finished syncing/uploading
-- these files can be large. Uses only cv2 (no PySpin/camera needed), so
it works on any machine with this repo's dependencies installed; doesn't
need to be run on the recording laptop specifically.

Checks:
  - The file actually opens and decodes frame-by-frame without error
    (catches truncation/corruption cv2.VideoCapture itself would hit).
  - Decoded frame count vs. the container's own reported frame count.
  - Runs of pixel-identical consecutive frames (a frozen/stalled capture
    would show up as a long run; a couple of identical frames back-to-back
    can happen normally with a static scene, so this only warns past a
    configurable threshold).
  - If --segments is given: cross-checks decoded frame count against
    that segment's row in the recording's own _segments.csv, and computes
    the ACTUALLY achieved fps from (frame_count / (closed_at - opened_at))
    -- independent of whatever fps is merely stored as the container's
    playback-speed metadata.

Usage:
    python3 scripts/verify_avi.py path/to/recording-0000.avi
    python3 scripts/verify_avi.py path/to/recording-0000.avi --segments path/to/recording_segments.csv
    python3 scripts/verify_avi.py path/to/recording-0000.avi --segments ... --stride 10  # faster, samples frozen-frame check
"""
import argparse
import csv
import os
import sys

import cv2
import numpy as np


def find_segment_row(segments_csv: str | None, avi_path: str) -> dict | None:
    """Find the row in segments.csv matching this AVI's filename, if any."""
    if not segments_csv:
        return None
    name = os.path.basename(avi_path)
    with open(segments_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["segment_file"] == name:
                return row
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("avi_path")
    ap.add_argument("--segments", help="path to the recording's _segments.csv, for cross-checking")
    ap.add_argument(
        "--stride", type=int, default=1,
        help="apply the frozen-frame check every Nth frame instead of every frame (default: every frame)",
    )
    ap.add_argument(
        "--max-frozen-run", type=int, default=5,
        help="warn if this many consecutive checked frames are pixel-identical (default: 5)",
    )
    args = ap.parse_args()

    if not os.path.isfile(args.avi_path):
        print(f"FAILED: {args.avi_path} does not exist")
        sys.exit(1)

    cap = cv2.VideoCapture(args.avi_path)
    if not cap.isOpened():
        print(f"FAILED to open {args.avi_path} -- likely corrupt or an unsupported codec on this machine")
        sys.exit(1)

    container_fps = cap.get(cv2.CAP_PROP_FPS)
    container_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    file_size = os.path.getsize(args.avi_path)

    print(f"File: {args.avi_path} ({file_size / 1e6:.1f} MB)")
    print(f"Container reports: {container_frame_count} frames, {container_fps:.2f} fps, {width}x{height}")
    print()

    decoded = 0
    frozen_run = 0
    max_frozen_run_seen = 0
    frozen_warnings = 0
    prev_checked_frame = None
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        decoded += 1
        if idx % args.stride == 0:
            if (
                prev_checked_frame is not None
                and frame.shape == prev_checked_frame.shape
                and np.array_equal(frame, prev_checked_frame)
            ):
                frozen_run += 1
                max_frozen_run_seen = max(max_frozen_run_seen, frozen_run)
                if frozen_run == args.max_frozen_run:
                    frozen_warnings += 1
            else:
                frozen_run = 0
            prev_checked_frame = frame
        idx += 1
        if decoded % 5000 == 0:
            print(f"  ...{decoded} frames decoded so far")
    cap.release()

    print()
    match_str = "MATCH" if decoded == container_frame_count else "MISMATCH -- possible truncation/corruption"
    print(f"Decoded successfully: {decoded} / {container_frame_count} container-reported frames ({match_str})")

    frozen_str = ""
    if frozen_warnings:
        frozen_str = (
            f"  -- {frozen_warnings} run(s) reached the {args.max_frozen_run}-frame "
            "threshold: possible frozen/stalled capture"
        )
    print(f"Longest run of pixel-identical consecutive checked frames: {max_frozen_run_seen}{frozen_str}")

    seg = find_segment_row(args.segments, args.avi_path)
    if seg is None:
        print()
        print("(no --segments given, or no matching row found -- pass --segments <path> "
              "for a stronger cross-check against the recording's own frame count/timing)")
        return

    expected_frames = int(seg["frame_count"])
    opened_at = float(seg["opened_at"])
    closed_at = float(seg["closed_at"])
    duration_s = closed_at - opened_at
    real_fps = expected_frames / duration_s if duration_s > 0 else float("nan")

    print()
    print(f"segments.csv says: {expected_frames} frames over {duration_s:.2f}s = {real_fps:.2f} fps actually achieved")
    count_match = "OK" if decoded == expected_frames else (
        f"MISMATCH: file has {decoded}, segments.csv says {expected_frames}"
    )
    print(f"Frame count vs segments.csv: {count_match}")
    if abs(real_fps - container_fps) > 1.0:
        print(
            f"NOTE: container's stored fps ({container_fps:.2f}) differs from the actually-achieved "
            f"rate ({real_fps:.2f}) -- this is expected/normal: the container fps is a playback-speed "
            "setting (recording_fps), not a live measurement, so the two aren't required to match."
        )


if __name__ == "__main__":
    main()
