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
  - Which codec the file really uses (MJPG = compressed, GREY = uncompressed) and
    whether the decoded frames are black. A writer path that "works" but decodes
    to all-black frames (seen with OpenCV's built-in MJPEG encoder on grayscale
    input) still has the right frame count and timing, so brightness is checked
    explicitly: the exit status is non-zero for an all-black or truncated video.
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


def fourcc_name(cap) -> str:
    code = int(cap.get(cv2.CAP_PROP_FOURCC))
    if code == 0:
        # OpenCV reports 0 for raw (BI_RGB-style) video, which is how the
        # uncompressed GREY recordings read back.
        return "raw/uncompressed"
    name = "".join(chr((code >> (8 * i)) & 0xFF) for i in range(4))
    return name if name.isprintable() and name.strip() else f"0x{code:08x}"


def scan_video(avi_path: str, *, stride: int = 1, max_frozen_run: int = 5, black_mean: float = 5.0) -> dict:
    """Decode a video and summarise it. `opened` is False if it cannot be read at all.

    A checked frame (every `stride`-th) counts as near-black when its mean pixel
    value is below `black_mean` (0-255).
    """
    result = {"opened": False, "path": avi_path}
    if not os.path.isfile(avi_path):
        return result
    cap = cv2.VideoCapture(avi_path)
    if not cap.isOpened():
        return result
    result.update(
        opened=True,
        file_size=os.path.getsize(avi_path),
        container_fps=cap.get(cv2.CAP_PROP_FPS),
        container_frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        fourcc=fourcc_name(cap),
    )
    decoded = checked = black_frames = frozen_run = max_frozen_run_seen = frozen_warnings = 0
    mean_min, mean_max = float("inf"), float("-inf")
    prev_checked_frame = None
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        decoded += 1
        if idx % stride == 0:
            checked += 1
            mean = float(frame.mean())
            mean_min, mean_max = min(mean_min, mean), max(mean_max, mean)
            if mean < black_mean:
                black_frames += 1
            if (
                prev_checked_frame is not None
                and frame.shape == prev_checked_frame.shape
                and np.array_equal(frame, prev_checked_frame)
            ):
                frozen_run += 1
                max_frozen_run_seen = max(max_frozen_run_seen, frozen_run)
                if frozen_run == max_frozen_run:
                    frozen_warnings += 1
            else:
                frozen_run = 0
            prev_checked_frame = frame
        idx += 1
        if decoded % 5000 == 0:
            print(f"  ...{decoded} frames decoded so far")
    cap.release()
    result.update(
        decoded=decoded,
        checked=checked,
        black_frames=black_frames,
        mean_min=None if not checked else mean_min,
        mean_max=None if not checked else mean_max,
        max_frozen_run_seen=max_frozen_run_seen,
        frozen_warnings=frozen_warnings,
        all_black=bool(checked) and black_frames == checked,
    )
    return result


def quick_check(avi_path: str, *, black_mean: float = 5.0) -> dict:
    """Cheap per-segment check: opens, frame count, codec, brightness of 3 sampled frames.

    Reads the first, middle and last frame instead of decoding everything, so it
    can cover every segment of a long run in seconds.
    """
    result = {"opened": False, "path": avi_path}
    if not os.path.isfile(avi_path):
        return result
    cap = cv2.VideoCapture(avi_path)
    if not cap.isOpened():
        return result
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    means = []
    for position in sorted({0, frames // 2, max(0, frames - 1)}) if frames else []:
        cap.set(cv2.CAP_PROP_POS_FRAMES, position)
        ok, frame = cap.read()
        if ok:
            means.append(float(frame.mean()))
    result.update(
        opened=True,
        frames=frames,
        fourcc=fourcc_name(cap),
        means=means,
        all_black=bool(means) and all(m < black_mean for m in means),
        unreadable=bool(frames) and not means,
    )
    cap.release()
    return result


def scan_folder(folder: str) -> int:
    """Cheap check of EVERY .avi in a folder (first/middle/last frame of each).

    Use it to audit a whole recording folder, e.g. one made while the MJPEG
    writer could silently fall back to a backend that decodes black.
    Returns 1 if any file cannot be opened/decoded or is black.
    """
    names = sorted(n for n in os.listdir(folder) if n.lower().endswith(".avi"))
    if not names:
        print(f"no .avi files in {folder}")
        return 1
    bad = 0
    print(f"{'file':<62} {'frames':>7} {'codec':<17} {'brightness':<14} status")
    for name in names:
        info = quick_check(os.path.join(folder, name))
        if not info["opened"] or info.get("unreadable"):
            status, bad = "CANNOT DECODE", bad + 1
            print(f"{name:<62} {'-':>7} {'-':<17} {'-':<14} {status}")
            continue
        status = "BLACK" if info["all_black"] else "ok"
        bad += status != "ok"
        bright = "/".join(f"{m:.0f}" for m in info["means"]) or "-"
        print(f"{name:<62} {info['frames']:>7} {info['fourcc']:<17} {bright:<14} {status}")
    print(f"\n{len(names)} file(s) checked, {bad} with problems")
    return 1 if bad else 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("avi_path")
    ap.add_argument("--segments", help="path to the recording's _segments.csv, for cross-checking")
    ap.add_argument(
        "--stride", type=int, default=1,
        help="apply the frozen-frame/brightness checks every Nth frame instead of every frame (default: every frame)",
    )
    ap.add_argument(
        "--max-frozen-run", type=int, default=5,
        help="warn if this many consecutive checked frames are pixel-identical (default: 5)",
    )
    args = ap.parse_args()

    if os.path.isdir(args.avi_path):
        sys.exit(scan_folder(args.avi_path))

    if not os.path.isfile(args.avi_path):
        print(f"FAILED: {args.avi_path} does not exist")
        sys.exit(1)

    info = scan_video(args.avi_path, stride=args.stride, max_frozen_run=args.max_frozen_run)
    if not info["opened"]:
        print(f"FAILED to open {args.avi_path} -- likely corrupt or an unsupported codec on this machine")
        sys.exit(1)

    container_fps = info["container_fps"]
    container_frame_count = info["container_frame_count"]
    decoded = info["decoded"]
    print(f"File: {args.avi_path} ({info['file_size'] / 1e6:.1f} MB)")
    print(f"Container reports: {container_frame_count} frames, {container_fps:.2f} fps, "
          f"{info['width']}x{info['height']}, codec {info['fourcc']}")
    print()

    exit_code = 0
    match_str = "MATCH" if decoded == container_frame_count else "MISMATCH -- possible truncation/corruption"
    if decoded != container_frame_count:
        exit_code = 1
    print(f"Decoded successfully: {decoded} / {container_frame_count} container-reported frames ({match_str})")

    if info["checked"]:
        print(
            f"Brightness (mean pixel 0-255) over {info['checked']} checked frame(s): "
            f"min {info['mean_min']:.1f}, max {info['mean_max']:.1f}; "
            f"{info['black_frames']} near-black"
        )
        if info["all_black"]:
            print("FAILED: every checked frame is (nearly) black -- the video decodes to a black image "
                  "even though it has the right number of frames. Do not use this recording.")
            exit_code = 1
        elif info["black_frames"]:
            print(f"WARNING: {info['black_frames']} checked frame(s) are near-black (lens cap, lights off, "
                  "or a capture problem).")

    frozen_str = ""
    if info["frozen_warnings"]:
        frozen_str = (
            f"  -- {info['frozen_warnings']} run(s) reached the {args.max_frozen_run}-frame "
            "threshold: possible frozen/stalled capture"
        )
    print(f"Longest run of pixel-identical consecutive checked frames: {info['max_frozen_run_seen']}{frozen_str}")

    seg = find_segment_row(args.segments, args.avi_path)
    if seg is None:
        print()
        print("(no --segments given, or no matching row found -- pass --segments <path> "
              "for a stronger cross-check against the recording's own frame count/timing)")
        sys.exit(exit_code)

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
    if decoded != expected_frames:
        exit_code = 1
    if abs(real_fps - container_fps) > 1.0:
        print(
            f"NOTE: container's stored fps ({container_fps:.2f}) differs from the actually-achieved "
            f"rate ({real_fps:.2f}) -- this is expected/normal: the container fps is a playback-speed "
            "setting (recording_fps), not a live measurement, so the two aren't required to match."
        )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
