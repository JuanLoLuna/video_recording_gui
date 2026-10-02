#!/usr/bin/env python3
"""Headless two-camera recording through the REAL controllers (plan step 9).

Runs the same backend the GUI will use -- one CameraController per camera,
bound by serial, a shared Spinnaker System, a CameraGroup doing the two-phase
start / ordered stop, real SessionPaths with per-camera names -- with no GUI,
so a backend problem cannot hide behind GUI work. Then it reconciles every
camera's sidecars (backend/session_verify.py).

Run ON THE RIG (recorder GUI and SpinView closed):

    python scripts/multi_controller_smoke.py --seconds 120 --segment-seconds 20 --output-dir D:\\smoke

Useful variants:
    --codec mjpg                     MJPEG instead of uncompressed GREY
    --serials 23227865 26134271      use only these cameras, primary first (optional; default: every connected camera)
    --seconds 600                    the 10-minute stress run
    --fault-serial 26134271          you WILL unplug/replug that camera during the run:
                                     it must show a timeline break + reinit, the other camera none

For --segment-seconds < 900 many short segments are written, exercising rotation.
Uncompressed segments also roll at the 3 GB byte ceiling regardless. Output is
deleted afterwards only if you pass --cleanup. Exit code 0 = every check passed.
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


EXPECTED_CODEC = {"grey": "raw/uncompressed", "mjpg": "MJPG"}


def queue_limit(codec: str, fps: float) -> int:
    """Append-queue peak (frames) at which a steady run is called too slow.

    The plan's criterion (< 5 uncompressed, < 7 MJPEG) was set at 30 fps, i.e. a
    backlog of about 0.17 s / 0.23 s. A backlog is a duration, so the limit in
    frames grows with the frame rate: at 60 fps the same 0.17 s is ~10 frames.
    (At 60 fps a pre-armed writer open that has to be retried pinned to FFMPEG
    holds the append thread for ~0.1 s, i.e. ~6-7 queued frames, with nothing lost.)
    """
    base = 7 if codec == "mjpg" else 5
    return round(base * max(1.0, fps / 30.0))


def check_videos(args, p, report) -> tuple[list[str], list[str]]:
    """Decode a camera's first and last segment: right codec, every frame, not black.

    Frame count and timing can look perfect while the pictures are black (that
    happened with OpenCV's built-in MJPEG encoder on grayscale), so the pixels
    themselves are checked. Returns (problems, info lines).
    """
    import csv

    import verify_avi  # scripts/ is on sys.path when this file is run as a script

    with open(p.segments_csv, newline="", encoding="utf-8") as handle:
        manifest = {row["segment_file"]: int(row["frame_count"]) for row in csv.DictReader(handle)}
    problems, lines = [], []
    # A run that stops exactly on a rotation boundary leaves a final segment with
    # no frames (e.g. 5400 frames at 600 per segment = 9 full segments + an empty
    # 10th). That is legitimate, so check the first and last NON-EMPTY segment.
    non_empty = [i for i in range(report.segment_count) if manifest.get(p.video_final(i).name, 0) > 0]
    empty = report.segment_count - len(non_empty)
    if empty:
        lines.append(f"{empty} empty segment(s) with 0 frames (stopped on a rotation boundary) -- not decoded")
    if not non_empty:
        problems.append("no segment contains any frames")
        return problems, lines
    for index in sorted({non_empty[0], non_empty[-1]}):
        path = p.video_final(index)
        info = verify_avi.scan_video(str(path), stride=args.video_stride)
        if not info["opened"]:
            problems.append(f"{path.name} cannot be opened/decoded")
            continue
        if not info["checked"]:
            problems.append(f"{path.name} decoded no frames (segments.csv says {manifest.get(path.name)})")
            continue
        per_frame = info["file_size"] / max(1, info["decoded"]) / 1024
        lines.append(
            f"video {path.name}: {info['fourcc']} {info['width']}x{info['height']}, {info['decoded']} frames, "
            f"{per_frame:.0f} KB/frame, brightness {info['mean_min']:.0f}..{info['mean_max']:.0f}"
        )
        if info["fourcc"] != EXPECTED_CODEC[args.codec]:
            problems.append(f"{path.name} is {info['fourcc']}, expected {EXPECTED_CODEC[args.codec]} for --codec {args.codec}")
        if info["decoded"] != info["container_frame_count"]:
            problems.append(f"{path.name} decoded {info['decoded']} of {info['container_frame_count']} frames")
        if path.name in manifest and info["decoded"] != manifest[path.name]:
            problems.append(f"{path.name} has {info['decoded']} frames, segments.csv says {manifest[path.name]}")
        if info["all_black"]:
            problems.append(f"{path.name} decodes to BLACK frames")
    # Every OTHER segment gets the cheap check (first/middle/last frame): a single
    # bad segment in the middle of a run (e.g. one written by a fallback encoder)
    # would otherwise go unnoticed because only the first and last are fully decoded.
    scanned = {non_empty[0], non_empty[-1]}
    others = [i for i in non_empty if i not in scanned]
    bad = 0
    for index in others:
        path = p.video_final(index)
        quick = verify_avi.quick_check(str(path))
        expected_frames = manifest.get(path.name)
        if not quick["opened"] or quick.get("unreadable"):
            problems.append(f"{path.name} cannot be opened/decoded")
            bad += 1
            continue
        if quick["fourcc"] != EXPECTED_CODEC[args.codec]:
            problems.append(f"{path.name} is {quick['fourcc']}, expected {EXPECTED_CODEC[args.codec]}")
            bad += 1
        if expected_frames is not None and quick["frames"] != expected_frames:
            problems.append(f"{path.name} has {quick['frames']} frames, segments.csv says {expected_frames}")
            bad += 1
        if quick["all_black"]:
            problems.append(f"{path.name} decodes to BLACK frames")
            bad += 1
    if others:
        lines.append(f"quick-checked {len(others)} other segment(s): {len(others) - bad} OK, {bad} with problems")
    return problems, lines


class StopHeartbeat:
    """While the cameras are being stopped, say what each one is waiting for.

    Stopping closes the last segment of every camera and can take a while (a
    grey 60 fps run leaves ~30 GB to close); a rig run once printed nothing
    after the last progress line and there was no way to tell where it was.
    This prints, every few seconds, whether each camera is still recording and
    how many frames/segments are still queued, so a hang names its own cause.
    """

    def __init__(self, slots, interval_s: float = 5.0, out=None) -> None:
        self.slots = slots
        self.interval_s = interval_s
        self.out = out or (lambda message: print(message, flush=True))
        self._done = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._started_at = time.monotonic()

    def start(self) -> "StopHeartbeat":
        self._started_at = time.monotonic()
        self._thread.start()
        return self

    def stop(self) -> None:
        self._done.set()
        self._thread.join(timeout=2.0)

    def snapshot(self) -> str:
        parts = []
        for slot in self.slots:
            c = slot.controller
            parts.append(
                f"#{slot.serial}: recording={c.recording_active} acquiring={c.acquiring} "
                f"appendQ={c._append_queue.qsize()} closerQ={c._closer_queue.unfinished_tasks}"
            )
        return " | ".join(parts)

    def _run(self) -> None:
        while not self._done.wait(self.interval_s):
            self.out(f"  ...still stopping after {time.monotonic() - self._started_at:.0f}s: {self.snapshot()}")


class _Tee:
    """Write to the console and to a log file, so a run can be shared as text."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
        return len(text)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seconds", type=float, default=120.0, help="recording duration")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--codec", choices=["grey", "mjpg"], default="grey")
    ap.add_argument("--segment-seconds", type=float, default=20.0, help="rotation interval (production: 900)")
    ap.add_argument("--output-dir", default="smoke_output")
    ap.add_argument("--serials", nargs="+", help="camera serials, primary first")
    ap.add_argument("--fault-serial", help="serial you will unplug/replug mid-run")
    ap.add_argument("--fault-queue-tolerance", type=int, default=300,
                    help="in a --fault-serial run, the append-queue peak (frames) tolerated on the cameras that were "
                         "NOT unplugged, provided it drains back to < 5 (default 300 = 10 s at 30 fps)")
    ap.add_argument("--video-stride", type=int, default=5,
                    help="decode-check brightness/frozen frames every Nth frame of the first and last segment "
                         "of each camera (default 5; every frame is still decoded)")
    ap.add_argument("--cleanup", action="store_true", help="delete the recorded files afterwards")
    args = ap.parse_args()

    # Read when recording starts, so it must be set before start_recording_all().
    os.environ["SLEEVE_VIDEO_GUI_SEGMENT_SECONDS"] = str(args.segment_seconds)

    # Everything printed is also saved next to the recordings, so the whole run
    # (not just what fits on a screen) can be sent as one text file.
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(args.output_dir) / f"smoke_log_{datetime.now():%Y%m%d_%H%M%S}.txt"
    # Line-buffered: every line reaches the file as it is printed, so a run that is
    # killed, hangs or crashes still leaves everything it said (a block-buffered
    # file loses the final results exactly when they matter most).
    with open(log_path, "w", encoding="utf-8", buffering=1) as log_file:
        real_stdout, real_stderr = sys.stdout, sys.stderr
        sys.stdout = _Tee(real_stdout, log_file)
        sys.stderr = _Tee(real_stderr, log_file)
        try:
            print(f"(saving this output to {log_path})")
            try:
                return _main_with_args(args)
            except Exception:
                # Into the log as well as the console: a crash must not leave a
                # log that simply stops.
                traceback.print_exc()
                return 1
        finally:
            sys.stdout, sys.stderr = real_stdout, real_stderr


def _main_with_args(args) -> int:

    from backend.camera_control import CameraController, enumerate_cameras
    from backend.camera_group import CameraGroup, CameraSlot
    from backend.camera_registry import format_camera_summary, parse_serials_env, select_cameras

    selection = select_cameras(enumerate_cameras(), args.serials or parse_serials_env())
    print(format_camera_summary(selection))
    for warning in selection.warnings:
        print(f"  WARNING: {warning}")
    for note in selection.notes:
        print(f"  note: {note}")
    if not selection.bound:
        return 2
    if selection.missing:
        # Plan decision 5: a manual start with a configured camera missing is
        # refused. A "PASS" with fewer cameras than asked for would be a lie.
        print(f"Configured camera(s) not found: {', '.join(selection.missing)} -- refusing to run.")
        return 2
    if args.fault_serial and args.fault_serial not in [c.serial for c in selection.bound]:
        print(f"--fault-serial {args.fault_serial} is not one of the bound cameras")
        return 2

    slots = []
    for cam in selection.bound:
        controller = CameraController(serial=cam.serial, tag=cam.tag)
        controller.target_frame_rate = args.fps
        controller.set_compression_enabled(args.codec == "mjpg")
        slots.append(CameraSlot(controller, serial=cam.serial, model=cam.model, tag=cam.tag, is_primary=cam.is_primary))
    group = CameraGroup(slots)

    try:
        return run_session(args, slots, group)
    finally:
        # Ctrl+C or an exception anywhere must not leave cameras streaming.
        # stop_all() is safe to repeat (per-owner Spinnaker references).
        group.stop_all()


def run_session(args, slots, group) -> int:
    from backend.disk_guard import assess_disk, estimate_bytes_per_hour, sample_disk_usage
    from backend.recording_paths import SessionPaths, resolve_output_dir
    from backend.session_verify import verify_camera_outputs

    result = group.start_all()
    print(f"start_all: ok={result.ok} {result.message}")
    if not result.ok:
        return 1

    fps_by = {s.serial: (s.controller.get_acquisition_frame_rate() or args.fps) for s in slots}
    for s in slots:
        print(f"  #{s.serial} {s.model}: applied {fps_by[s.serial]:.2f} fps, link frame size "
              f"{s.controller.get_stream_rate()}")
    rate = estimate_bytes_per_hour([r for r in (s.controller.get_stream_rate() for s in slots) if r])
    out_dir = resolve_output_dir(args.output_dir)
    verdict = assess_disk(sample_disk_usage(out_dir, at_s=time.monotonic()), bytes_per_hour=rate)
    print(f"  estimated {rate / 1e9:.0f} GB/h; disk: {verdict.level} - {verdict.reason}")

    started_at = datetime.now()
    paths = {s.serial: SessionPaths.for_session(out_dir, started_at, camera_tag=s.tag) for s in slots}
    result = group.start_recording_all(lambda s: paths[s.serial], lambda s: fps_by[s.serial])
    print(f"start_recording_all: ok={result.ok} {result.message}")
    if not result.ok:
        return 1

    cpu0, wall0 = time.process_time(), time.monotonic()
    try:
        import psutil
        proc = psutil.Process()
    except ImportError:
        proc = None
    deadline = time.monotonic() + args.seconds
    next_report = time.monotonic() + 10.0
    queue_max = {s.serial: 0 for s in slots}  # sampled every 0.25 s, not only at report time
    queue_max_at = {s.serial: 0.0 for s in slots}
    started_mono = time.monotonic()
    # Longest single write() and GetNextImage() per camera, from the controller's
    # own loop timing. A ~5000 ms write() = a disk/USB stall; a small write() but
    # a delayed queue = the writer thread was starved (e.g. the GIL).
    append_ms_max = {s.serial: 0.0 for s in slots}
    grab_ms_max = {s.serial: 0.0 for s in slots}
    if args.fault_serial:
        print(f"\n>>> During the run, UNPLUG camera #{args.fault_serial} for ~20-30 s, then replug it. <<<")
    try:
        while time.monotonic() < deadline:
            time.sleep(0.25)
            for s in slots:
                depth = s.controller._append_queue.qsize()
                if depth > queue_max[s.serial]:
                    queue_max[s.serial] = depth
                    queue_max_at[s.serial] = time.monotonic() - started_mono
                timing = s.controller.get_and_reset_loop_timing_samples()
                append_ms_max[s.serial] = max([append_ms_max[s.serial], *timing.get("append_ms", [])])
                grab_ms_max[s.serial] = max([grab_ms_max[s.serial], *timing.get("grab_ms", [])])
            if time.monotonic() >= next_report:
                next_report += 10.0
                line = f"  t+{args.seconds - (deadline - time.monotonic()):4.0f}s "
                for s in slots:
                    st = s.controller.get_acquisition_stats()
                    line += (f"| #{s.serial}: rec={s.controller.frame_counter} gaps={st['camera_frame_gaps']} "
                             f"err={st['acquisition_errors']} reinit={st['camera_reinits']} "
                             f"q={s.controller._append_queue.qsize()}(max {queue_max[s.serial]}) ")
                if proc is not None:
                    line += f"| rss={proc.memory_info().rss / 1e6:.0f}MB"
                print(line, flush=True)
    except KeyboardInterrupt:
        print("interrupted -- stopping and verifying what was recorded")
    cpu_cores = (time.process_time() - cpu0) / max(1e-9, time.monotonic() - wall0)

    # stop_all() always joins the writer queue, so the depth must be read first:
    # a queue that never drained would otherwise look like 0 here.
    depth_at_end = {s.serial: s.controller._append_queue.qsize() for s in slots}
    print("capture finished; stopping the cameras (this closes the last segments)...", flush=True)
    t_stop = time.monotonic()
    heartbeat = StopHeartbeat(slots).start()
    try:
        stopped = group.stop_all()
    finally:
        heartbeat.stop()
    print(f"stop_all: ok={stopped.ok} in {time.monotonic() - t_stop:.1f}s {stopped.message}", flush=True)
    final_stats = {s.serial: s.controller.get_acquisition_stats() for s in slots}

    print("\n" + "=" * 90)
    all_ok = stopped.ok
    for s in slots:
        p = paths[s.serial]
        print(f"verifying #{s.serial} (decoding its video)...", flush=True)
        report = verify_camera_outputs(
            p.metadata_csv, p.segments_csv, p.events_jsonl,
            expect_stem=p.stem, expect_serial=s.serial, expected_fps=fps_by[s.serial],
        )
        stats = final_stats[s.serial]
        problems = list(report.problems)
        video_problems, video_lines = check_videos(args, p, report)
        problems += video_problems
        leftovers = (
            [f for f in p.incomplete_dir.iterdir() if f.name.startswith(p.stem + "_part") or f.name.startswith("UNEXPECTED")]
            if p.incomplete_dir.exists() else []
        )
        if leftovers:
            problems.append(f".incomplete/ still holds {len(leftovers)} file(s)")
        for index in range(report.segment_count):
            if not p.video_final(index).exists():
                problems.append(f"missing segment file {p.video_final(index).name}")
        faulted = s.serial == args.fault_serial
        # An unplug makes every GetNextImage raise, so the faulted camera's
        # grab errors are expected; the others must be clean.
        counters = ("append_failures",) if faulted else (
            "append_failures", "incomplete_images", "acquisition_errors")
        for name in counters:
            if stats[name]:
                problems.append(f"{name}={stats[name]}")
        final_depth = depth_at_end[s.serial]
        for index, reason in s.controller.segment_pixel_problems:
            problems.append(f"the app itself flagged segment {index}: {reason}")
        if s.controller.closer_failures:
            problems.append(
                f"{s.controller.closer_failures} segment(s) failed to finalize "
                "(their files are left in .incomplete/)"
            )
        notes = []
        if faulted or not args.fault_serial:
            limit = queue_limit(args.codec, fps_by[s.serial]) if not args.fault_serial else None
        else:
            limit = args.fault_queue_tolerance
        if limit is not None and queue_max[s.serial] >= limit:
            problems.append(
                f"append queue peaked at {queue_max[s.serial]} "
                f"({f'plan criterion: < {limit}' if not args.fault_serial else f'tolerance {limit} during a fault run'})"
            )
        elif args.fault_serial and not faulted and queue_max[s.serial] >= 5:
            notes.append(
                f"append queue peaked at {queue_max[s.serial]} at t+{queue_max_at[s.serial]:.0f}s during the fault run "
                f"(tolerated; drained to {final_depth})"
            )
        if final_depth >= 5:
            problems.append(f"append queue still held {final_depth} frames when capture ended (writer never caught up)")
        if not faulted and report.max_capture_gap_s > 0.5:
            problems.append(
                f"capture stalled {report.max_capture_gap_s:.2f}s before frame {report.max_capture_gap_row} "
                "(an untouched camera should never pause)"
            )
        if faulted:
            if report.timeline_breaks < 1 or stats["camera_reinits"] < 1:
                problems.append("expected a timeline break and a reinit after the unplug; saw none")
        else:
            if stats["camera_reinits"] or report.timeline_breaks:
                problems.append("this camera was NOT unplugged but shows a reinit / timeline break")
            if stats["camera_frame_gaps"]:
                problems.append(f"camera_frame_gaps={stats['camera_frame_gaps']}")
        all_ok &= not problems
        print(f"#{s.serial} {s.model}  [{p.stem}]  {'PASS' if not problems else 'FAIL'}")
        print(f"    rows={report.metadata_rows} segments={report.segment_count} (sum {report.segment_frame_sum}) "
              f"rolls={report.roll_reasons} breaks={report.timeline_breaks} reinits={stats['camera_reinits']}")
        fps_txt = f"{report.effective_fps:.3f}" if report.effective_fps else "?"
        delta = f"{report.median_timestamp_delta_ms:.2f}" if report.median_timestamp_delta_ms else "?"
        print(f"    effective fps {fps_txt} (applied {fps_by[s.serial]:.2f}); median camera timestamp delta {delta} ms; "
              f"frame-id gaps {report.camera_frame_id_gaps}")
        print(f"    longest write() {append_ms_max[s.serial]:.0f} ms, longest grab {grab_ms_max[s.serial]:.0f} ms, "
              f"longest gap between captured frames {report.max_capture_gap_s * 1000:.0f} ms "
              f"(before frame {report.max_capture_gap_row}), queue peak {queue_max[s.serial]} at t+{queue_max_at[s.serial]:.0f}s")
        if s.controller.writer_open_retries:
            notes.append(
                f"{s.controller.writer_open_retries} MJPEG writer open(s) did not get the FFMPEG backend "
                "first time and were retried pinned to FFMPEG (recovered)"
            )
        for line in video_lines:
            print(f"    {line}")
        for note in notes:
            print(f"    note: {note}")
        for problem in problems:
            print(f"    !! {problem}")
        print(f"    verify video: python scripts/verify_avi.py \"{p.video_final(0)}\" --segments \"{p.segments_csv}\"")
    print(f"\nprocess CPU {cpu_cores:.2f} cores over the run")
    print("RESULT:", "PASS" if all_ok else "FAIL")

    if args.cleanup:
        for s in slots:
            p = paths[s.serial]
            for f in list(out_dir.glob(f"{p.stem}*")):
                f.unlink(missing_ok=True)
        try:
            (out_dir / ".incomplete").rmdir()
        except OSError:
            pass
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
