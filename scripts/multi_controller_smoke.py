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
    --serials 23227865 26134271      pin cameras/order (else SLEEVE_VIDEO_GUI_CAMERA_SERIALS, else all)
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
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


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
    ap.add_argument("--cleanup", action="store_true", help="delete the recorded files afterwards")
    args = ap.parse_args()

    # Read when recording starts, so it must be set before start_recording_all().
    os.environ["SLEEVE_VIDEO_GUI_SEGMENT_SECONDS"] = str(args.segment_seconds)

    from backend.camera_control import CameraController, enumerate_cameras
    from backend.camera_group import CameraGroup, CameraSlot
    from backend.camera_registry import format_camera_summary, parse_serials_env, select_cameras

    selection = select_cameras(enumerate_cameras(), args.serials or parse_serials_env())
    print(format_camera_summary(selection))
    for warning in selection.warnings:
        print(f"  WARNING: {warning}")
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

    t_stop = time.monotonic()
    stopped = group.stop_all()
    print(f"stop_all: ok={stopped.ok} in {time.monotonic() - t_stop:.1f}s {stopped.message}")
    final_stats = {s.serial: s.controller.get_acquisition_stats() for s in slots}

    print("\n" + "=" * 90)
    all_ok = stopped.ok
    for s in slots:
        p = paths[s.serial]
        report = verify_camera_outputs(
            p.metadata_csv, p.segments_csv, p.events_jsonl,
            expect_stem=p.stem, expect_serial=s.serial, expected_fps=fps_by[s.serial],
        )
        stats = final_stats[s.serial]
        problems = list(report.problems)
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
        final_depth = s.controller._append_queue.qsize()
        notes = []
        if faulted or not args.fault_serial:
            limit = 5 if not args.fault_serial else None
        else:
            limit = args.fault_queue_tolerance
        if limit is not None and queue_max[s.serial] >= limit:
            problems.append(
                f"append queue peaked at {queue_max[s.serial]} "
                f"({'plan criterion: < 5' if limit == 5 else f'tolerance {limit} during a fault run'})"
            )
        elif args.fault_serial and not faulted and queue_max[s.serial] >= 5:
            notes.append(
                f"append queue peaked at {queue_max[s.serial]} at t+{queue_max_at[s.serial]:.0f}s during the fault run "
                f"(tolerated; drained to {final_depth})"
            )
        if final_depth >= 5:
            problems.append(f"append queue still holds {final_depth} frames after stop (writer never caught up)")
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
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
