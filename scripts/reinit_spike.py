#!/usr/bin/env python3
"""Rig spike: can ONE camera recover while the OTHER keeps streaming?

Answers the plan's biggest unknown (risks R1/R2 in
docs/plans/2026-10-01-multi-camera.md) BEFORE the controller is changed:

  R1  After camera B is unplugged and replugged, can it be re-discovered with
      system.GetCameras() while camera A is still streaming? If not, does
      system.UpdateCameras() fix it?
  R2  Are two simultaneous CameraLists, and a fresh GetCameras() during A's
      acquisition, safe? (A must not lose a single frame throughout.)

Run ON THE RIG with the recorder GUI and SpinView CLOSED:

    python scripts/reinit_spike.py --seconds 25

Both cameras start streaming (A and B are the two serials given with --serials,
else the two lowest serials found; A keeps streaming, B is the one to unplug). When the script says UNPLUG, pull camera B's USB cable. When it
says REPLUG, plug it back in (a DIFFERENT port is a stronger test). The script
tears down only B, then keeps trying to re-find it by serial and restart it,
while A is never touched. A's frame ids are checked for gaps the whole time.

Nothing is written to disk. Read the SUMMARY block at the end; paste it back.
"""
from __future__ import annotations

import argparse
import sys
import threading
import time


class Stream:
    """Grab loop for one camera; counts frames, FrameID gaps and errors."""

    def __init__(self, name: str, cam):
        self.name = name
        self.cam = cam
        self.frames = 0
        self.gaps = 0
        self.errors = 0
        self.frame_ids_seen = 0
        self.max_interval_s = 0.0
        self.last_frame_at = time.perf_counter()
        self._last_id = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=5.0)

    def seconds_since_frame(self) -> float:
        return time.perf_counter() - self.last_frame_at

    def _run(self):
        while not self._stop.is_set():
            try:
                image = self.cam.GetNextImage(500)
            except Exception:
                self.errors += 1
                self._stop.wait(0.05)
                continue
            if image.IsIncomplete():
                image.Release()
                continue
            try:
                frame_id = int(image.GetChunkData().GetFrameID())
            except Exception:
                frame_id = None
            image.Release()
            self.frames += 1
            now = time.perf_counter()
            # A stall shows up as one very long interval even though no frame
            # id is missing, which gap counting alone cannot see.
            self.max_interval_s = max(self.max_interval_s, now - self.last_frame_at)
            self.last_frame_at = now
            if frame_id is not None:
                self.frame_ids_seen += 1
            if frame_id is not None and self._last_id is not None and frame_id > self._last_id + 1:
                self.gaps += frame_id - self._last_id - 1
            if frame_id is not None:
                self._last_id = frame_id


def serial_of(PySpin, cam) -> str:
    node = PySpin.CStringPtr(cam.GetTLDeviceNodeMap().GetNode("DeviceSerialNumber"))
    return node.GetValue() if PySpin.IsReadable(node) else "?"


def open_camera(PySpin, system, serial, fps):
    """New CameraList + GetBySerial + Init + minimal config + BeginAcquisition.

    Returns (cam_list, cam). The list is kept alive by the caller, like a
    CameraController keeps its own self.cam_list.
    """
    cam_list = system.GetCameras()
    cam = None
    try:
        cam = cam_list.GetBySerial(serial)
        cam.Init()
        return cam_list, _configure_and_begin(PySpin, cam, fps)
    except Exception:
        teardown(cam_list, [cam] if cam is not None else [])
        raise


def _configure_and_begin(PySpin, cam, fps):
    nodemap = cam.GetNodeMap()
    chunk = PySpin.CBooleanPtr(nodemap.GetNode("ChunkModeActive"))
    if PySpin.IsWritable(chunk):
        chunk.SetValue(True)
        selector = PySpin.CEnumerationPtr(nodemap.GetNode("ChunkSelector"))
        enable = PySpin.CBooleanPtr(nodemap.GetNode("ChunkEnable"))
        try:
            selector.SetIntValue(selector.GetEntryByName("FrameID").GetValue())
            enable.SetValue(True)
        except Exception:
            pass
    mode = PySpin.CEnumerationPtr(nodemap.GetNode("AcquisitionMode"))
    mode.SetIntValue(mode.GetEntryByName("Continuous").GetValue())
    enable_rate = PySpin.CBooleanPtr(nodemap.GetNode("AcquisitionFrameRateEnable"))
    if PySpin.IsWritable(enable_rate):
        enable_rate.SetValue(True)
    rate = PySpin.CFloatPtr(nodemap.GetNode("AcquisitionFrameRate"))
    if PySpin.IsWritable(rate):
        rate.SetValue(min(float(rate.GetMax()), max(float(rate.GetMin()), fps)))
    cam.BeginAcquisition()
    return cam


def teardown(cam_list, cam_ref):
    """EndAcquisition/DeInit the camera, DROP its reference, then Clear the list.

    cam_ref is a one-element list so the caller's reference can be removed
    here -- this mirrors CameraController._reinitialize_camera, which sets
    self.cam = None before cam_list.Clear(). Clearing a list while Python still
    holds the camera can make Spinnaker keep a stale device object alive,
    which is exactly the condition this spike is testing.
    """
    if cam_ref:
        cam = cam_ref.pop()
        for step in (lambda: cam.EndAcquisition(), lambda: cam.DeInit()):
            try:
                step()
            except Exception:
                pass
        del cam
    try:
        cam_list.Clear()
    except Exception:
        pass


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--serials", nargs=2, metavar=("A", "B"), help="A keeps streaming; B is the one you unplug")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--seconds", type=float, default=25.0, help="healthy streaming before asking you to unplug B")
    ap.add_argument("--wait", type=float, default=180.0, help="max seconds to wait for each unplug / replug")
    args = ap.parse_args()

    import PySpin

    system = PySpin.System.GetInstance()
    summary: dict[str, object] = {}
    stream_a = stream_b = None
    lists = {}
    cams = {}
    try:
        probe = system.GetCameras()
        found = [serial_of(PySpin, probe[i]) for i in range(probe.GetSize())]
        probe.Clear()
        if len(found) < 2:
            print(f"Need both cameras attached; found {found}")
            return 2
        serial_a, serial_b = args.serials or sorted(found)[:2]
        if serial_a not in found or serial_b not in found:
            print(f"Need both cameras attached; found {found}")
            return 2
        print(f"A (keeps streaming): {serial_a}    B (you unplug): {serial_b}\n")

        lists["a"], cam_a = open_camera(PySpin, system, serial_a, args.fps)
        lists["b"], cam_b = open_camera(PySpin, system, serial_b, args.fps)
        stream_a, stream_b = Stream("A", cam_a), Stream("B", cam_b)
        cams = {"a": cam_a, "b": cam_b}
        stream_a.start()
        stream_b.start()
        t_streaming = time.perf_counter()
        print(f"Both streaming for {args.seconds:.0f}s ...")
        time.sleep(args.seconds)
        print(f"  A: {stream_a.frames} frames, {stream_a.gaps} gaps | B: {stream_b.frames} frames, {stream_b.gaps} gaps")
        summary["baseline_A_gaps"] = stream_a.gaps
        summary["baseline_B_gaps"] = stream_b.gaps
        baseline_a_rate = stream_a.frames / max(1e-9, time.perf_counter() - t_streaming)

        print(f"\n>>> UNPLUG camera B ({serial_b}) now. <<<")
        deadline = time.perf_counter() + args.wait
        while stream_b.seconds_since_frame() < 3.0 and time.perf_counter() < deadline:
            time.sleep(0.2)
        if stream_b.seconds_since_frame() < 3.0:
            print("B never stalled -- was it unplugged? Aborting.")
            summary["result"] = "B never stalled"
            return 1
        print(f"B stalled (no frames for 3s). A kept going: {stream_a.frames} frames, {stream_a.gaps} gaps, {stream_a.errors} errors.")

        # Tear down ONLY B, exactly as a camera-only reinit would.
        stream_b.stop()
        ref_b = [cams.pop("b")]
        cam_b = None
        stream_b.cam = None
        teardown(lists.pop("b"), ref_b)
        a_gaps_after_teardown = stream_a.gaps
        print("B torn down (EndAcquisition/DeInit/CameraList.Clear). A untouched.")
        summary["A_gaps_after_B_teardown"] = a_gaps_after_teardown - summary["baseline_A_gaps"]

        print("\n>>> REPLUG camera B now (a DIFFERENT port is a stronger test). <<<")
        print("Trying to re-find B every 2s: GetCameras() first, then UpdateCameras()+GetCameras().")
        t_start = time.perf_counter()
        rediscovered = None
        methods_tried = set()
        while time.perf_counter() - t_start < args.wait:
            for method in ("GetCameras", "UpdateCameras"):
                try:
                    if method == "UpdateCameras":
                        system.UpdateCameras()
                    cam_list = system.GetCameras()
                    serials_now = [serial_of(PySpin, cam_list[i]) for i in range(cam_list.GetSize())]
                    cam_list.Clear()
                    methods_tried.add(method)
                    if serial_b in serials_now:
                        rediscovered = method
                        break
                except Exception as exc:
                    print(f"  {method} raised {exc.__class__.__name__}: {exc}")
            if rediscovered:
                break
            time.sleep(2.0)

        if not rediscovered:
            print("B was NOT re-discovered while A streamed.")
            summary["result"] = "NOT rediscovered"
        else:
            t_found = time.perf_counter() - t_start
            print(f"B re-discovered via {rediscovered} after {t_found:.1f}s. Restarting it by serial ...")
            try:
                lists["b"], cam_b = open_camera(PySpin, system, serial_b, args.fps)
                cams["b"] = cam_b
                stream_b = Stream("B", cam_b)
                stream_b.start()
                time.sleep(5.0)
                print(f"  B after restart: {stream_b.frames} frames in 5s, {stream_b.gaps} gaps")
                summary["result"] = "RECOVERED" if stream_b.frames > 0 else "re-found but no frames"
                summary["method"] = rediscovered
                summary["seconds_to_rediscover_after_replug_prompt"] = round(t_found, 1)
                summary["B_frames_after_restart_5s"] = stream_b.frames
            except Exception as exc:
                print(f"  restart failed: {exc.__class__.__name__}: {exc}")
                summary["result"] = f"re-found but restart failed: {exc}"

        elapsed = time.perf_counter() - t_streaming
        a_rate = stream_a.frames / max(1e-9, elapsed)
        a_limit_s = max(0.25, 8.0 / args.fps)
        a_checks = {
            "no_frame_id_gaps": stream_a.gaps == 0,
            "no_grab_errors": stream_a.errors == 0,
            "frame_ids_were_readable": stream_a.frame_ids_seen > 0,
            "rate_held_within_3pct_of_baseline": a_rate >= 0.97 * baseline_a_rate,
            f"longest_frame_interval_under_{a_limit_s:.2f}s": stream_a.max_interval_s <= a_limit_s,
        }
        summary["A_total_frames"] = stream_a.frames
        summary["A_total_gaps_over_whole_test"] = stream_a.gaps
        summary["A_total_errors_over_whole_test"] = stream_a.errors
        summary["A_rate_fps (baseline)"] = f"{a_rate:.2f} ({baseline_a_rate:.2f})"
        summary["A_longest_interval_s"] = round(stream_a.max_interval_s, 3)
        summary["A_checks"] = ", ".join(f"{k}={'ok' if v else 'FAIL'}" for k, v in a_checks.items())
        summary["methods_tried"] = sorted(methods_tried)
        return 0 if summary.get("result") == "RECOVERED" and all(a_checks.values()) else 1
    finally:
        for stream in (stream_a, stream_b):
            if stream is not None:
                stream.stop()
        for key in list(cams):
            teardown(lists.pop(key, None) or _NullList(), [cams.pop(key)])
        for cam_list in lists.values():
            try:
                cam_list.Clear()
            except Exception:
                pass
        stream_a = stream_b = None
        try:
            system.ReleaseInstance()
        except Exception as exc:
            print(f"  (ReleaseInstance: {exc.__class__.__name__}: {exc})")
        print("\n===== SUMMARY =====")
        for key, value in summary.items():
            print(f"  {key}: {value}")
        print("  PASS = result RECOVERED and every A_check ok (A never lost a frame or stalled)")


class _NullList:
    def Clear(self):
        pass


if __name__ == "__main__":
    sys.exit(main())
