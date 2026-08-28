# backend/camera_control.py
import os
import queue
import threading
import time
from dataclasses import dataclass, replace as dataclass_replace
from pathlib import Path

import numpy as np
import PySpin

from backend.async_csv_writer import AsyncCsvWriter
from backend.frame_metadata import METADATA_FIELDS, metadata_row, resolve_sync_label
from backend.acquisition_watchdog import AcquisitionWatchdog, watchdog_config_for_frame_rate
from backend.timeline_break import (
    JsonlEventLog,
    SegmentTracker,
    estimate_frames_lost,
    session_header_record,
    session_stop_record,
    timeline_break_record,
)
from backend.recording_paths import SessionPaths
from backend.segment_policy import (
    BYTES_SAMPLE_INTERVAL_FRAMES,
    DEFAULT_SDK_MAX_FILE_SIZE_MB,
    reconcile_part_files,
    resolve_segment_seconds,
    segment_frames_for,
    should_prepare,
    should_roll,
)
from backend.segment_manifest import SegmentManifestEntry, SegmentManifestWriter, manifest_row


# Spinnaker's own buffer pool is the decoupling queue between frame arrival
# and disk writes: deepening it is what makes a rotation-boundary disk
# stall (up to ~1.6 GB of dirty page cache) survivable without dropping
# frames. At 1.31 MB/frame (1280x1024 Mono8), 150 buffers is ~197 MB of
# RAM for ~5s of stall tolerance. Confirmed reachable on the production
# camera (Phase 0 bench item 2).
STREAM_BUFFER_COUNT_TARGET = 150


@dataclass
class _CloserJob:
    """One segment handed from the acquisition thread to the closer thread."""

    writer: object  # PySpin.SpinVideo
    part_base: Path
    final_path: Path
    segment_index: int
    manifest_entry: SegmentManifestEntry


@dataclass(frozen=True)
class PreviewFrame:
    """An owned preview image plus timing captured along its pipeline."""

    image: np.ndarray
    sequence: int
    frame_id: int | None
    camera_timestamp: int | None
    retrieved_at: float
    published_at: float


def detect_first_camera():
    """
    Use Spinnaker (PySpin) to detect the first connected camera.

    Returns:
        (found: bool, message: str)

    - found = True  -> at least one camera found, message has vendor/model/serial
    - found = False -> no camera / error, message has a short explanation
    """
    try:
        system = PySpin.System.GetInstance()
    except Exception as exc:
        return False, f"Error: could not create Spinnaker system ({exc})"

    cam_list = system.GetCameras()
    num_cams = cam_list.GetSize()

    if num_cams == 0:
        cam_list.Clear()
        system.ReleaseInstance()
        return False, "No cameras detected."

    cam = cam_list[0]

    try:
        nodemap_tldevice = cam.GetTLDeviceNodeMap()

        def get_str(node_name: str) -> str:
            node = PySpin.CStringPtr(nodemap_tldevice.GetNode(node_name))
            if PySpin.IsReadable(node):
                return node.GetValue()
            return "<unavailable>"

        vendor = get_str("DeviceVendorName")
        model = get_str("DeviceModelName")
        serial = get_str("DeviceSerialNumber")

        msg = f"Camera: {vendor} {model} (S/N: {serial})"
        return True, msg

    except Exception as exc:
        return False, f"Error reading camera info: {exc}"

    finally:
        # Make sure we clean up even if something goes wrong
        cam = None
        cam_list.Clear()
        system.ReleaseInstance()

class CameraController:
    """
    Handles:
      - Connecting to first camera
      - Running an acquisition loop in a background thread
      - Providing latest frame for preview
      - Recording to AVI via SpinVideo (MJPEG)
      - Logging per-recorded-frame metadata to CSV

    All SpinVideo operations (Open, Append, Close) happen ONLY
    inside the acquisition thread to avoid crashes.
    """

    def __init__(self):
        # Spinnaker objects
        self.system = None
        self.cam_list = None
        self.cam = None
        self.acquiring = False

        # Threading
        self._acq_thread = None
        self._stop_event = threading.Event()

        # Guards self.cam handle swaps and all GenICam node access, so a
        # fault-recovery reinit (acquisition thread) can never race a GUI
        # slider callback (get/set_image_param, get/set_frame_rate) into a
        # use-after-free on the native Spinnaker object. Never held across
        # GetNextImage()/Append() -- those must stay off this lock so a
        # slow grab can't block the GUI thread.
        self._camera_lock = threading.RLock()
        # Set for the duration of a reinit; GUI-thread accessors check this
        # and return immediately rather than blocking on _camera_lock, so a
        # multi-second camera reinit never freezes the GUI.
        self._recovering = threading.Event()
        self._watchdog: AcquisitionWatchdog | None = None
        self._segment_tracker = SegmentTracker()
        self._camera_reinits = 0
        # Timeline-break sidecar for the current recording session (opened in
        # start_recording, closed on stop). None while not recording, so a
        # reinit during preview-only acquisition just doesn't log a break --
        # there is no session timeline to protect yet.
        self._event_log: JsonlEventLog | None = None

        # --- Video segment rotation (Phase 2) ---
        self._session_paths: SessionPaths | None = None
        self._segment_index = 0
        self._frames_in_segment = 0
        self._bytes_in_segment = 0
        # Wall-clock (time.time()), matching closed_at/first_system_time/
        # last_system_time in the manifest row -- NOT time.monotonic(),
        # whose reference point is arbitrary and isn't comparable to those.
        self._segment_opened_at = 0.0
        self._segment_first_record_frame_index: int | None = None
        self._segment_first_system_time: float | None = None
        # Pre-armed next writer, opened ~60 frames before the roll so the
        # (small) Open()/header cost lands off the boundary frame.
        self._pending_writer = None
        self._pending_writer_segment_index: int | None = None
        self._prepared_next_segment = False
        # _max_frames_per_segment is set below, alongside target_frame_rate
        # (which it derives from) -- see that assignment for the real value.
        # Set by _recover_camera on a successful reinit: forces the NEXT
        # append to roll into a fresh segment, so a fault's gap always
        # lands between segments rather than inside one.
        self._pending_fault_roll = False
        self._pending_fault_roll_gap_s: float | None = None
        self._mark_next_frame_segment_resume = False
        # Value written into each frame's "segment" metadata column.
        # Deliberately NOT read directly from self._segment_tracker: the
        # tracker bumps the instant a reinit succeeds (for prompt
        # events.jsonl logging), which can be one or more frames before
        # segment_file actually rolls over. Copying the tracker's value
        # into this field only at the moment of the actual file swap (see
        # _maybe_rotate_segment) keeps "segment" and "segment_file"
        # changing on the exact same row, matching frame_metadata.py's
        # documented invariant.
        self._metadata_segment = 0
        # One row per segment (~960/session); reused across record start/
        # stop cycles like _metadata_writer, started fresh in start_recording.
        self._segment_manifest_writer = SegmentManifestWriter()
        # Retired writers are Close()'d, renamed, and manifest-logged off
        # the acquisition thread -- Close() can take long enough (flushing
        # up to ~1.6 GB of dirty page cache) that doing it inline would
        # risk dropping frames at every rotation boundary. Lives for the
        # whole app (daemon thread, started lazily), not per-session.
        self._closer_queue: queue.Queue = queue.Queue()
        self._closer_thread: threading.Thread | None = None

        # Latest frame for preview
        self._latest_frame = None
        self._latest_preview_frame: PreviewFrame | None = None
        self._frame_lock = threading.Lock()

        # Acquisition health used by the GUI diagnostics. These counters are
        # intentionally separate from recording metadata so preview can be
        # diagnosed before and after a recording.
        self._acquisition_stats_lock = threading.Lock()
        self._preview_sequence = 0
        self._last_camera_frame_id = None
        self._camera_frame_gaps = 0
        self._incomplete_images = 0
        self._acquisition_errors = 0
        self._append_failures = 0

        # Recording state/flags (thread-safe)
        self.recording_active = False          # true while SpinVideo is open
        self.record_start_requested = False    # GUI asks to start
        self.record_stop_requested = False     # GUI asks to stop

        self.avi_recorder = None
        self.recording_fps = 30.0
        # Target acquisition frame rate (fps). Applied in start(); can be changed
        # live via set_frame_rate(). recording_fps follows it so AVI playback
        # speed matches the real capture rate.
        self.target_frame_rate = 30.0
        self._max_frames_per_segment = segment_frames_for(
            self.target_frame_rate, resolve_segment_seconds()
        )
        # Streams one row per recorded frame to disk incrementally, rather
        # than buffering the whole session in RAM and writing once on
        # stop (which lost 100% of metadata on any crash and grew
        # unbounded over a multi-day session). Never drops a row -- see
        # backend/async_csv_writer.py.
        self._metadata_writer = AsyncCsvWriter(
            METADATA_FIELDS,
            max_pending_rows=2048,
            flush_every_rows=300,
            flush_every_seconds=5.0,
            drop_when_full=False,
            thread_name="frame-metadata-writer",
        )
        self.frame_counter = 0

        # --- Sync marker state (for CSV logging) ---
        self._sync_lock = threading.Lock()
        self._sync_window_end = 0.0  # wall-clock time until which sync_pulse=True
        self._sync_label = None  # label for the current sync window
        # --- Label event state (for CSV logging) ---
        self._label_lock = threading.Lock()
        self._pending_label_event = None  # "label_start" / "label_end"
        self._pending_adl_id = None
        self._pending_adl_label = None

    # ------------------------------------------------------------------
    # Camera start/stop
    # ------------------------------------------------------------------
    def start(self):
        """
        Initialize Spinnaker, open first camera, set continuous mode,
        and start acquisition thread.
        """
        if self.acquiring:
            return True, "Preview already running."

        try:
            self.system = PySpin.System.GetInstance()
            self.cam_list = self.system.GetCameras()
            num_cams = self.cam_list.GetSize()

            if num_cams == 0:
                self._cleanup_system()
                return False, "No cameras detected."

            self.cam = self.cam_list[0]
            self.cam.Init()
            self._configure_camera_nodes()

            self.cam.BeginAcquisition()
            self.acquiring = True
            self._stop_event.clear()
            self._reset_acquisition_stats()
            self._watchdog = AcquisitionWatchdog(
                watchdog_config_for_frame_rate(self.target_frame_rate),
                now=time.monotonic(),
            )
            self._camera_reinits = 0

            self._acq_thread = threading.Thread(
                target=self._acquisition_loop,
                daemon=True,
            )
            self._acq_thread.start()

            return True, "Preview started."

        except Exception as exc:
            self.stop()
            return False, f"Error starting preview: {exc}"

    def stop(self):
        """
        Clean shutdown:
          - Request recording stop (if active) and wait briefly
          - Stop acquisition thread
          - DeInit camera, clear camera list, release system
        """
        # If recording is active or queued, request stop and give loop time.
        # Since Phase 2, finishing a stop means closing a segment (possibly
        # flushing a large dirty-page-cache write) AND draining the closer
        # thread's queue -- both can now take meaningfully longer than the
        # old single-file case did, so this is deliberately generous rather
        # than the previous fixed ~1s. Below, EndAcquisition/DeInit/
        # ReleaseInstance must not run while any of that is still in
        # flight, or the acquisition/closer threads can hit a use-after-free
        # on the native Spinnaker objects.
        if self.recording_active or self.record_start_requested:
            self.record_stop_requested = True
            stop_deadline = time.monotonic() + 90.0
            while self.recording_active and time.monotonic() < stop_deadline:
                time.sleep(0.05)

        # Tell acquisition loop to stop
        self._stop_event.set()

        # Break GetNextImage()
        if self.cam is not None and self.acquiring:
            try:
                self.cam.EndAcquisition()
            except Exception:
                pass

        # Wait for thread to exit
        if self._acq_thread is not None:
            try:
                self._acq_thread.join(timeout=2.0)
            except Exception:
                pass
            self._acq_thread = None

        # DeInit camera
        if self.cam is not None:
            try:
                self.cam.DeInit()
            except Exception:
                pass
            self.cam = None

        # Clear cam list
        if self.cam_list is not None:
            try:
                self.cam_list.Clear()
            except Exception:
                pass
            self.cam_list = None

        # Release system
        self._cleanup_system()

        self.acquiring = False
        self._latest_frame = None
        self._latest_preview_frame = None

    def _cleanup_system(self):
        if self.system is not None:
            try:
                self.system.ReleaseInstance()
            except Exception:
                pass
            self.system = None

    def _enable_chunk_data(self):
        """
        Enable chunk mode and request some common chunks (Timestamp, FrameID, FrameCounter).
        This is called AFTER cam.Init() and BEFORE BeginAcquisition().
        """
        nodemap = self.cam.GetNodeMap()

        # 1) Turn on chunk mode
        chunk_mode_active = PySpin.CBooleanPtr(nodemap.GetNode("ChunkModeActive"))
        if not PySpin.IsWritable(chunk_mode_active):
            print("ChunkModeActive not writable; skipping chunk setup.")
            return

        chunk_mode_active.SetValue(True)
        print("Chunk mode activated.")

        # 2) Enable specific chunks if they exist
        chunk_selector = PySpin.CEnumerationPtr(nodemap.GetNode("ChunkSelector"))
        chunk_enable = PySpin.CBooleanPtr(nodemap.GetNode("ChunkEnable"))

        if not (PySpin.IsReadable(chunk_selector) and PySpin.IsWritable(chunk_selector)):
            print("ChunkSelector not usable; skipping chunk setup.")
            return

        for name in ["Timestamp", "FrameID"]:
            try:
                entry = chunk_selector.GetEntryByName(name)
                if not PySpin.IsReadable(entry):
                    continue

                chunk_selector.SetIntValue(entry.GetValue())
                if PySpin.IsWritable(chunk_enable):
                    chunk_enable.SetValue(True)
            except Exception as exc:
                # This chunk name might simply not exist on this model
                print(f"Could not enable chunk '{name}': {exc}")
                continue

    def _configure_camera_nodes(self) -> None:
        """Chunk data + acquisition mode + frame rate.

        Called from start() and from _reinitialize_camera() so a camera
        recovered after a fault ends up configured identically to how it
        started, instead of silently reverting to whatever the driver
        defaults to. Must be called AFTER cam.Init() and BEFORE
        cam.BeginAcquisition(), same constraint as _enable_chunk_data().
        """
        self._enable_chunk_data()
        self._configure_stream_buffers()

        nodemap = self.cam.GetNodeMap()
        acq_mode = PySpin.CEnumerationPtr(nodemap.GetNode("AcquisitionMode"))
        continuous_entry = acq_mode.GetEntryByName("Continuous")
        acq_mode.SetIntValue(continuous_entry.GetValue())

        frame_rate_enable = PySpin.CBooleanPtr(nodemap.GetNode("AcquisitionFrameRateEnable"))
        if PySpin.IsWritable(frame_rate_enable):
            frame_rate_enable.SetValue(True)
        frame_rate = PySpin.CFloatPtr(nodemap.GetNode("AcquisitionFrameRate"))
        if PySpin.IsWritable(frame_rate):
            lo, hi = float(frame_rate.GetMin()), float(frame_rate.GetMax())
            target = min(hi, max(lo, float(self.target_frame_rate)))
            frame_rate.SetValue(target)
        # Read back whatever the camera actually applied so recording fps
        # (AVI playback speed) matches the true capture rate.
        if PySpin.IsReadable(frame_rate):
            actual = float(frame_rate.GetValue())
            self.target_frame_rate = actual
            self.recording_fps = actual

    def _configure_stream_buffers(self) -> None:
        """Deepen the transport-layer buffer pool (see STREAM_BUFFER_COUNT_TARGET).

        TLStream nodes (StreamBufferCountMode/StreamBufferCountManual) are
        accessed via GetTLStreamNodeMap(), a separate nodemap from the
        regular GenICam one used everywhere else in this file.
        """
        try:
            tl_nodemap = self.cam.GetTLStreamNodeMap()
            mode = PySpin.CEnumerationPtr(tl_nodemap.GetNode("StreamBufferCountMode"))
            if PySpin.IsWritable(mode):
                manual_entry = mode.GetEntryByName("Manual")
                if PySpin.IsReadable(manual_entry):
                    mode.SetIntValue(manual_entry.GetValue())
            count = PySpin.CIntegerPtr(tl_nodemap.GetNode("StreamBufferCountManual"))
            if PySpin.IsWritable(count):
                target = min(int(count.GetMax()), STREAM_BUFFER_COUNT_TARGET)
                count.SetValue(target)
        except Exception as exc:
            print(f"[camera] could not configure stream buffer count: {exc}")

    def _reinitialize_camera(self) -> tuple[bool, str]:
        """Best-effort full camera reinit after a fault. Acquisition thread only.

        Tears down and rebuilds the whole Spinnaker object chain (System ->
        CameraList -> Camera), the same sequence stop() already performs,
        then re-applies _configure_camera_nodes() and restarts acquisition.
        Called repeatedly by the watchdog's backoff loop until it succeeds
        -- see AcquisitionWatchdog's "no give_up action" design note.
        """
        # Set _recovering BEFORE taking the lock: GUI-thread accessors check
        # this flag first and return immediately without ever touching
        # _camera_lock, so they should never see the flag clear while
        # blocked waiting on the lock. (Setting it after acquiring the
        # lock would leave a narrow window where an accessor's flag check
        # passes and it then blocks on the lock instead.)
        self._recovering.set()
        try:
            with self._camera_lock:
                if self.cam is not None:
                    try:
                        self.cam.EndAcquisition()
                    except Exception:
                        pass
                    try:
                        self.cam.DeInit()
                    except Exception:
                        pass
                    self.cam = None
                if self.cam_list is not None:
                    try:
                        self.cam_list.Clear()
                    except Exception:
                        pass
                    self.cam_list = None
                if self.system is not None:
                    try:
                        self.system.ReleaseInstance()
                    except Exception:
                        pass
                    self.system = None

                try:
                    self.system = PySpin.System.GetInstance()
                    self.cam_list = self.system.GetCameras()
                    if self.cam_list.GetSize() == 0:
                        return False, "No camera detected during reinit."
                    self.cam = self.cam_list[0]
                    self.cam.Init()
                    self._configure_camera_nodes()
                    self.cam.BeginAcquisition()
                    with self._acquisition_stats_lock:
                        self._camera_reinits += 1
                    return True, "Camera reinitialized."
                except Exception as exc:
                    self.cam = None
                    return False, f"{exc.__class__.__name__}: {exc}"
        finally:
            self._recovering.clear()

    def _watchdog_grab_timeout_ms(self) -> int:
        return int(self._watchdog.config.grab_timeout_s * 1000)

    def _apply_watchdog_decision(self, decision) -> None:
        """Acquisition thread only. Acts on a "sleep" or "reinit" decision.

        ("continue" needs no action -- the caller just proceeds.)
        """
        if decision.action == "sleep":
            # Interruptible: returns immediately if stop() sets the event,
            # so a growing backoff never delays shutdown.
            self._stop_event.wait(decision.sleep_s)
        elif decision.action == "reinit":
            self._recover_camera(decision)

    def _recover_camera(self, decision) -> None:
        """Acquisition thread only. Attempt one reinit and log the outcome.

        On success, marks a timeline break in the current recording's
        events sidecar (if recording) so the gap is documented -- but
        deliberately does NOT touch frame_counter or the metadata writer:
        record_frame_index must stay unbroken across the break, per the
        session's one-continuous-CSV design.
        """
        ok, msg = self._reinitialize_camera()
        now = time.monotonic()
        if not ok:
            print(f"[camera] reinit failed: {msg}")
        else:
            print(f"[camera] reinit succeeded ({decision.reason})")
            if self._event_log is not None:
                gap_s = decision.stalled_for_s if decision.stalled_for_s else None
                frames_lost = (
                    estimate_frames_lost(gap_s, self.target_frame_rate)
                    if gap_s is not None
                    else None
                )
                brk = self._segment_tracker.begin_break(
                    cause="camera_reinit",
                    mono_ns=int(now * 1e9),
                    wall_ns=int(time.time() * 1e9),
                    note=f"{decision.reason}; do not fit across this",
                    gap_s=gap_s,
                    frames_lost_estimate=frames_lost,
                    record_frame_index=self.frame_counter,
                )
                try:
                    self._event_log.write(timeline_break_record(brk))
                except Exception as exc:
                    print("Error writing timeline break:", exc)
                if self.recording_active:
                    # Force the NEXT successful append to roll into a
                    # fresh segment (see should_roll(fault=True) and
                    # _maybe_rotate_segment), so the gap always lands
                    # between segments rather than inside one.
                    self._pending_fault_roll = True
                    self._pending_fault_roll_gap_s = gap_s
        reinit_decision = self._watchdog.note_reinit_result(now=now, ok=ok)
        self._apply_watchdog_decision(reinit_decision)

    # ------------------------------------------------------------------
    # Video segment rotation (acquisition thread only, while recording)
    # ------------------------------------------------------------------

    def _open_segment_writer(self, segment_index: int):
        """Open a new SpinVideo writer for segment_index at its part path.

        SpinVideo always appends its own "-0000" suffix to whatever base
        path it's given, so this writes to
        ".../.incomplete/<basename>_part{NNNN}-0000.avi"; the closer
        thread renames it to the final "<basename>-{NNNN}.avi" after
        Close() succeeds.
        """
        part_base = self._session_paths.video_part_base(segment_index)
        part_base.parent.mkdir(parents=True, exist_ok=True)
        writer = PySpin.SpinVideo()
        # SDK-level net, well above our own max_bytes ceiling -- should
        # never fire first. reconcile_part_files/the closer thread handle
        # it gracefully if it ever does.
        writer.SetMaximumFileSize(DEFAULT_SDK_MAX_FILE_SIZE_MB)
        opt = PySpin.MJPGOption()
        opt.frameRate = self.recording_fps
        opt.quality = 75
        writer.Open(str(part_base), opt)
        return writer

    def _maybe_rotate_segment(self) -> None:
        """Acquisition thread only. Called after each successful Append().

        Periodically samples the in-progress segment's on-disk size,
        pre-arms the next writer shortly before the boundary, and swaps
        writers when a roll condition fires. See backend/segment_policy.py
        for the roll/prepare decision logic.
        """
        if self._frames_in_segment % BYTES_SAMPLE_INTERVAL_FRAMES == 0:
            try:
                part_base = self._session_paths.video_part_base(self._segment_index)
                current_avi = part_base.with_name(part_base.name + "-0000.avi")
                self._bytes_in_segment = current_avi.stat().st_size
            except OSError:
                pass

        if not self._prepared_next_segment and should_prepare(
            frames_in_segment=self._frames_in_segment,
            max_frames=self._max_frames_per_segment,
        ):
            try:
                next_index = self._segment_index + 1
                self._pending_writer = self._open_segment_writer(next_index)
                self._pending_writer_segment_index = next_index
                self._prepared_next_segment = True
            except Exception as exc:
                print(f"[camera] could not pre-arm next segment: {exc}")

        decision = should_roll(
            frames_in_segment=self._frames_in_segment,
            bytes_in_segment=self._bytes_in_segment,
            max_frames=self._max_frames_per_segment,
            fault=self._pending_fault_roll,
        )
        if not decision.should_roll:
            return

        fault_gap_s = None
        if decision.reason == "fault":
            fault_gap_s = self._pending_fault_roll_gap_s
            self._pending_fault_roll = False
            self._pending_fault_roll_gap_s = None

        if self._pending_writer is not None and self._prepared_next_segment:
            new_writer = self._pending_writer
            new_index = self._pending_writer_segment_index
        else:
            new_index = self._segment_index + 1
            try:
                new_writer = self._open_segment_writer(new_index)
            except Exception as exc:
                print(f"[camera] could not open segment {new_index}: {exc}")
                # Keep recording into the current (oversized) segment
                # rather than losing the writer entirely.
                return

        self._pending_writer = None
        self._pending_writer_segment_index = None
        self._prepared_next_segment = False

        self._retire_segment(
            writer=self.avi_recorder,
            segment_index=self._segment_index,
            roll_reason=decision.reason,
            timeline_break=(decision.reason == "fault"),
            gap_s=fault_gap_s,
        )

        self.avi_recorder = new_writer
        self._segment_index = new_index
        self._frames_in_segment = 0
        self._bytes_in_segment = 0
        self._segment_opened_at = time.time()
        self._segment_first_record_frame_index = None
        self._segment_first_system_time = None
        if decision.reason == "fault":
            # Never reuse "record_start" -- downstream filters drop
            # unknown sync_label values, so this new value is inert to
            # anything that doesn't know about it yet.
            self._mark_next_frame_segment_resume = True
            # _segment_tracker.current_segment was already bumped when the
            # reinit succeeded (for prompt events.jsonl logging), possibly
            # several frames before this swap. Copy it into the per-row
            # "segment" column only now, so it changes on the exact same
            # row as segment_file -- not one or more frames earlier.
            self._metadata_segment = self._segment_tracker.current_segment

    def _retire_segment(
        self,
        *,
        writer,
        segment_index: int,
        roll_reason: str,
        timeline_break: bool,
        gap_s: float | None,
    ) -> None:
        """Hand a segment's writer off to the closer thread with its manifest entry."""
        final_path = self._session_paths.video_final(segment_index)
        part_base = self._session_paths.video_part_base(segment_index)
        entry = SegmentManifestEntry(
            segment_index=segment_index,
            segment_file=final_path.name,
            frame_count=self._frames_in_segment,
            first_record_frame_index=self._segment_first_record_frame_index,
            last_record_frame_index=self.frame_counter,
            first_system_time=self._segment_first_system_time,
            last_system_time=time.time(),
            opened_at=self._segment_opened_at,
            roll_reason=roll_reason,
            timeline_break=timeline_break,
            gap_s=gap_s,
        )
        self._closer_queue.put(
            _CloserJob(
                writer=writer,
                part_base=part_base,
                final_path=final_path,
                segment_index=segment_index,
                manifest_entry=entry,
            )
        )

    def _start_closer_thread(self) -> None:
        self._closer_thread = threading.Thread(
            target=self._closer_loop, name="segment-closer", daemon=True
        )
        self._closer_thread.start()

    def _closer_loop(self) -> None:
        while True:
            job = self._closer_queue.get()
            try:
                self._run_closer_job(job)
            finally:
                self._closer_queue.task_done()

    def _run_closer_job(self, job: "_CloserJob") -> None:
        start = time.monotonic()
        try:
            job.writer.Close()
        except Exception as exc:
            print(f"[camera] error closing segment {job.segment_index}: {exc}")
        close_duration_s = time.monotonic() - start

        part_files = reconcile_part_files(job.part_base)
        total_bytes = 0
        if not part_files:
            print(
                f"[camera] segment {job.segment_index}: no part file found "
                "after close (recording may be incomplete)"
            )
        else:
            total_bytes += self._safe_rename(part_files[0], job.final_path)
            for extra in part_files[1:]:
                # Spinnaker's own SetMaximumFileSize net fired mid-segment
                # (should never happen -- it's set well above our own
                # ceiling). Never silently orphan it: rename with an
                # unmistakable prefix rather than trying to claim another
                # slot in the live segment_index sequence, which risks a
                # race against the acquisition thread's own counter.
                print(f"[camera] WARNING: unexpected extra part file for segment {job.segment_index}: {extra}")
                total_bytes += self._safe_rename(extra, extra.with_name("UNEXPECTED_" + extra.name))

        entry = dataclass_replace(
            job.manifest_entry,
            closed_at=time.time(),
            close_duration_s=close_duration_s,
            bytes=total_bytes,
        )
        self._segment_manifest_writer.submit(manifest_row(entry))

    def _safe_rename(self, source: Path, destination: Path) -> int:
        size = 0
        try:
            size = source.stat().st_size
        except OSError:
            pass
        for attempt in range(5):
            try:
                os.replace(source, destination)
                return size
            except OSError as exc:
                if attempt == 4:
                    print(f"[camera] error finalizing {source} -> {destination}: {exc}")
                    return size
                time.sleep(0.2 * (attempt + 1))
        return size

    # ------------------------------------------------------------------
    # Recording control (GUI thread): only set flags
    # ------------------------------------------------------------------

    def start_recording(self, session_paths: SessionPaths, fps: float = 30.0):
        """
        Request recording to start. The acquisition thread will
        actually open SpinVideo and begin appending frames.

        Returns:
            (ok: bool, message: str)
        """
        if not self.acquiring or self.cam is None:
            return False, "Cannot record: camera is not acquiring."

        if self.recording_active or self.record_start_requested:
            return True, "Recording already starting or in progress."

        # Open the metadata CSV synchronously, on the calling (GUI) thread,
        # so a bad output path fails the start immediately instead of
        # silently losing every frame's metadata for the whole session.
        self._metadata_writer.start(session_paths.metadata_csv)
        if not self._metadata_writer.wait_until_open():
            return False, f"Cannot open metadata CSV: {self._metadata_writer.last_error}"

        try:
            self._event_log = JsonlEventLog(session_paths.events_jsonl)
            self._event_log.write(
                session_header_record(
                    mono_ns=int(time.monotonic() * 1e9),
                    wall_ns=int(time.time() * 1e9),
                    recording_basename=session_paths.basename,
                )
            )
        except Exception as exc:
            self._metadata_writer.stop()
            if self._event_log is not None:
                try:
                    self._event_log.close()
                except Exception:
                    pass
                self._event_log = None
            return False, f"Cannot open events log {session_paths.events_jsonl}: {exc}"

        self._segment_manifest_writer.start(session_paths.segments_csv)
        if not self._segment_manifest_writer.wait_until_open():
            self._metadata_writer.stop()
            self._event_log.close()
            self._event_log = None
            return False, f"Cannot open segments CSV: {self._segment_manifest_writer.last_error}"

        # Lives for the app's lifetime, not per-session -- start it once.
        if self._closer_thread is None or not self._closer_thread.is_alive():
            self._start_closer_thread()

        self._segment_tracker.reset()
        self._session_paths = session_paths
        self._segment_index = 0
        self._frames_in_segment = 0
        self._bytes_in_segment = 0
        self._segment_opened_at = time.time()
        self._segment_first_record_frame_index = None
        self._segment_first_system_time = None
        self._pending_writer = None
        self._pending_writer_segment_index = None
        self._prepared_next_segment = False
        self._pending_fault_roll = False
        self._pending_fault_roll_gap_s = None
        self._mark_next_frame_segment_resume = False
        self._metadata_segment = 0
        self._max_frames_per_segment = segment_frames_for(fps, resolve_segment_seconds())

        self.recording_fps = fps
        self.record_start_requested = True
        self.record_stop_requested = False
        # Reset recording frame counter
        self.frame_counter = 0
        with self._acquisition_stats_lock:
            self._append_failures = 0
            self._camera_reinits = 0

        return True, f"Recording requested: {session_paths.basename}"

    def stop_recording(self):
        """
        Request recording to stop. The acquisition thread will
        close SpinVideo and write CSV.
        """
        if not self.recording_active and not self.record_start_requested:
            return
        self.record_stop_requested = True

    # ------------------------------------------------------------------
    # Acquisition loop (runs in background thread)
    # ------------------------------------------------------------------

    def _acquisition_loop(self):
        # Deliberately does NOT require self.cam is not None: self.cam is
        # None for the whole duration between a fault tearing the handle
        # down and a reinit rebuilding it (see _reinitialize_camera / the
        # "no camera handle" branch below), and the loop must keep running
        # through that window to retry -- otherwise a single failed reinit
        # would silently end the loop and never resume.
        while not self._stop_event.is_set() and self.acquiring:
            # --------------------------------------------------
            # START recording (open SpinVideo) if requested
            # --------------------------------------------------
            if self.record_start_requested and not self.recording_active:
                try:
                    self.avi_recorder = self._open_segment_writer(0)
                    self.recording_active = True
                except Exception as exc:
                    print("Error starting recording:", exc)
                    self.avi_recorder = None
                    self.recording_active = False
                finally:
                    self.record_start_requested = False

            # --------------------------------------------------
            # STOP recording (close SpinVideo + write CSV) if requested
            # --------------------------------------------------
            if self.record_stop_requested and self.recording_active:
                if self.avi_recorder is not None:
                    self._retire_segment(
                        writer=self.avi_recorder,
                        segment_index=self._segment_index,
                        roll_reason="session_stop",
                        timeline_break=False,
                        gap_s=None,
                    )
                    self.avi_recorder = None

                # Discard any pre-armed next writer -- it will never be used.
                if self._pending_writer is not None:
                    try:
                        self._pending_writer.Close()
                    except Exception:
                        pass
                    try:
                        unused_base = self._session_paths.video_part_base(
                            self._pending_writer_segment_index
                        )
                        for unused_file in reconcile_part_files(unused_base):
                            unused_file.unlink(missing_ok=True)
                    except Exception:
                        pass
                    self._pending_writer = None
                    self._pending_writer_segment_index = None
                self._prepared_next_segment = False

                # Block until every queued segment (including the one just
                # retired above) is Close()'d, renamed, and manifest-logged.
                # Otherwise stop()'s DeInit()/ReleaseInstance() could run
                # concurrently with a Close() still in flight on the closer
                # thread -- the acquisition thread has no more frames to
                # grab at this point, so waiting here costs nothing.
                self._closer_queue.join()

                # Drains any queued/overflowed rows and closes the CSV.
                # Rows themselves were already written incrementally during
                # recording -- see the frame-append block below.
                self._metadata_writer.stop()
                writer_error = self._metadata_writer.last_error
                if writer_error:
                    print("Error writing metadata CSV:", writer_error)

                self._segment_manifest_writer.stop()
                manifest_error = self._segment_manifest_writer.last_error
                if manifest_error:
                    print("Error writing segments CSV:", manifest_error)

                if self._event_log is not None:
                    try:
                        with self._acquisition_stats_lock:
                            reinits = self._camera_reinits
                        self._event_log.write(
                            session_stop_record(
                                mono_ns=int(time.monotonic() * 1e9),
                                wall_ns=int(time.time() * 1e9),
                                total_segments=self._segment_tracker.current_segment,
                                camera_reinits=reinits,
                            )
                        )
                    except Exception as exc:
                        print("Error writing session stop record:", exc)
                    try:
                        self._event_log.close()
                    except Exception as exc:
                        print("Error closing events log:", exc)
                    self._event_log = None

                # Reset recording state
                self.recording_active = False
                self.record_stop_requested = False
                self._session_paths = None

            # --------------------------------------------------
            # Grab next frame from camera, with fault recovery
            # --------------------------------------------------
            # A stall (camera stopped delivering, no exception at all) can
            # only be caught by checking elapsed time independently of
            # whether the grab itself raises -- poll before attempting it.
            stall_decision = self._watchdog.poll(now=time.monotonic())
            if stall_decision.action == "reinit":
                self._recover_camera(stall_decision)
                continue

            with self._camera_lock:
                cam = self.cam
            if cam is None:
                error_decision = self._watchdog.note_error(
                    now=time.monotonic(), error="no camera handle"
                )
                self._apply_watchdog_decision(error_decision)
                continue

            try:
                # grabTimeout is milliseconds (Spinnaker CameraBase::GetNextImage);
                # bounding it is what turns a silent forever-block into a
                # detectable, recoverable error instead.
                grab_timeout_ms = self._watchdog_grab_timeout_ms()
                image = cam.GetNextImage(grab_timeout_ms)
            except Exception as exc:
                with self._acquisition_stats_lock:
                    self._acquisition_errors += 1
                error_decision = self._watchdog.note_error(now=time.monotonic(), error=str(exc))
                self._apply_watchdog_decision(error_decision)
                continue

            self._watchdog.note_frame_ok(now=time.monotonic())
            retrieved_at = time.monotonic()

            if image.IsIncomplete():
                with self._acquisition_stats_lock:
                    self._incomplete_images += 1
                image.Release()
                continue

            # Read camera identity/timing for every complete frame, including
            # preview-only frames. Previously these were read only while recording.
            timestamp_us = None
            frame_id = None
            try:
                chunk_data = image.GetChunkData()
                if hasattr(chunk_data, "GetTimestamp"):
                    try:
                        timestamp_us = chunk_data.GetTimestamp()
                    except Exception:
                        timestamp_us = None
                if hasattr(chunk_data, "GetFrameID"):
                    try:
                        frame_id = chunk_data.GetFrameID()
                    except Exception:
                        frame_id = None
            except Exception:
                pass

            with self._acquisition_stats_lock:
                self._preview_sequence += 1
                preview_sequence = self._preview_sequence
                if frame_id is not None and self._last_camera_frame_id is not None:
                    frame_delta = int(frame_id) - int(self._last_camera_frame_id)
                    if frame_delta > 1:
                        self._camera_frame_gaps += frame_delta - 1
                if frame_id is not None:
                    self._last_camera_frame_id = int(frame_id)

            # --------------------------------------------------
            # If recording, append frame + log metadata
            # --------------------------------------------------
            if self.recording_active and self.avi_recorder is not None:
                # Determine if this frame is within a sync window
                now = time.time()
                with self._sync_lock:
                    sync_this_frame = now <= self._sync_window_end
                    sync_label = self._sync_label if sync_this_frame else None
                with self._label_lock:
                    if self._pending_label_event is not None:
                        label_event = self._pending_label_event
                        adl_id = self._pending_adl_id
                        adl_label = self._pending_adl_label
                        self._pending_label_event = None
                        self._pending_adl_id = None
                        self._pending_adl_label = None
                    else:
                        label_event = None
                        adl_id = None
                        adl_label = None

                # Append before incrementing/logging: a failed Append must
                # not advance record_frame_index, or the CSV's frame index
                # permanently desynchronises from the AVI's actual frame
                # count for the rest of the (possibly multi-day) session.
                try:
                    self.avi_recorder.Append(image)
                except Exception as exc:
                    print("Error appending frame:", exc)
                    with self._acquisition_stats_lock:
                        self._append_failures += 1
                else:
                    self.frame_counter += 1
                    self._frames_in_segment += 1
                    if self._frames_in_segment == 1:
                        self._segment_first_record_frame_index = self.frame_counter
                        self._segment_first_system_time = time.time()

                    if self._mark_next_frame_segment_resume:
                        row_sync_label = "segment_resume"
                        self._mark_next_frame_segment_resume = False
                    else:
                        row_sync_label = resolve_sync_label(label_event, sync_label)

                    self._metadata_writer.submit(
                        metadata_row(
                            {
                                "record_frame_index": self.frame_counter,
                                "camera_frame_id": frame_id,
                                "timestamp_us": timestamp_us,
                                "system_time": time.time(),
                                "sync_pulse": sync_this_frame,
                                "sync_label": row_sync_label,
                                "adl_id": adl_id,
                                "adl_label": adl_label,
                                "segment": self._metadata_segment,
                                "segment_file": self._session_paths.video_final(self._segment_index).name,
                                "segment_frame_index": self._frames_in_segment,
                            }
                        )
                    )
                    self._maybe_rotate_segment()

            # --------------------------------------------------
            # Preview: store latest frame
            # --------------------------------------------------
            try:
                arr = image.GetNDArray()
                arr = np.array(arr, copy=True)
                published_at = time.monotonic()
                preview_frame = PreviewFrame(
                    image=arr,
                    sequence=preview_sequence,
                    frame_id=int(frame_id) if frame_id is not None else None,
                    camera_timestamp=(
                        int(timestamp_us) if timestamp_us is not None else None
                    ),
                    retrieved_at=retrieved_at,
                    published_at=published_at,
                )
                with self._frame_lock:
                    self._latest_frame = arr
                    self._latest_preview_frame = preview_frame
            except Exception:
                pass

            image.Release()

    # ------------------------------------------------------------------
    # Preview API for Qt
    # ------------------------------------------------------------------

    def get_latest_frame(self):
        """
        Return a copy of the latest acquired frame as a NumPy array,
        or None if no frame is available yet.
        """
        with self._frame_lock:
            if self._latest_frame is None:
                return None
            return self._latest_frame.copy()

    def get_latest_preview_frame(
        self,
        after_sequence: int | None = None,
    ) -> PreviewFrame | None:
        """Return a new owned preview frame, or ``None`` if it is unchanged."""
        with self._frame_lock:
            latest = self._latest_preview_frame
            if latest is None or (
                after_sequence is not None and latest.sequence == after_sequence
            ):
                return None
            return PreviewFrame(
                image=latest.image.copy(),
                sequence=latest.sequence,
                frame_id=latest.frame_id,
                camera_timestamp=latest.camera_timestamp,
                retrieved_at=latest.retrieved_at,
                published_at=latest.published_at,
            )

    def _reset_acquisition_stats(self) -> None:
        with self._acquisition_stats_lock:
            self._preview_sequence = 0
            self._last_camera_frame_id = None
            self._camera_frame_gaps = 0
            self._incomplete_images = 0
            self._acquisition_errors = 0

    def get_acquisition_stats(self) -> dict[str, int]:
        """Return a thread-safe acquisition-health snapshot."""
        with self._acquisition_stats_lock:
            return {
                "complete_frames": self._preview_sequence,
                "camera_frame_gaps": self._camera_frame_gaps,
                "incomplete_images": self._incomplete_images,
                "acquisition_errors": self._acquisition_errors,
                "append_failures": self._append_failures,
                "metadata_overflow_rows": self._metadata_writer.overflow_rows,
                "camera_reinits": self._camera_reinits,
            }

    # ------------------------------------------------------------------
    # Image controls (GenICam; GUI thread while acquiring)
    # ------------------------------------------------------------------

    def get_image_param_limits(self, param_name: str) -> tuple[float, float, float] | None:
        """
        Return (min, max, current) as floats for a GenICam node name
        (e.g. 'Gain', 'Gamma', 'BlackLevel'), or None if unavailable.

        Gamma is often gated by GammaEnable; this enables it when writable
        so limits/current can be read.
        """
        if self.cam is None or not self.acquiring or self._recovering.is_set():
            return None
        with self._camera_lock:
            if self.cam is None:
                return None
            try:
                nodemap = self.cam.GetNodeMap()
                if param_name == "Gamma":
                    ge = PySpin.CBooleanPtr(nodemap.GetNode("GammaEnable"))
                    if PySpin.IsWritable(ge):
                        ge.SetValue(True)
                node = nodemap.GetNode(param_name)
                if node is None or not PySpin.IsReadable(node):
                    return None
                fn = PySpin.CFloatPtr(node)
                if PySpin.IsReadable(fn):
                    return (
                        float(fn.GetMin()),
                        float(fn.GetMax()),
                        float(fn.GetValue()),
                    )
                ir = PySpin.CIntegerPtr(node)
                if PySpin.IsReadable(ir):
                    return (
                        float(ir.GetMin()),
                        float(ir.GetMax()),
                        float(ir.GetValue()),
                    )
            except Exception:
                return None
        return None

    def set_image_param(self, param_name: str, value: float) -> bool:
        """Write Gain / Gamma / BlackLevel (float or integer node)."""
        if self.cam is None or not self.acquiring or self._recovering.is_set():
            return False
        with self._camera_lock:
            if self.cam is None:
                return False
            try:
                nodemap = self.cam.GetNodeMap()
                if param_name == "Gamma":
                    ge = PySpin.CBooleanPtr(nodemap.GetNode("GammaEnable"))
                    if PySpin.IsWritable(ge):
                        ge.SetValue(True)
                node = nodemap.GetNode(param_name)
                if node is None:
                    return False
                fn = PySpin.CFloatPtr(node)
                if PySpin.IsWritable(fn):
                    lo, hi = float(fn.GetMin()), float(fn.GetMax())
                    fn.SetValue(min(hi, max(lo, float(value))))
                    return True
                ir = PySpin.CIntegerPtr(node)
                if PySpin.IsWritable(ir):
                    lo, hi = int(ir.GetMin()), int(ir.GetMax())
                    iv = int(round(float(value)))
                    ir.SetValue(min(hi, max(lo, iv)))
                    return True
            except Exception as exc:
                print(f"[camera] set_image_param {param_name}: {exc}")
        return False

    # ------------------------------------------------------------------
    # Acquisition frame rate (GenICam; GUI thread while acquiring)
    # ------------------------------------------------------------------

    def get_frame_rate_limits(self) -> tuple[float, float, float] | None:
        """
        Return (min, max, current) fps for AcquisitionFrameRate, or None if
        the camera does not expose an adjustable frame rate.

        Enables AcquisitionFrameRateEnable first so the node is readable; the
        'current' value right after enabling reflects the camera's default
        (typically its max sustainable rate at the current exposure).
        """
        if self.cam is None or not self.acquiring or self._recovering.is_set():
            return None
        with self._camera_lock:
            if self.cam is None:
                return None
            try:
                nodemap = self.cam.GetNodeMap()
                enable = PySpin.CBooleanPtr(nodemap.GetNode("AcquisitionFrameRateEnable"))
                if PySpin.IsWritable(enable):
                    enable.SetValue(True)
                node = PySpin.CFloatPtr(nodemap.GetNode("AcquisitionFrameRate"))
                if PySpin.IsReadable(node):
                    return (
                        float(node.GetMin()),
                        float(node.GetMax()),
                        float(node.GetValue()),
                    )
            except Exception:
                return None
        return None

    def get_acquisition_frame_rate(self) -> float:
        """Current acquisition fps, or the last-known target if unreadable."""
        limits = self.get_frame_rate_limits()
        if limits is None:
            return float(self.target_frame_rate)
        return limits[2]

    def set_frame_rate(self, value: float) -> bool:
        """
        Set AcquisitionFrameRate (clamped to the camera's supported range) and
        keep recording_fps in sync so AVI playback speed matches capture.
        Returns True on success.
        """
        if self.cam is None or not self.acquiring or self._recovering.is_set():
            return False
        with self._camera_lock:
            if self.cam is None:
                return False
            try:
                nodemap = self.cam.GetNodeMap()
                enable = PySpin.CBooleanPtr(nodemap.GetNode("AcquisitionFrameRateEnable"))
                if PySpin.IsWritable(enable):
                    enable.SetValue(True)
                node = PySpin.CFloatPtr(nodemap.GetNode("AcquisitionFrameRate"))
                if PySpin.IsWritable(node):
                    lo, hi = float(node.GetMin()), float(node.GetMax())
                    node.SetValue(min(hi, max(lo, float(value))))
                    actual = float(node.GetValue())
                    self.target_frame_rate = actual
                    self.recording_fps = actual
                    return True
            except Exception as exc:
                print(f"[camera] set_frame_rate: {exc}")
        return False

    # ------------------------------------------------------------------
    # Sync pulse logic for logging
    # ------------------------------------------------------------------
    def notify_sync_pulse_window(self, width_s: float, label: str):
        """
        Notify that a sync pulse is active for the next `width_s` seconds.
        Any recorded frame whose system_time is <= this window end
        will be logged with sync_pulse=True and this label.
        """
        now = time.time()
        end_time = now + float(width_s)

        with self._sync_lock:
            # extend window if overlapping pulses
            self._sync_window_end = max(self._sync_window_end, end_time)
            self._sync_label = label

    def notify_label_event(self, label: str, adl_id, adl_label):
        """
        Notify that a label event occurred; it will be attached to the next
        recorded frame as sync_label plus ADL metadata.
        """
        with self._label_lock:
            self._pending_label_event = label
            self._pending_adl_id = adl_id
            self._pending_adl_label = adl_label
