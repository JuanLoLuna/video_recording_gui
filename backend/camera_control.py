# backend/camera_control.py
import os
import queue
import threading
import time
from dataclasses import dataclass, replace as dataclass_replace
from pathlib import Path

import cv2
import numpy as np
import PySpin

from backend.async_csv_writer import AsyncCsvWriter
from backend.camera_registry import CameraDescriptor
from backend.disk_guard import StreamRate
from backend.preview_scaling import fit_size
from backend.spinnaker_system import SharedSystemHolder, default_holder
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
from backend.recording_paths import SessionPaths, ensure_directory
from backend.segment_policy import (
    BYTES_SAMPLE_INTERVAL_FRAMES,
    reconcile_part_files,
    resolve_segment_seconds,
    segment_frames_for,
    should_prepare,
    should_roll,
)
from backend.segment_manifest import SegmentManifestEntry, SegmentManifestWriter, manifest_row
from backend.teardown import assess_teardown_readiness
from backend.timeline import TimelineBaseline, compute_wall_mono_skew_s


# Confirmed on the bench: this camera does NOT reliably shrink ExposureTime
# (or its allowed max) when AcquisitionFrameRate is raised -- exposure sat at
# 15ms with fps reading 99.96, a combination that can't physically sustain
# more than ~1/0.015 =~ 66 fps, let alone 100. Nothing enforces the other
# direction either. So this app clamps ExposureTime itself whenever fps
# changes, leaving this fraction of the new frame period as headroom for
# sensor readout (which isn't otherwise visible to us).
EXPOSURE_FRAME_PERIOD_HEADROOM = 0.9

# Spinnaker's own buffer pool is the decoupling queue between frame arrival
# and disk writes: deepening it is what makes a rotation-boundary disk
# stall (up to ~1.6 GB of dirty page cache) survivable without dropping
# frames. Sized in SECONDS rather than a fixed frame count, and multiplied
# against the camera's max achievable fps (see _configure_stream_buffers),
# not self.target_frame_rate -- StreamBufferCountManual is only writable
# before BeginAcquisition(), but the frame rate is normally raised well
# after that (GUI spin box calls set_frame_rate() post-connect), so the
# buffer can't be re-sized to match whatever fps the camera ends up
# running at. At 1.31 MB/frame (1280x1024 Mono8) and 100 fps, 5s is ~500
# buffers / ~655 MB of RAM.
STREAM_BUFFER_SECONDS_TARGET = 5.0

# fps at or above which "Exposure mode" is forced to Off (see the GUI's
# _apply_exposure_auto_lock_for_fps). ExposureAuto gives no guarantee of
# respecting the frame period -- confirmed on the bench: left on Continuous it
# converged to ~14.8 ms, exceeding the ~10 ms period 100 fps needs, and the
# exposure clamp below can only act once ExposureTime is writable, i.e. Off.
# Below this there is enough slack that Auto stays a free user choice. Lives
# here (not just in the GUI) because a camera that was unplugged and replugged
# comes back on its power-on default (usually Continuous), and the GUI's lock
# only runs when the frame-rate widgets change.
EXPOSURE_AUTO_LOCK_MIN_FPS = 30.0

# After each segment is finalized, its first frame is decoded from the FILE and
# compared with the frame that was handed to the writer. A segment that "worked"
# (right frame count, normal timing) but decodes black went unnoticed on the rig;
# this is the only check that looks at what was actually written. A scene darker
# than the minimum cannot be told apart from a black file, so it is skipped.
SEGMENT_PIXEL_CHECK_MIN_MEAN = 10.0
SEGMENT_PIXEL_BLACK_RATIO = 0.25


@dataclass
class _CloserJob:
    """One segment handed from the acquisition thread to the closer thread."""

    writer: object  # cv2.VideoWriter
    part_base: Path
    final_path: Path
    segment_index: int
    manifest_entry: SegmentManifestEntry
    # Mean pixel value of the segment's first frame as recorded (None = unknown / empty).
    first_frame_mean: float | None = None


@dataclass
class _AppendJob:
    """One frame handed from the acquisition thread to the append thread.

    `frame_array` is the same already-copied numpy array built for the
    preview path (see the acquisition loop's "Preview: store latest
    frame" section) -- reused rather than copied twice. It's safe to
    share across threads: nothing mutates it after creation, both the
    GUI (via _latest_frame) and the append thread only ever read it. This
    also means the acquisition thread Release()s the PySpin image
    immediately after that conversion, regardless of recording state --
    unlike the SpinVideo-based version of this job, it no longer needs to
    hold the native image buffer open until the append thread is done
    with it.

    sync_this_frame/sync_label/label_event/adl_id/adl_label/
    captured_wall_s/captured_mono_s are captured at grab time, not append
    time -- they describe when the frame was captured, and must not drift
    with however far behind the append queue is.
    """

    frame_array: np.ndarray
    frame_id: int | None
    timestamp_us: int | None
    sync_this_frame: bool
    sync_label: str | None
    label_event: str | None
    adl_id: object | None
    adl_label: object | None
    captured_wall_s: float
    captured_mono_s: float


@dataclass(frozen=True)
class PreviewFrame:
    """An owned preview image plus timing captured along its pipeline."""

    image: np.ndarray
    sequence: int
    frame_id: int | None
    camera_timestamp: int | None
    retrieved_at: float
    published_at: float


def _read_tl_string(nodemap, node_name: str) -> str:
    node = PySpin.CStringPtr(nodemap.GetNode(node_name))
    if PySpin.IsReadable(node):
        return node.GetValue()
    return "<unavailable>"


def enumerate_cameras(holder: SharedSystemHolder | None = None) -> list[CameraDescriptor]:
    """Every Spinnaker camera currently visible, as plain descriptors.

    Takes its own short-lived reference on the shared System, so it is safe to
    call while other controllers are streaming (validated on the rig:
    GetCameras() during another camera's acquisition does not disturb it).
    Raises if the Spinnaker System itself cannot be created.
    """
    holder = holder or default_holder()
    owner = object()
    system = holder.acquire(owner)
    try:
        cam_list = system.GetCameras()
        try:
            found: list[CameraDescriptor] = []
            for index in range(cam_list.GetSize()):
                cam = cam_list[index]
                try:
                    nodemap = cam.GetTLDeviceNodeMap()
                    found.append(
                        CameraDescriptor(
                            serial=_read_tl_string(nodemap, "DeviceSerialNumber"),
                            model=_read_tl_string(nodemap, "DeviceModelName"),
                            vendor=_read_tl_string(nodemap, "DeviceVendorName"),
                        )
                    )
                except Exception:
                    # Still listed, so the user can see something is wrong
                    # with it; select_cameras() ignores unreadable serials.
                    found.append(CameraDescriptor(serial="<unavailable>"))
                finally:
                    cam = None
            return found
        finally:
            cam_list.Clear()
    finally:
        holder.release(owner)


def detect_first_camera():
    """
    Use Spinnaker (PySpin) to detect the first connected camera.

    Returns:
        (found: bool, message: str)

    - found = True  -> at least one camera found, message has vendor/model/serial
    - found = False -> no camera / error, message has a short explanation
    """
    try:
        cameras = enumerate_cameras()
    except Exception as exc:
        return False, f"Error: could not list cameras ({exc})"
    if not cameras:
        return False, "No cameras detected."
    first = cameras[0]
    return True, f"Camera: {first.vendor} {first.model} (S/N: {first.serial})"


class CameraController:
    """
    Handles:
      - Connecting to first camera
      - Running an acquisition loop in a background thread
      - Providing latest frame for preview
      - Recording to AVI via cv2.VideoWriter (uncompressed grayscale by
        default; MJPEG opt-in -- see _use_compression)
      - Logging per-recorded-frame metadata to CSV

    Recording was PySpin's SpinVideo until profiling showed its Append()
    costing a ~fixed ~18ms/frame regardless of codec or actual disk speed
    (independently measured at well under 4ms even with a forced fsync,
    via scripts/disk_write_latency_probe.py) -- overhead internal to
    SpinVideo itself. scripts/cv2_videowriter_latency_probe.py found
    cv2.VideoWriter's uncompressed grayscale codec writing the same real
    frame size in ~2ms.

    Open() runs on whichever thread opens that particular segment: the
    acquisition thread for the first segment of a session (start_recording),
    the append thread for every one after (_maybe_rotate_segment).
    write() and release() are both still handed off to their own
    dedicated threads (_append_queue/_append_thread,
    _closer_queue/_closer_thread) -- write() is now fast, but keeping it
    off the acquisition thread means an occasional slow frame (a disk
    hiccup, or MJPEG mode) still can't block the next grab.
    """

    def __init__(self, *, serial=None, tag=None, system_holder=None):
        # Identity. `serial` pins this controller to one physical camera
        # (GetBySerial) -- enumeration order is not stable, so with several
        # cameras nothing may ever rely on cam_list[0]. With serial=None the
        # legacy single-camera behaviour is kept (first camera found), and the
        # serial it finds is pinned for the rest of this controller's life so
        # a fault recovery re-finds the SAME camera.
        self.requested_serial = str(serial) if serial else None
        self.serial = self.requested_serial
        self.model = ""
        self.tag = tag
        # The Spinnaker System is shared by every controller in the process;
        # controllers never call GetInstance/ReleaseInstance themselves.
        self._holder = system_holder or default_holder()

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
        self._timeline_baseline = TimelineBaseline(
            session_start_wall_s=0.0, session_start_mono_s=0.0
        )
        # One row per segment (~960/session); reused across record start/
        # stop cycles like _metadata_writer, started fresh in start_recording.
        self._segment_manifest_writer = SegmentManifestWriter()
        # Retired writers are Close()'d, renamed, and manifest-logged off
        # the acquisition thread -- Close() can take long enough (flushing
        # up to ~1.6 GB of dirty page cache) that doing it inline would
        # risk dropping frames at every rotation boundary. Lives for the
        # whole app (daemon thread, started lazily), not per-session.
        self._closer_queue: queue.Queue = queue.Queue()
        # Segments whose finalize step raised (the closer thread keeps going).
        self.closer_failures = 0
        # (segment_index, reason) for segments whose finished file does not decode to
        # the picture that was recorded. Cleared by every prepare_recording().
        self.segment_pixel_problems: list[tuple[int, str]] = []
        self._segment_first_frame_mean: float | None = None
        # MJPEG writer opens that had to be retried because FFMPEG did not take the file.
        self.writer_open_retries = 0
        self._closer_thread: threading.Thread | None = None
        # Append() (+ the per-frame bookkeeping/rotation contingent on it
        # succeeding) profiled at a ~fixed ~18ms/frame regardless of codec,
        # consistent with a per-call disk sync cost rather than genuine
        # CPU/bandwidth work -- moved off the acquisition thread for the
        # same reason Close() already is. Ownership of frame_counter,
        # avi_recorder, and all _segment_*/_pending_writer* state transfers
        # to this thread once recording starts; the acquisition thread's
        # STOP-recording path joins this queue before reading any of it
        # back (see _acquisition_loop). Lives for the whole app (daemon
        # thread, started lazily), not per-session.
        self._append_queue: queue.Queue = queue.Queue()
        self._append_thread: threading.Thread | None = None

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

        # Per-stage acquisition-loop timing, one raw sample per frame --
        # separate lock from _acquisition_stats_lock since this is populated
        # every frame (not just on recording-relevant events) and read/reset
        # once a second by the GUI's diagnostics sampler (see
        # get_and_reset_loop_timing_samples). Added to find which stage of
        # GetNextImage -> Append -> GetNDArray is actually responsible for
        # the ~50fps ceiling observed even when the sensor itself produces
        # frames faster (confirmed via camera_frame_id in a real test).
        self._loop_timing_lock = threading.Lock()
        self._loop_timing_samples: dict[str, list[float]] = {
            "grab_ms": [],
            "append_ms": [],
            "ndarray_ms": [],
        }

        # Recording state/flags (thread-safe)
        self.recording_active = False          # true while the video writer is open
        self.record_start_requested = False    # GUI asks to start
        self.record_stop_requested = False     # GUI asks to stop
        # SessionPaths handed to prepare_recording() and not yet begun/aborted.
        self._prepared = None
        # Why the last ACCEPTED start failed asynchronously (segment 0 would not
        # open); None when it did not. Cleared by every prepare_recording().
        self.last_start_error: str | None = None

        self.avi_recorder = None
        # MJPEG trades smaller files for an encode cost inside Append() that
        # profiling measured at ~17-20ms/frame on one test machine -- enough
        # to cap throughput well under 100fps there, but likely fine at
        # lower frame rates (more slack in the frame period) or on faster
        # hardware. Left False (uncompressed) by default since it's the
        # only choice verified not to cap the recording rate; the GUI lets
        # this be turned on deliberately, with a live append_ms-based
        # warning if it turns out too slow for the chosen fps.
        self._use_compression = False
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
            thread_name="frame-metadata-writer"
            + (f"-{self.requested_serial or self.tag}" if (self.requested_serial or self.tag) else ""),
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
    # Identity helpers
    # ------------------------------------------------------------------
    @property
    def _log_prefix(self) -> str:
        return f"[camera {self.serial}]" if self.serial else "[camera]"

    @property
    def _thread_suffix(self) -> str:
        return f"-{self.serial}" if self.serial else ""

    def _pick_camera(self, cam_list):
        """The camera this controller is bound to, or None if it isn't present.

        Pinned/requested serial -> GetBySerial; otherwise (legacy single
        camera, first start only) the first camera in the list.
        """
        want = self.serial
        if not want:
            return cam_list[0] if cam_list.GetSize() > 0 else None
        try:
            cam = cam_list.GetBySerial(want)
        except Exception:
            return None
        try:
            if cam is None or not cam.IsValid():
                return None
        except Exception:
            return None
        return cam

    def _pin_identity(self) -> None:
        """Record the serial/model of the camera just Init()'d."""
        try:
            nodemap = self.cam.GetTLDeviceNodeMap()
            serial = _read_tl_string(nodemap, "DeviceSerialNumber")
            model = _read_tl_string(nodemap, "DeviceModelName")
        except Exception:
            return
        if serial != "<unavailable>" and not self.serial:
            self.serial = serial
        if model != "<unavailable>":
            self.model = model

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
            self.system = self._holder.acquire(self)
            self.cam_list = self.system.GetCameras()
            num_cams = self.cam_list.GetSize()

            if num_cams == 0:
                self._cleanup_system()
                return False, "No cameras detected."

            self.cam = self._pick_camera(self.cam_list)
            if self.cam is None:
                self.cam_list.Clear()
                self.cam_list = None
                self._cleanup_system()
                return False, f"Camera {self.serial} not found."
            self.cam.Init()
            self._pin_identity()
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
                name=f"acquisition{self._thread_suffix}",
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

        Returns (ok, message). ok is False only in the rare case the
        acquisition thread did not exit in time -- see the teardown-safety
        note below.
        """
        # A recording that was prepared but never begun holds open sidecars and
        # would otherwise leave the controller refusing "already prepared".
        self.abort_prepared()

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
        thread_alive = False
        if self._acq_thread is not None:
            try:
                self._acq_thread.join(timeout=2.0)
            except Exception:
                pass
            thread_alive = self._acq_thread.is_alive()
            if not thread_alive:
                self._acq_thread = None

        self.acquiring = False
        self._latest_frame = None
        self._latest_preview_frame = None

        # The 90s wait above already covers an ordinary large-segment close;
        # reaching here with the thread still alive means it is genuinely
        # stuck, not just slow. DeInit/ReleaseInstance while it might still
        # be touching self.cam/self.system is a use-after-free in native
        # code, not a catchable exception -- so deliberately leak the handle
        # instead of releasing it. self.cam/self.cam_list/self.system are
        # left as-is (not cleared) so a caller can tell teardown didn't
        # finish, rather than reporting a clean stop that didn't happen.
        decision = assess_teardown_readiness(acquisition_thread_alive=thread_alive)
        if not decision.safe_to_release:
            print(f"{self._log_prefix} teardown deferred: {decision.reason}")
            return False, decision.reason

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

        # The serial found by a legacy (no-serial) controller is pinned only
        # while it runs, so that fault recovery re-finds the SAME camera. After
        # a clean stop it must go: otherwise swapping in a different camera and
        # pressing Preview again would look for the old serial forever.
        self.serial = self.requested_serial
        if self.requested_serial is None:
            self.model = ""

        return True, ""

    def _cleanup_system(self):
        # Releases THIS controller's reference only; the System itself is
        # released when the last controller lets go (and never while another
        # camera's deferred teardown still holds its reference). Idempotent.
        self._holder.release(self)
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
            print(f"{self._log_prefix} ChunkModeActive not writable; skipping chunk setup.")
            return

        chunk_mode_active.SetValue(True)
        print(f"{self._log_prefix} Chunk mode activated.")

        # 2) Enable specific chunks if they exist
        chunk_selector = PySpin.CEnumerationPtr(nodemap.GetNode("ChunkSelector"))
        chunk_enable = PySpin.CBooleanPtr(nodemap.GetNode("ChunkEnable"))

        if not (PySpin.IsReadable(chunk_selector) and PySpin.IsWritable(chunk_selector)):
            print(f"{self._log_prefix} ChunkSelector not usable; skipping chunk setup.")
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
                print(f"{self._log_prefix} Could not enable chunk '{name}': {exc}")
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

        nodemap = self.cam.GetNodeMap()
        acq_mode = PySpin.CEnumerationPtr(nodemap.GetNode("AcquisitionMode"))
        continuous_entry = acq_mode.GetEntryByName("Continuous")
        acq_mode.SetIntValue(continuous_entry.GetValue())

        frame_rate_enable = PySpin.CBooleanPtr(nodemap.GetNode("AcquisitionFrameRateEnable"))
        if PySpin.IsWritable(frame_rate_enable):
            frame_rate_enable.SetValue(True)
        frame_rate = PySpin.CFloatPtr(nodemap.GetNode("AcquisitionFrameRate"))

        # Must run before frame_rate.SetValue() below: it sizes the buffer
        # pool off the camera's max achievable fps (frame_rate.GetMax()),
        # not self.target_frame_rate -- the GUI's fps spin box normally
        # raises the rate well after this point via set_frame_rate(), which
        # only touches AcquisitionFrameRate and can't resize buffers live
        # once BeginAcquisition() has run. Sizing off the ceiling up front
        # means the buffer is correct no matter what fps gets dialed in
        # later, instead of silently assuming the 30fps default.
        self._configure_stream_buffers(frame_rate)

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
            if actual >= EXPOSURE_AUTO_LOCK_MIN_FPS:
                self._force_exposure_auto_off(nodemap)
            self._clamp_exposure_to_frame_period(actual)

    def _force_exposure_auto_off(self, nodemap) -> None:
        """Set ExposureAuto to Off if the camera has it (best effort, never raises)."""
        try:
            node = PySpin.CEnumerationPtr(nodemap.GetNode("ExposureAuto"))
            if PySpin.IsWritable(node):
                node.SetIntValue(node.GetEntryByName("Off").GetValue())
        except Exception as exc:
            print(f"{self._log_prefix} could not force ExposureAuto Off: {exc}")

    def _clamp_exposure_to_frame_period(self, fps: float) -> None:
        """Cap ExposureTime so it fits the frame period implied by `fps`.

        See EXPOSURE_FRAME_PERIOD_HEADROOM: the camera doesn't reliably do
        this itself, so whenever the acquisition frame rate changes -- at
        connect (_configure_camera_nodes) or later via set_frame_rate() --
        this brings exposure back into a range the sensor can actually
        sustain, rather than silently capping real throughput while
        AcquisitionFrameRate reads back whatever was requested.

        Callers already hold _camera_lock (an RLock) or run single-threaded
        during connect, so this doesn't acquire it itself.
        """
        if self.cam is None or fps <= 0:
            return
        try:
            nodemap = self.cam.GetNodeMap()
            raw_node = nodemap.GetNode("ExposureTime")
            if raw_node is None:
                return
            node = PySpin.CFloatPtr(raw_node)
            if not PySpin.IsReadable(node) or not PySpin.IsWritable(node):
                return
            period_us = 1_000_000.0 / fps
            safe_max_us = period_us * EXPOSURE_FRAME_PERIOD_HEADROOM
            current_us = float(node.GetValue())
            if current_us > safe_max_us:
                lo = float(node.GetMin())
                new_value = max(lo, safe_max_us)
                node.SetValue(new_value)
                print(
                    f"{self._log_prefix} clamped ExposureTime {current_us:.0f}us -> "
                    f"{new_value:.0f}us to fit {fps:.1f} fps "
                    f"({period_us:.0f}us period)"
                )
        except Exception as exc:
            print(f"{self._log_prefix} could not clamp exposure to frame period: {exc}")

    def _configure_stream_buffers(self, frame_rate_node=None) -> None:
        """Deepen the transport-layer buffer pool (see STREAM_BUFFER_SECONDS_TARGET).

        TLStream nodes (StreamBufferCountMode/StreamBufferCountManual) are
        accessed via GetTLStreamNodeMap(), a separate nodemap from the
        regular GenICam one used everywhere else in this file.

        `frame_rate_node` is the (already-fetched) AcquisitionFrameRate node
        from the regular nodemap, used to read the camera's max achievable
        fps so the buffer is sized for the fastest rate it could ever run
        at, not whatever self.target_frame_rate happens to be right now.
        """
        try:
            max_fps = float(self.target_frame_rate)
            if frame_rate_node is not None and PySpin.IsReadable(frame_rate_node):
                max_fps = max(max_fps, float(frame_rate_node.GetMax()))

            tl_nodemap = self.cam.GetTLStreamNodeMap()
            mode = PySpin.CEnumerationPtr(tl_nodemap.GetNode("StreamBufferCountMode"))
            if PySpin.IsWritable(mode):
                manual_entry = mode.GetEntryByName("Manual")
                if PySpin.IsReadable(manual_entry):
                    mode.SetIntValue(manual_entry.GetValue())
            count = PySpin.CIntegerPtr(tl_nodemap.GetNode("StreamBufferCountManual"))
            if PySpin.IsWritable(count):
                buffer_max = int(count.GetMax())
                requested = round(max_fps * STREAM_BUFFER_SECONDS_TARGET)
                target = min(buffer_max, requested)
                count.SetValue(target)
                applied = int(count.GetValue()) if PySpin.IsReadable(count) else target
                print(
                    f"{self._log_prefix} stream buffer count: applied={applied} "
                    f"requested={requested} max={buffer_max} "
                    f"(sized for {max_fps:.1f} fps x {STREAM_BUFFER_SECONDS_TARGET:.0f}s)"
                )
        except Exception as exc:
            print(f"{self._log_prefix} could not configure stream buffer count: {exc}")

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
                self.system = None

                try:
                    if not self._holder.holds(self):
                        self.system = self._holder.acquire(self)
                    elif self._holder.restart_if_sole_owner(self):
                        # Nobody else is streaming: the full System ->
                        # CameraList -> Camera rebuild that passed the real
                        # unplug/replug tests in the phase 1 and 3 reports.
                        self.system = self._holder.system
                    else:
                        # Another camera is streaming off this System: never
                        # touch it. Rebuild only this camera's own handles
                        # (validated on the rig by scripts/reinit_spike.py).
                        self.system = self._holder.system or self._holder.acquire(self)
                    self.cam_list = self.system.GetCameras()
                    self.cam = self._pick_camera(self.cam_list)
                    if self.cam is None and self.serial:
                        # The bus may not have refreshed yet: ask once, retry
                        # (also when the list came back empty).
                        try:
                            self.cam_list.Clear()
                            self.system.UpdateCameras()
                            self.cam_list = self.system.GetCameras()
                            self.cam = self._pick_camera(self.cam_list)
                        except Exception:
                            self.cam = None
                    if self.cam is None:
                        if self.cam_list.GetSize() == 0:
                            return False, "No camera detected during reinit."
                        return False, f"Camera {self.serial} not present during reinit."
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

        frame_counter and _pending_fault_roll[_gap_s] are read/written
        here from the acquisition thread, but owned day-to-day by the
        append thread (see _run_append_job) -- both are simple attributes
        (GIL-atomic) and the only consequence of a stale read is the
        timeline-break record citing a frame_counter a queue-depth behind
        the append thread's true progress, or the fault-roll flag being
        noticed a few queued frames later than this exact instant. Neither
        needs a lock to be correct.
        """
        ok, msg = self._reinitialize_camera()
        now = time.monotonic()
        if not ok:
            print(f"{self._log_prefix} reinit failed: {msg}")
        else:
            print(f"{self._log_prefix} reinit succeeded ({decision.reason})")
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
                    print(f"{self._log_prefix} Error writing timeline break:", exc)
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
    # Video segment rotation (append thread only, while recording -- see
    # _run_append_job/_append_queue)
    # ------------------------------------------------------------------

    def _get_frame_dimensions(self) -> tuple[int, int]:
        """(width, height) from the camera's GenICam nodes.

        Unlike SpinVideo (which inferred size from the first appended
        frame), cv2.VideoWriter requires it up front at construction.
        Locked like every other GenICam accessor in this file, so a
        concurrent reinit can't tear self.cam down mid-read; raises if
        the handle isn't available (e.g. mid-reinit) rather than
        returning a sentinel -- callers (_open_segment_writer, via
        start_recording/_maybe_rotate_segment) already wrap segment-open
        failures in try/except the same way.
        """
        with self._camera_lock:
            if self.cam is None:
                raise RuntimeError("camera not available")
            nodemap = self.cam.GetNodeMap()
            width_node = PySpin.CIntegerPtr(nodemap.GetNode("Width"))
            height_node = PySpin.CIntegerPtr(nodemap.GetNode("Height"))
            return int(width_node.GetValue()), int(height_node.GetValue())

    def _open_segment_writer(self, segment_index: int):
        """Open a new cv2.VideoWriter for segment_index at its part path.

        Was PySpin's SpinVideo until profiling showed its Append() costing
        a ~fixed ~18ms/frame regardless of codec (MJPEG vs uncompressed
        AVIOption) or actual disk speed (independently measured at well
        under 4ms even with a forced fsync, via
        scripts/disk_write_latency_probe.py) -- pointing to overhead
        internal to SpinVideo itself, not disk I/O or encoding. An
        SDK-independent probe (scripts/cv2_videowriter_latency_probe.py)
        found cv2.VideoWriter with an uncompressed grayscale codec
        ("GREY") writes the same real frame size in ~2.2ms.

        Constructs the "-0000.avi" suffixed path ourselves (SpinVideo used
        to append that suffix automatically) so the closer thread's
        rename/reconcile logic keeps working unchanged.
        """
        part_base = self._session_paths.video_part_base(segment_index)
        ensure_directory(part_base.parent)
        part_path = part_base.with_name(part_base.name + "-0000.avi")
        width, height = self._get_frame_dimensions()
        # MJPEG profiled at ~14-15ms/frame through cv2.VideoWriter in
        # grayscale (isColor=False) -- confirmed to actually open on the
        # one system tested, and ~34% faster than color mode (~22ms):
        # unlike the raw "DIB " tag (which failed to open in grayscale),
        # JPEG natively supports single-component grayscale, so this
        # isn't the same limitation. Still too slow for 100fps, but a
        # legitimate opt-in at lower frame rates (_use_compression) for
        # much smaller files. "GREY" (uncompressed grayscale) is the fast
        # default.
        if self._use_compression:
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        else:
            fourcc = cv2.VideoWriter_fourcc(*"GREY")
        is_color = False
        writer = self._open_cv2_writer(part_path, fourcc, width, height, is_color)
        if not writer.isOpened():
            writer.release()
            raise RuntimeError(f"cv2.VideoWriter could not open {part_path}")
        return writer

    def _open_cv2_writer(self, path: Path, fourcc: int, width: int, height: int, is_color: bool):
        """Construct the cv2.VideoWriter with the plain isColor-only constructor.

        This deliberately does NOT pass VIDEOWRITER_PROP_QUALITY /
        VIDEOWRITER_PROP_IS_COLOR via the (fourcc, fps, frameSize, params)
        overload. Measured on the rig (OpenCV 4.11.0, Windows): the FFMPEG
        backend rejects that params list ("unsupported parameters in
        VideoWriter"), and OpenCV then silently falls through to its built-in
        CV_MJPEG backend, which took ~800ms per 1280x1024 frame (vs ~10ms on
        FFMPEG) and let the append queue grow without bound. The plain
        constructor stays on FFMPEG. See scripts/multi_camera_probe.py
        (--no-quality-param) for the comparison.

        MJPEG quality is therefore FFMPEG's fixed default. It cannot be set on
        this path (writer.set(VIDEOWRITER_PROP_QUALITY) returns False and the
        file size is unchanged), so the app has no quality control. Measured on
        these cameras: about 42 dB PSNR, 10-14x smaller than raw.
        """
        writer = cv2.VideoWriter(
            str(path), fourcc, self.recording_fps, (width, height), isColor=is_color
        )
        if not self._use_compression:
            return writer  # uncompressed GREY: the validated path, unchanged

        # MJPEG must be written by the FFMPEG backend. If FFMPEG cannot open the
        # file (seen on the rig around a segment boundary, when a transient
        # filesystem error hit), OpenCV silently falls through to its built-in
        # CV_MJPEG writer, which was measured to be far slower and, for
        # grayscale input, to decode as black frames. A "working" recording with
        # black pictures is the worst outcome, so: discard that writer and retry
        # pinned to FFMPEG so it cannot fall through; refuse if that fails too.
        backend = self._writer_backend(writer)
        for attempt in range(3):
            if backend == "FFMPEG":
                return writer
            writer.release()
            self.writer_open_retries += 1
            print(
                f"{self._log_prefix} MJPEG writer for {path.name} opened on backend {backend!r}, "
                f"not FFMPEG; retrying pinned to FFMPEG (attempt {attempt + 1}/3)"
            )
            try:
                path.unlink(missing_ok=True)  # the discarded writer's partial file
            except OSError:
                pass
            time.sleep(0.1 * (attempt + 1))
            writer = cv2.VideoWriter(
                str(path), cv2.CAP_FFMPEG, fourcc, self.recording_fps, (width, height), is_color
            )
            backend = self._writer_backend(writer)
        if backend == "FFMPEG":
            return writer
        writer.release()
        raise RuntimeError(
            f"MJPEG writer for {path.name} could not be opened on the FFMPEG backend "
            f"(last backend: {backend!r}); refusing to record with a different backend"
        )

    @staticmethod
    def _writer_backend(writer) -> str:
        """Backend name of an OPEN writer, or "not opened"."""
        try:
            if not writer.isOpened():
                return "not opened"
            return writer.getBackendName()
        except Exception:
            return "unknown"

    def _maybe_rotate_segment(self) -> None:
        """Append thread only. Called after each successful Append().

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
                print(f"{self._log_prefix} could not pre-arm next segment: {exc}")

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
                print(f"{self._log_prefix} could not open segment {new_index}: {exc}")
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
                first_frame_mean=(
                    self._segment_first_frame_mean if self._frames_in_segment > 0 else None
                ),
            )
        )

    def _start_closer_thread(self) -> None:
        self._closer_thread = threading.Thread(
            target=self._closer_loop, name=f"segment-closer{self._thread_suffix}", daemon=True
        )
        self._closer_thread.start()

    def _closer_loop(self) -> None:
        while True:
            job = self._closer_queue.get()
            try:
                self._run_closer_job(job)
            except Exception as exc:
                # One bad segment must never end this thread: it is the only
                # consumer of the queue, so a dead closer silently stops every
                # later segment from being finalized and makes stop() wait out
                # its whole deadline (seen on the rig after one transient
                # WinError 3). The part file, if any, stays in .incomplete/.
                self.closer_failures += 1
                print(
                    f"{self._log_prefix} ERROR finalizing segment {job.segment_index}: "
                    f"{exc.__class__.__name__}: {exc} -- continuing; its file (if any) "
                    "stays in .incomplete/"
                )
                try:
                    self._segment_manifest_writer.submit(
                        manifest_row(
                            dataclass_replace(
                                job.manifest_entry, closed_at=time.time(), close_duration_s=0.0, bytes=0
                            )
                        )
                    )
                except Exception:
                    pass
            finally:
                self._closer_queue.task_done()

    def _run_closer_job(self, job: "_CloserJob") -> None:
        start = time.monotonic()
        try:
            job.writer.release()
        except Exception as exc:
            print(f"{self._log_prefix} error closing segment {job.segment_index}: {exc}")
        close_duration_s = time.monotonic() - start

        part_files = reconcile_part_files(job.part_base)
        total_bytes = 0
        if not part_files:
            print(
                f"{self._log_prefix} segment {job.segment_index}: no part file found "
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
                print(f"{self._log_prefix} WARNING: unexpected extra part file for segment {job.segment_index}: {extra}")
                total_bytes += self._safe_rename(extra, extra.with_name("UNEXPECTED_" + extra.name))

        entry = dataclass_replace(
            job.manifest_entry,
            closed_at=time.time(),
            close_duration_s=close_duration_s,
            bytes=total_bytes,
        )
        self._segment_manifest_writer.submit(manifest_row(entry))

        # Look at what was actually written, after the file is final and the
        # manifest row is out, so it can never delay either.
        if part_files and job.first_frame_mean is not None:
            problem = self._check_segment_pixels(job.final_path, job.first_frame_mean)
            if problem:
                self.segment_pixel_problems.append((job.segment_index, problem))
                print(f"{self._log_prefix} ERROR: segment {job.segment_index} ({job.final_path.name}) {problem}")

    @staticmethod
    def _check_segment_pixels(path: Path, recorded_mean: float) -> str | None:
        """None if the finished file decodes to roughly the picture that was recorded."""
        if recorded_mean < SEGMENT_PIXEL_CHECK_MIN_MEAN:
            return None  # too dark to tell a dark scene from a black file
        capture = cv2.VideoCapture(str(path))
        try:
            if not capture.isOpened():
                return "cannot be opened after it was finalized"
            ok, frame = capture.read()
            if not ok:
                return "has a first frame that cannot be decoded"
            decoded_mean = float(frame.mean())
        finally:
            capture.release()
        if decoded_mean < SEGMENT_PIXEL_BLACK_RATIO * recorded_mean:
            return (
                f"decodes (nearly) black: first frame mean {decoded_mean:.1f} "
                f"vs {recorded_mean:.1f} when recorded"
            )
        return None

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
                    print(f"{self._log_prefix} error finalizing {source} -> {destination}: {exc}")
                    return size
                time.sleep(0.2 * (attempt + 1))
        return size

    def _start_append_thread(self) -> None:
        self._append_thread = threading.Thread(
            target=self._append_loop, name=f"segment-appender{self._thread_suffix}", daemon=True
        )
        self._append_thread.start()

    def _append_loop(self) -> None:
        while True:
            job = self._append_queue.get()
            try:
                self._run_append_job(job)
            except Exception as exc:
                # Same reasoning as the closer loop: an unexpected error in
                # one frame's bookkeeping (e.g. a failed segment rotation) must
                # not kill the only thread that writes frames.
                print(f"{self._log_prefix} ERROR in frame append: {exc.__class__.__name__}: {exc} -- continuing")
                with self._acquisition_stats_lock:
                    self._append_failures += 1
            finally:
                self._append_queue.task_done()

    def _run_append_job(self, job: "_AppendJob") -> None:
        """Append thread only. write() + the bookkeeping/rotation that's
        contingent on it succeeding -- owns frame_counter, avi_recorder,
        and all _segment_*/_pending_writer* state from here on (see
        _append_queue's comment in __init__).

        Capture-time values (frame_id/timestamp_us/sync_*/label_*/
        captured_wall_s/captured_mono_s) come from the job instead of
        being read fresh -- they describe when the frame was captured,
        not whenever this thread got around to it. Unlike the SpinVideo
        version of this method, there's no image to Release() here --
        job.frame_array is a plain numpy array the acquisition thread
        already released the native PySpin buffer for.

        Caveat carried over from cv2.VideoWriter itself: write() doesn't
        reliably raise on failure the way SpinVideo's Append() did (e.g.
        a full disk may fail silently rather than throwing) -- the
        try/except here still catches whatever it can, but append_failures
        is not a complete backstop the way it was before.
        """
        # No color conversion needed: both codecs are opened isColor=False
        # (see _open_segment_writer) -- MJPEG grayscale was confirmed to
        # actually open and profiled faster than converting to BGR first.
        t_append_start = time.monotonic()
        try:
            self.avi_recorder.write(job.frame_array)
        except Exception as exc:
            print(f"{self._log_prefix} Error appending frame:", exc)
            with self._acquisition_stats_lock:
                self._append_failures += 1
            self._record_loop_timing(
                "append_ms", (time.monotonic() - t_append_start) * 1000.0
            )
            return

        self._record_loop_timing(
            "append_ms", (time.monotonic() - t_append_start) * 1000.0
        )

        self.frame_counter += 1
        self._frames_in_segment += 1
        if self._frames_in_segment == 1:
            self._segment_first_record_frame_index = self.frame_counter
            self._segment_first_system_time = job.captured_wall_s
            self._segment_first_frame_mean = float(job.frame_array.mean())

        if self._mark_next_frame_segment_resume:
            row_sync_label = "segment_resume"
            self._mark_next_frame_segment_resume = False
        else:
            row_sync_label = resolve_sync_label(job.label_event, job.sync_label)

        self._metadata_writer.submit(
            metadata_row(
                {
                    "record_frame_index": self.frame_counter,
                    "camera_frame_id": job.frame_id,
                    "timestamp_us": job.timestamp_us,
                    "system_time": job.captured_wall_s,
                    "sync_pulse": job.sync_this_frame,
                    "sync_label": row_sync_label,
                    "adl_id": job.adl_id,
                    "adl_label": job.adl_label,
                    "segment": self._metadata_segment,
                    "segment_file": self._session_paths.video_final(self._segment_index).name,
                    "segment_frame_index": self._frames_in_segment,
                    "monotonic_s": job.captured_mono_s,
                    "wall_mono_skew_s": compute_wall_mono_skew_s(
                        self._timeline_baseline,
                        wall_s=job.captured_wall_s,
                        mono_s=job.captured_mono_s,
                    ),
                }
            )
        )
        self._maybe_rotate_segment()

    # ------------------------------------------------------------------
    # Recording control (GUI thread): only set flags
    # ------------------------------------------------------------------

    def start_recording(self, session_paths: SessionPaths, fps: float = 30.0):
        """
        Request recording to start. The acquisition thread will
        actually open the video writer and begin appending frames.

        Equivalent to prepare_recording() followed by begin_recording(); the
        two-phase form exists so a group of cameras can open every sidecar
        first and only then let any of them start (see CameraGroup).

        Returns:
            (ok: bool, message: str)
        """
        ok, message = self.prepare_recording(session_paths, fps=fps)
        if not ok or self._prepared is None:
            return ok, message  # a refusal, or "already in progress"
        return self.begin_recording()

    def prepare_recording(self, session_paths: SessionPaths, fps: float = 30.0):
        """Phase 1 of a start: open the sidecars and reset per-session state.

        Nothing is recorded yet -- the acquisition thread only starts writing
        video after begin_recording() raises the start flag. A failure here
        leaves the controller exactly as it was, so a refusal costs nothing.
        Pair with begin_recording() (go) or abort_prepared() (cancel).
        """
        if not self.acquiring or self.cam is None:
            return False, "Cannot record: camera is not acquiring."

        if self.record_stop_requested and (self.recording_active or self.record_start_requested):
            # The previous recording is still draining/closing its last
            # segment (can take seconds for a ~3 GB file), or a start was
            # accepted and then immediately stopped before the acquisition
            # thread acted on it. Reporting success here used to leave the GUI
            # "recording" while nothing was.
            return False, "Previous recording is still closing; try again in a moment."

        if self.recording_active or self.record_start_requested:
            return True, "Recording already starting or in progress."

        if self._prepared is not None:
            return False, "A recording is already prepared; begin or abort it first."

        self.last_start_error = None
        self.segment_pixel_problems = []
        # An events log left open by an earlier failed/aborted attempt would
        # be replaced below and its handle (and, on Windows, file lock) leaked.
        if self._event_log is not None:
            try:
                self._event_log.close()
            except Exception:
                pass
            self._event_log = None

        # Everything this call creates, so a failure removes exactly that and
        # nothing belonging to an earlier session with the same name.
        created: list[Path] = []

        def fail(message: str):
            for step in (self._metadata_writer.stop, self._segment_manifest_writer.stop):
                try:
                    step()
                except Exception:
                    pass
            if self._event_log is not None:
                try:
                    self._event_log.close()
                except Exception:
                    pass
                self._event_log = None
            for path in created:
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    pass
            return False, message

        # The events log goes FIRST: it is opened with mode "x" and refuses to
        # overwrite, so a retry with a name that already exists fails here,
        # before the metadata CSV (opened with "w", which truncates) is
        # touched. Otherwise a same-name retry wiped the earlier session's
        # per-frame metadata and only then discovered the clash.
        try:
            self._event_log = JsonlEventLog(session_paths.events_jsonl)
            created.append(session_paths.events_jsonl)
            self._event_log.write(
                session_header_record(
                    mono_ns=int(time.monotonic() * 1e9),
                    wall_ns=int(time.time() * 1e9),
                    # The per-camera stem (basename + camera tag), so a tagged
                    # camera's header names ITS files; `session` is the shared
                    # basename and camera_serial/model say which camera this is.
                    recording_basename=session_paths.stem,
                    camera_serial=self.serial,
                    camera_model=self.model or None,
                    session=session_paths.basename,
                )
            )
        except Exception as exc:
            if self._event_log is None:
                created.clear()  # the open itself failed: nothing of ours exists
            return fail(f"Cannot open events log {session_paths.events_jsonl}: {exc}")

        # Open the metadata CSV synchronously, on the calling (GUI) thread,
        # so a bad output path fails the start immediately instead of
        # silently losing every frame's metadata for the whole session.
        self._metadata_writer.start(session_paths.metadata_csv)
        created.append(session_paths.metadata_csv)
        if not self._metadata_writer.wait_until_open():
            return fail(f"Cannot open metadata CSV: {self._metadata_writer.last_error}")

        self._segment_manifest_writer.start(session_paths.segments_csv)
        created.append(session_paths.segments_csv)
        if not self._segment_manifest_writer.wait_until_open():
            return fail(f"Cannot open segments CSV: {self._segment_manifest_writer.last_error}")

        # Lives for the app's lifetime, not per-session -- start them once.
        if self._closer_thread is None or not self._closer_thread.is_alive():
            self._start_closer_thread()
        if self._append_thread is None or not self._append_thread.is_alive():
            self._start_append_thread()

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
        self._timeline_baseline = TimelineBaseline(
            session_start_wall_s=time.time(),
            session_start_mono_s=time.perf_counter(),
        )

        # A label or sync window left over from before this session (or from
        # a previous one) must not be stamped on this session's first frame.
        with self._sync_lock:
            self._sync_window_end = 0.0
            self._sync_label = None
        with self._label_lock:
            self._pending_label_event = None
            self._pending_adl_id = None
            self._pending_adl_label = None

        self.recording_fps = fps
        # Reset recording frame counter
        self.frame_counter = 0
        with self._acquisition_stats_lock:
            self._append_failures = 0
            self._camera_reinits = 0

        self._prepared = session_paths
        return True, f"Recording prepared: {session_paths.stem}"

    def begin_recording(self):
        """Phase 2 of a start: let the acquisition thread open the writer."""
        if self._prepared is None:
            if self.recording_active or self.record_start_requested:
                return True, "Recording already starting or in progress."
            return False, "Nothing prepared to record."
        session_paths = self._prepared
        self._prepared = None
        self.record_start_requested = True
        self.record_stop_requested = False
        return True, f"Recording requested: {session_paths.stem}"

    def _discard_failed_start(self) -> None:
        """Acquisition thread: segment 0 could not be opened after an accepted start.

        No frame was ever written, so close the sidecars prepare_recording()
        opened and delete them. Without this they stay open (on Windows that
        also locks the files) and the next prepare would replace a still-open
        events log. The failure is left in last_start_error for the GUI/group
        to report, since the start already returned ok.
        """
        session_paths = self._session_paths
        for step in (self._metadata_writer.stop, self._segment_manifest_writer.stop):
            try:
                step()
            except Exception:
                pass
        if self._event_log is not None:
            try:
                self._event_log.close()
            except Exception:
                pass
            self._event_log = None
        self._session_paths = None
        if session_paths is not None:
            self._remove_unused_sidecars(session_paths)

    @staticmethod
    def _remove_unused_sidecars(session_paths: SessionPaths) -> None:
        """Delete sidecars opened for a session that never recorded a frame.

        They hold only a header. Leaving them would make an immediate retry
        with the same session name fail on the events log, which refuses to
        overwrite an existing file, and would leave a stray, empty session
        behind for the downstream pipeline to trip over.
        """
        for path in (
            session_paths.metadata_csv,
            session_paths.segments_csv,
            session_paths.events_jsonl,
        ):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass

    def abort_prepared(self) -> None:
        """Cancel a prepare_recording() that will not be followed by begin."""
        if self._prepared is None:
            return
        session_paths = self._prepared
        self._prepared = None
        for step in (
            self._metadata_writer.stop,
            self._segment_manifest_writer.stop,
        ):
            try:
                step()
            except Exception:
                pass
        if self._event_log is not None:
            try:
                self._event_log.close()
            except Exception:
                pass
            self._event_log = None
        self._session_paths = None
        self._remove_unused_sidecars(session_paths)

    def stop_recording(self):
        """
        Request recording to stop. The acquisition thread will
        close the video writer and write CSV.
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
        cam = image = None
        while not self._stop_event.is_set() and self.acquiring:
            # Drop last iteration's camera handle and image BEFORE anything
            # below can start a reinit: _reinitialize_camera sets self.cam to
            # None and clears the CameraList, which only works if this loop is
            # not still holding the old device object (see
            # scripts/reinit_spike.py, which drops its reference first too).
            cam = image = None
            # --------------------------------------------------
            # START recording (open the video writer) if requested
            # --------------------------------------------------
            if self.record_start_requested and not self.recording_active:
                try:
                    self.avi_recorder = self._open_segment_writer(0)
                    self.recording_active = True
                except Exception as exc:
                    print(f"{self._log_prefix} Error starting recording:", exc)
                    self.avi_recorder = None
                    self.recording_active = False
                    self.last_start_error = f"{exc.__class__.__name__}: {exc}"
                    self._discard_failed_start()
                finally:
                    self.record_start_requested = False

            # --------------------------------------------------
            # STOP recording (close the video writer + write CSV) if requested
            # --------------------------------------------------
            if self.record_stop_requested and self.recording_active:
                # Every already-queued frame (and any rotation it triggers)
                # must finish appending -- updating avi_recorder/
                # _segment_index/_frames_in_segment/etc. as it goes -- before
                # those are read below to retire the final segment.
                # recording_active only flips False further down, so the
                # acquisition thread has already stopped feeding this queue
                # by the time we get here; the wait is bounded.
                self._append_queue.join()

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
                        self._pending_writer.release()
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
                    print(f"{self._log_prefix} Error writing metadata CSV:", writer_error)

                self._segment_manifest_writer.stop()
                manifest_error = self._segment_manifest_writer.last_error
                if manifest_error:
                    print(f"{self._log_prefix} Error writing segments CSV:", manifest_error)

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
                        print(f"{self._log_prefix} Error writing session stop record:", exc)
                    try:
                        self._event_log.close()
                    except Exception as exc:
                        print(f"{self._log_prefix} Error closing events log:", exc)
                    self._event_log = None

                # Reset recording state. recording_active goes False only after
                # the paths are gone, and record_stop_requested last, so a
                # prepare_recording() landing in this window is refused rather
                # than having its paths cleared underneath it.
                self._session_paths = None
                self.recording_active = False
                self.record_stop_requested = False

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

            t_grab_start = time.monotonic()
            try:
                # grabTimeout is milliseconds (Spinnaker CameraBase::GetNextImage);
                # bounding it is what turns a silent forever-block into a
                # detectable, recoverable error instead.
                grab_timeout_ms = self._watchdog_grab_timeout_ms()
                image = cam.GetNextImage(grab_timeout_ms)
            except Exception as exc:
                cam = image = None  # may be about to be torn down by a reinit
                with self._acquisition_stats_lock:
                    self._acquisition_errors += 1
                error_decision = self._watchdog.note_error(now=time.monotonic(), error=str(exc))
                self._apply_watchdog_decision(error_decision)
                continue

            self._watchdog.note_frame_ok(now=time.monotonic())
            retrieved_at = time.monotonic()
            self._record_loop_timing("grab_ms", (retrieved_at - t_grab_start) * 1000.0)

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
            # Preview: store latest frame
            # --------------------------------------------------
            # arr, once built, is reused below as the append job's
            # payload too (see _AppendJob) instead of converting twice --
            # safe to share since nothing mutates it after this point.
            arr = None
            try:
                t_ndarray_start = time.monotonic()
                arr = image.GetNDArray()
                arr = np.array(arr, copy=True)
                published_at = time.monotonic()
                self._record_loop_timing(
                    "ndarray_ms", (published_at - t_ndarray_start) * 1000.0
                )
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
                arr = None

            # Unlike the SpinVideo version of this loop, `image` is never
            # handed to another thread -- release it here, unconditionally,
            # now that every read of it (chunk data above, GetNDArray here)
            # is done.
            image.Release()

            # --------------------------------------------------
            # If recording, hand the frame off to the append thread
            # --------------------------------------------------
            # write() profiled at ~2ms/frame for cv2.VideoWriter's
            # uncompressed grayscale codec vs. SpinVideo's Append() at a
            # ~fixed ~18ms/frame regardless of codec -- see
            # scripts/cv2_videowriter_latency_probe.py. Still handed to a
            # dedicated thread (rather than called inline) so a slow
            # frame (disk hiccup, MJPEG mode) can never block the next
            # grab.
            if arr is not None and self.recording_active and self.avi_recorder is not None:
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

                # Captured here, not on the append thread: these describe
                # when the frame was actually captured, and must not drift
                # with however far behind the append thread's queue is.
                self._append_queue.put(
                    _AppendJob(
                        frame_array=arr,
                        frame_id=frame_id,
                        timestamp_us=timestamp_us,
                        sync_this_frame=sync_this_frame,
                        sync_label=sync_label,
                        label_event=label_event,
                        adl_id=adl_id,
                        adl_label=adl_label,
                        captured_wall_s=time.time(),
                        captured_mono_s=time.perf_counter(),
                    )
                )

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
        max_size: tuple[int, int] | None = None,
    ) -> PreviewFrame | None:
        """Return a new owned preview frame, or ``None`` if it is unchanged.

        With max_size=(w, h) the image is downscaled to fit (aspect kept, never
        upscaled) BEFORE it reaches the GUI thread. Only a reference is taken
        under the lock -- the acquisition loop never mutates a published array
        (see the "Preview: store latest frame" note there) -- so the lock is
        held for microseconds instead of a full-frame copy.
        """
        with self._frame_lock:
            latest = self._latest_preview_frame
            if latest is None or (
                after_sequence is not None and latest.sequence == after_sequence
            ):
                return None
        image = latest.image
        if max_size is not None:
            height, width = image.shape[:2]
            target_w, target_h = fit_size(width, height, max_size[0], max_size[1])
            if (target_w, target_h) != (width, height):
                image = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_AREA)
            else:
                image = image.copy()
        else:
            image = image.copy()
        return PreviewFrame(
            image=image,
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

    def _record_loop_timing(self, name: str, elapsed_ms: float) -> None:
        with self._loop_timing_lock:
            self._loop_timing_samples[name].append(elapsed_ms)

    def get_and_reset_loop_timing_samples(self) -> dict[str, list[float]]:
        """Pop and clear this interval's per-stage acquisition-loop timing
        samples (grab_ms/append_ms/ndarray_ms) -- one raw sample per frame
        processed since the last call. The GUI's once-a-second diagnostics
        sampler turns these into mean/p95 (see preview_diagnostics.py).
        """
        with self._loop_timing_lock:
            samples = {name: values for name, values in self._loop_timing_samples.items()}
            for name in self._loop_timing_samples:
                self._loop_timing_samples[name] = []
        return samples

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

    def get_exposure_time_limits(self) -> tuple[float, float, float] | None:
        """(min, max, current) for ExposureTime, with `max` additionally
        capped to the current frame period (see EXPOSURE_FRAME_PERIOD_HEADROOM
        / _clamp_exposure_to_frame_period).

        The raw node max is a sensor-wide ceiling with no relation to the
        current fps -- often multiple seconds, vs. the ~10ms window that
        actually matters at 100fps -- so without this, a GUI slider built
        from get_image_param_limits("ExposureTime") spends almost its
        entire travel on values that can't sustain the current frame rate,
        making fine adjustment in the range that matters effectively
        impossible by drag.
        """
        limits = self.get_image_param_limits("ExposureTime")
        if limits is None:
            return None
        mn, mx, cur = limits
        fps = self.get_acquisition_frame_rate()
        if fps and fps > 0:
            period_max = (1_000_000.0 / fps) * EXPOSURE_FRAME_PERIOD_HEADROOM
            mx = min(mx, max(mn, period_max))
        return mn, mx, cur

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
                print(f"{self._log_prefix} set_image_param {param_name}: {exc}")
        return False

    def get_bool_param(self, param_name: str) -> bool | None:
        """Return a GenICam boolean node's current value, or None if
        unavailable (e.g. 'AcquisitionFrameRateEnable', 'TriggerMode' is an
        enum not bool -- use get_enum_param for that one).
        """
        if self.cam is None or not self.acquiring or self._recovering.is_set():
            return None
        with self._camera_lock:
            if self.cam is None:
                return None
            try:
                nodemap = self.cam.GetNodeMap()
                raw_node = nodemap.GetNode(param_name)
                if raw_node is None:
                    return None
                node = PySpin.CBooleanPtr(raw_node)
                if not PySpin.IsReadable(node):
                    return None
                return bool(node.GetValue())
            except Exception:
                return None
        return None

    def get_enum_param(self, param_name: str) -> tuple[str, list[str]] | None:
        """Return (current_entry_name, available_entry_names) for a GenICam
        enumeration node (e.g. 'ExposureAuto', 'GainAuto'), or None if
        unavailable. Used to drive the GUI's Auto/Off mode dropdowns.
        """
        if self.cam is None or not self.acquiring or self._recovering.is_set():
            return None
        with self._camera_lock:
            if self.cam is None:
                return None
            try:
                nodemap = self.cam.GetNodeMap()
                raw_node = nodemap.GetNode(param_name)
                if raw_node is None:
                    return None
                node = PySpin.CEnumerationPtr(raw_node)
                if not PySpin.IsReadable(node):
                    return None
                # GetEntries() returns raw INode pointers, not IEnumEntryPtr
                # -- .GetSymbolic() only exists on the latter, so each entry
                # must be re-wrapped with CEnumEntryPtr first.
                entries = []
                for raw_entry in node.GetEntries():
                    entry = PySpin.CEnumEntryPtr(raw_entry)
                    if PySpin.IsReadable(entry):
                        entries.append(entry.GetSymbolic())
                current = node.GetCurrentEntry().GetSymbolic()
                return current, entries
            except Exception:
                return None
        return None

    def set_enum_param(self, param_name: str, entry_name: str) -> bool:
        """Write a GenICam enumeration node (e.g. set ExposureAuto to 'Off')."""
        if self.cam is None or not self.acquiring or self._recovering.is_set():
            return False
        with self._camera_lock:
            if self.cam is None:
                return False
            try:
                nodemap = self.cam.GetNodeMap()
                raw_node = nodemap.GetNode(param_name)
                if raw_node is None:
                    return False
                node = PySpin.CEnumerationPtr(raw_node)
                if not PySpin.IsWritable(node):
                    return False
                # Same as get_enum_param: GetEntryByName() returns a raw
                # INode, not IEnumEntryPtr -- .GetValue() needs the wrap.
                raw_entry = node.GetEntryByName(entry_name)
                if raw_entry is None:
                    return False
                entry = PySpin.CEnumEntryPtr(raw_entry)
                if not PySpin.IsReadable(entry):
                    return False
                node.SetIntValue(entry.GetValue())
                return True
            except Exception as exc:
                print(f"{self._log_prefix} set_enum_param {param_name}={entry_name}: {exc}")
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

    def get_stream_rate(self) -> StreamRate | None:
        """What this camera will write per hour: frame geometry, rate, codec.

        None until the camera is acquiring (or while it is mid-recovery), so a
        caller falls back to a default estimate rather than guessing.
        """
        if self.cam is None or not self.acquiring or self._recovering.is_set():
            return None
        with self._camera_lock:
            if self.cam is None:
                return None
            try:
                nodemap = self.cam.GetNodeMap()
                width = int(PySpin.CIntegerPtr(nodemap.GetNode("Width")).GetValue())
                height = int(PySpin.CIntegerPtr(nodemap.GetNode("Height")).GetValue())
                fmt_node = PySpin.CEnumerationPtr(nodemap.GetNode("PixelFormat"))
                symbolic = fmt_node.GetCurrentEntry().GetSymbolic() if PySpin.IsReadable(fmt_node) else ""
            except Exception:
                return None
        bytes_per_pixel = 2 if "16" in symbolic else 1
        return StreamRate(
            width=width,
            height=height,
            bytes_per_pixel=bytes_per_pixel,
            fps=float(self.target_frame_rate),
            compressed=bool(self._use_compression),
        )

    def get_acquisition_frame_rate(self) -> float:
        """Current acquisition fps, or the last-known target if unreadable."""
        limits = self.get_frame_rate_limits()
        if limits is None:
            return float(self.target_frame_rate)
        return limits[2]

    def get_compression_enabled(self) -> bool:
        return self._use_compression

    def set_compression_enabled(self, enabled: bool) -> bool:
        """Choose MJPEG vs uncompressed for segment writers opened from now
        on. Takes effect at the next start_recording() or segment rotation
        -- an already-open writer keeps whatever codec it was opened with.
        Refuses while recording_active, same as set_frame_rate: changing
        the currently-open segment's codec isn't possible, and silently
        queuing the change for a future segment without the GUI's
        knowledge would be confusing.
        """
        if self.recording_active:
            return False
        self._use_compression = bool(enabled)
        return True

    def get_device_link_throughput_limit(self) -> tuple[int, int] | None:
        """Return (current, max) DeviceLinkThroughputLimit in bytes/sec, or
        None if the camera doesn't expose it.

        This caps the sustained data rate independent of AcquisitionFrameRate
        -- a GigE/USB3 Vision camera can accept and read back a fixed fps
        setting (e.g. 100) while still silently throttling actual delivery
        to whatever this limit allows, with no error surfaced anywhere.
        """
        if self.cam is None or not self.acquiring or self._recovering.is_set():
            return None
        with self._camera_lock:
            if self.cam is None:
                return None
            try:
                nodemap = self.cam.GetNodeMap()
                node = PySpin.CIntegerPtr(nodemap.GetNode("DeviceLinkThroughputLimit"))
                if node is None or not PySpin.IsReadable(node):
                    return None
                return int(node.GetValue()), int(node.GetMax())
            except Exception:
                return None
        return None

    def get_diagnostics_camera_state(self) -> dict[str, float | None]:
        """Exposure/gain/frame-rate/throughput gauges for the diagnostics CSV.

        Nothing here writes to the camera -- this is read-only, so it's
        safe to poll once a second regardless of whether ExposureAuto/
        GainAuto are enabled. The point is to catch two independent ways
        the camera can silently fall short of the requested fps:
        AcquisitionFrameRate's achievable ceiling (limits[1], "GetMax()")
        sagging below the requested rate as auto-exposure/auto-gain drift
        exposure upward over a recording (the camera can't outrun
        1 / ExposureTime, and neither control is ever set by this app --
        see gui/main.py's Gain/Gamma/BlackLevel-only controls, so whatever
        the driver defaults to governs unchecked); and
        DeviceLinkThroughputLimit capping actual delivery independent of
        AcquisitionFrameRate, which can read back the full requested value
        while still being throttled underneath; and a third: AcquisitionFrame-
        RateEnable / TriggerMode silently NOT being what this app assumes,
        which would make the camera ignore AcquisitionFrameRate entirely
        and free-run at whatever exposure+readout allows (or wait on an
        external trigger) -- indistinguishable from the outside except by
        reading these back directly, since GetValue() on the rate/exposure/
        gain nodes themselves only echoes the requested setpoint, not
        confirmation the sensor is actually obeying it.
        """
        frame_rate_limits = self.get_frame_rate_limits()
        exposure_limits = self.get_image_param_limits("ExposureTime")
        gain_limits = self.get_image_param_limits("Gain")
        throughput_limit = self.get_device_link_throughput_limit()
        frame_rate_enable = self.get_bool_param("AcquisitionFrameRateEnable")
        trigger_mode = self.get_enum_param("TriggerMode")
        return {
            "acquisition_frame_rate_fps": (
                None if frame_rate_limits is None else frame_rate_limits[2]
            ),
            "frame_rate_ceiling_fps": (
                None if frame_rate_limits is None else frame_rate_limits[1]
            ),
            "exposure_time_us": None if exposure_limits is None else exposure_limits[2],
            "gain_db": None if gain_limits is None else gain_limits[2],
            "device_link_throughput_limit_bps": (
                None if throughput_limit is None else throughput_limit[0]
            ),
            "acquisition_frame_rate_enable": frame_rate_enable,
            "trigger_mode": None if trigger_mode is None else trigger_mode[0],
            # Not camera state, but sampled alongside it for convenience:
            # direct evidence of whether the async append thread (see
            # _append_queue/_run_append_job) is keeping pace. Near 0 means
            # appends finish about as fast as frames arrive; a growing
            # value means the acquisition thread is now capturing faster
            # than Append() can drain -- proof the two are decoupled
            # (frames queueing) rather than serialized (frames dropped).
            "append_queue_depth": self._append_queue.qsize(),
        }

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
                    self._clamp_exposure_to_frame_period(actual)
                    return True
            except Exception as exc:
                print(f"{self._log_prefix} set_frame_rate: {exc}")
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
