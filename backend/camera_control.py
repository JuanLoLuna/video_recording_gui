# backend/camera_control.py
import threading
import time
import csv
import numpy as np
import PySpin

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

        # Latest frame for preview
        self._latest_frame = None
        self._frame_lock = threading.Lock()

        # Recording state/flags (thread-safe)
        self.recording_active = False          # true while SpinVideo is open
        self.record_start_requested = False    # GUI asks to start
        self.record_stop_requested = False     # GUI asks to stop

        self.avi_recorder = None
        self.recording_fps = 30.0
        self.record_filename = None
        self.metadata_records = []
        self.frame_counter = 0

        # --- Sync marker state (for CSV logging) ---
        self._sync_lock = threading.Lock()
        self._sync_window_end = 0.0  # wall-clock time until which sync_pulse=True
        self._sync_label = None  # label for the current sync window

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

            # Enable chunk metadata
            self._enable_chunk_data()

            # Acquisition mode: Continuous
            nodemap = self.cam.GetNodeMap()
            acq_mode = PySpin.CEnumerationPtr(nodemap.GetNode("AcquisitionMode"))
            continuous_entry = acq_mode.GetEntryByName("Continuous")
            acq_mode.SetIntValue(continuous_entry.GetValue())

            self.cam.BeginAcquisition()
            self.acquiring = True
            self._stop_event.clear()

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
        # If recording is active or queued, request stop and give loop time
        if self.recording_active or self.record_start_requested:
            self.record_stop_requested = True
            # wait up to ~1s for recording_active to go False
            for _ in range(100):
                if not self.recording_active:
                    break
                time.sleep(0.01)

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

    # ------------------------------------------------------------------
    # Recording control (GUI thread): only set flags
    # ------------------------------------------------------------------

    def start_recording(self, filename: str, fps: float = 30.0):
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

        self.record_filename = filename
        self.recording_fps = fps
        self.metadata_records = []
        self.record_start_requested = True
        self.record_stop_requested = False
        # Reset recording frame counter
        self.frame_counter = 0

        return True, f"Recording requested: {filename}"

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
        while (
            not self._stop_event.is_set()
            and self.acquiring
            and self.cam is not None
        ):
            # --------------------------------------------------
            # START recording (open SpinVideo) if requested
            # --------------------------------------------------
            if self.record_start_requested and not self.recording_active:
                try:
                    self.avi_recorder = PySpin.SpinVideo()
                    opt = PySpin.MJPGOption()
                    opt.frameRate = self.recording_fps
                    opt.quality = 75
                    self.avi_recorder.Open(self.record_filename, opt)
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
                try:
                    if self.avi_recorder is not None:
                        self.avi_recorder.Close()
                except Exception as exc:
                    print("Error closing recorder:", exc)

                # Write metadata CSV (simple csv module, no pandas)
                if self.record_filename and self.metadata_records:
                    csv_path = self.record_filename.rsplit(".", 1)[0] + "_metadata.csv"
                    fieldnames = [
                        "record_frame_index",
                        "camera_frame_id",
                        "timestamp_us",
                        "system_time",
                        "sync_pulse",
                        "sync_label",
                    ]

                    try:
                        with open(csv_path, "w", newline="") as f:
                            writer = csv.DictWriter(f, fieldnames=fieldnames)
                            writer.writeheader()

                            for rec in self.metadata_records:
                                if not isinstance(rec, dict):
                                    continue

                                row = {
                                    "record_frame_index": int(rec.get("record_frame_index", 0)),
                                    "camera_frame_id": (
                                        "" if rec.get("camera_frame_id") is None
                                        else int(rec.get("camera_frame_id"))
                                    ),
                                    "timestamp_us": (
                                        "" if rec.get("timestamp_us") is None
                                        else int(rec.get("timestamp_us"))
                                    ),
                                    "system_time": float(rec.get("system_time", 0.0)),
                                    "sync_pulse": bool(rec.get("sync_pulse", False)),
                                    "sync_label": "" if rec.get("sync_label") is None else str(rec.get("sync_label")),
                                }
                                writer.writerow(row)
                    except Exception as exc:
                        print("Error writing metadata CSV:", exc)

                # Reset recording state
                self.recording_active = False
                self.record_stop_requested = False
                self.avi_recorder = None
                self.metadata_records = []
                self.record_filename = None

            # --------------------------------------------------
            # Grab next frame from camera
            # --------------------------------------------------
            try:
                image = self.cam.GetNextImage()
            except Exception:
                continue

            if image.IsIncomplete():
                image.Release()
                continue

            # --------------------------------------------------
            # If recording, append frame + log metadata
            # --------------------------------------------------
            if self.recording_active and self.avi_recorder is not None:
                # Determine if this frame is within a sync window
                now = time.time()
                with self._sync_lock:
                    sync_this_frame = now <= self._sync_window_end
                    sync_label = self._sync_label if sync_this_frame else None

                # Increment only for recorded frames
                self.frame_counter += 1
                try:
                    self.avi_recorder.Append(image)
                except Exception as exc:
                    print("Error appending frame:", exc)

                # --- Check chunk data ---
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

                self.metadata_records.append(
                    {
                        "record_frame_index": int(self.frame_counter),
                        "camera_frame_id": int(frame_id) if frame_id is not None else None,
                        "timestamp_us": int(timestamp_us) if timestamp_us is not None else None,
                        "system_time": float(time.time()),
                        "sync_pulse": bool(sync_this_frame),
                        "sync_label": sync_label,
                    }
                )

            # --------------------------------------------------
            # Preview: store latest frame
            # --------------------------------------------------
            try:
                arr = image.GetNDArray()
                arr = np.array(arr, copy=True)
                with self._frame_lock:
                    self._latest_frame = arr
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
