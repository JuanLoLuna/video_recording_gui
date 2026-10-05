import atexit
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# Allow direct script execution via `python gui/main.py` by exposing the repo root.
if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[1]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)

import cv2
from PySide6.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QPushButton,
    QLabel,
    QComboBox,
    QCheckBox,
    QProgressBar,
    QScrollArea,
    QSlider,
    QDoubleSpinBox,
    QFrame,
    QMessageBox,
    QFileDialog,
)
from PySide6.QtCore import QTimer, Qt, QUrl
from PySide6.QtGui import QDesktopServices
from enum import Enum, auto
from datetime import datetime

from backend.audio_control import (
    MicLevelPreview,
    SessionAudioRecorder,
    list_audio_input_devices,
    list_audio_output_devices,
)
from backend.camera_control import (
    EXPOSURE_AUTO_LOCK_MIN_FPS,
    CameraController,
    enumerate_cameras,
)
from backend.camera_group import CameraGroup, CameraSlot
from backend.camera_registry import (
    format_camera_summary,
    intersect_ranges,
    parse_serials_env,
    select_cameras,
)
from backend.ni_control import (
    NIDaqDO,
    DOLine,
    list_do_lines,
    ni_output_enabled,
    NI_DISABLED_MESSAGE,
)
from backend.pulse_manager import PulseManager
from backend.preview_diagnostics import (
    AsyncDiagnosticsCsvLogger,
    PreviewDiagnosticsAccumulator,
)
from backend.power_status import assess_power_safety, read_power_status
from backend.power_policy import PowerPolicyState, next_power_action
from backend.recording_warnings import RecordingWarningTracker, format_duration_s
from backend.recording_paths import SessionPaths, resolve_output_dir
from backend.disk_guard import (
    assess_disk,
    estimate_bytes_per_hour,
    resolve_planned_hours,
    sample_disk_usage,
)
from backend.power_keepalive import (
    KeepAwakeRequest,
    KeepAwakeState,
    apply_keep_awake,
    assess_keepalive,
    release_keep_awake,
)

from backend.compression_policy import describe_policy, suggested_compression
from backend.calibration_store import (
    CalibrationStore,
    assess_camera,
    assess_session,
    default_calibration_dir,
    session_snapshot,
    write_session_snapshot,
)
from backend.sustained import SustainedCondition
from gui.calibration_status import CalibrationStatusBar
from gui.camera_preview import CameraPreviewTile, frame_to_qimage

SYNC_WIDTH_RECORD = 0.100  # 100 ms

# The "Compress recordings (MJPEG)" checkbox is pre-selected by
# backend/compression_policy.py (one camera: MJPEG up to 30 fps; several cameras:
# uncompressed until MJPEG is validated for them). A suggestion, never a limit.

# append_queue_depth at/above which the live compression warning fires
# (see _sample_preview_diagnostics). Chosen over an append_ms-percentile
# threshold after a real 30fps/MJPEG test: append_ms_p95 sat right at the
# frame period (even briefly over it) for the whole session -- inflated
# by OS timer-tick granularity (~15.6ms quantization observed on Windows)
# -- while append_queue_depth stayed at 0 throughout and not one frame
# was actually dropped. Queue depth is ground truth for whether the
# writer is actually falling behind capture; a latency percentile is
# only a noisy proxy for that, and a deep buffer easily absorbs
# occasional spikes p95 would flag anyway.
COMPRESSION_QUEUE_DEPTH_WARNING = 5

# EXPOSURE_AUTO_LOCK_MIN_FPS (imported from backend.camera_control) is the fps at or
# above which "Exposure mode" is force-set to Off and locked rather than left as
# a free choice; see the comment there for why.

# A writer backlog (of this many frames for any codec, or COMPRESSION_QUEUE_DEPTH_
# WARNING for MJPEG) must HOLD for this long to count. Deliberately NOT a one-
# sample threshold: a USB device plugged into the shared dock stalled the SSD for
# ~5 s on the rig (queue peaked at 150, drained by itself, nothing lost), and
# "MJPEG is falling behind, turn off compression" would have been false advice.
WRITER_BACKLOG_FRAMES = 30
WRITER_BACKLOG_SECONDS = 10.0

EXPERIMENT_LABELS = {
    "Long Term ADLs": [
        (1, "Pick up coins from purses"),
        (2, "Pick up wooden blocks"),
        (3, "Pick up nuts and put in bolts"),
        (4, "Unscrew lid of jars"),
        (5, "Cut play-doh"),
        (6, "Writing"),
        (7, "Pick up telephone and put in ear"),
        (8, "Pour water from pure pack"),
        (9, "Pour water from jug"),
        (10, "Pour water from cup"),
        (11, "Typing on smartphone"),
        (12, "Scrolling on smartphone"),
        (13, "Typing on keyboard"),
        (14, "Start sensors recording"),
        (15, "Dynamometer hand grip baseline"),
        (16, "Dynamometer hand grip active"),
    ],
    "OCD Sleeve": [
        (101, "Symptom provocation"),
        (102, "Relax"),
        (103, "Compulsion"),
        (104, "Control"),
    ],
}

@dataclass
class _SlotRuntime:
    """GUI-side state for one camera: its tile, diagnostics accumulator and CSV logger."""

    tile: CameraPreviewTile
    diagnostics: PreviewDiagnosticsAccumulator
    logger: AsyncDiagnosticsCsvLogger
    last_seq: int | None = None
    # Writer-queue conditions that only count once they have held for a while.
    backlog: SustainedCondition = field(
        default_factory=lambda: SustainedCondition(WRITER_BACKLOG_SECONDS)
    )
    mjpeg_behind: SustainedCondition = field(
        default_factory=lambda: SustainedCondition(WRITER_BACKLOG_SECONDS)
    )


class AppState(Enum):
    IDLE = auto()
    CAMERA_DETECTED = auto()
    PREVIEWING = auto()
    RECORDING = auto()

def detect_camera():
    """
    Try to find the first available camera (index 0–4).
    Print basic info to the terminal.
    Returns True if a camera is found, False otherwise.
    """
    print("=== Detecting camera ===")
    for index in range(5):
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            fps = cap.get(cv2.CAP_PROP_FPS)

            print(f"Found camera at index {index}")
            print(f"  Resolution: {int(width)} x {int(height)}")
            print(f"  FPS (reported): {fps:.2f}")

            cap.release()
            print("=== Detection done ===\n")
            return True

        cap.release()

    print("No suitable camera found.")
    print("=== Detection done ===\n")
    return False


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Camera Preview")
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        # Make the window usable on smaller displays (Windows laptops):
        # put all controls in a scrollable container so nothing is inaccessible.
        root_layout = QVBoxLayout(self)
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        root_layout.addWidget(scroll)

        content = QWidget()
        scroll.setWidget(content)
        layout = QVBoxLayout(content)

        self.status_label = QLabel("Press the button to detect a camera.")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        # --- Recording power safety (always visible, including while recording) ---
        self.power_status_frame = QFrame()
        self.power_status_frame.setObjectName("powerStatusFrame")
        power_status_layout = QHBoxLayout(self.power_status_frame)
        power_status_layout.setContentsMargins(8, 5, 8, 5)
        self.power_status_label = QLabel("Power safety: checking…")
        self.power_status_label.setWordWrap(True)
        power_status_layout.addWidget(self.power_status_label, stretch=1)
        self.open_power_settings_button = QPushButton("Open Power Settings")
        self.open_power_settings_button.clicked.connect(self._open_power_settings)
        self.open_power_settings_button.setVisible(sys.platform.startswith("win"))
        power_status_layout.addWidget(self.open_power_settings_button)
        layout.addWidget(self.power_status_frame)

        # --- Recording integrity warning (dismissible; reappears on new issues) ---
        self.recording_warning_frame = QFrame()
        self.recording_warning_frame.setObjectName("recordingWarningFrame")
        recording_warning_layout = QHBoxLayout(self.recording_warning_frame)
        recording_warning_layout.setContentsMargins(8, 5, 8, 5)
        self.recording_warning_label = QLabel("")
        self.recording_warning_label.setWordWrap(True)
        recording_warning_layout.addWidget(self.recording_warning_label, stretch=1)
        self.dismiss_recording_warning_button = QPushButton("Dismiss")
        self.dismiss_recording_warning_button.clicked.connect(
            self._on_dismiss_recording_warning_clicked
        )
        recording_warning_layout.addWidget(self.dismiss_recording_warning_button)
        self.recording_warning_frame.setVisible(False)
        layout.addWidget(self.recording_warning_frame)

        # --- Setup & options (hidden while recording) ---
        self._setup_options_expanded = True
        self.setup_options_toggle = QPushButton("▼ Setup & options")
        self.setup_options_toggle.setStyleSheet(
            "QPushButton { text-align: left; padding: 6px 8px; }"
        )
        self.setup_options_toggle.clicked.connect(self._on_setup_options_toggle)
        layout.addWidget(self.setup_options_toggle)

        self.setup_panel = QWidget()
        setup_inner = QVBoxLayout(self.setup_panel)
        setup_inner.setContentsMargins(0, 0, 0, 0)

        self.detect_button = QPushButton("Detect camera")
        self.detect_button.clicked.connect(self.on_detect_clicked)
        setup_inner.addWidget(self.detect_button)

        # --- Output folder (persists for this session only; set
        # SLEEVE_VIDEO_GUI_OUTPUT_DIR to persist across launches -- there is
        # no QSettings precedent in this repo) ---
        output_dir_row = QHBoxLayout()
        output_dir_row.addWidget(QLabel("Output folder"))
        self.output_dir_label = QLabel()
        self.output_dir_label.setWordWrap(True)
        self.output_dir_label.setStyleSheet("color: #555;")
        output_dir_row.addWidget(self.output_dir_label, stretch=1)
        self.choose_output_dir_button = QPushButton("Browse…")
        self.choose_output_dir_button.clicked.connect(self.on_choose_output_dir_clicked)
        output_dir_row.addWidget(self.choose_output_dir_button)
        setup_inner.addLayout(output_dir_row)

        # --- Collapsible camera image tuning (Spinnaker GenICam) ---
        self._camera_tuning_expanded = False
        self.camera_tuning_toggle = QPushButton(
            "▶ Camera image — exposure, gain, gamma, black level"
        )
        self.camera_tuning_toggle.setStyleSheet(
            "QPushButton { text-align: left; padding: 6px 8px; }"
        )
        self.camera_tuning_toggle.clicked.connect(self._on_camera_tuning_toggle)
        # Which camera the controls below act on (shown only with several cameras).
        self.tuning_camera_row = QWidget()
        tuning_camera_layout = QHBoxLayout(self.tuning_camera_row)
        tuning_camera_layout.setContentsMargins(0, 0, 0, 0)
        tuning_camera_layout.addWidget(QLabel("Adjust camera"))
        self.tuning_camera_combo = QComboBox()
        tuning_camera_layout.addWidget(self.tuning_camera_combo, stretch=1)
        self.tuning_camera_combo.currentIndexChanged.connect(self._on_tuning_camera_changed)
        self.tuning_camera_row.setVisible(False)
        setup_inner.addWidget(self.tuning_camera_row)

        setup_inner.addWidget(self.camera_tuning_toggle)

        self.camera_tuning_panel = QWidget()
        tuning_layout = QGridLayout(self.camera_tuning_panel)
        tuning_layout.setContentsMargins(12, 4, 8, 4)
        tuning_layout.setHorizontalSpacing(16)

        # Auto/manual mode for exposure and gain, side by side: this is what
        # actually makes the sliders below stick. With ExposureAuto/GainAuto
        # left at the camera's default (typically "Continuous"), the sensor
        # re-writes ExposureTime/Gain on its own cycle and silently overrides
        # any value the sliders set -- reported as "the Gain slider does
        # nothing." Switching to "Off" here is what hands control to the
        # sliders.
        self._auto_mode_meta: dict[str, dict] = {}
        for col, (title, nodename) in enumerate(
            (("Exposure mode", "ExposureAuto"), ("Gain mode", "GainAuto"))
        ):
            row = QHBoxLayout()
            row.addWidget(QLabel(title))
            combo = QComboBox()
            combo.setEnabled(False)
            row.addWidget(combo, stretch=1)
            container = QWidget()
            container.setLayout(row)
            tuning_layout.addWidget(container, 0, col)
            self._auto_mode_meta[nodename] = {"combo": combo}
            combo.currentTextChanged.connect(
                lambda text, n=nodename: self._on_auto_mode_changed(n, text)
            )

        # Sliders in two columns to keep the panel compact.
        self._image_slider_resolution = 1000
        self._slider_meta: dict[str, dict] = {}
        slider_params = (
            ("Exposure (µs)", "ExposureTime"),
            ("Gain", "Gain"),
            ("Gamma", "Gamma"),
            ("Black level", "BlackLevel"),
        )
        for i, (title, nodename) in enumerate(slider_params):
            row = QHBoxLayout()
            row.addWidget(QLabel(title))
            sl = QSlider(Qt.Orientation.Horizontal)
            sl.setRange(0, self._image_slider_resolution)
            sl.setEnabled(False)
            val_lbl = QLabel("—")
            val_lbl.setMinimumWidth(64)
            val_lbl.setAlignment(
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
            )
            row.addWidget(sl, stretch=1)
            row.addWidget(val_lbl)

            # ExposureTime's usable range is bounded by the current frame
            # rate (get_exposure_time_limits), not the sensor's raw max --
            # explain that here, since otherwise "why did my range shrink"
            # is non-obvious.
            range_caption = None
            if nodename == "ExposureTime":
                container_layout = QVBoxLayout()
                container_layout.setContentsMargins(0, 0, 0, 0)
                container_layout.setSpacing(0)
                container_layout.addLayout(row)
                range_caption = QLabel("")
                range_caption.setWordWrap(True)
                range_caption.setStyleSheet("color: #777; font-size: 10px;")
                container_layout.addWidget(range_caption)
                container = QWidget()
                container.setLayout(container_layout)
            else:
                container = QWidget()
                container.setLayout(row)
            tuning_layout.addWidget(container, 1 + i // 2, i % 2)
            self._slider_meta[nodename] = {
                "slider": sl,
                "label": val_lbl,
                "range_caption": range_caption,
                "min": 0.0,
                "max": 1.0,
                "steps": self._image_slider_resolution,
                "supported": False,
            }
            sl.valueChanged.connect(
                lambda v, n=nodename: self._on_image_slider_changed(n, v)
            )

        hint_row = 1 + (len(slider_params) + 1) // 2
        self.camera_tuning_hint = QLabel(
            "Start preview to enable these controls (requires Spinnaker / GenICam nodes)."
        )
        self.camera_tuning_hint.setWordWrap(True)
        self.camera_tuning_hint.setStyleSheet("color: #555; font-size: 11px;")
        tuning_layout.addWidget(self.camera_tuning_hint, hint_row, 0, 1, 2)
        self.camera_tuning_panel.setVisible(False)
        setup_inner.addWidget(self.camera_tuning_panel)

        # --- Acquisition frame rate (fps) ---
        # Drives both the camera sensor rate and the AVI playback rate, so they
        # stay matched. Range/current are read from the camera once preview
        # starts; locked while recording to avoid changing capture speed mid-file.
        fps_row = QHBoxLayout()
        fps_row.addWidget(QLabel("Frame rate"))
        self.frame_rate_spin = QDoubleSpinBox()
        self.frame_rate_spin.setDecimals(1)
        self.frame_rate_spin.setSingleStep(1.0)
        self.frame_rate_spin.setRange(1.0, 1000.0)
        self.frame_rate_spin.setValue(30.0)
        self.frame_rate_spin.setSuffix(" fps")
        self.frame_rate_spin.setEnabled(False)
        self.frame_rate_spin.valueChanged.connect(self._on_frame_rate_changed)
        fps_row.addWidget(self.frame_rate_spin)
        fps_row.addSpacing(20)

        # --- Video compression (MJPEG vs uncompressed) ---
        # Uncompressed is the safe fallback: MJPEG's encoding cost
        # profiled at ~21-22ms/frame on one test machine (cv2.VideoWriter's
        # MJPG codec), enough by itself to cap throughput well under
        # 100fps there. Defaults to
        # checked/unchecked based on the current fps (see
        # _apply_compression_default_for_fps) -- a starting suggestion,
        # not enforced: _sample_preview_diagnostics warns live (via the
        # append_queue_depth diagnostic) if it turns out too slow for
        # whatever fps is actually chosen, regardless of this checkbox's
        # state.
        # True once the user has clicked the checkbox directly (as opposed
        # to it being set programmatically by the fps-based default) --
        # from then on, _apply_compression_default_for_fps leaves it alone.
        self._compression_manually_set = False
        # True whenever fps >= EXPOSURE_AUTO_LOCK_MIN_FPS -- unlike
        # _compression_manually_set, there's no user override for this one:
        # see _apply_exposure_auto_lock_for_fps.
        self._exposure_auto_locked = False
        self.compression_checkbox = QCheckBox("Compress recordings (MJPEG)")
        self.compression_checkbox.setToolTip(
            "MJPEG at a fixed quality (about 42 dB PSNR measured on these cameras; the "
            "study's existing video is MJPEG too). Unticked = uncompressed 8-bit "
            "grayscale: lossless, but roughly 10-14x larger and needs a fast disk."
        )
        self.compression_checkbox.setEnabled(False)
        self.compression_checkbox.toggled.connect(self._on_compression_toggled)
        fps_row.addWidget(self.compression_checkbox)

        fps_row.addStretch(1)
        setup_inner.addLayout(fps_row)

        self.frame_rate_hint = QLabel(
            "Start preview to read the camera's supported range and current rate."
        )
        self.frame_rate_hint.setWordWrap(True)
        self.frame_rate_hint.setStyleSheet("color: #555; font-size: 11px;")
        setup_inner.addWidget(self.frame_rate_hint)

        self.compression_hint = QLabel(describe_policy(1))
        self.compression_hint.setWordWrap(True)
        self.compression_hint.setStyleSheet("color: #555; font-size: 11px;")
        setup_inner.addWidget(self.compression_hint)

        # --- Sync / NI-DAQ status ---
        self.sync_label = QLabel("Sync not available — no DAQ connected")
        setup_inner.addWidget(self.sync_label)

        daq_row = QHBoxLayout()
        self.daq_line_combo = QComboBox()
        self.daq_line_combo.addItem("No lines detected", None)
        self.daq_line_combo.setMinimumWidth(200)
        daq_row.addWidget(self.daq_line_combo)

        self.scan_daq_button = QPushButton("Scan")
        self.scan_daq_button.clicked.connect(self.on_scan_daq_clicked)
        daq_row.addWidget(self.scan_daq_button)

        self.connect_daq_button = QPushButton("Connect")
        self.connect_daq_button.setEnabled(False)
        self.connect_daq_button.clicked.connect(self.on_connect_daq_clicked)
        daq_row.addWidget(self.connect_daq_button)

        setup_inner.addLayout(daq_row)

        # --- Microphone (optional WAV alongside video) ---
        self.audio_label = QLabel("Microphone: scan and choose a device, or leave as no audio.")
        setup_inner.addWidget(self.audio_label)
        audio_row = QHBoxLayout()
        self.audio_input_combo = QComboBox()
        self.audio_input_combo.setMinimumWidth(280)
        self.audio_input_combo.addItem("No audio", None)
        self.audio_input_combo.currentIndexChanged.connect(
            self._on_audio_input_device_changed
        )
        audio_row.addWidget(self.audio_input_combo)
        self.scan_audio_button = QPushButton("Scan")
        self.scan_audio_button.clicked.connect(self.on_scan_audio_clicked)
        audio_row.addWidget(self.scan_audio_button)
        setup_inner.addLayout(audio_row)

        self.audio_monitor_checkbox = QCheckBox(
            "Hear live mic in headphones (Preview mic + recording; avoids speaker feedback)"
        )
        self.audio_monitor_checkbox.setChecked(True)
        self.audio_monitor_checkbox.stateChanged.connect(self._on_audio_monitor_changed)
        self.mic_preview_button = QPushButton("Preview mic")
        self.mic_preview_button.clicked.connect(self.on_mic_preview_clicked)
        monitor_preview_row = QHBoxLayout()
        monitor_preview_row.addWidget(self.audio_monitor_checkbox, stretch=1)
        monitor_preview_row.addWidget(self.mic_preview_button)
        setup_inner.addLayout(monitor_preview_row)

        output_row = QHBoxLayout()
        output_row.addWidget(QLabel("Audio Output"))
        self.audio_output_combo = QComboBox()
        self.audio_output_combo.setMinimumWidth(280)
        self.audio_output_combo.addItem("Default output", None)
        self.audio_output_combo.currentIndexChanged.connect(
            self._on_audio_output_device_changed
        )
        output_row.addWidget(self.audio_output_combo)
        self.scan_output_button = QPushButton("Scan")
        self.scan_output_button.clicked.connect(self.on_scan_output_clicked)
        output_row.addWidget(self.scan_output_button)
        setup_inner.addLayout(output_row)

        layout.addWidget(self.setup_panel)

        # --- Session bar (always visible): transport, level, ADL — stays compact while recording ---
        self.session_bar = QWidget()
        session_layout = QVBoxLayout(self.session_bar)
        session_layout.setContentsMargins(0, 4, 0, 0)
        session_layout.setSpacing(6)

        # Large blinking "recording" cue — hard to miss even from across the room.
        self.rec_indicator_label = QLabel("● RECORDING")
        self.rec_indicator_label.setAlignment(Qt.AlignCenter)
        self.rec_indicator_label.setStyleSheet(
            "QLabel { color: white; background-color: #c62828; border-radius: 4px;"
            " font-size: 20px; font-weight: bold; padding: 6px; }"
        )
        self.rec_indicator_label.setVisible(False)
        session_layout.addWidget(self.rec_indicator_label)
        self._rec_blink_on = True
        self.rec_blink_timer = QTimer(self)
        self.rec_blink_timer.setInterval(600)
        self.rec_blink_timer.timeout.connect(self._toggle_rec_blink)

        self.preview_button = QPushButton("Start Preview")
        self.preview_button.clicked.connect(self.on_preview_clicked)

        self.record_button = QPushButton("Start Recording")
        self.record_button.clicked.connect(self.on_record_clicked)

        self.sync_button = QPushButton("Sync Pulse")
        self.sync_button.setEnabled(False)
        self.sync_button.clicked.connect(self.on_sync_pulse_clicked)

        controls_row = QHBoxLayout()
        controls_row.addWidget(self.preview_button)
        controls_row.addWidget(self.record_button)
        controls_row.addWidget(self.sync_button)
        session_layout.addLayout(controls_row)

        level_row = QHBoxLayout()
        level_row.addWidget(QLabel("Mic level"))
        self.audio_level_bar = QProgressBar()
        self.audio_level_bar.setRange(0, 100)
        self.audio_level_bar.setValue(0)
        self.audio_level_bar.setTextVisible(False)
        self.audio_level_bar.setFixedHeight(14)
        self.audio_level_bar.setStyleSheet(
            "QProgressBar { border: 1px solid #888; border-radius: 3px; background: #e8e8e8; }"
            "QProgressBar::chunk { background-color: #2e7d32; border-radius: 2px; }"
        )
        level_row.addWidget(self.audio_level_bar, stretch=1)
        session_layout.addLayout(level_row)

        # Disable DAQ controls on non-Windows
        if not sys.platform.startswith("win"):
            self.scan_daq_button.setEnabled(False)
            self.connect_daq_button.setEnabled(False)
            self.sync_label.setText("Sync not available on this OS")

        # Handle to the DAQ controller (set on connect)
        self.daq = None
        self.pulse_manager = None
        self._session_audio: SessionAudioRecorder | None = None
        self._mic_preview: MicLevelPreview | None = None
        self._mic_preview_active = False

        self.mic_level_timer = QTimer(self)
        self.mic_level_timer.timeout.connect(self._update_mic_level_bar)
        self.mic_level_timer.start(50)

        # --- Image preview: one tile per detected camera (stretches between
        # setup and session bar). Before Detect there is a single empty tile. ---
        self.preview_area = QWidget()
        self.preview_layout = QHBoxLayout(self.preview_area)
        self.preview_layout.setContentsMargins(0, 0, 0, 0)
        self.preview_layout.addWidget(CameraPreviewTile())
        layout.addWidget(self.preview_area, stretch=1)

        self.preview_health_label = QLabel("Preview pipeline: waiting for preview")
        self.preview_health_label.setStyleSheet("color: #555; font-size: 11px;")
        self.preview_health_label.setWordWrap(True)  # two cameras' status can be long
        self.preview_health_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        layout.addWidget(self.preview_health_label)

        # --- 3D calibration status (informational; never blocks recording) ---
        self._calibration_store = CalibrationStore(default_calibration_dir())
        self._cameras_detected_at: datetime | None = None
        # Filled by the Calibration window's live fixed-board check: serial ->
        # camera moved since setup (bool) / board reprojection px with the
        # stored intrinsics. Empty = not measured.
        self._calibration_moved: dict[str, bool] = {}
        self._calibration_live_rms: dict[str, float] = {}
        self.calibration_status = None
        self.calibration_bar = CalibrationStatusBar()
        self.calibration_bar.calibrate_button.clicked.connect(self._open_calibration_window)
        layout.addWidget(self.calibration_bar)

        # --- ADL label strip (session bar) ---
        self._label_frame_base_style = (
            "QFrame#labelMarkerFrame { border: 1px solid #bbb; border-radius: 4px; "
            "padding: 4px 8px; background: #f0f0f0; }"
        )
        self.label_marker_frame = QFrame()
        self.label_marker_frame.setObjectName("labelMarkerFrame")
        self.label_marker_frame.setStyleSheet(self._label_frame_base_style)
        self.label_marker_frame.setMaximumHeight(40)
        marker_layout = QHBoxLayout(self.label_marker_frame)
        marker_layout.setContentsMargins(6, 2, 6, 2)
        self.label_last_event = QLabel("—")
        self.label_last_event.setStyleSheet("font-weight: 600;")
        self.label_last_event.setWordWrap(False)
        marker_layout.addWidget(self.label_last_event, stretch=1)
        session_layout.addWidget(self.label_marker_frame)

        # Label-pair feedback is kept per label, so changing the dropdown shows
        # the completed-pair count for that specific label.
        self._label_pair_counts: dict[int, int] = {}
        self._open_label_ids: set[int] = set()

        adl_row = QHBoxLayout()
        self.experiment_dropdown = QComboBox()
        for experiment_name in EXPERIMENT_LABELS:
            self.experiment_dropdown.addItem(experiment_name)
        self.experiment_dropdown.currentIndexChanged.connect(self.on_experiment_changed)
        adl_row.addWidget(self.experiment_dropdown)

        self.adl_dropdown = QComboBox()
        self.adl_dropdown.currentIndexChanged.connect(self._update_label_pair_counter)
        adl_row.addWidget(self.adl_dropdown)
        self.label_pair_counter = QLabel("Completed pairs: 0")
        self.label_pair_counter.setStyleSheet("font-weight: 600;")
        self.label_pair_counter.setToolTip(
            "Number of completed start/end pairs for the selected label in this recording."
        )
        adl_row.addWidget(self.label_pair_counter)
        self._populate_label_dropdown(self.experiment_dropdown.currentText())
        session_layout.addLayout(adl_row)

        layout.addWidget(self.session_bar)

        # --- Cameras: Detect builds one CameraController per camera, grouped. ---
        # Inert until then; it only keeps accessors like self.camera safe to
        # call before any camera has been detected.
        self._placeholder_camera = CameraController()
        self.cameras: CameraGroup | None = None
        self._slot_rt: dict[str, _SlotRuntime] = {}
        self._tuning_serial: str | None = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.preview_running = False

        self._recording_warnings = RecordingWarningTracker()

        # Output folder: None means "let resolve_output_dir() pick" (env
        # var, then cwd). Set by on_choose_output_dir_clicked; not
        # persisted across launches -- use SLEEVE_VIDEO_GUI_OUTPUT_DIR for
        # that.
        self._configured_output_dir: str | None = None
        self._refresh_output_dir_label()
        self.preview_diagnostics_timer = QTimer(self)
        self.preview_diagnostics_timer.timeout.connect(
            self._sample_preview_diagnostics
        )

        self._current_power_status = read_power_status()
        self._power_assessment = assess_power_safety(self._current_power_status)
        self._power_policy_state = PowerPolicyState()
        self._power_paused = False

        # Sleep prevention while recording (Windows-only; no-op elsewhere).
        self._keepalive_request = KeepAwakeRequest(system=True, display=False, away_mode=True)
        self._keepalive_state = KeepAwakeState(supported=sys.platform.startswith("win"))

        self.state = AppState.IDLE
        self._apply_state()

        self.power_status_timer = QTimer(self)
        self.power_status_timer.timeout.connect(self._refresh_power_status)
        self.power_status_timer.start(500)
        self._refresh_power_status()

        # count manual sync pulses during a run
        self.manual_sync_count = 0

    def on_scan_audio_clicked(self):
        """List host audio input devices (external mic, etc.)."""
        if self._mic_preview_active:
            self._stop_mic_preview()
        self.audio_input_combo.clear()
        devices = list_audio_input_devices()
        self.audio_input_combo.addItem("No audio", None)
        for idx, label in devices:
            self.audio_input_combo.addItem(label, idx)
        if devices:
            self.audio_label.setText(f"Microphone: {len(devices)} input device(s) found.")
        else:
            self.audio_label.setText(
                "Microphone: none found (install sounddevice/soundfile, or check mic permissions)."
            )
        self._apply_state()

    def on_scan_output_clicked(self):
        """List host audio output devices (headphones, speakers, etc.)."""
        if self._mic_preview_active:
            self._stop_mic_preview()
        self.audio_output_combo.clear()
        devices = list_audio_output_devices()
        self.audio_output_combo.addItem("Default output", None)
        for idx, label in devices:
            self.audio_output_combo.addItem(label, idx)
        # Keep existing label text; just refresh state.
        self._apply_state()

    def _populate_label_dropdown(self, experiment_name: str):
        self.adl_dropdown.clear()
        self.adl_dropdown.addItem("Select label...", None)
        for adl_id, adl_label in EXPERIMENT_LABELS.get(experiment_name, []):
            self.adl_dropdown.addItem(adl_label, adl_id)

    def on_experiment_changed(self, _index=None):
        self._populate_label_dropdown(self.experiment_dropdown.currentText())

    def _update_label_pair_counter(self, _index=None) -> None:
        """Show the completed start/end-pair count for the selected label."""
        adl_id = self.adl_dropdown.currentData()
        count = self._label_pair_counts.get(adl_id, 0) if adl_id is not None else 0
        self.label_pair_counter.setText(f"Completed pairs: {count}")

    def _apply_state(self):
        if self.state == AppState.IDLE:
            self.detect_button.setEnabled(True)
            self.preview_button.setEnabled(False)
            self.preview_button.setText("Start Preview")
            self.record_button.setEnabled(False)
            self.record_button.setText("Start Recording")

        elif self.state == AppState.CAMERA_DETECTED:
            self.detect_button.setEnabled(True)
            self.preview_button.setEnabled(True)
            self.preview_button.setText("Start Preview")
            self.record_button.setEnabled(False)
            self.record_button.setText("Start Recording")

        elif self.state == AppState.PREVIEWING:
            self.detect_button.setEnabled(False)
            self.preview_button.setText("Stop Preview")
            if self._power_paused:
                # Stopping preview here would drop state to CAMERA_DETECTED,
                # and auto-resume only fires from PREVIEWING -- an accidental
                # click would silently strand the run with no way to resume.
                self.preview_button.setEnabled(False)
                self.record_button.setEnabled(False)
                self.record_button.setText("Paused for power safety — resuming automatically")
            else:
                self.preview_button.setEnabled(True)
                self.record_button.setEnabled(True)
                self.record_button.setText("Start Recording")

        elif self.state == AppState.RECORDING:
            self.detect_button.setEnabled(False)
            # While recording, keep preview button disabled
            self.preview_button.setEnabled(False)
            self.preview_button.setText("Stop Preview")
            self.record_button.setEnabled(True)
            self.record_button.setText("Stop Recording")
            self._stop_mic_preview()
            self.audio_input_combo.setEnabled(False)
            self.scan_audio_button.setEnabled(False)
            self.audio_monitor_checkbox.setEnabled(False)
            self.mic_preview_button.setEnabled(False)

        if self.state != AppState.RECORDING:
            self.audio_monitor_checkbox.setEnabled(True)
            if self._mic_preview_active:
                self.audio_input_combo.setEnabled(False)
                self.scan_audio_button.setEnabled(False)
                self.mic_preview_button.setEnabled(True)
                self.mic_preview_button.setText("Stop mic preview")
                self.audio_output_combo.setEnabled(False)
                self.scan_output_button.setEnabled(False)
            else:
                self.audio_input_combo.setEnabled(True)
                self.scan_audio_button.setEnabled(True)
                has_dev = self.audio_input_combo.currentData() is not None
                self.mic_preview_button.setEnabled(has_dev)
                self.mic_preview_button.setText("Preview mic")
                want_monitor = self.audio_monitor_checkbox.isChecked()
                self.audio_output_combo.setEnabled(want_monitor)
                self.scan_output_button.setEnabled(want_monitor)

        self._update_camera_tuning_widgets_enabled()
        self._update_frame_rate_widget_enabled()
        self._update_recording_chrome()
        self._apply_power_safety_to_controls()
        self.calibration_bar.calibrate_button.setEnabled(
            self.cameras is not None and self.state in (AppState.CAMERA_DETECTED, AppState.PREVIEWING)
        )

    def _open_power_settings(self) -> None:
        """Open Windows Power & battery settings without changing them."""
        if not sys.platform.startswith("win"):
            return
        if not QDesktopServices.openUrl(QUrl("ms-settings:powersleep")):
            self.status_label.setText("Could not open Windows Power Settings.")

    def _update_power_status_display(self) -> None:
        colors = {
            "safe": ("#e8f5e9", "#2e7d32"),
            "warning": ("#fff8e1", "#e65100"),
            "danger": ("#ffebee", "#b71c1c"),
            "neutral": ("#f0f0f0", "#555"),
        }
        background, foreground = colors.get(
            self._power_assessment.level,
            colors["neutral"],
        )
        keepalive_assessment = assess_keepalive(
            self._keepalive_state, recording=self.state == AppState.RECORDING
        )
        text = self._power_assessment.summary
        tooltip = self._power_assessment.reason or ""
        if self._power_paused:
            text = f"{text} — recording paused, will resume automatically"
            background, foreground = colors["danger"]
        if self.state == AppState.RECORDING:
            text = f"{text} — {keepalive_assessment.summary}"
            if keepalive_assessment.reason:
                tooltip = (tooltip + "\n" if tooltip else "") + keepalive_assessment.reason
            # A "danger"/"warning" keepalive state during recording should be
            # visible even when the power assessment itself reads "safe".
            level_rank = {"safe": 0, "neutral": 0, "warning": 1, "danger": 2}
            if level_rank.get(keepalive_assessment.level, 0) > level_rank.get(
                self._power_assessment.level, 0
            ):
                background, foreground = colors.get(
                    keepalive_assessment.level, colors["neutral"]
                )
        self.power_status_label.setText(text)
        self.power_status_label.setToolTip(tooltip)
        self.power_status_frame.setStyleSheet(
            "QFrame#powerStatusFrame {"
            f" background: {background}; border: 1px solid {foreground};"
            " border-radius: 4px; }"
            f"QLabel {{ color: {foreground}; font-weight: 600; }}"
        )

    def _apply_power_safety_to_controls(self) -> None:
        """Gate recording starts without disabling the active Stop button."""
        if not hasattr(self, "record_button"):
            return
        reason = self._power_assessment.reason or ""
        if self.state == AppState.PREVIEWING:
            self.record_button.setEnabled(not self._power_assessment.recording_blocked)
            self.record_button.setToolTip(reason)
        elif self.state == AppState.RECORDING:
            self.record_button.setEnabled(True)
            self.record_button.setToolTip("Stop the current recording.")
        else:
            self.record_button.setToolTip(reason)

    def _refresh_power_status(self) -> None:
        """Refresh the banner and safely stop if power becomes recording-unsafe."""
        self._current_power_status = read_power_status()
        self._power_assessment = assess_power_safety(self._current_power_status)

        # Self-heal sleep prevention: some intervening OS/driver event can
        # clear a prior SetThreadExecutionState request. Re-assert while
        # recording, at negligible cost, rather than leaving it lapsed.
        if self.state == AppState.RECORDING and not self._keepalive_state.active:
            self._keepalive_state = apply_keep_awake(self._keepalive_request)

        self._update_power_status_display()
        self._apply_power_safety_to_controls()

        self._power_policy_state, decision = next_power_action(
            self._power_policy_state,
            recording_blocked=self._power_assessment.recording_blocked,
            recording=self.state == AppState.RECORDING,
            reason=self._power_assessment.reason or self._power_assessment.summary,
        )

        if decision.action == "pause":
            self._power_paused = True
            print(
                f"[power] {datetime.now().isoformat()} pausing recording "
                f"automatically: {decision.reason}"
            )
            self._stop_recording_session(
                f"Recording paused automatically: {decision.reason}"
            )
        elif decision.action == "resume":
            self._power_paused = False
            print(
                f"[power] {datetime.now().isoformat()} resuming recording "
                "automatically after sustained-safe power"
            )
            if not self._begin_recording_session(bypass_confirmation=True):
                # Couldn't actually start a new session (camera/disk issue) --
                # go back to waiting for the next sustained-safe window
                # instead of silently giving up for the rest of the run.
                print(
                    f"[power] {datetime.now().isoformat()} resume attempt "
                    "failed to start a new session; will retry"
                )
                self._power_policy_state = PowerPolicyState(paused=True)
                self._power_paused = True
                self._apply_state()

    def _confirm_power_safe_to_record(self) -> bool:
        """Re-read power immediately before recording and request consent if low."""
        self._current_power_status = read_power_status()
        self._power_assessment = assess_power_safety(self._current_power_status)
        self._update_power_status_display()
        self._apply_power_safety_to_controls()

        if self._power_assessment.recording_blocked:
            QMessageBox.warning(
                self,
                "Recording blocked by power safety",
                self._power_assessment.reason or self._power_assessment.summary,
            )
            return False
        if not self._power_assessment.requires_confirmation:
            return True

        choice = QMessageBox.question(
            self,
            "Low battery warning",
            f"{self._power_assessment.summary}\n\n"
            f"{self._power_assessment.reason}\n\nStart recording anyway?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        return choice == QMessageBox.StandardButton.Yes

    def on_choose_output_dir_clicked(self) -> None:
        current = self._configured_output_dir or str(resolve_output_dir(create=False))
        chosen = QFileDialog.getExistingDirectory(self, "Choose output folder", current)
        if chosen:
            self._configured_output_dir = chosen
            self._refresh_output_dir_label()

    def _refresh_output_dir_label(self) -> None:
        # create=False: just show where a recording would land, without
        # side-effecting the filesystem merely by displaying it.
        shown = self._configured_output_dir or str(resolve_output_dir(create=False))
        self.output_dir_label.setText(shown)
        self.output_dir_label.setToolTip(shown)

    def _estimated_bytes_per_hour(self) -> int:
        """Sum of every camera's real frame size x fps (default rate if none is readable)."""
        rates = [r for r in (s.controller.get_stream_rate() for s in self._slots()) if r is not None]
        return estimate_bytes_per_hour(rates)

    def _confirm_disk_safe_to_record(
        self, output_dir, *, bypass_confirmation: bool = False
    ) -> bool:
        """Refuse (or confirm) a run whose planned duration can't fit.

        bypass_confirmation is set for the automatic power-resume path: a
        hard block still refuses to start, but a soft low-disk warning that
        would normally ask a human is accepted rather than blocking an
        unattended resume on a dialog nobody is there to answer.
        """
        try:
            sample = sample_disk_usage(output_dir, at_s=time.monotonic())
        except OSError as exc:
            if not bypass_confirmation:
                QMessageBox.warning(
                    self,
                    "Could not check free disk space",
                    f"Continuing without a disk-space check.\n\n{exc}",
                )
            return True

        verdict = assess_disk(
            sample,
            bytes_per_hour=self._estimated_bytes_per_hour(),
            planned_hours=resolve_planned_hours(),
        )
        if verdict.recording_blocked:
            if not bypass_confirmation:
                QMessageBox.warning(self, "Recording blocked by disk space", verdict.reason)
            return False
        if verdict.requires_confirmation:
            if bypass_confirmation:
                return True
            title = (
                "Low disk space warning" if verdict.level == "warning"
                else "Disk space may not be enough"
            )
            choice = QMessageBox.question(
                self,
                title,
                f"{verdict.reason}\n\nStart recording anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            return choice == QMessageBox.StandardButton.Yes
        return True

    def _refresh_setup_options_toggle_text(self) -> None:
        if not hasattr(self, "setup_options_toggle"):
            return
        arrow = "▼" if self._setup_options_expanded else "▶"
        self.setup_options_toggle.setText(f"{arrow} Setup & options")

    def _on_setup_options_toggle(self) -> None:
        if self.state == AppState.RECORDING:
            return
        self._setup_options_expanded = not self._setup_options_expanded
        self.setup_panel.setVisible(self._setup_options_expanded)
        self._refresh_setup_options_toggle_text()

    def _update_recording_chrome(self) -> None:
        """Hide setup chrome while recording; keep session bar (video, transport, ADL)."""
        if not hasattr(self, "session_bar"):
            return
        rec = self.state == AppState.RECORDING
        if rec:
            self.setup_options_toggle.setVisible(False)
            self.setup_panel.setVisible(False)
            self.preview_button.setVisible(False)
            self.status_label.setMaximumHeight(72)
        else:
            self.setup_options_toggle.setVisible(True)
            self.setup_panel.setVisible(self._setup_options_expanded)
            self.preview_button.setVisible(True)
            self.status_label.setMaximumHeight(16_777_215)
        self._refresh_setup_options_toggle_text()

        if rec:
            self.setWindowTitle("● REC — Camera Preview")
            self._rec_blink_on = True
            self.rec_indicator_label.setVisible(True)
            if not self.rec_blink_timer.isActive():
                self.rec_blink_timer.start()
        else:
            self.setWindowTitle("Camera Preview")
            self.rec_blink_timer.stop()
            self.rec_indicator_label.setVisible(False)

    def _toggle_rec_blink(self) -> None:
        """Blink the REC banner so it stays noticeable even in peripheral vision."""
        self._rec_blink_on = not self._rec_blink_on
        self.rec_indicator_label.setStyleSheet(
            "QLabel { color: white; background-color: %s; border-radius: 4px;"
            " font-size: 20px; font-weight: bold; padding: 6px; }"
            % ("#c62828" if self._rec_blink_on else "#7a1414")
        )

    def _on_camera_tuning_toggle(self) -> None:
        self._camera_tuning_expanded = not self._camera_tuning_expanded
        self.camera_tuning_panel.setVisible(self._camera_tuning_expanded)
        arrow = "▼" if self._camera_tuning_expanded else "▶"
        self.camera_tuning_toggle.setText(
            f"{arrow} Camera image — exposure, gain, gamma, black level"
        )

    def _on_auto_mode_changed(self, node: str, entry_name: str) -> None:
        meta = self._auto_mode_meta.get(node)
        if meta is None or not meta["combo"].isEnabled() or not entry_name:
            return
        self.camera.set_enum_param(node, entry_name)
        # Auto mode governs the paired slider (ExposureAuto -> ExposureTime,
        # GainAuto -> Gain): re-sync so it reflects whatever the camera
        # actually holds now, and is only editable when mode is "Off".
        self._sync_image_sliders_from_camera()

    def _on_image_slider_changed(self, node: str, slider_value: int) -> None:
        meta = self._slider_meta.get(node)
        if meta is None:
            return
        sl = meta["slider"]
        if not sl.isEnabled():
            return
        mn, mx = float(meta["min"]), float(meta["max"])
        steps = float(meta["steps"])
        if mx <= mn or steps <= 0:
            return
        value = mn + (slider_value / steps) * (mx - mn)
        meta["label"].setText(f"{value:.4g}")
        self.camera.set_image_param(node, value)

    # Sliders whose value the camera will silently override while the
    # paired Auto mode isn't "Off" -- see _on_auto_mode_changed.
    _AUTO_MODE_FOR_SLIDER = {"ExposureTime": "ExposureAuto", "Gain": "GainAuto"}

    def _slider_can_tune(self, node: str, acquiring: bool) -> bool:
        if not acquiring:
            return False
        auto_node = self._AUTO_MODE_FOR_SLIDER.get(node)
        if auto_node is None:
            return True
        auto_meta = self._auto_mode_meta.get(auto_node)
        if auto_meta is None or not auto_meta["combo"].isEnabled():
            return True
        return auto_meta["combo"].currentText() == "Off"

    def _sync_image_sliders_from_camera(self) -> None:
        """Read limits from the open camera and align sliders (call after preview starts)."""
        steps = self._image_slider_resolution
        for node, meta in self._slider_meta.items():
            sl = meta["slider"]
            lbl = meta["label"]
            caption = meta.get("range_caption")
            if node == "ExposureTime":
                limits = self.camera.get_exposure_time_limits()
            else:
                limits = self.camera.get_image_param_limits(node)
            if limits is None:
                meta["supported"] = False
                sl.blockSignals(True)
                sl.setEnabled(False)
                lbl.setText("N/A")
                sl.blockSignals(False)
                if caption is not None:
                    caption.setText("")
                continue
            mn, mx, cur = limits
            meta["min"], meta["max"] = mn, mx
            meta["steps"] = steps
            meta["supported"] = True
            sl.blockSignals(True)
            sl.setRange(0, steps)
            if mx <= mn:
                sl.setEnabled(False)
                lbl.setText(f"{cur:.4g}")
            else:
                acquiring = self.state in (
                    AppState.PREVIEWING,
                    AppState.RECORDING,
                )
                sl.setEnabled(self._slider_can_tune(node, acquiring))
                t = (cur - mn) / (mx - mn)
                sl.setValue(int(round(min(1.0, max(0.0, t)) * steps)))
                lbl.setText(f"{cur:.4g}")
            if caption is not None:
                fps = self.camera.get_acquisition_frame_rate()
                caption.setText(
                    f"Range limited by frame rate: {mn/1000:.2f}–{mx/1000:.2f} ms "
                    f"at {fps:.1f} fps (90% of the {1000/fps:.2f} ms frame period)."
                )
            sl.blockSignals(False)

    def _sync_auto_mode_combos_from_camera(self) -> None:
        """Read ExposureAuto/GainAuto options + current entry (call after preview starts)."""
        for node, meta in self._auto_mode_meta.items():
            combo = meta["combo"]
            result = self.camera.get_enum_param(node)
            combo.blockSignals(True)
            if result is None:
                combo.clear()
                combo.setEnabled(False)
            else:
                current, entries = result
                if [combo.itemText(i) for i in range(combo.count())] != entries:
                    combo.clear()
                    combo.addItems(entries)
                combo.setCurrentText(current)
                can_tune = self.state in (AppState.PREVIEWING, AppState.RECORDING)
                combo.setEnabled(can_tune)
            combo.blockSignals(False)

    def _sync_frame_rate_from_camera(self) -> None:
        """Read the cameras' frame-rate range/current and fill the spin box.

        With several cameras the range is what ALL of them support, so one
        value is always valid everywhere.
        """
        spin = self.frame_rate_spin
        controllers = [slot.controller for slot in self._slots()] or [self.camera]
        all_limits = [controller.get_frame_rate_limits() for controller in controllers]
        shared = intersect_ranges(all_limits)
        if shared is None:
            spin.blockSignals(True)
            spin.setEnabled(False)
            spin.blockSignals(False)
            self.frame_rate_hint.setText("Frame rate not adjustable on this camera.")
            return
        mn, mx = shared
        own = self.camera.get_frame_rate_limits()
        cur = min(mx, max(mn, own[2] if own is not None else mn))
        spin.blockSignals(True)
        spin.setRange(mn, mx)
        spin.setValue(cur)
        spin.blockSignals(False)
        self._update_frame_rate_widget_enabled()
        self.frame_rate_hint.setStyleSheet("color: #555; font-size: 11px;")
        who = "Cameras support" if len(controllers) > 1 else "Camera supports"
        self.frame_rate_hint.setText(
            f"{who} {mn:.1f}–{mx:.1f} fps. Current rate: {cur:.1f} fps "
            f"(also sets AVI playback speed)."
        )

    def _update_frame_rate_ceiling_hint(self, ceiling_fps: float | str | None) -> None:
        """Warn once a second if the camera can no longer sustain the requested
        rate -- e.g. ExposureTime has grown too long for the target fps.
        """
        if ceiling_fps is None or ceiling_fps == "" or not hasattr(self, "frame_rate_spin"):
            return
        requested = self.frame_rate_spin.value()
        ceiling = float(ceiling_fps)
        if ceiling + 0.5 < requested:
            self.frame_rate_hint.setStyleSheet("color: #b00020; font-size: 11px; font-weight: 600;")
            self.frame_rate_hint.setText(
                f"Camera can only sustain ~{ceiling:.1f} fps right now (requested "
                f"{requested:.1f} fps) -- check exposure/gain mode below."
            )
        else:
            self.frame_rate_hint.setStyleSheet("color: #555; font-size: 11px;")
            self.frame_rate_hint.setText(
                f"Current rate: {requested:.1f} fps (also sets AVI playback speed). "
                f"Max sustainable right now: {ceiling:.1f} fps."
            )

    def _on_frame_rate_changed(self, value: float) -> None:
        if not self.frame_rate_spin.isEnabled():
            return
        if self.cameras is not None:
            previous = {
                slot.serial: slot.controller.get_acquisition_frame_rate() for slot in self._slots()
            }
            results = self.cameras.broadcast("set_frame_rate", float(value))
            rejected = [r.slot.label for r in results if not (r.ok and r.value)]
            if rejected:
                # One camera refused (e.g. it is mid-recovery): put the ones that
                # accepted back, so every camera keeps sharing one rate.
                for r in results:
                    if r.ok and r.value:
                        r.slot.controller.set_frame_rate(previous[r.slot.serial])
                self.frame_rate_spin.blockSignals(True)
                self.frame_rate_spin.setValue(previous.get(self.camera.serial, value))
                self.frame_rate_spin.blockSignals(False)
                self.status_label.setText(
                    f"Frame rate NOT changed to {float(value):.1f} fps: "
                    + ", ".join(rejected) + " did not accept it."
                )
                return
        else:
            if not self.camera.set_frame_rate(float(value)):
                return
            rejected = []
        actual = self.camera.get_acquisition_frame_rate()
        # Camera may snap to a nearby achievable rate; reflect what it accepted.
        if abs(actual - float(value)) > 0.05:
            self.frame_rate_spin.blockSignals(True)
            self.frame_rate_spin.setValue(actual)
            self.frame_rate_spin.blockSignals(False)
        message = f"Frame rate set to {actual:.1f} fps (also sets AVI playback speed)."
        if self._is_multi_camera():
            message = "Frame rate set: " + ", ".join(
                f"{slot.label} {slot.controller.get_acquisition_frame_rate():.1f} fps"
                for slot in self._slots()
            ) + " (also sets AVI playback speed)."
        if rejected:
            message += " NOT applied to: " + ", ".join(rejected) + "."
        self.frame_rate_hint.setText(message)
        # The backend may have just clamped ExposureTime to fit the new
        # frame period (see CameraController._clamp_exposure_to_frame_period)
        # -- refresh the slider so it doesn't show a stale, now-wrong value.
        self._sync_image_sliders_from_camera()
        self._apply_compression_default_for_fps(actual)
        self._apply_exposure_auto_lock_for_fps(actual)

    def _on_compression_toggled(self, checked: bool) -> None:
        if not self.compression_checkbox.isEnabled():
            return
        # Only reached for a genuine click -- programmatic changes elsewhere
        # always go through blockSignals(). From here on the user's choice
        # sticks; the fps-based default stops touching this checkbox.
        self._compression_manually_set = True
        if not self._set_compression_on_all(checked):
            # Refused (recording started between the click and here) --
            # put the checkbox back without re-entering this handler.
            self.compression_checkbox.blockSignals(True)
            self.compression_checkbox.setChecked(not checked)
            self.compression_checkbox.blockSignals(False)

    def _apply_compression_default_for_fps(self, fps: float) -> None:
        """Suggest compression on/off based on fps -- a starting point
        only. Once the user has touched the checkbox directly this
        session, their choice sticks and this stops overriding it.
        """
        if self._compression_manually_set:
            return
        desired = suggested_compression(max(1, len(self._slots())), fps)
        if self.compression_checkbox.isChecked() != desired:
            if self._set_compression_on_all(desired):
                self.compression_checkbox.blockSignals(True)
                self.compression_checkbox.setChecked(desired)
                self.compression_checkbox.blockSignals(False)

    def _apply_exposure_auto_lock_for_fps(self, fps: float) -> None:
        """Force Exposure mode to Off and lock the dropdown at/above
        EXPOSURE_AUTO_LOCK_MIN_FPS -- no user override, unlike
        _apply_compression_default_for_fps. See that constant's comment
        for why: ExposureAuto gives no guarantee of fitting the frame
        period, and our own exposure-clamp can't act while it's engaged.
        """
        self._exposure_auto_locked = fps >= EXPOSURE_AUTO_LOCK_MIN_FPS
        auto_meta = self._auto_mode_meta.get("ExposureAuto")
        if auto_meta is None:
            return
        combo = auto_meta["combo"]
        if self._exposure_auto_locked:
            combo.setToolTip(
                f"Locked to Off at {fps:.0f} fps (>= {EXPOSURE_AUTO_LOCK_MIN_FPS:.0f} fps): "
                "auto exposure isn't guaranteed to fit the frame period."
            )
            # EVERY camera, every time: the combo shows only the camera the
            # controls point at, so "it already reads Off" says nothing about
            # the others, and a camera left on auto exposure at >= 30 fps is
            # exactly the failure this lock exists to prevent.
            self._call_on_all_cameras("set_enum_param", "ExposureAuto", "Off")
            if combo.count() > 0 and combo.currentText() != "Off":
                combo.blockSignals(True)
                combo.setCurrentText("Off")
                combo.blockSignals(False)
                self._sync_image_sliders_from_camera()
        else:
            combo.setToolTip("")
        self._update_camera_tuning_widgets_enabled()

    def _update_frame_rate_widget_enabled(self) -> None:
        """Adjustable only while previewing — locked when idle or recording."""
        if not hasattr(self, "frame_rate_spin"):
            return
        self.frame_rate_spin.setEnabled(self.state == AppState.PREVIEWING)
        # Same lock as frame rate: changing codec only makes sense before
        # the next segment writer opens, not mid-recording.
        if hasattr(self, "compression_checkbox"):
            self.compression_checkbox.setEnabled(self.state == AppState.PREVIEWING)

    def _update_camera_tuning_widgets_enabled(self) -> None:
        if not self._slider_meta:
            return
        acquiring = self.state in (AppState.PREVIEWING, AppState.RECORDING)
        if hasattr(self, "camera_tuning_hint"):
            self.camera_tuning_hint.setVisible(not acquiring)
        for node, meta in self._auto_mode_meta.items():
            can_enable = acquiring and meta["combo"].count() > 0
            if node == "ExposureAuto" and self._exposure_auto_locked:
                can_enable = False
            meta["combo"].setEnabled(can_enable)
        for node, meta in self._slider_meta.items():
            sl = meta["slider"]
            if not meta.get("supported"):
                sl.setEnabled(False)
                continue
            sl.setEnabled(self._slider_can_tune(node, acquiring))

    def _stop_mic_preview(self) -> None:
        if self._mic_preview is not None:
            try:
                self._mic_preview.stop()
            except Exception as exc:
                print("Error stopping mic preview:", exc)
            self._mic_preview = None
        self._mic_preview_active = False
        self.audio_level_bar.setValue(0)

    def _on_audio_input_device_changed(self, _index: int | None = None) -> None:
        if self._mic_preview_active:
            self._stop_mic_preview()
        self._apply_state()

    def _on_audio_output_device_changed(self, _index: int | None = None) -> None:
        if self._mic_preview_active:
            self._stop_mic_preview()
        self._apply_state()

    def _on_audio_monitor_changed(self, _state: int | None = None) -> None:
        if self._mic_preview_active:
            self._stop_mic_preview()
        self._apply_state()

    def on_mic_preview_clicked(self) -> None:
        if self.state == AppState.RECORDING:
            return
        if self._mic_preview_active:
            self._stop_mic_preview()
            self.audio_label.setText(
                "Microphone: scan and choose a device, or leave as no audio."
            )
            self._apply_state()
            return
        dev = self.audio_input_combo.currentData()
        if dev is None:
            return
        out_dev = self.audio_output_combo.currentData()
        self._mic_preview = MicLevelPreview()
        try:
            self._mic_preview.start(
                int(dev),
                monitor=self.audio_monitor_checkbox.isChecked(),
                output_device=int(out_dev) if out_dev is not None else None,
            )
            self._mic_preview_active = True
            self.audio_label.setText("Microphone: preview active (no file saved).")
            QTimer.singleShot(450, self._notify_preview_listen_status)
        except Exception as exc:
            self._mic_preview = None
            self._mic_preview_active = False
            print(f"Mic preview failed: {exc}")
            self.audio_label.setText(f"Mic preview failed: {exc}")
        self._apply_state()

    def _notify_preview_listen_status(self) -> None:
        """If user wanted hear-through but monitor failed, explain in the UI."""
        if not self._mic_preview_active or self._mic_preview is None:
            return
        if not self.audio_monitor_checkbox.isChecked():
            return
        if self._mic_preview.had_duplex_output:
            return
        self.audio_label.setText(
            "Mic preview: level only — hear-through did not open. "
            "Turn the checkbox on, set the OS default output to your headphones, "
            "raise system/app volume, then stop preview and start again."
        )

    def _reset_label_marker_session(self) -> None:
        """Reset label feedback when a new recording starts."""
        self.label_last_event.setText("—")
        self._label_pair_counts.clear()
        self._open_label_ids.clear()
        self._update_label_pair_counter()

    def _record_label_pair_event(self, label_event: str, adl_id) -> bool:
        """Track valid start/end pairs and return whether one just completed."""
        if adl_id is None:
            return False
        if label_event == "label_start":
            self._open_label_ids.add(adl_id)
            return False
        if adl_id not in self._open_label_ids:
            return False

        self._open_label_ids.remove(adl_id)
        self._label_pair_counts[adl_id] = self._label_pair_counts.get(adl_id, 0) + 1
        self._update_label_pair_counter()
        return True

    def _reset_label_marker_panel_style(self) -> None:
        self.label_marker_frame.setStyleSheet(self._label_frame_base_style)

    def _flash_label_marker_panel(self, label_event: str) -> None:
        if label_event == "label_start":
            self.label_marker_frame.setStyleSheet(
                "QFrame#labelMarkerFrame { border: 2px solid #2e7d32; border-radius: 4px; "
                "padding: 4px 8px; background: #c8e6c9; }"
            )
        else:
            self.label_marker_frame.setStyleSheet(
                "QFrame#labelMarkerFrame { border: 2px solid #e65100; border-radius: 4px; "
                "padding: 4px 8px; background: #ffe0b2; }"
            )
        QTimer.singleShot(220, self._reset_label_marker_panel_style)

    def _append_label_marker_activity(
        self,
        label_event: str,
        adl_id,
        adl_label: str | None,
    ) -> None:
        if label_event == "label_start":
            verb = "START"
        else:
            verb = "END"
        if adl_id is None:
            name = "(no ADL)"
        else:
            name = (adl_label or "").strip() or f"ID {adl_id}"
        self.label_last_event.setText(f"{verb} — {name}")
        self._flash_label_marker_panel(label_event)

    def _notify_recording_monitor_status(self, wav_path: str) -> None:
        """had_duplex_output is set on a background thread; re-check after open."""
        if self.state != AppState.RECORDING:
            return
        if self._session_audio is None:
            return
        if not self.audio_monitor_checkbox.isChecked():
            return
        if self._session_audio.had_duplex_output:
            self.audio_label.setText(f"Microphone: recording to {wav_path}")
            return
        self.audio_label.setText(
            f"Microphone: recording to {wav_path} (monitor unavailable — "
            "check Windows sound output device / exclusive mode)."
        )

    def _update_mic_level_bar(self) -> None:
        if self._mic_preview_active and self._mic_preview is not None:
            self.audio_level_bar.setValue(self._mic_preview.level_0_100())
        elif self.state == AppState.RECORDING and self._session_audio is not None:
            self.audio_level_bar.setValue(self._session_audio.level_0_100())
        else:
            v = self.audio_level_bar.value()
            if v > 0:
                self.audio_level_bar.setValue(max(0, v - 14))

    # ------------------------------------------------------------------
    # Cameras: detection, grouping, per-camera GUI state
    # ------------------------------------------------------------------
    @property
    def camera(self) -> CameraController:
        """The camera the tuning controls act on (the primary until another is picked)."""
        if self.cameras is not None:
            slot = self.cameras.slot_for_serial(self._tuning_serial) or self.cameras.default_slot
            return slot.controller
        return self._placeholder_camera

    def _slots(self) -> list[CameraSlot]:
        return self.cameras.slots if self.cameras is not None else []

    def _is_multi_camera(self) -> bool:
        return len(self._slots()) > 1

    def _call_on_all_cameras(self, method_name: str, *args) -> bool:
        """Call controller.<method_name>(*args) on every camera; True only if all accepted."""
        if self.cameras is None:
            return bool(getattr(self._placeholder_camera, method_name)(*args))
        results = self.cameras.broadcast(method_name, *args)
        return all(r.ok and bool(r.value) for r in results)

    def _set_compression_on_all(self, enabled: bool) -> bool:
        """Change the codec on every camera, or on none: if one refuses (it is
        still closing its last recording), the ones that accepted go back."""
        slots = self._slots()
        if not slots:
            return bool(self._placeholder_camera.set_compression_enabled(enabled))
        previous = {slot.serial: slot.controller.get_compression_enabled() for slot in slots}
        results = self.cameras.broadcast("set_compression_enabled", enabled)
        if all(r.ok and r.value for r in results):
            return True
        for r in results:
            if r.ok and r.value:
                r.slot.controller.set_compression_enabled(previous[r.slot.serial])
        return False

    def _refresh_compression_hint(self) -> None:
        self.compression_hint.setText(describe_policy(max(1, len(self._slots()))))

    def _clear_camera_group(self) -> None:
        self.cameras = None
        self._slot_rt = {}
        self._tuning_serial = None
        self.tuning_camera_row.setVisible(False)
        self._rebuild_preview_tiles()
        self._refresh_compression_hint()

    def _install_camera_group(self, selection) -> None:
        slots = [
            CameraSlot(
                CameraController(serial=cam.serial, tag=cam.tag),
                serial=cam.serial,
                model=cam.model,
                tag=cam.tag,
                is_primary=cam.is_primary,
            )
            for cam in selection.bound
        ]
        self.cameras = CameraGroup(slots)
        self._tuning_serial = self.cameras.default_slot.serial
        self._rebuild_preview_tiles()
        self.tuning_camera_combo.blockSignals(True)
        self.tuning_camera_combo.clear()
        for slot in slots:
            self.tuning_camera_combo.addItem(slot.label, slot.serial)
        self.tuning_camera_combo.setCurrentIndex(
            max(0, self.tuning_camera_combo.findData(self._tuning_serial))
        )
        self.tuning_camera_combo.blockSignals(False)
        self.tuning_camera_row.setVisible(len(slots) > 1)
        self._refresh_compression_hint()

    # ------------------------------------------------------------------
    # 3D calibration status (backend.calibration_store; never blocks recording)
    # ------------------------------------------------------------------
    def _refresh_calibration_status(self) -> None:
        """Re-read the stored calibrations and show whether 3D pose is available."""
        slots = self._slots()
        if not slots:
            self.calibration_status = None
            self.calibration_bar.show_idle()
            return
        try:
            now = datetime.now()
            statuses = {}
            for slot in slots:
                fingerprint = slot.controller.get_sensor_fingerprint() if self.preview_running else None
                statuses[slot.serial] = assess_camera(
                    slot.serial,
                    slot.label,
                    self._calibration_store.current_intrinsics(slot.serial),
                    fingerprint,
                    now,
                    live_rms_px=self._calibration_live_rms.get(slot.serial),
                )
            self.calibration_status = assess_session(
                [(slot.serial, slot.label) for slot in slots],
                statuses,
                self._calibration_store.current_setup(),
                setup_valid_since=self._cameras_detected_at,
                moved=self._calibration_moved or None,
            )
        except Exception as exc:  # a broken calibration folder must never break the recorder
            print(f"[calibration] status check failed: {exc}")
            self.calibration_status = None
            self.calibration_bar.show_idle(f"3D pose: could not read calibrations ({exc})")
            return
        self.calibration_bar.show_status(self.calibration_status)

    def _write_calibration_snapshot(self, basename_path: Path) -> None:
        """<basename>_calibration.json next to the video: the 3D status at start and the records behind it."""
        if self.calibration_status is None:
            return
        path = basename_path.with_name(f"{basename_path.name}_calibration.json")
        try:
            snapshot = session_snapshot(
                self._calibration_store, [slot.serial for slot in self._slots()], self.calibration_status
            )
            write_session_snapshot(path, snapshot)
        except Exception as exc:  # recording goes on; the status line already says what is known
            print(f"[calibration] could not write {path}: {exc}")

    def _open_calibration_window(self) -> None:
        QMessageBox.information(
            self,
            "Calibration",
            "The calibration window is not built yet (plan steps 13-14).\n\n"
            f"Calibrations are stored in:\n{self._calibration_store.root}",
        )

    def _rebuild_preview_tiles(self) -> None:
        """One tile (and one diagnostics accumulator + logger) per camera."""
        while self.preview_layout.count():
            widget = self.preview_layout.takeAt(0).widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()
        slots = self._slots()
        if not slots:
            self.preview_layout.addWidget(CameraPreviewTile())
            return
        multi = len(slots) > 1
        self._slot_rt = {}
        for slot in slots:
            tile = CameraPreviewTile(min_size=(320, 240) if multi else (480, 320))
            tile.set_caption(slot.label)
            tile.set_caption_visible(multi)
            self.preview_layout.addWidget(tile)
            self._slot_rt[slot.serial] = _SlotRuntime(
                tile=tile,
                diagnostics=PreviewDiagnosticsAccumulator(),
                logger=AsyncDiagnosticsCsvLogger(),
            )

    def _on_tuning_camera_changed(self, index: int) -> None:
        serial = self.tuning_camera_combo.itemData(index)
        if serial is None or serial == self._tuning_serial:
            return
        self._tuning_serial = serial
        if self.preview_running:
            self._sync_auto_mode_combos_from_camera()
            self._sync_image_sliders_from_camera()
            # The combo sync re-enables the dropdown from the new camera's
            # state; put the lock back.
            self._apply_exposure_auto_lock_for_fps(self.camera.get_acquisition_frame_rate())

    def _reset_all_diagnostics(self) -> None:
        for slot in self._slots():
            rt = self._slot_rt[slot.serial]
            rt.diagnostics.reset(slot.controller.get_acquisition_stats())
            rt.backlog.reset()
            rt.mjpeg_behind.reset()

    def _apply_camera_selection(self, discovered, selection) -> tuple[bool, str]:
        """(found, status message) for a Detect click; installs the camera group on success."""
        if not selection.bound:
            self._clear_camera_group()
            message = format_camera_summary(selection)
            if selection.warnings:
                message += "\n" + "\n".join(f"Warning: {w}" for w in selection.warnings)
            return False, message
        if selection.missing:
            # A configured camera is absent: a manual start is refused rather than
            # quietly recording with fewer cameras than asked for. (An unattended
            # power-resume records with whatever is present; it never passes here.)
            self._clear_camera_group()
            return False, (
                format_camera_summary(selection)
                + " Check the connection or SLEEVE_VIDEO_GUI_CAMERA_SERIALS."
            )
        self._install_camera_group(selection)
        if len(selection.bound) == 1:
            cam = selection.bound[0]
            vendor = next((d.vendor for d in discovered if d.serial == cam.serial), "")
            message = f"Camera: {vendor} {cam.model} (S/N: {cam.serial})"
        else:
            message = format_camera_summary(selection)
        if selection.warnings:
            message += "\n" + "\n".join(f"Warning: {w}" for w in selection.warnings)
        if selection.notes:
            message += "\n" + "\n".join(f"Note: {n}" for n in selection.notes)
        return True, message

    def on_detect_clicked(self):
        if self.state not in (AppState.IDLE, AppState.CAMERA_DETECTED):
            return

        self.status_label.setText("Detecting camera...")
        try:
            discovered = enumerate_cameras()
            selection = select_cameras(discovered, parse_serials_env())
        except Exception as exc:
            self._clear_camera_group()
            found, message = False, f"Error: could not list cameras ({exc})"
        else:
            found, message = self._apply_camera_selection(discovered, selection)
        self.status_label.setText(message)

        if found:
            self.state = AppState.CAMERA_DETECTED
            # Cameras may have been re-mounted before this Detect: an older setup
            # only counts again once the live fixed-board check confirms it.
            self._cameras_detected_at = datetime.now()
        else:
            self.state = AppState.IDLE
        self._calibration_moved = {}
        self._calibration_live_rms = {}

        self._apply_state()
        self._refresh_calibration_status()

    def on_scan_daq_clicked(self):
        """Scan for connected NI-DAQ devices and populate the dropdown."""
        self.daq_line_combo.clear()

        # NI output is released to the SyncService heartbeat by default. Say so
        # explicitly rather than letting it look like a hardware fault.
        if not ni_output_enabled():
            self.daq_line_combo.addItem("Disabled by configuration", None)
            self.connect_daq_button.setEnabled(False)
            self.sync_button.setEnabled(False)
            self.sync_label.setText(NI_DISABLED_MESSAGE)
            return

        lines = list_do_lines()
        if lines:
            for line in lines:
                self.daq_line_combo.addItem(line, line)
            self.connect_daq_button.setEnabled(True)
            self.sync_label.setText(f"Found {len(lines)} DO line(s) — select one and click Connect")
        else:
            self.daq_line_combo.addItem("No lines detected", None)
            self.connect_daq_button.setEnabled(False)
            self.sync_label.setText("No NI devices found — check connections and drivers")

    def on_connect_daq_clicked(self):
        """Connect to the selected NI-DAQ line and update UI."""
        selected_line = self.daq_line_combo.currentData()
        if selected_line is None:
            self.sync_label.setText("No DO line selected")
            return
        try:
            cfg = DOLine(line=selected_line, idle_low=True)
            self.daq = NIDaqDO(cfg)
            self.daq.start()

            self.pulse_manager = PulseManager(daq=self.daq, default_width_s=0.010)
            self.pulse_manager.start()

            self.sync_label.setText(f"Sync available — connected to {selected_line}")
            self.connect_daq_button.setEnabled(False)
            self.scan_daq_button.setEnabled(False)
            self.daq_line_combo.setEnabled(False)
            self.sync_button.setEnabled(True)

        except Exception as e:
            self.sync_label.setText(f"Sync not available — {e.__class__.__name__}: {e}")
            self.daq = None
            self.pulse_manager = None
            self.sync_button.setEnabled(False)

    def on_sync_pulse_clicked(self):
        """Send a sync pulse.

        - If recording: send a sync pulse AND mark its window in the metadata.
        - If NOT recording: send a simple test pulse only (no logging).
        """
        # If NOT recording, just send a test pulse (no logging window)
        if self.state != AppState.RECORDING:
            if self.pulse_manager is not None and sys.platform.startswith("win"):
                try:
                    # Use a short test pulse
                    self.pulse_manager.request_pulse(
                        width_s=SYNC_WIDTH_RECORD,
                        label="test_pulse",
                    )
                    self.sync_label.setText("Test pulse sent (no logging, not recording).")
                except Exception as e:
                    self.sync_label.setText(f"Test pulse failed: {e}")
            else:
                self.sync_label.setText("DAQ not connected.")
            return

        # === Recording case: send sync pulse and mark window ===
        # Increment which manual sync this is
        self.manual_sync_count += 1

        # Map count -> pulse width
        width = SYNC_WIDTH_RECORD * (self.manual_sync_count + 1)
        label = f"manual_sync_{width * 1000:.0f}_ms"

        try:
            # hardware pulse
            if self.pulse_manager is not None and sys.platform.startswith("win"):
                self.pulse_manager.request_pulse(width_s=width, label=label)

            # logging window
            self.cameras.notify_sync_pulse_window(width_s=width, label=label)

            self.sync_label.setText(f"Sync pulse sent ({label}, {width * 1000:.0f} ms).")
        except Exception as e:
            self.sync_label.setText(f"Pulse failed: {e}")

    def on_preview_clicked(self):
        if not self.preview_running:
            if self.cameras is None:
                self.status_label.setText("Detect a camera first.")
                return
            result = self.cameras.start_all()
            self.status_label.setText(result.message)
            if not result.ok:
                return

            # This controls PREVIEW redraw rate, NOT camera fps. Redrawing faster
            # than the camera produces frames just burns GUI-thread time on
            # repeated smooth-scaled repaints and makes the UI feel laggy.
            # 33 ms ~ 30 fps matches the default capture rate.
            self.timer.start(33)
            self.preview_diagnostics_timer.start(1000)
            self.preview_running = True
            for rt in self._slot_rt.values():
                rt.last_seq = None
            self._reset_all_diagnostics()
            self.preview_health_label.setText("Preview pipeline: measuring…")
            self.preview_button.setText("Stop Preview")
            self.state = AppState.PREVIEWING
            self._apply_state()
            # Fresh session: let the fps-based default pick the checkbox
            # again rather than carrying over a manual choice from before.
            self._compression_manually_set = False
            self._sync_auto_mode_combos_from_camera()
            self._sync_image_sliders_from_camera()
            self._sync_frame_rate_from_camera()
            self._apply_compression_default_for_fps(self.camera.get_acquisition_frame_rate())
            self._apply_exposure_auto_lock_for_fps(self.camera.get_acquisition_frame_rate())
            self._refresh_calibration_status()


        else:
            # Only allow stopping preview when NOT recording
            if self.state == AppState.RECORDING:
                return  # safety, shouldn't happen if buttons are disabled correctly
            self.timer.stop()
            self.preview_diagnostics_timer.stop()
            self._stop_preview_diagnostics_logging()
            result = self.cameras.stop_all()
            if not result.ok:
                print(f"[camera] stop() deferred cleanup: {result.message}")
            self.preview_running = False
            for slot in self._slots():
                rt = self._slot_rt[slot.serial]
                rt.tile.clear()
                rt.tile.set_caption(slot.label)
                rt.last_seq = None
            self.preview_health_label.setText("Preview pipeline: waiting for preview")
            self.status_label.setText("Preview stopped.")
            self.frame_rate_hint.setText(
                "Start preview to read the camera's supported range and current rate."
            )
            self.state = AppState.CAMERA_DETECTED
            self._apply_state()

    def on_record_clicked(self):
        # Start recording
        if self.state == AppState.PREVIEWING:
            self._begin_recording_session()

        # Stop recording
        elif self.state == AppState.RECORDING:
            self._stop_recording_session("Recording stopped.")

    def _begin_recording_session(self, *, bypass_confirmation: bool = False) -> bool:
        """Start a brand-new recording session.

        Used both by the manual Start Recording button and by automatic
        power-resume. bypass_confirmation skips every human-facing dialog
        for the automatic path -- the sustained-safe debounce in
        next_power_action() already stands in for that confirmation, and
        nobody is there to answer a dialog at 3am. Returns whether a new
        session actually started.
        """
        if self.state != AppState.PREVIEWING:
            return False

        if not bypass_confirmation:
            if not self._confirm_power_safe_to_record():
                return False

        # SessionPaths is the single place deriving every session
        # artifact name (video segments, wav, metadata, diagnostics,
        # segments manifest, events) from one stem, so they can't
        # drift out of sync with each other. Every camera shares the
        # session's timestamp; each camera's files carry its own tag.
        output_dir = resolve_output_dir(explicit=self._configured_output_dir)
        if not self._confirm_disk_safe_to_record(
            output_dir, bypass_confirmation=bypass_confirmation
        ):
            return False
        started_at = datetime.now()
        paths_by_serial = {
            slot.serial: SessionPaths.for_session(output_dir, started_at, camera_tag=slot.tag)
            for slot in self._slots()
        }
        # The WAV is shared by all cameras and never tagged.
        session_paths = paths_by_serial[self.cameras.default_slot.serial]

        self._keepalive_state = apply_keep_awake(self._keepalive_request)

        self.record_button.setEnabled(False)
        self.record_button.setText("Starting recording… please wait")
        self.status_label.setText("Starting recording…")
        QApplication.processEvents()

        # Use each camera's real acquisition rate so AVI playback speed matches.
        # An unattended power-resume records with whichever cameras still accept
        # (best_effort); a manual start is all-or-nothing.
        result = self.cameras.start_recording_all(
            lambda slot: paths_by_serial[slot.serial],
            lambda slot: slot.controller.get_acquisition_frame_rate() or 30.0,
            best_effort=bypass_confirmation,
        )
        self.status_label.setText(result.message)
        QApplication.processEvents()

        if not result.ok:
            self._apply_state()
            return False

        self._refresh_calibration_status()
        self._write_calibration_snapshot(session_paths.output_dir / session_paths.basename)

        self._recording_warnings.reset(now_s=time.monotonic())
        self._update_recording_warning_banner()
        # (A camera that did not start is reported on every diagnostics tick by
        # _sample_preview_diagnostics, so the banner cannot quietly fade to
        # "Recovered" while it is still not recording.)
        self._start_preview_diagnostics_logging(
            paths_by_serial, [o.serial for o in result.outcomes if o.ok]
        )

        self._stop_mic_preview()
        # Parallel WAV sharing the same session stem as the video segments.
        audio_device = self.audio_input_combo.currentData()
        self._session_audio = None
        if audio_device is not None:
            wav_path = str(session_paths.wav)
            try:
                self._session_audio = SessionAudioRecorder()
                out_dev = self.audio_output_combo.currentData()
                self._session_audio.start(
                    wav_path,
                    int(audio_device),
                    monitor=self.audio_monitor_checkbox.isChecked(),
                    output_device=int(out_dev) if out_dev is not None else None,
                )
                format_choice = self._session_audio.audio_format_choice
                format_note = f" ({format_choice.reason})" if format_choice is not None else ""
                self.audio_label.setText(
                    f"Microphone: recording to {wav_path}{format_note}"
                )
                if self.audio_monitor_checkbox.isChecked():
                    QTimer.singleShot(
                        500,
                        lambda w=wav_path: self._notify_recording_monitor_status(w),
                    )
            except Exception as exc:
                self._session_audio = None
                print(f"Audio recording failed: {exc}")
                self.audio_label.setText(f"Microphone: failed ({exc}) — video only")

        # 1) fire a 100 ms hardware pulse
        if self.pulse_manager is not None and sys.platform.startswith("win"):
            try:
                self.pulse_manager.request_pulse(
                    width_s=SYNC_WIDTH_RECORD,
                    label="record_start",
                )
            except Exception as e:
                print("Record-start pulse failed:", e)

        # 2) tell the camera to mark frames in this window only
        # if we actually sent a hardware pulse
        if self.pulse_manager is not None and sys.platform.startswith("win"):
            self.cameras.notify_sync_pulse_window(
                width_s=SYNC_WIDTH_RECORD,
                label="record_start",
            )

        self.manual_sync_count = 0  # reset manual counter
        self._reset_label_marker_session()

        self.state = AppState.RECORDING
        self._apply_state()
        return True

    def _stop_recording_session(self, status_message: str) -> None:
        """Use the existing stop path for manual and power-safety stops."""
        if self.state != AppState.RECORDING:
            return
        self._keepalive_state = release_keep_awake()
        if self._session_audio is not None:
            try:
                self._session_audio.stop()
            except Exception as exc:
                print("Error stopping audio recording:", exc)
            self._session_audio = None
        self.cameras.stop_recording_all()
        self._stop_preview_diagnostics_logging()
        self.status_label.setText(status_message)
        self.audio_label.setText(
            "Microphone: scan and choose a device, or leave as no audio."
        )
        self.state = AppState.PREVIEWING
        self._apply_state()

    def _start_preview_diagnostics_logging(self, paths_by_serial, serials=None) -> None:
        """Start a sidecar diagnostics CSV for each camera that is recording this session."""
        self._reset_all_diagnostics()
        for serial, rt in self._slot_rt.items():
            if serials is None or serial in serials:
                rt.logger.start(paths_by_serial[serial].diagnostics_csv)

    def _stop_preview_diagnostics_logging(self) -> None:
        """Flush the current interval and finish every camera's diagnostics sidecar."""
        if any(rt.logger.is_running for rt in self._slot_rt.values()):
            self._sample_preview_diagnostics()
        for rt in self._slot_rt.values():
            rt.logger.stop()
        if self.preview_running:
            self._reset_all_diagnostics()

    def _update_recording_warning_banner(self) -> None:
        """Reflect RecordingWarningTracker state in the dismissible banner.

        Unlike the old permanent-latch text on preview_health_label, this
        can return to a calm "recovered" state after the acquisition
        watchdog fixes a fault, and can be dismissed once read.
        """
        state = self._recording_warnings.summarize(now_s=time.monotonic())
        self.recording_warning_frame.setVisible(state.visible)
        if not state.visible:
            return
        colors = {
            "active": ("#ffebee", "#b71c1c"),
            "recovered": ("#fff8e1", "#e65100"),
        }
        background, foreground = colors.get(state.level, ("#f0f0f0", "#555"))
        self.recording_warning_label.setText(f"{state.headline}\n{state.detail}")
        self.recording_warning_frame.setStyleSheet(
            "QFrame#recordingWarningFrame {"
            f" background: {background}; border: 1px solid {foreground};"
            " border-radius: 4px; }"
            f"QLabel {{ color: {foreground}; font-weight: 600; }}"
        )

    def _on_dismiss_recording_warning_clicked(self) -> None:
        self._recording_warnings.dismiss(now_s=time.monotonic())
        self._update_recording_warning_banner()

    def _sample_preview_diagnostics(self) -> None:
        """Update GUI health once per second and queue a CSV row per camera if recording."""
        slots = self._slots()
        if not slots:
            return
        multi = len(slots) > 1
        audio_stats: dict[str, int] = {}
        if self._session_audio is not None:
            audio_health = self._session_audio.health_snapshot()
            audio_stats = {
                "audio_xruns": audio_health.total_xruns,
                "audio_reconnects": audio_health.reconnects,
                "audio_silence_frames_inserted": audio_health.silence_frames_inserted,
            }

        samples = []  # (slot, runtime, row, camera_state)
        for slot in slots:
            rt = self._slot_rt[slot.serial]
            stats = dict(slot.controller.get_acquisition_stats())
            stats.update(audio_stats)
            camera_state = slot.controller.get_diagnostics_camera_state()
            loop_timing = slot.controller.get_and_reset_loop_timing_samples()
            row = rt.diagnostics.sample(
                stats, camera_state=camera_state, loop_timing=loop_timing
            )
            if rt.logger.is_running:
                rt.logger.submit(row)
            samples.append((slot, rt, row, camera_state))

        ceilings = [
            float(cs["frame_rate_ceiling_fps"])
            for _, _, _, cs in samples
            if cs.get("frame_rate_ceiling_fps") not in (None, "")
        ]
        self._update_frame_rate_ceiling_hint(min(ceilings) if ceilings else None)

        now_s = time.monotonic()
        recording = self.state == AppState.RECORDING
        warn_parts: list[str] = []
        segments: list[str] = []
        worst_color = "#2e7d32"
        interval_s = 0.0
        audio_reconnects = 0

        for slot, rt, row, _camera_state in samples:
            prefix = f"{slot.label}: " if multi else ""
            tag = f"[{slot.label}] " if multi else ""
            age_value = row["preview_age_ms"]
            interval_s = max(interval_s, float(row["interval_s"]))
            rendered_fps = float(row["rendered_fps"])
            frame_gaps = int(row["camera_frame_gaps"])
            incomplete = int(row["incomplete_images"])
            errors = int(row["acquisition_errors"])
            append_failures = int(row["append_failures"])
            camera_reinits = int(row["camera_reinits"])
            audio_reconnects = max(audio_reconnects, int(row["audio_reconnects"]))

            if recording:
                controller = slot.controller
                if not (controller.recording_active or controller.record_start_requested):
                    # An accepted start can still fail on the acquisition thread
                    # (segment 0 will not open), and a best-effort resume starts
                    # without a camera that refused. Say so on every tick.
                    self._recording_warnings.note_issue(
                        f"{tag}NOT recording: "
                        f"{controller.last_start_error or 'this camera did not start'}",
                        now_s=now_s,
                    )
                if controller.segment_pixel_problems:
                    index, reason = controller.segment_pixel_problems[-1]
                    self._recording_warnings.note_issue(
                        f"{tag}VIDEO PROBLEM: segment {index} {reason} "
                        f"({len(controller.segment_pixel_problems)} affected so far). "
                        "Do not rely on that video.",
                        now_s=now_s,
                    )
                if controller.closer_failures:
                    self._recording_warnings.note_issue(
                        f"{tag}{controller.closer_failures} segment(s) could not be finalized "
                        "-- their files are left in .incomplete/ inside the output folder",
                        now_s=now_s,
                    )
                if frame_gaps or incomplete or errors or append_failures:
                    self._recording_warnings.note_issue(
                        f"{tag}{frame_gaps} frame gap(s), {incomplete} incomplete image(s), "
                        f"{errors} acquisition error(s), {append_failures} append failure(s)",
                        now_s=now_s,
                    )
                if camera_reinits:
                    self._recording_warnings.note_issue(
                        f"{tag}camera reconnected after a fault ({camera_reinits} reinit(s))",
                        now_s=now_s,
                    )
                # MJPEG is opt-in specifically because its encode cost is
                # hardware-dependent (see set_compression_enabled) -- flag it
                # live from append_queue_depth (see COMPRESSION_QUEUE_DEPTH_WARNING
                # for why that's used instead of an append_ms latency percentile).
                # Both writer-queue warnings need the backlog to HOLD (see
                # WRITER_BACKLOG_SECONDS): a brief stall that drains is not a fault.
                queue_depth = row["append_queue_depth"]
                depth = int(queue_depth) if queue_depth != "" else 0
                mjpeg_slow = (
                    slot.controller.get_compression_enabled()
                    and depth >= COMPRESSION_QUEUE_DEPTH_WARNING
                )
                if rt.mjpeg_behind.update(mjpeg_slow, now_s):
                    self._recording_warnings.note_issue(
                        f"{tag}MJPEG encoding is falling behind capture (append queue depth "
                        f"{depth}) -- frames are piling up waiting to be written "
                        "and may start dropping. Consider turning off compression for "
                        "this frame rate.",
                        now_s=now_s,
                    )
                if rt.backlog.update(depth >= WRITER_BACKLOG_FRAMES, now_s):
                    self._recording_warnings.note_issue(
                        f"{tag}{depth} frames have been waiting to be written for "
                        f"over {WRITER_BACKLOG_SECONDS:.0f} s -- the disk or encoder is "
                        "not keeping up.",
                        now_s=now_s,
                    )

            if age_value == "":
                segments.append(f"{prefix}no new frame ({rendered_fps:.1f} displayed fps)")
                color = "#b71c1c"
            else:
                age_ms = float(age_value)
                segments.append(f"{prefix}{age_ms:.0f} ms, {rendered_fps:.1f} displayed fps")
                color = "#2e7d32" if age_ms <= 250.0 else "#e65100"
            if multi:
                rt.tile.set_caption(f"{slot.label} — {segments[-1]}", color)
            if color == "#b71c1c" or (color == "#e65100" and worst_color == "#2e7d32"):
                worst_color = color
            if frame_gaps or incomplete or errors:
                warn_parts.append(
                    f"{prefix}{frame_gaps} gap(s), {incomplete} incomplete, {errors} error(s)"
                )

        if recording and audio_reconnects:
            self._recording_warnings.note_issue(
                f"microphone reconnected after a dropout ({audio_reconnects} time(s))",
                now_s=now_s,
            )
        self._update_recording_warning_banner()

        # "as of HH:MM:SS" is the one thing on this label that only a live
        # Qt event loop can advance: a fully hung loop stops this timer from
        # firing at all, so every other field would otherwise keep showing
        # its last (possibly healthy-looking) reading forever with no way
        # to tell a frozen GUI from a genuinely quiet recording.
        as_of = datetime.now().strftime("%H:%M:%S")
        text = f"as of {as_of} — Preview pipeline: " + " | ".join(segments)
        color = worst_color

        if recording:
            warning_state = self._recording_warnings.summarize(now_s=time.monotonic())
            if warning_state.healthy_for_s is not None:
                text = f"Healthy for {format_duration_s(warning_state.healthy_for_s)} — {text}"
            self.setWindowTitle(f"SmartSleeve Recorder — Recording ({as_of})")
        elif self._power_paused:
            self.setWindowTitle("SmartSleeve Recorder — Paused for power safety")
        else:
            self.setWindowTitle("SmartSleeve Recorder")

        # A delayed one-second diagnostics timer indicates that the Qt event
        # loop itself was unable to run, even if it catches up with a fresh
        # frame immediately afterward.
        if interval_s > 1.5:
            text += f" — GUI stalled about {(interval_s - 1.0):.1f} s"
            color = "#b71c1c"
        if warn_parts:
            text += " — camera warnings: " + "; ".join(warn_parts)
            color = "#b71c1c"

        logger_error = next(
            (rt.logger.last_error for rt in self._slot_rt.values() if rt.logger.last_error), None
        )
        if logger_error:
            text += f" — diagnostics log error: {logger_error}"
            color = "#b71c1c"
        self.preview_health_label.setText(text)
        self.preview_health_label.setStyleSheet(
            f"color: {color}; font-size: 11px;"
        )

    def update_frame(self):
        for slot in self._slots():
            rt = self._slot_rt[slot.serial]
            started = time.perf_counter()
            preview_frame = slot.controller.get_latest_preview_frame(
                after_sequence=rt.last_seq, max_size=rt.tile.target_size()
            )
            if preview_frame is None:
                if rt.last_seq is not None:
                    rt.diagnostics.note_repeated_frame()
                continue

            qimg = frame_to_qimage(preview_frame.image)
            if qimg is None:
                # Unexpected pixel layout: skip the frame rather than break the timer.
                continue
            rt.tile.show_image(qimg)
            displayed_at = time.monotonic()
            rt.last_seq = preview_frame.sequence
            rt.diagnostics.note_displayed_frame(
                frame_id=preview_frame.frame_id,
                sequence=preview_frame.sequence,
                retrieved_at=preview_frame.retrieved_at,
                published_at=preview_frame.published_at,
                displayed_at=displayed_at,
            )
            rt.diagnostics.note_render_ms((time.perf_counter() - started) * 1000.0)

    def keyPressEvent(self, event):
        if event.isAutoRepeat():
            return
        if self.state != AppState.RECORDING:
            return

        if event.key() == Qt.Key.Key_S:
            label_event = "label_start"
        elif event.key() == Qt.Key.Key_E:
            label_event = "label_end"
        else:
            return

        adl_id = self.adl_dropdown.currentData()
        adl_label = self.adl_dropdown.currentText() if adl_id is not None else None
        self.cameras.notify_label_event(label_event, adl_id, adl_label)
        self._append_label_marker_activity(label_event, adl_id, adl_label)
        pair_completed = self._record_label_pair_event(label_event, adl_id)
        pair_message = " Completed pair." if pair_completed else ""
        self.status_label.setText(
            f"Logged {label_event} ({adl_label if adl_label else 'no ADL selected'})."
            f"{pair_message}"
        )

    def closeEvent(self, event):
        """Ensure all hardware and timers are properly stopped."""
        try:
            release_keep_awake()
        except Exception as e:
            print("Error releasing sleep prevention on close:", e)
        # Stop recording/preview/camera first
        try:
            if self.state == AppState.RECORDING:
                if self._session_audio is not None:
                    try:
                        self._session_audio.stop()
                    except Exception as exc:
                        print("Error stopping audio on close:", exc)
                    self._session_audio = None
                if self.cameras is not None:
                    self.cameras.stop_recording_all()
        except Exception as e:
            print("Error stopping recording on close:", e)
        try:
            if self.preview_running:
                self.timer.stop()
                self.preview_diagnostics_timer.stop()
                self._stop_preview_diagnostics_logging()
        except Exception as e:
            print("Error stopping timer on close:", e)
        try:
            if self.cameras is not None:
                result = self.cameras.stop_all()
                if not result.ok:
                    print(f"[camera] stop() deferred cleanup on close: {result.message}")
        except Exception as e:
            print("Error stopping camera on close:", e)

        # Then stop PulseManager (which also stops DAQ)
        try:
            if self.pulse_manager is not None:
                self.pulse_manager.stop()
                self.pulse_manager = None
        except Exception as e:
            print("Error stopping PulseManager on close:", e)

        # If for some reason PulseManager was never started but DAQ was:
        try:
            if self.daq is not None:
                self.daq.stop()
                self.daq = None
        except Exception as e:
            print("Error stopping DAQ on close:", e)

        try:
            self.power_status_timer.stop()
            self._stop_mic_preview()
            self.mic_level_timer.stop()
        except Exception as e:
            print("Error stopping mic preview on close:", e)

        super().closeEvent(event)

def main():
    app = QApplication(sys.argv)
    window = MainWindow()

    # Register cleanup on crash or normal exit
    def _cleanup_on_exit():
        try:
            if getattr(window, "daq", None) is not None:
                window.daq.stop()
                print("DAQ disconnected (atexit).")
        except Exception as e:
            print("Error during DAQ atexit cleanup:", e)

    atexit.register(_cleanup_on_exit)

    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
