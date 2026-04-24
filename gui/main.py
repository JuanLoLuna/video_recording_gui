import atexit
import sys
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
    QPushButton,
    QLabel,
    QComboBox,
    QCheckBox,
    QProgressBar,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QFrame,
    QPlainTextEdit,
)
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QImage, QPixmap
from enum import Enum, auto
from datetime import datetime

from backend.audio_control import (
    MicLevelPreview,
    SessionAudioRecorder,
    list_audio_input_devices,
    list_audio_output_devices,
)
from backend.camera_control import detect_first_camera, CameraController
from backend.ni_control import NIDaqDO, DOLine, list_do_lines
from backend.pulse_manager import PulseManager

SYNC_WIDTH_RECORD = 0.100  # 100 ms

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
        layout.addWidget(self.status_label)

        self.detect_button = QPushButton("Detect camera")
        self.detect_button.clicked.connect(self.on_detect_clicked)
        layout.addWidget(self.detect_button)

        # --- Collapsible camera image tuning (Spinnaker GenICam) ---
        self._camera_tuning_expanded = False
        self.camera_tuning_toggle = QPushButton(
            "▶ Camera image — gain, gamma, black level"
        )
        self.camera_tuning_toggle.setStyleSheet(
            "QPushButton { text-align: left; padding: 6px 8px; }"
        )
        self.camera_tuning_toggle.clicked.connect(self._on_camera_tuning_toggle)
        layout.addWidget(self.camera_tuning_toggle)

        self.camera_tuning_panel = QWidget()
        tuning_layout = QVBoxLayout(self.camera_tuning_panel)
        tuning_layout.setContentsMargins(12, 4, 8, 4)
        self._image_slider_resolution = 1000
        self._slider_meta: dict[str, dict] = {}
        for title, nodename in (
            ("Gain", "Gain"),
            ("Gamma", "Gamma"),
            ("Black level", "BlackLevel"),
        ):
            row = QHBoxLayout()
            row.addWidget(QLabel(title))
            sl = QSlider(Qt.Orientation.Horizontal)
            sl.setRange(0, self._image_slider_resolution)
            sl.setEnabled(False)
            val_lbl = QLabel("—")
            val_lbl.setMinimumWidth(76)
            val_lbl.setAlignment(
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
            )
            row.addWidget(sl, stretch=1)
            row.addWidget(val_lbl)
            tuning_layout.addLayout(row)
            self._slider_meta[nodename] = {
                "slider": sl,
                "label": val_lbl,
                "min": 0.0,
                "max": 1.0,
                "steps": self._image_slider_resolution,
                "supported": False,
            }
            sl.valueChanged.connect(
                lambda v, n=nodename: self._on_image_slider_changed(n, v)
            )

        self.camera_tuning_hint = QLabel(
            "Start preview to enable these controls (requires Spinnaker / GenICam nodes)."
        )
        self.camera_tuning_hint.setWordWrap(True)
        self.camera_tuning_hint.setStyleSheet("color: #555; font-size: 11px;")
        tuning_layout.addWidget(self.camera_tuning_hint)
        self.camera_tuning_panel.setVisible(False)
        layout.addWidget(self.camera_tuning_panel)

        # --- Sync / NI-DAQ status ---
        self.sync_label = QLabel("Sync not available — no DAQ connected")
        layout.addWidget(self.sync_label)

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

        layout.addLayout(daq_row)

        # --- Microphone (optional WAV alongside video) ---
        self.audio_label = QLabel("Microphone: scan and choose a device, or leave as no audio.")
        layout.addWidget(self.audio_label)
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
        layout.addLayout(audio_row)

        self.audio_monitor_checkbox = QCheckBox(
            "Hear live mic in headphones (Preview mic + recording; avoids speaker feedback)"
        )
        self.audio_monitor_checkbox.setChecked(True)
        self.audio_monitor_checkbox.stateChanged.connect(self._on_audio_monitor_changed)
        layout.addWidget(self.audio_monitor_checkbox)

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
        layout.addLayout(output_row)

        mic_preview_row = QHBoxLayout()
        self.mic_preview_button = QPushButton("Preview mic")
        self.mic_preview_button.clicked.connect(self.on_mic_preview_clicked)
        mic_preview_row.addWidget(self.mic_preview_button)
        mic_preview_row.addWidget(QLabel("Input level"))
        self.audio_level_bar = QProgressBar()
        self.audio_level_bar.setRange(0, 100)
        self.audio_level_bar.setValue(0)
        self.audio_level_bar.setTextVisible(False)
        self.audio_level_bar.setFixedHeight(14)
        self.audio_level_bar.setStyleSheet(
            "QProgressBar { border: 1px solid #888; border-radius: 3px; background: #e8e8e8; }"
            "QProgressBar::chunk { background-color: #2e7d32; border-radius: 2px; }"
        )
        mic_preview_row.addWidget(self.audio_level_bar, stretch=1)
        layout.addLayout(mic_preview_row)

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

        # --- Image preview label ---
        self.image_label = QLabel("No video")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # Avoid fixed size so the window can shrink; keep a reasonable minimum.
        self.image_label.setMinimumSize(480, 320)
        self.image_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(self.image_label)

        # --- Controls row: Preview / Record / Sync ---
        self.preview_button = QPushButton("Start Preview")
        self.preview_button.clicked.connect(self.on_preview_clicked)

        self.record_button = QPushButton("Start Recording")
        self.record_button.clicked.connect(self.on_record_clicked)

        # Sync pulse button (formerly "test pulse")
        self.sync_button = QPushButton("Sync Pulse")
        self.sync_button.setEnabled(False)  # enabled after DAQ connects
        self.sync_button.clicked.connect(self.on_sync_pulse_clicked)

        controls_row = QHBoxLayout()
        controls_row.addWidget(self.preview_button)
        controls_row.addWidget(self.record_button)
        controls_row.addWidget(self.sync_button)

        layout.addLayout(controls_row)

        # --- ADL label feedback (keys S / E while recording → CSV metadata) ---
        self._label_frame_base_style = (
            "QFrame#labelMarkerFrame { border: 1px solid #bbb; border-radius: 6px; "
            "padding: 6px; background: #f5f5f5; }"
        )
        self.label_marker_frame = QFrame()
        self.label_marker_frame.setObjectName("labelMarkerFrame")
        self.label_marker_frame.setStyleSheet(self._label_frame_base_style)
        marker_layout = QVBoxLayout(self.label_marker_frame)
        marker_layout.setSpacing(6)
        self.label_hint = QLabel(
            "While recording: press S for label START and E for END (logged with the next "
            "frames). Select an ADL in the row below first."
        )
        self.label_hint.setWordWrap(True)
        marker_layout.addWidget(self.label_hint)
        last_row = QHBoxLayout()
        self.label_last_event = QLabel("No marks yet.")
        self.label_last_event.setStyleSheet("font-weight: 600;")
        last_row.addWidget(self.label_last_event, stretch=1)
        self.label_count = QLabel("Marks this take: 0")
        last_row.addWidget(self.label_count)
        marker_layout.addLayout(last_row)
        self.label_history = QPlainTextEdit()
        self.label_history.setReadOnly(True)
        self.label_history.setFixedHeight(88)
        self.label_history.setPlaceholderText("Label events appear here during recording.")
        self.label_history.document().setMaximumBlockCount(80)
        marker_layout.addWidget(self.label_history)
        layout.addWidget(self.label_marker_frame)
        self._label_mark_count = 0

        # --- Experiment + label dropdowns (bottom of GUI) ---
        adl_row = QHBoxLayout()
        self.experiment_dropdown = QComboBox()
        for experiment_name in EXPERIMENT_LABELS:
            self.experiment_dropdown.addItem(experiment_name)
        self.experiment_dropdown.currentIndexChanged.connect(self.on_experiment_changed)
        adl_row.addWidget(self.experiment_dropdown)

        self.adl_dropdown = QComboBox()
        adl_row.addWidget(self.adl_dropdown)
        self._populate_label_dropdown(self.experiment_dropdown.currentText())
        layout.addLayout(adl_row)

        # --- Camera controller + timer ---
        self.camera = CameraController()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.preview_running = False

        self.state = AppState.IDLE
        self._apply_state()

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
            self.preview_button.setEnabled(True)
            self.preview_button.setText("Stop Preview")
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

        if hasattr(self, "label_hint"):
            if self.state == AppState.RECORDING:
                self.label_hint.setText(
                    "Recording — press S for label START and E for END (written to the "
                    "metadata CSV on the next frames). Choose the ADL in the row below first."
                )
            else:
                self.label_hint.setText(
                    "While recording: press S for label START and E for END (logged with the "
                    "next frames). Select an ADL in the row below first."
                )

    def _on_camera_tuning_toggle(self) -> None:
        self._camera_tuning_expanded = not self._camera_tuning_expanded
        self.camera_tuning_panel.setVisible(self._camera_tuning_expanded)
        arrow = "▼" if self._camera_tuning_expanded else "▶"
        self.camera_tuning_toggle.setText(
            f"{arrow} Camera image — gain, gamma, black level"
        )

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

    def _sync_image_sliders_from_camera(self) -> None:
        """Read limits from the open camera and align sliders (call after preview starts)."""
        steps = self._image_slider_resolution
        for node, meta in self._slider_meta.items():
            sl = meta["slider"]
            lbl = meta["label"]
            limits = self.camera.get_image_param_limits(node)
            if limits is None:
                meta["supported"] = False
                sl.blockSignals(True)
                sl.setEnabled(False)
                lbl.setText("N/A")
                sl.blockSignals(False)
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
                can_tune = self.state in (
                    AppState.PREVIEWING,
                    AppState.RECORDING,
                )
                sl.setEnabled(can_tune)
                t = (cur - mn) / (mx - mn)
                sl.setValue(int(round(t * steps)))
                lbl.setText(f"{cur:.4g}")
            sl.blockSignals(False)

    def _update_camera_tuning_widgets_enabled(self) -> None:
        if not self._slider_meta:
            return
        acquiring = self.state in (AppState.PREVIEWING, AppState.RECORDING)
        if hasattr(self, "camera_tuning_hint"):
            self.camera_tuning_hint.setVisible(not acquiring)
        for meta in self._slider_meta.values():
            sl = meta["slider"]
            if not meta.get("supported"):
                sl.setEnabled(False)
                continue
            sl.setEnabled(acquiring)

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
        """Clear the on-screen label log when a new recording starts."""
        self._label_mark_count = 0
        self.label_count.setText("Marks this take: 0")
        self.label_last_event.setText("No marks logged yet in this recording.")
        self.label_history.clear()

    def _reset_label_marker_panel_style(self) -> None:
        self.label_marker_frame.setStyleSheet(self._label_frame_base_style)

    def _flash_label_marker_panel(self, label_event: str) -> None:
        if label_event == "label_start":
            self.label_marker_frame.setStyleSheet(
                "QFrame#labelMarkerFrame { border: 2px solid #2e7d32; border-radius: 6px; "
                "padding: 6px; background: #c8e6c9; }"
            )
        else:
            self.label_marker_frame.setStyleSheet(
                "QFrame#labelMarkerFrame { border: 2px solid #e65100; border-radius: 6px; "
                "padding: 6px; background: #ffe0b2; }"
            )
        QTimer.singleShot(220, self._reset_label_marker_panel_style)

    def _append_label_marker_activity(
        self,
        label_event: str,
        adl_id,
        adl_label: str | None,
    ) -> None:
        self._label_mark_count += 1
        self.label_count.setText(f"Marks this take: {self._label_mark_count}")
        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]  # trim to ms
        if adl_id is None:
            name = "(no ADL selected)"
        else:
            name = (adl_label or "").strip() or f"ID {adl_id}"
        short = f"{label_event} — {name}"
        self.label_last_event.setText(f"Last: {short}")
        self.label_history.appendPlainText(f"{ts}  {label_event}  —  {name}")
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

    def on_detect_clicked(self):
        if self.state not in (AppState.IDLE, AppState.CAMERA_DETECTED):
            return

        self.status_label.setText("Detecting camera...")
        found, message = detect_first_camera()
        self.status_label.setText(message)

        if found:
            self.state = AppState.CAMERA_DETECTED
        else:
            self.state = AppState.IDLE

        self._apply_state()

    def on_scan_daq_clicked(self):
        """Scan for connected NI-DAQ devices and populate the dropdown."""
        self.daq_line_combo.clear()
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
            self.camera.notify_sync_pulse_window(width_s=width, label=label)

            self.sync_label.setText(f"Sync pulse sent ({label}, {width * 1000:.0f} ms).")
        except Exception as e:
            self.sync_label.setText(f"Pulse failed: {e}")

    def on_preview_clicked(self):
        if not self.preview_running:
            ok, msg = self.camera.start()
            self.status_label.setText(msg)
            if not ok:
                return

            # This controls PREVIEW fps, not camera fps.
            # 33 ms ~ 30 fps, 16 ms ~ 60 fps, 8 ms ~ 90 fps.
            self.timer.start(8)
            self.preview_running = True
            self.preview_button.setText("Stop Preview")
            self.state = AppState.PREVIEWING
            self._apply_state()
            self._sync_image_sliders_from_camera()


        else:
            # Only allow stopping preview when NOT recording
            if self.state == AppState.RECORDING:
                return  # safety, shouldn't happen if buttons are disabled correctly
            self.timer.stop()
            self.camera.stop()
            self.preview_running = False
            self.image_label.setPixmap(QPixmap())
            self.image_label.setText("No video")
            self.status_label.setText("Preview stopped.")
            self.state = AppState.CAMERA_DETECTED
            self._apply_state()

    def on_record_clicked(self):
        # Start recording
        if self.state == AppState.PREVIEWING:
            # SpinVideo adds its own suffix; avoid a double ".avi"
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"recording_{timestamp}"

            self.record_button.setEnabled(False)
            self.record_button.setText("Starting recording… please wait")
            self.status_label.setText("Starting recording…")
            QApplication.processEvents()

            ok, msg = self.camera.start_recording(filename, fps=30.0)
            self.status_label.setText(msg)
            QApplication.processEvents()

            if not ok:
                self._apply_state()
                return

            self._stop_mic_preview()
            # Parallel WAV with same session base name as video (recording_<ts>.avi / .wav)
            audio_device = self.audio_input_combo.currentData()
            self._session_audio = None
            if audio_device is not None:
                wav_path = f"{filename}.wav"
                try:
                    self._session_audio = SessionAudioRecorder()
                    out_dev = self.audio_output_combo.currentData()
                    self._session_audio.start(
                        wav_path,
                        int(audio_device),
                        monitor=self.audio_monitor_checkbox.isChecked(),
                        output_device=int(out_dev) if out_dev is not None else None,
                    )
                    self.audio_label.setText(
                        f"Microphone: recording to {wav_path}"
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
                self.camera.notify_sync_pulse_window(
                    width_s=SYNC_WIDTH_RECORD,
                    label="record_start",
                )

            self.manual_sync_count = 0  # reset manual counter
            self._reset_label_marker_session()

            self.state = AppState.RECORDING
            self._apply_state()

        # Stop recording
        elif self.state == AppState.RECORDING:
            if self._session_audio is not None:
                try:
                    self._session_audio.stop()
                except Exception as exc:
                    print("Error stopping audio recording:", exc)
                self._session_audio = None
            self.camera.stop_recording()
            self.status_label.setText("Recording stopped.")
            self.audio_label.setText(
                "Microphone: scan and choose a device, or leave as no audio."
            )
            self.state = AppState.PREVIEWING
            self._apply_state()

    def update_frame(self):
        frame = self.camera.get_latest_frame()
        if frame is None:
            return

        # Handle grayscale vs color
        if frame.ndim == 2:
            height, width = frame.shape
            bytes_per_line = width
            qimg = QImage(
                frame.data, width, height, bytes_per_line, QImage.Format.Format_Grayscale8
            ).copy()
        else:
            height, width, channels = frame.shape
            if channels == 3:
                bytes_per_line = 3 * width
                qimg = QImage(
                    frame.data,
                    width,
                    height,
                    bytes_per_line,
                    QImage.Format.Format_RGB888,
                ).rgbSwapped().copy()
            else:
                # Fallback: just bail if format is unexpected
                return

        pixmap = QPixmap.fromImage(qimg)
        self.image_label.setPixmap(pixmap.scaled(
            self.image_label.size(), Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        ))

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
        self.camera.notify_label_event(label_event, adl_id, adl_label)
        self._append_label_marker_activity(label_event, adl_id, adl_label)
        self.status_label.setText(
            f"Logged {label_event} ({adl_label if adl_label else 'no ADL selected'})."
        )

    def closeEvent(self, event):
        """Ensure all hardware and timers are properly stopped."""
        # Stop recording/preview/camera first
        try:
            if self.state == AppState.RECORDING:
                if self._session_audio is not None:
                    try:
                        self._session_audio.stop()
                    except Exception as exc:
                        print("Error stopping audio on close:", exc)
                    self._session_audio = None
                self.camera.stop_recording()
        except Exception as e:
            print("Error stopping recording on close:", e)
        try:
            if self.preview_running:
                self.timer.stop()
        except Exception as e:
            print("Error stopping timer on close:", e)
        try:
            self.camera.stop()
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
