import atexit
import sys
import cv2
from PySide6.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
)
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QImage, QPixmap
from enum import Enum, auto
from datetime import datetime

from backend.camera_control import detect_first_camera, CameraController
from backend.ni_control import NIDaqDO, DOLine
from backend.pulse_manager import PulseManager

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

        layout = QVBoxLayout(self)

        self.status_label = QLabel("Press the button to detect a camera.")
        layout.addWidget(self.status_label)

        self.detect_button = QPushButton("Detect camera")
        self.detect_button.clicked.connect(self.on_detect_clicked)
        layout.addWidget(self.detect_button)

        # --- Sync / NI-DAQ status ---
        self.sync_label = QLabel("Sync not available — no DAQ connected")
        layout.addWidget(self.sync_label)

        self.connect_daq_button = QPushButton("Connect DAQ")
        self.connect_daq_button.clicked.connect(self.on_connect_daq_clicked)
        layout.addWidget(self.connect_daq_button)

        # Disable the button on non-Windows
        if not sys.platform.startswith("win"):
            self.connect_daq_button.setEnabled(False)
            self.sync_label.setText("Sync not available on this OS")

        # Handle to the DAQ controller (set on connect)
        self.daq = None
        self.pulse_manager = None

        # --- Image preview label ---
        self.image_label = QLabel("No video")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setFixedSize(640, 480)
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

        # --- Camera controller + timer ---
        self.camera = CameraController()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.preview_running = False

        self.state = AppState.IDLE
        self._apply_state()

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

    def on_connect_daq_clicked(self):
        """Connect to NI-DAQ and update UI."""
        try:
            cfg = DOLine(line="Dev1/port0/line0", idle_low=True)
            self.daq = NIDaqDO(cfg)
            self.daq.start()

            # Start PulseManager on top of the DAQ
            self.pulse_manager = PulseManager(daq=self.daq, default_width_s=0.010)
            self.pulse_manager.start()

            # Success → update UI and disable button
            self.sync_label.setText("Sync available — DAQ connected")
            self.connect_daq_button.setEnabled(False)
            self.sync_button.setEnabled(True)

        except Exception as e:
            # Keep it silent in UI per your preference; show brief text
            self.sync_label.setText(f"Sync not available — {e.__class__.__name__}")
            self.daq = None
            self.pulse_manager = None
            self.sync_button.setEnabled(False)

    def on_sync_pulse_clicked(self):
        """Send a single TTL sync pulse through PulseManager."""
        try:
            if self.pulse_manager is not None:
                self.pulse_manager.request_pulse(label="manual_sync")
                self.sync_label.setText("Sync pulse sent!")
            else:
                self.sync_label.setText("DAQ not connected.")
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
            # Simple auto filename: recording_YYYYmmdd_HHMMSS.avi
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"recording_{timestamp}.avi"

            ok, msg = self.camera.start_recording(filename, fps=30.0)
            self.status_label.setText(msg)

            if ok:
                self.state = AppState.RECORDING
                self._apply_state()

        # Stop recording
        elif self.state == AppState.RECORDING:
            self.camera.stop_recording()
            self.status_label.setText("Recording stopped.")
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

    def closeEvent(self, event):
        """Ensure all hardware and timers are properly stopped."""
        # Stop recording/preview/camera first
        try:
            if self.state == AppState.RECORDING:
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
