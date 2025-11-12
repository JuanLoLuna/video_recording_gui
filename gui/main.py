import sys
import cv2
from PySide6.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QPushButton,
    QLabel,
)
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QImage, QPixmap
from enum import Enum, auto
from datetime import datetime

from backend.camera_control import detect_first_camera, CameraController

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

        # --- Image preview label ---
        self.image_label = QLabel("No video")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setFixedSize(640, 480)
        layout.addWidget(self.image_label)

        # --- Preview button ---
        self.preview_button = QPushButton("Start Preview")
        self.preview_button.clicked.connect(self.on_preview_clicked)
        layout.addWidget(self.preview_button)

        # --- Record button ---
        self.record_button = QPushButton("Start Recording")
        self.record_button.clicked.connect(self.on_record_clicked)
        layout.addWidget(self.record_button)

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
        if self.state == AppState.RECORDING:
            self.camera.stop_recording()
        if self.preview_running:
            self.timer.stop()
        self.camera.stop()
        super().closeEvent(event)

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
