"""The "3D pose" line under the camera previews, with the Calibrate… button.

Informational only: it never blocks recording. The text comes from
backend.calibration_store.assess_session().
"""

from __future__ import annotations

from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton

from backend.calibration_store import SessionStatus

_READY_STYLE = "QFrame#calibrationStatusFrame { border: 1px solid #a5d6a7; border-radius: 4px; background: #e8f5e9; }"
_NOT_READY_STYLE = "QFrame#calibrationStatusFrame { border: 1px solid #ffcc80; border-radius: 4px; background: #fff3e0; }"
_IDLE_STYLE = "QFrame#calibrationStatusFrame { border: 1px solid #ddd; border-radius: 4px; background: #f7f7f7; }"


class CalibrationStatusBar(QFrame):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("calibrationStatusFrame")
        row = QHBoxLayout(self)
        row.setContentsMargins(8, 4, 8, 4)
        self.label = QLabel()
        self.label.setWordWrap(True)
        row.addWidget(self.label, stretch=1)
        self.calibrate_button = QPushButton("Calibrate…")
        self.calibrate_button.setToolTip("Calibrate a camera, or set up the cameras for 3D for this session")
        row.addWidget(self.calibrate_button)
        self.show_idle()

    def show_idle(self, text: str = "3D pose: detect cameras first") -> None:
        self.setStyleSheet(_IDLE_STYLE)
        self.label.setText(text)
        self.label.setToolTip("")

    def show_status(self, status: SessionStatus) -> None:
        self.setStyleSheet(_READY_STYLE if status.ready else _NOT_READY_STYLE)
        self.label.setText(status.headline)
        notes = [f"[{c.label}] {n}" for c in status.cameras for n in c.notes]
        self.label.setToolTip("\n".join(notes))
