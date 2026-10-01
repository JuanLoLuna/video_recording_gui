"""Per-camera preview tiles.

A tile is a frame view plus a one-line caption. With one camera the caption is
hidden, so the window looks exactly as it did before multi-camera support; with
several, each camera gets its own tile side by side, captioned with its model,
serial and live health.

Frames reach a tile already downscaled to its on-screen size (the controller's
get_latest_preview_frame(max_size=...)), so the GUI thread only converts and
paints a small image instead of copying and smooth-scaling a full-resolution
frame for every camera every 33 ms.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QLabel, QSizePolicy, QVBoxLayout, QWidget


def frame_to_qimage(frame: np.ndarray) -> QImage | None:
    """A grayscale (2-D) or 3-channel BGR uint8 frame as an owned QImage.

    Returns None for any other layout, so a caller can skip the frame rather
    than crash the GUI timer (the old code did the same: "just bail").
    """
    if frame.ndim == 2:
        height, width = frame.shape
        return QImage(
            frame.data, width, height, width, QImage.Format.Format_Grayscale8
        ).copy()
    if frame.ndim == 3 and frame.shape[2] == 3:
        height, width, _ = frame.shape
        return QImage(
            frame.data, width, height, 3 * width, QImage.Format.Format_RGB888
        ).rgbSwapped().copy()
    return None


class CameraPreviewTile(QWidget):
    def __init__(self, parent: QWidget | None = None, *, min_size: tuple[int, int] = (480, 320)):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        self.image_label = QLabel("No video")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(*min_size)
        self.image_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        layout.addWidget(self.image_label, stretch=1)

        self.caption = QLabel("")
        self.caption.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.caption.setStyleSheet("color: #555; font-size: 11px;")
        self.caption.setVisible(False)
        layout.addWidget(self.caption)

    def target_size(self) -> tuple[int, int]:
        """Pixel size the next frame should be downscaled to fit."""
        return (max(1, self.image_label.width()), max(1, self.image_label.height()))

    def show_image(self, image: QImage) -> None:
        pixmap = QPixmap.fromImage(image)
        self.image_label.setPixmap(
            pixmap.scaled(
                self.image_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                # The frame is already close to this size; a fast transform is
                # all that is left to do on the GUI thread.
                Qt.TransformationMode.FastTransformation,
            )
        )

    def clear(self, text: str = "No video") -> None:
        self.image_label.setPixmap(QPixmap())
        self.image_label.setText(text)

    def set_caption(self, text: str, color: str = "#555") -> None:
        self.caption.setText(text)
        self.caption.setStyleSheet(f"color: {color}; font-size: 11px;")

    def set_caption_visible(self, visible: bool) -> None:
        self.caption.setVisible(visible)
