"""Sizing for preview tiles.

The preview used to copy and smooth-scale a full-resolution frame on the GUI
thread every 33 ms. With two cameras (one of them 1280x1024) that is twice the
work on the one thread that must also keep the window responsive, so each
tile is downscaled from the camera's frame to its on-screen size first.
"""

from __future__ import annotations


def fit_size(src_w: int, src_h: int, max_w: int, max_h: int) -> tuple[int, int]:
    """Largest (w, h) that fits in max_w x max_h at src_w:src_h's aspect ratio.

    Never upscales (a preview tile bigger than the frame shows the frame at
    native size), and never returns a zero dimension, so cv2.resize and QImage
    can always be handed the result.
    """
    if src_w <= 0 or src_h <= 0:
        raise ValueError(f"source size must be positive, got {src_w}x{src_h}")
    if max_w <= 0 or max_h <= 0:
        return (1, 1)
    scale = min(max_w / src_w, max_h / src_h, 1.0)
    return (max(1, int(src_w * scale)), max(1, int(src_h * scale)))
