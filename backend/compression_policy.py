"""Which codec the GUI pre-selects for the "Compress recordings (MJPEG)" box.

The box is only a suggestion; the user can always change it before recording.

  - One camera: MJPEG at or below 30 fps, uncompressed above that. Measured on
    the rig (plain FFMPEG MJPG, ~42 dB PSNR, 3-8 ms/frame); the study's existing
    video is MJPEG as well.
  - Several cameras: uncompressed (GREY) at every rate by default. Two cameras
    with MJPEG passed the writer-stage probe at 30 and 60 fps, but the full
    recording stack (controllers, rotation, fault recovery, GUI) has so far only
    been run on the rig with uncompressed video. Raise
    MULTI_CAMERA_MJPEG_MAX_FPS (e.g. to 30.0) once that run has passed.
"""

from __future__ import annotations

ONE_CAMERA_MJPEG_MAX_FPS = 30.0
# None = never suggest MJPEG when several cameras record (it can still be ticked).
MULTI_CAMERA_MJPEG_MAX_FPS: float | None = None


def mjpeg_max_fps(camera_count: int) -> float | None:
    """Highest frame rate at which MJPEG is suggested, or None for 'never'."""
    return ONE_CAMERA_MJPEG_MAX_FPS if camera_count <= 1 else MULTI_CAMERA_MJPEG_MAX_FPS


def suggested_compression(camera_count: int, fps: float) -> bool:
    limit = mjpeg_max_fps(camera_count)
    return limit is not None and fps <= limit


def describe_policy(camera_count: int) -> str:
    """One sentence for the hint under the checkbox."""
    limit = mjpeg_max_fps(camera_count)
    if limit is None:
        return (
            "With several cameras the default is uncompressed at every frame rate "
            "(MJPEG has not yet been validated for them in the full recorder). "
            "Tick the box to use MJPEG: much smaller files, fixed quality. "
            "Locked while recording."
        )
    return (
        f"Suggested at or below {limit:.0f} fps: much smaller files, fixed quality; "
        "above that, uncompressed is suggested because encoding can cap the "
        "achievable frame rate. Locked while recording."
    )
