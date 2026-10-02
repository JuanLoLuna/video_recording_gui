"""Which codec the GUI pre-selects for the "Compress recordings (MJPEG)" box.

The box is only a suggestion; the user can always change it before recording.

  - MJPEG at or below 30 fps, uncompressed above that, for one camera AND for
    several. Measured on the rig (plain FFMPEG MJPG, ~42 dB PSNR, 3-8 ms/frame;
    the study's existing video is MJPEG as well).
  - Several cameras were held back at "uncompressed" until MJPEG passed the full
    recorder (controllers, rotation, fault recovery, GUI) with two cameras. It
    did on 2026-10-02: 10 min at 30 fps, 5 min at 60 fps, and an unplug/replug
    fault run, all with the video decoded and checked afterwards. Set
    MULTI_CAMERA_MJPEG_MAX_FPS back to None to make "uncompressed" the default
    again for several cameras.
"""

from __future__ import annotations

ONE_CAMERA_MJPEG_MAX_FPS = 30.0
# None = never suggest MJPEG when several cameras record (it can still be ticked).
# 30.0 since the two-camera MJPEG validation of 2026-10-02.
MULTI_CAMERA_MJPEG_MAX_FPS: float | None = 30.0


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
            "With several cameras the default is uncompressed at every frame rate. "
            "Tick the box to use MJPEG: much smaller files, fixed quality. "
            "Locked while recording."
        )
    return (
        f"Suggested at or below {limit:.0f} fps: much smaller files, fixed quality; "
        "above that, uncompressed is suggested because encoding can cap the "
        "achievable frame rate. Locked while recording."
    )
