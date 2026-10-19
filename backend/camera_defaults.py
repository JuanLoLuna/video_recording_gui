"""Image settings every camera starts with (Two-camera setup #2).

Chosen in Two-camera setup #1 (sleeve_adl docs/3d-setup-experiments.md, 2026-10-14) on
the two-camera rig with its battery IR light: 8 ms exposure on both cameras (4 and 2 ms
gave more keypoint jumps from the extra gain noise), gain set so the work area is well
exposed and not clipped. Auto exposure and auto gain are off so nothing drifts during a
session. The one-button auto setup (backend/auto_setup.py) re-tunes the gain per session.

SLEEVE_VIDEO_GUI_NO_IMAGE_DEFAULTS=1 leaves the cameras as they power on.
"""

from __future__ import annotations

import os

DEFAULT_EXPOSURE_US = 8000.0

# Gain (dB) chosen per rig camera; other cameras of the same model get the model's value.
DEFAULT_GAIN_DB_BY_SERIAL = {
    "23227865": 2.5,   # Firefly FFY-U3-04S2M
    "26134271": 3.4,   # Blackfly S BFS-U3-13Y3M
}
DEFAULT_GAIN_DB_BY_MODEL = (("Firefly", 2.5), ("Blackfly", 3.4))

NO_DEFAULTS_ENV = "SLEEVE_VIDEO_GUI_NO_IMAGE_DEFAULTS"


def image_defaults_enabled() -> bool:
    return os.environ.get(NO_DEFAULTS_ENV, "").strip() not in ("1", "true", "yes")


def default_gain_db(serial: str, model: str) -> float | None:
    """The starting gain for this camera, or None to leave its gain as it is."""
    if serial in DEFAULT_GAIN_DB_BY_SERIAL:
        return DEFAULT_GAIN_DB_BY_SERIAL[serial]
    for name, gain in DEFAULT_GAIN_DB_BY_MODEL:
        if name.lower() in (model or "").lower():
            return gain
    return None
