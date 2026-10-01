"""Which cameras a session uses, and how each is named.

Pure logic, no PySpin: the controller layer enumerates real cameras into
CameraDescriptors and hands them here, so selection can be tested anywhere.

Selection rules
  - SLEEVE_VIDEO_GUI_CAMERA_SERIALS="26134271,23227865" pins both WHICH cameras
    are used and their ORDER. The first listed serial is the primary camera.
  - Unset: every detected camera, sorted by serial; the lowest serial is the
    primary. Enumeration order is NOT stable (it varies with USB port and
    boot), which is why nothing here ever depends on it.

Naming rules (see backend/recording_paths.py)
  - The primary camera is unsuffixed, so a one-camera setup produces exactly
    the file names it always did.
  - Every other camera gets "cam<serial>".
  - A tag is decided by the camera's place in the *intended* order, not by who
    is currently plugged in: with the env var set, a camera keeps its name
    even while another one is missing. Without the env var and with a single
    camera attached, that camera is the primary (legacy names).
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from backend.recording_paths import camera_tag_for_serial

CAMERA_SERIALS_ENV = "SLEEVE_VIDEO_GUI_CAMERA_SERIALS"

_SPLIT_RE = re.compile(r"[,\s;]+")


@dataclass(frozen=True)
class CameraDescriptor:
    """A camera the enumeration layer found."""

    serial: str
    model: str = ""
    vendor: str = ""


@dataclass(frozen=True)
class BoundCamera:
    serial: str
    model: str
    tag: str | None  # None = primary (unsuffixed names)
    is_primary: bool

    @property
    def label(self) -> str:
        """Short human label for the GUI, e.g. "Blackfly S BFS-U3-13Y3M #26134271"."""
        return f"{self.model} #{self.serial}" if self.model else f"#{self.serial}"


@dataclass(frozen=True)
class CameraSelection:
    bound: tuple[BoundCamera, ...]
    missing: tuple[str, ...]  # configured serials that were not detected
    unused: tuple[CameraDescriptor, ...]  # detected but not configured

    @property
    def multi_camera(self) -> bool:
        return len(self.bound) > 1


def parse_serials_env(env: Mapping[str, str] | None = None) -> list[str]:
    """Serials from CAMERA_SERIALS_ENV: order kept, blanks and repeats dropped."""
    env_map = os.environ if env is None else env
    raw = env_map.get(CAMERA_SERIALS_ENV, "")
    seen: list[str] = []
    for item in _SPLIT_RE.split(raw):
        serial = item.strip()
        if serial and serial not in seen:
            seen.append(serial)
    return seen


def select_cameras(
    discovered: Iterable[CameraDescriptor],
    configured: Sequence[str] | None = None,
) -> CameraSelection:
    by_serial: dict[str, CameraDescriptor] = {}
    for descriptor in discovered:
        by_serial.setdefault(str(descriptor.serial), descriptor)

    configured_serials = [str(s) for s in (configured or [])]
    if configured_serials:
        ordering = configured_serials
        missing = tuple(s for s in ordering if s not in by_serial)
        unused = tuple(
            by_serial[s] for s in sorted(by_serial) if s not in configured_serials
        )
        present = [s for s in ordering if s in by_serial]
    else:
        ordering = sorted(by_serial)
        missing = ()
        unused = ()
        present = list(ordering)

    primary_serial = ordering[0] if ordering else None
    bound = tuple(
        BoundCamera(
            serial=serial,
            model=by_serial[serial].model,
            tag=None if serial == primary_serial else camera_tag_for_serial(serial),
            is_primary=serial == primary_serial,
        )
        for serial in present
    )
    return CameraSelection(bound=bound, missing=missing, unused=unused)


def format_camera_summary(selection: CameraSelection) -> str:
    """One status-line sentence for the GUI's Detect button."""
    if not selection.bound:
        if selection.missing:
            return "No configured camera found (missing: " + ", ".join(selection.missing) + ")."
        return "No cameras detected."
    parts = [
        f"{cam.label}{' (primary)' if cam.is_primary and selection.multi_camera else ''}"
        for cam in selection.bound
    ]
    count = len(selection.bound)
    text = f"{count} camera{'s' if count != 1 else ''}: " + "; ".join(parts)
    if selection.missing:
        text += ". MISSING: " + ", ".join(selection.missing)
    return text


def intersect_ranges(
    ranges: Iterable[tuple[float, float] | None],
) -> tuple[float, float] | None:
    """The (lo, hi) range every camera supports; None if there is none.

    None entries (a camera that could not report a range) are ignored rather
    than collapsing the answer, matching how the GUI already treats unreadable
    nodes as "N/A".
    """
    usable = [r for r in ranges if r is not None]
    if not usable:
        return None
    lo = max(r[0] for r in usable)
    hi = min(r[1] for r in usable)
    return (lo, hi) if lo <= hi else None
