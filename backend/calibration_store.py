"""Where calibrations live, and whether they still apply (plan step 11).

Pure: no PySpin, no Qt, no OpenCV. Records are JSON (human-readable, diffable;
the lab's .npz layout is an export, plan step 15).

Layout under the calibration directory:

    intrinsics/<serial>/<YYYYMMDD_HHMMSS>.json   one per calibration, never overwritten
    intrinsics/<serial>/current.json             {"file": "<name>.json"} -> the active one
    setups/<YYYYMMDD_HHMMSS>.json                one per session setup
    setups/current.json

Intrinsics are tied to the camera settings that change them (the sensor
fingerprint: resolution, ROI offset, binning, decimation, mirroring, pixel
format). The lens is manual, so focus/zoom changes cannot be read from the
camera; only a live check against the board can catch those ("suspect").

Nothing here ever blocks recording: assess_session() returns what the GUI
shows ("3D pose: ready" or the reason it is not) and what goes into the
session's files.
"""

from __future__ import annotations

import json
import os
import sys
import uuid
from dataclasses import MISSING as MISSING_DEFAULT
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Mapping, Sequence

CALIBRATION_DIR_ENV = "SLEEVE_VIDEO_GUI_CALIBRATION_DIR"
SCHEMA_VERSION = 1

# Camera nodes whose value changes what a pixel means. Types are fixed per
# node so the controller can read them with the right PySpin pointer.
FINGERPRINT_NODES: dict[str, str] = {
    "Width": "int",
    "Height": "int",
    "OffsetX": "int",
    "OffsetY": "int",
    "BinningHorizontal": "int",
    "BinningVertical": "int",
    "DecimationHorizontal": "int",
    "DecimationVertical": "int",
    "ReverseX": "bool",
    "ReverseY": "bool",
    "PixelFormat": "enum",
}

# An intrinsics calibration older than this gets a note (not a block): lenses
# get bumped, and an occasional re-check is cheap.
OLD_AFTER_DAYS = 180


def default_calibration_dir(env: Mapping[str, str] | None = None, platform: str | None = None) -> Path:
    env = os.environ if env is None else env
    platform = sys.platform if platform is None else platform
    override = env.get(CALIBRATION_DIR_ENV, "").strip()
    if override:
        return Path(override).expanduser()
    if platform.startswith("win"):
        base = env.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        return Path(base) / "SleeveVideoGUI" / "calibration"
    return Path.home() / ".local" / "share" / "SleeveVideoGUI" / "calibration"


def stamp(now: datetime) -> str:
    return now.strftime("%Y%m%d_%H%M%S")


def _parse_record(cls, data, container_fields: dict):
    """cls(**known fields of data), or None if data is not a usable record of this schema.

    Extra keys are ignored (forward compatible) and missing optional ones take
    their defaults. A record that is valid JSON but lacks a required field, has
    the wrong kind of value in one, or comes from a newer schema is "unreadable":
    None, so a damaged file can never stop the window or a recording.
    """
    if not isinstance(data, dict):
        return None
    version = data.get("schema_version", SCHEMA_VERSION)
    if not isinstance(version, int) or isinstance(version, bool) or version > SCHEMA_VERSION:
        return None
    required = [name for name, f in cls.__dataclass_fields__.items()
                if f.default is MISSING_DEFAULT and f.default_factory is MISSING_DEFAULT]
    if any(name not in data for name in required):
        return None
    for name, kind in container_fields.items():
        if name in data and data[name] is not None and not isinstance(data[name], kind):
            return None
        if name in required and data[name] is None:
            return None
    try:
        return cls(**{k: data[k] for k in cls.__dataclass_fields__ if k in data})
    except (TypeError, ValueError):
        return None


def fingerprint_differences(stored: Mapping, live: Mapping) -> list[str]:
    """Settings that differ, as 'Name stored -> live'. A node either side could not read is not a difference."""
    out = []
    for name in FINGERPRINT_NODES:
        a, b = stored.get(name), live.get(name)
        if a is None or b is None:
            continue
        if a != b:
            out.append(f"{name} {a} -> {b}")
    return out


# --------------------------------------------------------------------------
# Records
# --------------------------------------------------------------------------

@dataclass
class IntrinsicsRecord:
    serial: str
    model: str
    K: list                       # 3x3, nested lists
    D: list                       # distortion coefficients
    image_size: list              # [width, height]
    fingerprint: dict
    board: dict                   # calibration.BoardConfig.to_dict()
    rms_px: float
    n_views: int
    per_view_rms_px: list = field(default_factory=list)
    coverage: dict = field(default_factory=dict)
    loose: bool = False           # saved although it failed the threshold
    created_at: str = ""          # ISO 8601, local time
    app_commit: str | None = None
    id: str = ""                  # file stem, set by the store
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "IntrinsicsRecord | None":
        return _parse_record(cls, data, {"K": list, "D": list, "image_size": list, "fingerprint": dict,
                                         "board": dict, "per_view_rms_px": list, "coverage": dict,
                                         "rms_px": (int, float), "created_at": str})


@dataclass
class CameraSetup:
    serial: str
    R: list                       # board -> camera rotation, 3x3
    t: list                       # board -> camera translation, metres
    rms_px: float
    intrinsics_id: str            # which IntrinsicsRecord the pose was solved with


@dataclass
class SetupRecord:
    cameras: list                 # [CameraSetup as dict]
    board: dict                   # the FIXED board's config (its frame is the world frame)
    baseline_mm: float | None
    triangulation_rms_mm: float | None
    passed: bool
    verify: dict | None = None    # calibration.VerifyResult fields + "passed"
    created_at: str = ""
    sync_mode: str = "free"
    app_commit: str | None = None
    id: str = ""
    schema_version: int = SCHEMA_VERSION

    @property
    def serials(self) -> list[str]:
        return [c["serial"] for c in self.cameras]

    def camera(self, serial: str) -> dict | None:
        return next((c for c in self.cameras if c["serial"] == serial), None)

    @property
    def verified(self) -> bool:
        return bool(self.verify and self.verify.get("passed"))

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "SetupRecord | None":
        record = _parse_record(cls, data, {"cameras": list, "board": dict, "verify": dict,
                                                  "created_at": str})
        if record is not None and not all(isinstance(c, dict) and "serial" in c for c in record.cameras):
            return None
        return record


# --------------------------------------------------------------------------
# Store
# --------------------------------------------------------------------------

def _write_json_atomic(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # A name of its own, so two writers (two app instances) never share a temp file.
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
    try:
        with open(tmp, "x", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _id_order(stem: str) -> tuple:
    """Sort key for record ids '<YYYYMMDD>_<HHMMSS>[_<n>]': chronological, '_10' after '_2'."""
    parts = stem.split("_")
    if len(parts) in (2, 3) and all(p.isdigit() for p in parts):
        return (parts[0] + parts[1], int(parts[2]) if len(parts) == 3 else 1, stem)
    return (stem, 0, stem)


def _read_json(path: Path) -> dict | None:
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


class CalibrationStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    # -- helpers -----------------------------------------------------------
    @staticmethod
    def _claim_name(folder: Path, base: str) -> str:
        """Create '<name>.json' exclusively (an empty placeholder) and return the name.

        Exclusive creation is what makes "never overwritten" hold with two
        writers: the loser of a race gets FileExistsError and takes the next name.
        """
        folder.mkdir(parents=True, exist_ok=True)
        name, n = base, 1
        while True:
            try:
                with open(folder / f"{name}.json", "x"):
                    return name
            except FileExistsError:
                n += 1
                name = f"{base}_{n}"

    def _write_new_record(self, folder: Path, record, base: str) -> Path:
        record.id = self._claim_name(folder, base)
        path = folder / f"{record.id}.json"
        try:
            _write_json_atomic(path, record.to_dict())   # replaces the placeholder
        except BaseException:
            try:
                path.unlink()   # never leave an empty claim behind
            except OSError:
                pass
            raise
        return path

    def _set_current(self, folder: Path, file_stem: str) -> None:
        _write_json_atomic(folder / "current.json", {"file": f"{file_stem}.json"})

    def _current(self, folder: Path) -> dict | None:
        pointer = _read_json(folder / "current.json")
        if not pointer or not isinstance(pointer.get("file"), str):
            return None
        return _read_json(folder / pointer["file"])

    # -- intrinsics --------------------------------------------------------
    def intrinsics_dir(self, serial: str) -> Path:
        return self.root / "intrinsics" / str(serial)

    def save_intrinsics(self, record: IntrinsicsRecord, now: datetime | None = None,
                        make_current: bool = True) -> Path:
        now = now or datetime.now()
        folder = self.intrinsics_dir(record.serial)
        record.created_at = record.created_at or now.isoformat(timespec="seconds")
        path = self._write_new_record(folder, record, stamp(now))
        if make_current:
            self._set_current(folder, record.id)
        return path

    def set_current_intrinsics(self, record: IntrinsicsRecord) -> Path:
        """Make an already-saved record the active one (the second half of save_intrinsics)."""
        folder = self.intrinsics_dir(record.serial)
        self._set_current(folder, record.id)
        return folder / f"{record.id}.json"

    def current_intrinsics(self, serial: str) -> IntrinsicsRecord | None:
        data = self._current(self.intrinsics_dir(serial))
        return None if data is None else IntrinsicsRecord.from_dict(data)  # None = unreadable

    def intrinsics_history(self, serial: str) -> list[IntrinsicsRecord]:
        folder = self.intrinsics_dir(serial)
        out = []
        for path in sorted(folder.glob("*.json"), key=lambda p: _id_order(p.stem)):
            if path.name == "current.json":
                continue
            record = IntrinsicsRecord.from_dict(_read_json(path))
            if record is not None:  # unreadable records are skipped
                out.append(record)
        return out

    # -- setups ------------------------------------------------------------
    @property
    def setups_dir(self) -> Path:
        return self.root / "setups"

    def save_setup(self, record: SetupRecord, now: datetime | None = None) -> Path:
        now = now or datetime.now()
        record.created_at = record.created_at or now.isoformat(timespec="seconds")
        path = self._write_new_record(self.setups_dir, record, stamp(now))
        self._set_current(self.setups_dir, record.id)
        return path

    def current_setup(self) -> SetupRecord | None:
        data = self._current(self.setups_dir)
        return None if data is None else SetupRecord.from_dict(data)


# --------------------------------------------------------------------------
# Assessment
# --------------------------------------------------------------------------

MISSING = "missing"
MISMATCH = "mismatch"
LOOSE = "loose"
SUSPECT = "suspect"
READY = "ready"


@dataclass
class CameraStatus:
    serial: str
    label: str
    state: str                       # missing | mismatch | loose | suspect | ready
    reason: str = ""                 # why it is not ready (plain language)
    notes: list = field(default_factory=list)  # never block: "calibrated 200 days ago", ...
    intrinsics_id: str | None = None

    @property
    def ready(self) -> bool:
        return self.state == READY


def assess_camera(serial: str, label: str, record: IntrinsicsRecord | None,
                  live_fingerprint: Mapping | None, now: datetime,
                  live_rms_px: float | None = None, suspect_rms_px: float = 1.0) -> CameraStatus:
    if record is None:
        return CameraStatus(serial, label, MISSING, "camera not calibrated")
    notes = []
    if live_fingerprint is None:
        notes.append("camera settings checked when Preview starts")
    else:
        diffs = fingerprint_differences(record.fingerprint, live_fingerprint)
        # The stored image size is checked on its own too, so a record whose
        # fingerprint is empty or lacks Width/Height still notices a resolution change.
        if len(record.image_size) == 2:
            for name, stored in zip(("Width", "Height"), record.image_size):
                live = live_fingerprint.get(name)
                if live is not None and stored != live and not any(d.startswith(f"{name} ") for d in diffs):
                    diffs.append(f"{name} {stored} -> {live}")
        if diffs:
            return CameraStatus(serial, label, MISMATCH,
                                "camera settings changed since calibration (" + "; ".join(diffs) + ")",
                                notes, record.id)
    if not record.fingerprint:
        notes.append("saved without a record of the camera settings: changes to them cannot be detected")
    if record.loose:
        return CameraStatus(serial, label, LOOSE,
                            f"calibration saved although it failed ({record.rms_px:.2f} px)", notes, record.id)
    if live_rms_px is not None and live_rms_px > suspect_rms_px:
        return CameraStatus(serial, label, SUSPECT,
                            f"lens may have moved: board fits {live_rms_px:.1f} px off with the saved calibration",
                            notes, record.id)
    try:
        age = now - datetime.fromisoformat(record.created_at)
        if age > timedelta(days=OLD_AFTER_DAYS):
            notes.append(f"calibrated {age.days} days ago")
    except ValueError:
        pass
    return CameraStatus(serial, label, READY, "", notes, record.id)


NO_SETUP = "no setup"
SETUP_OTHER_CAMERAS = "setup for other cameras"
SETUP_STALE = "setup uses an old calibration"
SETUP_FAILED = "setup failed"
SETUP_UNVERIFIED = "setup not verified"
SETUP_MOVED = "camera moved"
SETUP_OLD_SESSION = "setup from before this session"


@dataclass
class SessionStatus:
    ready: bool
    headline: str
    cameras: list                     # [CameraStatus]
    setup_state: str | None           # None = setup fine
    setup_id: str | None
    reasons: list                     # every blocking reason, "[label] reason"

    def to_dict(self) -> dict:
        return {
            "ready": self.ready,
            "headline": self.headline,
            "setup_state": self.setup_state,
            "setup_id": self.setup_id,
            "reasons": list(self.reasons),
            "cameras": [asdict(c) for c in self.cameras],
        }


def assess_session(cameras: Sequence[tuple[str, str]], camera_statuses: Mapping[str, CameraStatus],
                   setup: SetupRecord | None, *, setup_valid_since: datetime | None = None,
                   moved: Mapping[str, bool] | None = None) -> SessionStatus:
    """Combine per-camera status and the current setup into one line for the GUI.

    cameras: (serial, label) of every camera in use, in display order.
    setup_valid_since: when the cameras were last detected (they may have been
        re-mounted before that). Used only while the live fixed-board check
        has not answered for every camera: then a setup older than this is
        stale. None = do not apply this rule.
    moved: serial -> True/False from the live fixed-board check (the board is
        in view and the camera's pose vs the saved setup was measured). When
        every camera has an answer it decides on its own: nothing moved =
        an older setup is still good (no need to redo it every session).
    """
    statuses = [camera_statuses[s] for s, _ in cameras]
    reasons = [f"[{c.label}] {c.reason}" for c in statuses if not c.ready]
    setup_state = None
    serials = [s for s, _ in cameras]

    if len(cameras) < 2:
        reasons.insert(0, "needs two cameras")
        setup_state = None
    elif setup is None:
        setup_state = NO_SETUP
    elif sorted(setup.serials) != sorted(serials):
        setup_state = SETUP_OTHER_CAMERAS
    elif not setup.passed:
        setup_state = SETUP_FAILED
    elif any(c.ready and (setup.camera(c.serial) or {}).get("intrinsics_id") != c.intrinsics_id
             for c in statuses):
        setup_state = SETUP_STALE
    elif moved and any(moved.get(s) for s in serials):
        setup_state = SETUP_MOVED
    elif (not (moved and all(s in moved for s in serials))
          and setup_valid_since is not None and _created(setup) < setup_valid_since):
        setup_state = SETUP_OLD_SESSION
    elif not setup.verified:
        setup_state = SETUP_UNVERIFIED

    if setup_state is not None:
        detail = setup_state
        if setup_state == SETUP_MOVED:
            labels = [lbl for s, lbl in cameras if moved and moved.get(s)]
            detail = f"camera moved since setup ({', '.join(labels)})"
        reasons.append(detail)

    ready = not reasons
    headline = "3D pose: ready" if ready else "3D pose not available: " + "; ".join(reasons)
    return SessionStatus(ready=ready, headline=headline, cameras=statuses, setup_state=setup_state,
                         setup_id=None if setup is None else setup.id, reasons=reasons)


def session_snapshot(store: CalibrationStore, serials: Sequence[str], status: SessionStatus) -> dict:
    """What a recording carries about 3D calibration: the status at start and every record it rests on.

    Written next to the video as <basename>_calibration.json whether or not 3D
    is ready, so an analysis can always tell what was known at recording time.
    """
    setup = store.current_setup()
    return {
        "schema_version": SCHEMA_VERSION,
        "status": status.to_dict(),
        "setup": None if setup is None else setup.to_dict(),
        "intrinsics": {
            s: (None if (rec := store.current_intrinsics(s)) is None else rec.to_dict()) for s in serials
        },
    }


def failed_snapshot(error: str) -> dict:
    """The snapshot when the calibration status itself could not be computed: still says so, next to the video."""
    return {
        "schema_version": SCHEMA_VERSION,
        "status": {"ready": False, "headline": "3D pose: could not read calibrations", "error": error,
                   "reasons": [error], "cameras": []},
        "setup": None,
        "intrinsics": {},
    }


def write_session_snapshot(path: str | Path, snapshot: dict) -> None:
    _write_json_atomic(Path(path), snapshot)


def _created(setup: SetupRecord) -> datetime:
    try:
        return datetime.fromisoformat(setup.created_at)
    except ValueError:
        return datetime.min
