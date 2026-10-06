"""Pure ChArUco calibration logic: board, detection, intrinsics, board pose, setup check.

No PySpin and no Qt: everything takes numpy images / arrays and returns plain
results, so it is unit-tested against synthetic board images
(tests/test_calibration.py) and reused by the Calibration window.

Units: the board frame and every translation are in METRES (OpenCV's
CharucoBoard takes square/marker lengths in whatever unit you pass; this module
always passes metres). Millimetres appear only in reported metrics, named *_mm.

Board frame (OpenCV's): origin at the board's top-left outer corner, x along
the squares_x direction, y along squares_y, z = 0 on the board surface.

Conventions follow the lab's penncubed_analyze/calibrate_stereo.py (same
default board, same intrinsics threshold) so its outputs can be exported in
that script's layout.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Sequence

import cv2
import numpy as np

# Acceptance thresholds.
# Intrinsics: the lab script's THRESH_INTRINSIC_RMS_PX.
INTRINSIC_RMS_PX = 0.5
# Setup (per camera, board pose reprojection with stored intrinsics). The lab
# script's stereo RMS threshold.
SETUP_REPROJECTION_RMS_PX = 1.0
# Setup triangulation of the board corners vs the real board. The lab's 0.5 mm
# assumes a close behaviour rig; at ~1-1.5 m with these sensors one pixel is
# ~1-2 mm on the table, so the default is looser. Tune on the rig (plan step 16).
SETUP_TRIANGULATION_RMS_MM = 2.0
# Verification (verify_setup): the board, moved to a new static spot, must
# triangulate to its real size within this, and the spot must be at least this
# far (along the board normal) from where the setup board lies. Scale errors
# from wrong intrinsics grow with that distance: on synthetic data a 10 % focal
# error gives ~2 % scale error at 20 cm and almost none for a sideways move.
VERIFY_SCALE_ERROR_PCT = 1.0
VERIFY_MIN_DEPTH_CHANGE_M = 0.15
# A stored intrinsics whose live board reprojection error exceeds this is
# "suspect" (lens probably touched).
SUSPECT_REPROJECTION_RMS_PX = 1.0
# Fixed-board drift check: the camera moved if its pose relative to the board
# changed by more than either of these since the setup was captured.
SETUP_MOVED_TRANSLATION_MM = 5.0
SETUP_MOVED_ROTATION_DEG = 0.5

MIN_MARKERS = 4
MIN_CORNERS = 6
MIN_INTRINSIC_VIEWS = 10

# Coverage guidance for the intrinsics capture (plan step 13).
COVERAGE_GRID = 3            # image split into 3x3 regions
NEAR_FAR_SPLIT = 0.25        # board bbox area / image area above this = "near"
TILTED_DEG = 20.0            # board normal vs optical axis above this = "tilted"
TILT_DIRECTIONS = ("left", "right", "up", "down")


@dataclass(frozen=True)
class BoardConfig:
    """A ChArUco board, or (kind="grid") a plain ArUco marker grid.

    For a grid the fields mean: squares_x/squares_y = markers across/down,
    marker_length_m = marker side, square_length_m = marker pitch (side + gap).
    Grids have few, big markers: they are read from ~3x farther than a
    ChArUco board of the same paper size, which is what the reference boards
    need (they only detect camera moves; calibration accuracy comes from A).
    """
    squares_x: int = 5
    squares_y: int = 5
    square_length_m: float = 0.04
    marker_length_m: float = 0.03
    dictionary: str = "DICT_5X5_50"
    # Boards sharing a dictionary must not share marker ids, or a camera that
    # sees both cannot tell them apart (B1 uses ids 0-16, B2 17-33).
    first_marker_id: int = 0
    kind: str = "charuco"

    def __post_init__(self) -> None:
        if self.kind not in ("charuco", "grid"):
            raise ValueError(f"unknown board kind {self.kind!r}")
        minimum = 1 if self.kind == "grid" else 2
        if self.squares_x < minimum or self.squares_y < minimum or self.squares_x * self.squares_y < 2:
            raise ValueError("a ChArUco board needs at least 2x2 squares, a grid at least 2 markers")
        if not 0 < self.marker_length_m < self.square_length_m:
            raise ValueError("marker_length_m must be > 0 and smaller than square_length_m")
        if not hasattr(cv2.aruco, self.dictionary):
            raise ValueError(f"unknown ArUco dictionary {self.dictionary!r}")
        size = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, self.dictionary)).bytesList.shape[0]
        if self.first_marker_id < 0 or self.first_marker_id + self.marker_count > size:
            raise ValueError(f"marker ids {self.first_marker_id}..{self.first_marker_id + self.marker_count - 1} "
                             f"do not fit in {self.dictionary} ({size} markers)")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "BoardConfig":
        return cls(**{k: data[k] for k in cls.__dataclass_fields__ if k in data})

    @property
    def gap_m(self) -> float:
        """Grid only: the white space between markers, also left as a quiet zone around them."""
        return self.square_length_m - self.marker_length_m

    @property
    def image_origin_m(self) -> float:
        """Where render_board()'s top-left pixel lies in the board frame (a grid has a quiet zone)."""
        return -self.gap_m if self.kind == "grid" else 0.0

    @property
    def size_m(self) -> tuple[float, float]:
        """Printed size, including a grid's quiet zone."""
        if self.kind == "grid":
            return (self.squares_x * self.square_length_m + self.gap_m,
                    self.squares_y * self.square_length_m + self.gap_m)
        return self.squares_x * self.square_length_m, self.squares_y * self.square_length_m

    @property
    def corner_count(self) -> int:
        if self.kind == "grid":
            return 4 * self.marker_count
        return (self.squares_x - 1) * (self.squares_y - 1)

    @property
    def marker_count(self) -> int:
        if self.kind == "grid":
            return self.squares_x * self.squares_y
        return (self.squares_x * self.squares_y) // 2

    @property
    def centre_m(self) -> tuple[float, float, float]:
        """Centre of the board's pattern in its own frame (the origin is a corner of the pattern)."""
        if self.kind == "grid":
            return ((self.squares_x * self.square_length_m - self.gap_m) / 2,
                    (self.squares_y * self.square_length_m - self.gap_m) / 2, 0.0)
        return self.squares_x * self.square_length_m / 2, self.squares_y * self.square_length_m / 2, 0.0


# The boards in use (plan decision 10). Squares/markers are the NOMINAL print
# sizes; with_measured_square() rescales both to what the printer produced.
BOARD_PRESETS: dict[str, BoardConfig] = {
    "A4 board A - handheld (5x5 markers)": BoardConfig(7, 5, 0.036, 0.027, "DICT_5X5_50"),
    "A4 board B1 - camera 1 reference (4x4, ids 0-16)": BoardConfig(7, 5, 0.036, 0.027, "DICT_4X4_50", 0),
    "A4 board B2 - camera 2 reference (4x4, ids 17-33)": BoardConfig(7, 5, 0.036, 0.027, "DICT_4X4_50", 17),
    "Lab 5x5 board, 40 mm squares": BoardConfig(),
}
# Big-marker reference boards: 6 markers of 70 mm (2.6x the 27 mm of B1/B2), with
# 12 mm gaps and quiet zone: 258 x 176 mm, fits A4 landscape inside the margins.
BOARD_PRESETS["A4 board G1 - camera 1 reference, big markers (ids 34-39)"] = BoardConfig(
    3, 2, 0.082, 0.070, "DICT_4X4_50", 34, "grid")
BOARD_PRESETS["A4 board G2 - camera 2 reference, big markers (ids 40-45)"] = BoardConfig(
    3, 2, 0.082, 0.070, "DICT_4X4_50", 40, "grid")
# Board A with big markers, for setups where A is small in a low-resolution view
# (the Firefly read A's 5x5 markers at ~13 px only partly). 6x4 squares of 45 mm,
# 4x4 markers of 36 mm: each marker cell is 6.0 mm instead of 3.9 mm. Ids 50-61 of
# DICT_4X4_100 are not in the DICT_4X4_50 set the B/G boards use, and differ from
# every one of them by >= 4 bits in any rotation (4X4_50 corrects 1), so neither
# can be read as the other.
BOARD_PRESETS["A4 board A-big - setup, big markers (4x4, ids 50-61)"] = BoardConfig(
    6, 4, 0.045, 0.036, "DICT_4X4_100", 50)
HANDHELD_PRESET = "A4 board A - handheld (5x5 markers)"
A_BIG_PRESET = "A4 board A-big - setup, big markers (4x4, ids 50-61)"
SETUP_A_PRESETS = (HANDHELD_PRESET, A_BIG_PRESET)
REFERENCE_PRESETS = ("A4 board B1 - camera 1 reference (4x4, ids 0-16)",
                     "A4 board B2 - camera 2 reference (4x4, ids 17-33)",
                     "A4 board G1 - camera 1 reference, big markers (ids 34-39)",
                     "A4 board G2 - camera 2 reference, big markers (ids 40-45)")
FIXED_PRESET = REFERENCE_PRESETS[0]  # the first fixed board printed ("B") is B1


def with_measured_square(cfg: BoardConfig, measured_square_mm: float) -> BoardConfig:
    """The board as actually printed: a printer scales squares and markers by the same factor.

    For a ChArUco board measure one square; for a grid measure one MARKER (black outer edge).
    """
    nominal = cfg.marker_length_m if cfg.kind == "grid" else cfg.square_length_m
    scale = (measured_square_mm / 1000.0) / nominal
    return BoardConfig(cfg.squares_x, cfg.squares_y, cfg.square_length_m * scale,
                       cfg.marker_length_m * scale, cfg.dictionary, cfg.first_marker_id, cfg.kind)


# What the rig's printer actually produced (measured 2026-10-06), used as the default
# of the "measured size" fields. The presets above stay nominal: they are what gets printed.
PRINTED_SIZE_MM: dict[str, float] = {
    HANDHELD_PRESET: 35.86,
    REFERENCE_PRESETS[0]: 35.86,
    REFERENCE_PRESETS[1]: 35.86,
    REFERENCE_PRESETS[2]: 69.89,
    REFERENCE_PRESETS[3]: 69.89,
}


def default_measured_mm(preset_name: str) -> float:
    """Default for a measured-size field: the rig's measured print, else the nominal size."""
    return PRINTED_SIZE_MM.get(preset_name, measured_size_mm(BOARD_PRESETS[preset_name]))


def measured_size_mm(cfg: BoardConfig) -> float:
    """The length the operator measures on the print: one square (ChArUco) or one marker (grid)."""
    return (cfg.marker_length_m if cfg.kind == "grid" else cfg.square_length_m) * 1000.0


def make_board(cfg: BoardConfig):
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, cfg.dictionary))
    ids = np.arange(cfg.first_marker_id, cfg.first_marker_id + cfg.marker_count, dtype=np.int32)
    if cfg.kind == "grid":
        return cv2.aruco.GridBoard((cfg.squares_x, cfg.squares_y), cfg.marker_length_m, cfg.gap_m,
                                   dictionary, ids)
    return cv2.aruco.CharucoBoard(
        (cfg.squares_x, cfg.squares_y), cfg.square_length_m, cfg.marker_length_m, dictionary, ids
    )


def render_board(cfg: BoardConfig, px_per_m: float, margin_px: int = 0) -> np.ndarray:
    """Board image at `px_per_m` (e.g. dpi / 0.0254 for printing), white margin around it.

    A grid comes with its quiet zone (one gap wide) already around the markers:
    the top-left pixel is at cfg.image_origin_m in the board frame.
    """
    w_m, h_m = cfg.size_m
    w, h = round(w_m * px_per_m), round(h_m * px_per_m)
    if cfg.kind == "grid":
        # Each marker drawn at its exact board position (GridBoard.generateImage
        # rounds its own layout and can fail on sizes that do not divide evenly).
        board = make_board(cfg)
        img = np.full((h, w), 255, np.uint8)
        side = round(cfg.marker_length_m * px_per_m)
        for marker_id, obj in zip(np.asarray(board.getIds()).reshape(-1), board.getObjPoints()):
            x0, y0 = np.asarray(obj, float).reshape(4, 3)[0, :2] - cfg.image_origin_m
            col, row = round(x0 * px_per_m), round(y0 * px_per_m)
            img[row:row + side, col:col + side] = cv2.aruco.generateImageMarker(
                board.getDictionary(), int(marker_id), side, borderBits=1)
    else:
        img = make_board(cfg).generateImage((w, h), marginSize=0, borderBits=1)
    if margin_px:
        img = cv2.copyMakeBorder(img, margin_px, margin_px, margin_px, margin_px,
                                 cv2.BORDER_CONSTANT, value=255)
    return img


# --------------------------------------------------------------------------
# Detection
# --------------------------------------------------------------------------

@dataclass
class Detection:
    """ChArUco corners found in one image (corners: (N, 2) float32 px, ids: (N,) int)."""
    corners: np.ndarray
    ids: np.ndarray
    marker_count: int
    image_size: tuple[int, int]  # (width, height)
    corners_per_row: int | None = None  # squares_x - 1: turns an id into (row, column) of the board
    # A grid's corners are its markers' corners (id = 4 * marker index + corner): 2 markers
    # already give 8 corners spread over a plane, so fewer markers are enough.
    min_markers: int = MIN_MARKERS

    @property
    def spans_board(self) -> bool:
        """The corners cover at least 2 rows AND 2 columns of the board (unknown layout: assumed yes)."""
        if not self.corners_per_row or len(self.ids) == 0:
            return True
        return (len(np.unique(self.ids // self.corners_per_row)) >= 2
                and len(np.unique(self.ids % self.corners_per_row)) >= 2)

    @property
    def ok(self) -> bool:
        # A board edge-on or cut off at the image border can leave all the corners
        # in one row (6 on the A4 boards = MIN_CORNERS): collinear points have no pose.
        return self.marker_count >= self.min_markers and len(self.ids) >= MIN_CORNERS and self.spans_board

    def object_points(self, board) -> np.ndarray:
        if isinstance(board, cv2.aruco.CharucoBoard):
            return board.getChessboardCorners()[self.ids].astype(np.float32)
        return np.concatenate([np.asarray(m, np.float32).reshape(4, 3) for m in board.getObjPoints()])[self.ids]


class BoardDetector:
    """Reusable detector for one board (building the OpenCV detector is not free)."""

    def __init__(self, cfg: BoardConfig) -> None:
        self.cfg = cfg
        self.board = make_board(cfg)
        if cfg.kind == "grid":
            # Plain marker corners are pixel-accurate by default; the move check needs sub-pixel.
            params = cv2.aruco.DetectorParameters()
            params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
            self._detector = cv2.aruco.ArucoDetector(self.board.getDictionary(), params)
            self._index = {int(i): n for n, i in enumerate(np.asarray(self.board.getIds()).reshape(-1))}
        else:
            self._detector = cv2.aruco.CharucoDetector(self.board)

    def _detect_grid(self, gray: np.ndarray) -> Detection:
        h, w = gray.shape[:2]
        corners, ids, _rejected = self._detector.detectMarkers(gray)
        pts, pt_ids, n = [], [], 0
        for c, i in zip(corners, [] if ids is None else ids.reshape(-1)):
            index = self._index.get(int(i))
            if index is None:
                continue  # a marker of another board in the same dictionary
            n += 1
            pts.append(c.reshape(4, 2))
            pt_ids.extend(4 * index + k for k in range(4))
        if not pts:
            return Detection(np.zeros((0, 2), np.float32), np.zeros((0,), np.int32), 0, (w, h), None, 2)
        order = np.argsort(pt_ids)
        return Detection(np.concatenate(pts).astype(np.float32)[order], np.asarray(pt_ids, np.int32)[order],
                         n, (w, h), None, 2)

    def detect(self, image: np.ndarray) -> Detection:
        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if self.cfg.kind == "grid":
            return self._detect_grid(gray)
        h, w = gray.shape[:2]
        corners, ids, _marker_corners, marker_ids = self._detector.detectBoard(gray)
        n_markers = 0 if marker_ids is None else len(marker_ids)
        per_row = self.cfg.squares_x - 1
        if corners is None or ids is None or len(ids) == 0:
            return Detection(np.zeros((0, 2), np.float32), np.zeros((0,), np.int32), n_markers, (w, h), per_row)
        return Detection(corners.reshape(-1, 2).astype(np.float32), ids.reshape(-1).astype(np.int32),
                         n_markers, (w, h), per_row)


def corner_motion_px(a: Detection, b: Detection) -> float | None:
    """Mean displacement of the corners both detections share (stillness check)."""
    common, ia, ib = np.intersect1d(a.ids, b.ids, return_indices=True)
    if len(common) < MIN_CORNERS:
        return None
    return float(np.linalg.norm(a.corners[ia] - b.corners[ib], axis=1).mean())


# --------------------------------------------------------------------------
# Coverage (intrinsics capture guidance)
# --------------------------------------------------------------------------

def guess_camera_matrix(image_size: tuple[int, int]) -> np.ndarray:
    """A rough K (focal ~ image width) good enough to estimate board tilt before calibration."""
    w, h = image_size
    return np.array([[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]], dtype=np.float64)


@dataclass(frozen=True)
class ViewSignature:
    """Where/how the board sits in one view, for coverage and 'new pose' decisions."""
    cell: tuple[int, int]      # (col, row) of the corner centroid in the COVERAGE_GRID
    centroid: tuple[float, float]
    area_fraction: float       # board corners' bounding box / image area
    tilt_deg: float            # board normal vs optical axis (rough before calibration)
    normal: tuple[float, float, float] = (0.0, 0.0, 1.0)  # board normal in camera coordinates, z >= 0

    @property
    def tilt_direction(self) -> str | None:
        """Which way the board is turned ('left', 'right', 'up', 'down'), None if it is roughly flat."""
        if not self.tilted:
            return None
        nx, ny, _ = self.normal
        if abs(nx) >= abs(ny):
            return "left" if nx > 0 else "right"
        return "up" if ny > 0 else "down"

    @property
    def near(self) -> bool:
        return self.area_fraction >= NEAR_FAR_SPLIT

    @property
    def tilted(self) -> bool:
        return self.tilt_deg >= TILTED_DEG


def view_signature(det: Detection, board, K: np.ndarray | None = None,
                   D: np.ndarray | None = None) -> ViewSignature | None:
    if not det.ok:
        return None
    w, h = det.image_size
    cx, cy = det.corners.mean(axis=0)
    x0, y0 = det.corners.min(axis=0)
    x1, y1 = det.corners.max(axis=0)
    area = float((x1 - x0) * (y1 - y0) / (w * h))
    cell = (min(COVERAGE_GRID - 1, int(cx / w * COVERAGE_GRID)),
            min(COVERAGE_GRID - 1, int(cy / h * COVERAGE_GRID)))
    pose = solve_board_pose(det, board, guess_camera_matrix(det.image_size) if K is None else K,
                            np.zeros(5) if D is None else D)
    tilt = 0.0 if pose is None else pose.tilt_deg
    normal = (0.0, 0.0, 1.0) if pose is None else tuple(float(v) for v in pose.normal_cam)
    return ViewSignature(cell=cell, centroid=(float(cx), float(cy)), area_fraction=area, tilt_deg=tilt,
                         normal=normal)


def _normal_angle_deg(a: tuple, b: tuple) -> float:
    return math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(a, b))))))


def is_new_view(sig: ViewSignature, taken: Sequence[ViewSignature], image_size: tuple[int, int],
                min_shift_fraction: float = 0.1, min_tilt_change_deg: float = 10.0,
                min_scale_ratio: float = 1.25) -> bool:
    """True if `sig` differs enough from every view already taken.

    Tilt is compared as the angle between the board normals, not as |tilt|:
    tilting left and right (or up and down) by the same angle are different views.
    """
    diag = math.hypot(*image_size)
    for t in taken:
        shift = math.hypot(sig.centroid[0] - t.centroid[0], sig.centroid[1] - t.centroid[1]) / diag
        scale = max(sig.area_fraction, t.area_fraction) / max(1e-9, min(sig.area_fraction, t.area_fraction))
        if (shift < min_shift_fraction and _normal_angle_deg(sig.normal, t.normal) < min_tilt_change_deg
                and scale < min_scale_ratio):
            return False
    return True


@dataclass
class Coverage:
    cells: set = field(default_factory=set)
    near: int = 0
    far: int = 0
    tilted: int = 0
    flat: int = 0
    views: int = 0
    directions: dict = field(default_factory=dict)   # tilt direction -> number of views

    def add(self, sig: ViewSignature) -> None:
        self.cells.add(sig.cell)
        if sig.tilt_direction is not None:
            self.directions[sig.tilt_direction] = self.directions.get(sig.tilt_direction, 0) + 1
        self.near += sig.near
        self.far += not sig.near
        self.tilted += sig.tilted
        self.flat += not sig.tilted
        self.views += 1

    def missing(self, target_views: int = 30) -> list[str]:
        """Plain-language hints for what the operator should do next (empty = good)."""
        hints = []
        names = {0: "left", 1: "centre", 2: "right"}
        rows = {0: "top", 1: "middle", 2: "bottom"}
        empty = [(c, r) for r in range(COVERAGE_GRID) for c in range(COVERAGE_GRID) if (c, r) not in self.cells]
        if empty:
            hints.append("cover " + ", ".join(f"{rows[r]}-{names[c]}" for c, r in empty[:3])
                         + (" and more" if len(empty) > 3 else ""))
        if self.near < 5:
            hints.append("bring the board closer")
        if self.far < 5:
            hints.append("move the board further away")
        if self.tilted < 8:
            hints.append("tilt the board more (about 30 degrees, in different directions)")
        else:
            untried = [d for d in TILT_DIRECTIONS if not self.directions.get(d)]
            if untried:
                hints.append("also tilt the board so it faces " + " and ".join(untried))
        if self.views < target_views:
            hints.append(f"{target_views - self.views} more views")
        return hints


@dataclass
class CaptureUpdate:
    state: str                 # "no board" | "moving" | "steady" | "captured" | "seen already"
    signature: ViewSignature | None = None


class CaptureSession:
    """Automatic view selection for the intrinsics capture (no Qt, no camera).

    Feed it one detection at a time. A view is taken when the board has been
    still (corners moving < still_px) for still_s seconds AND it differs from
    every view already taken (is_new_view). Holding still in the same spot
    therefore never adds duplicates; the operator just moves on.
    """

    def __init__(self, board, still_px: float = 1.5, still_s: float = 0.5, target_views: int = 30) -> None:
        self.board = board
        self.still_px, self.still_s, self.target_views = still_px, still_s, target_views
        self.views: list[Detection] = []
        self.signatures: list[ViewSignature] = []
        self._prev: Detection | None = None
        self._steady_since: float | None = None

    @property
    def coverage(self) -> Coverage:
        cov = Coverage()
        for sig in self.signatures:
            cov.add(sig)
        return cov

    def hints(self) -> list[str]:
        return self.coverage.missing(self.target_views)

    def feed(self, det: Detection, now: float) -> CaptureUpdate:
        prev, self._prev = self._prev, det
        if not det.ok:
            self._steady_since = None
            return CaptureUpdate("no board")
        motion = corner_motion_px(prev, det) if prev is not None and prev.ok else None
        if motion is None or motion > self.still_px:
            self._steady_since = now
            return CaptureUpdate("moving")
        if self._steady_since is None:
            self._steady_since = now
        if now - self._steady_since < self.still_s:
            return CaptureUpdate("steady")
        sig = view_signature(det, self.board)
        if sig is None:
            return CaptureUpdate("no board")
        if not is_new_view(sig, self.signatures, det.image_size):
            return CaptureUpdate("seen already", sig)
        self._add(det, sig)
        return CaptureUpdate("captured", sig)

    def capture_now(self, det: Detection) -> bool:
        """Manual capture: any usable view, even one similar to an earlier one."""
        sig = view_signature(det, self.board)
        if sig is None:
            return False
        self._add(det, sig)
        return True

    def undo(self) -> bool:
        if not self.views:
            return False
        self.views.pop()
        self.signatures.pop()
        return True

    def drop(self, indices: Sequence[int]) -> None:
        keep = [i for i in range(len(self.views)) if i not in set(indices)]
        self.views = [self.views[i] for i in keep]
        self.signatures = [self.signatures[i] for i in keep]

    def _add(self, det: Detection, sig: ViewSignature) -> None:
        self.views.append(det)
        self.signatures.append(sig)


def outlier_views(per_view_rms_px: Sequence[float], factor: float = 2.0) -> list[int]:
    """Views whose error is more than `factor` x the median (blurred, bent board, wrong detection)."""
    if len(per_view_rms_px) < 3:
        return []
    median = float(np.median(per_view_rms_px))
    return [i for i, e in enumerate(per_view_rms_px) if e > factor * median]


# --------------------------------------------------------------------------
# Intrinsics
# --------------------------------------------------------------------------

@dataclass
class IntrinsicsResult:
    K: np.ndarray
    D: np.ndarray
    rms_px: float
    per_view_rms_px: list[float]
    image_size: tuple[int, int]
    n_views: int

    @property
    def passed(self) -> bool:
        return self.rms_px < INTRINSIC_RMS_PX


def unusable_views(views: Sequence[Detection], board) -> list[int]:
    """Indices of views that cannot be posed: not ok, corners in a line on the board, or no finite pose.

    Such a view makes cv2.calibrateCamera raise, and more views cannot fix that.
    """
    bad = []
    for i, v in enumerate(views):
        if not v.ok:
            bad.append(i)
            continue
        pts = v.object_points(board)[:, :2]
        if np.linalg.svd(pts - pts.mean(axis=0), compute_uv=False)[-1] < 1e-6:
            bad.append(i)  # one row or column of the board, whatever the Detection says
            continue
        pose = solve_board_pose(v, board, guess_camera_matrix(v.image_size), np.zeros(5))
        if pose is None or not math.isfinite(pose.rms_px):
            bad.append(i)
    return bad


def calibrate_intrinsics(views: Sequence[Detection], board) -> IntrinsicsResult:
    usable = [v for v in views if v.ok]
    if len(usable) < MIN_INTRINSIC_VIEWS:
        raise ValueError(f"need at least {MIN_INTRINSIC_VIEWS} usable views, have {len(usable)}")
    sizes = {v.image_size for v in usable}
    if len(sizes) != 1:
        raise ValueError(f"views have different image sizes: {sorted(sizes)}")
    image_size = usable[0].image_size
    obj = [v.object_points(board) for v in usable]
    img = [v.corners.reshape(-1, 1, 2) for v in usable]
    try:
        rms, K, D, rvecs, tvecs = cv2.calibrateCamera(obj, img, image_size, None, None)
    except cv2.error as exc:
        raise ValueError(f"OpenCV could not fit the camera to these views ({str(exc).splitlines()[0][:120]})") from exc
    per_view = []
    for o, i, r, t in zip(obj, img, rvecs, tvecs):
        proj, _ = cv2.projectPoints(o, r, t, K, D)
        per_view.append(float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - i.reshape(-1, 2)) ** 2, axis=1)))))
    return IntrinsicsResult(K=K, D=D.reshape(-1), rms_px=float(rms), per_view_rms_px=per_view,
                            image_size=image_size, n_views=len(usable))


# --------------------------------------------------------------------------
# Board pose / setup (extrinsics)
# --------------------------------------------------------------------------

@dataclass
class BoardPose:
    """Board -> camera transform: x_cam = R @ x_board + t (t in metres)."""
    R: np.ndarray
    t: np.ndarray
    rms_px: float
    n_corners: int

    @property
    def camera_centre_m(self) -> np.ndarray:
        """Camera position in the board frame."""
        return (-self.R.T @ self.t).reshape(3)

    @property
    def normal_cam(self) -> np.ndarray:
        """Board normal in camera coordinates, as a unit vector pointing away from the camera (z >= 0)."""
        n = self.R @ np.array([0.0, 0.0, 1.0])
        return -n if n[2] < 0 else n

    @property
    def tilt_deg(self) -> float:
        """Angle between the board normal and the camera's optical axis (0 = facing the camera)."""
        return float(math.degrees(math.acos(min(1.0, abs(self.normal_cam[2])))))


def solve_board_pose(det: Detection, board, K: np.ndarray, D: np.ndarray) -> BoardPose | None:
    if not det.ok:
        return None
    obj = det.object_points(board)
    try:
        ok, rvec, tvec = cv2.solvePnP(obj, det.corners, K, D, flags=cv2.SOLVEPNP_ITERATIVE)
    except cv2.error:  # degenerate corner sets can trip an OpenCV assertion
        return None
    if not ok:
        return None
    proj, _ = cv2.projectPoints(obj, rvec, tvec, K, D)
    rms = float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - det.corners) ** 2, axis=1))))
    R, _ = cv2.Rodrigues(rvec)
    return BoardPose(R=R, t=tvec.reshape(3), rms_px=rms, n_corners=len(det.ids))


def average_detections(dets: Sequence[Detection]) -> Detection | None:
    """Mean corner position per id over several frames of a STATIC board (noise reduction)."""
    ok = [d for d in dets if d.ok]
    if not ok:
        return None
    by_id: dict[int, list[np.ndarray]] = {}
    for d in ok:
        for i, c in zip(d.ids, d.corners):
            by_id.setdefault(int(i), []).append(c)
    # keep ids seen in at least half the frames
    ids = sorted(i for i, cs in by_id.items() if len(cs) * 2 >= len(ok))
    corners = np.array([np.mean(by_id[i], axis=0) for i in ids], np.float32).reshape(-1, 2)
    return Detection(corners, np.array(ids, np.int32), max(d.marker_count for d in ok), ok[0].image_size,
                     ok[0].corners_per_row, ok[0].min_markers)


def rotation_angle_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    cos = (np.trace(Ra.T @ Rb) - 1.0) / 2.0
    return float(math.degrees(math.acos(max(-1.0, min(1.0, cos)))))


@dataclass
class SetupResult:
    poses: dict                    # serial -> BoardPose
    baseline_mm: float | None      # distance between the first two camera centres
    triangulation_rms_mm: float | None
    triangulated_corners: int
    expected: tuple = ()           # serials compute_setup was given (a camera without a pose is dropped from `poses`)

    @property
    def missing(self) -> list:
        """Cameras that were given but whose pose could not be solved (board not found)."""
        return [s for s in self.expected if s not in self.poses]

    @property
    def per_camera_passed(self) -> dict:
        return {s: p.rms_px < SETUP_REPROJECTION_RMS_PX for s, p in self.poses.items()}

    @property
    def passed(self) -> bool:
        if not self.poses or not all(self.per_camera_passed.values()):
            return False
        if len(self.expected) >= 2 and self.missing:  # one of two cameras posed is not a 3D setup
            return False
        if len(self.poses) >= 2:
            return self.triangulation_rms_mm is not None and self.triangulation_rms_mm < SETUP_TRIANGULATION_RMS_MM
        return True


def triangulate_board(det_a: Detection, pose_a: BoardPose, K_a, D_a,
                      det_b: Detection, pose_b: BoardPose, K_b, D_b, board) -> tuple[float | None, int]:
    """Triangulate the corners both cameras see; RMS distance (mm) to the real board corners."""
    common, ia, ib = np.intersect1d(det_a.ids, det_b.ids, return_indices=True)
    if len(common) < MIN_CORNERS:
        return None, len(common)
    na = cv2.undistortPoints(det_a.corners[ia].reshape(-1, 1, 2), K_a, D_a).reshape(-1, 2).T
    nb = cv2.undistortPoints(det_b.corners[ib].reshape(-1, 1, 2), K_b, D_b).reshape(-1, 2).T
    Pa = np.hstack([pose_a.R, pose_a.t.reshape(3, 1)])
    Pb = np.hstack([pose_b.R, pose_b.t.reshape(3, 1)])
    X = cv2.triangulatePoints(Pa, Pb, na, nb)
    X = (X[:3] / X[3]).T  # board frame, metres
    truth = board.getChessboardCorners()[common]
    err_mm = np.linalg.norm(X - truth, axis=1) * 1000.0
    return float(np.sqrt(np.mean(err_mm ** 2))), len(common)


def compute_setup(dets: dict, intrinsics: dict, board) -> SetupResult:
    """dets / intrinsics: serial -> Detection / (K, D). Same static board seen by every camera."""
    poses = {}
    for serial, det in dets.items():
        K, D = intrinsics[serial]
        pose = solve_board_pose(det, board, K, D) if det is not None else None
        if pose is not None:
            poses[serial] = pose
    baseline = tri = None
    n_tri = 0
    serials = [s for s in dets if s in poses]
    if len(serials) >= 2:
        a, b = serials[:2]
        baseline = float(np.linalg.norm(poses[a].camera_centre_m - poses[b].camera_centre_m) * 1000.0)
        tri, n_tri = triangulate_board(dets[a], poses[a], *intrinsics[a], dets[b], poses[b], *intrinsics[b], board)
    return SetupResult(poses=poses, baseline_mm=baseline, triangulation_rms_mm=tri, triangulated_corners=n_tri,
                       expected=tuple(dets))


def kabsch_rms_mm(points_m: np.ndarray, truth_m: np.ndarray) -> float:
    """RMS distance (mm) after the best RIGID alignment of points to truth (no scaling)."""
    p, q = points_m - points_m.mean(axis=0), truth_m - truth_m.mean(axis=0)
    U, _, Vt = np.linalg.svd(p.T @ q)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    err = (R @ p.T).T - q
    return float(np.sqrt(np.mean(np.sum(err ** 2, axis=1))) * 1000.0)


@dataclass
class VerifyResult:
    rms_mm: float | None        # triangulated board vs real board, rigid fit only
    scale_error_pct: float | None
    depth_change_m: float | None  # new board centre's distance from the setup board's plane
    corners: int

    @property
    def problems(self) -> list[str]:
        if self.rms_mm is None:
            return [f"both cameras must see the board ({self.corners} common corners)"]
        out = []
        if self.depth_change_m < VERIFY_MIN_DEPTH_CHANGE_M:
            out.append(f"move the board at least {VERIFY_MIN_DEPTH_CHANGE_M * 100:.0f} cm toward the cameras "
                       "from where it was "
                       f"(now {self.depth_change_m * 100:.0f} cm)")
        if abs(self.scale_error_pct) >= VERIFY_SCALE_ERROR_PCT:
            out.append(f"board measures {self.scale_error_pct:+.1f} % off its real size")
        if self.rms_mm >= SETUP_TRIANGULATION_RMS_MM:
            out.append(f"board shape off by {self.rms_mm:.1f} mm RMS")
        return out

    @property
    def passed(self) -> bool:
        return not self.problems


def verify_setup(det_a: Detection, pose_a: BoardPose, K_a, D_a,
                 det_b: Detection, pose_b: BoardPose, K_b, D_b, board) -> VerifyResult:
    """Independent check of a saved setup.

    compute_setup's triangulation reuses the board the poses were solved from,
    so it stays self-consistent even with wrong intrinsics (a 20 % focal error
    still triangulates the setup board to < 1 mm). Here the board has been
    moved to a NEW static position: its corners are triangulated with the
    SAVED camera poses (not re-solved), and the result is compared to the
    board's real geometry with a rigid fit, so scale and shape errors show.
    The new spot must be off the setup board's plane (raised toward the
    cameras): a sideways move hides scale errors.
    """
    common, ia, ib = np.intersect1d(det_a.ids, det_b.ids, return_indices=True)
    if len(common) < MIN_CORNERS:
        return VerifyResult(None, None, None, len(common))
    na = cv2.undistortPoints(det_a.corners[ia].reshape(-1, 1, 2), K_a, D_a).reshape(-1, 2).T
    nb = cv2.undistortPoints(det_b.corners[ib].reshape(-1, 1, 2), K_b, D_b).reshape(-1, 2).T
    X = cv2.triangulatePoints(np.hstack([pose_a.R, pose_a.t.reshape(3, 1)]),
                              np.hstack([pose_b.R, pose_b.t.reshape(3, 1)]), na, nb)
    X = (X[:3] / X[3]).T
    truth = board.getChessboardCorners()[common].astype(np.float64)

    def spread(p):
        return np.mean(np.linalg.norm(p - p.mean(axis=0), axis=1))

    scale = (spread(X) / spread(truth) - 1.0) * 100.0
    depth = float(abs(X[:, 2].mean()))  # board frame: z = 0 is the setup board's plane
    return VerifyResult(kabsch_rms_mm(X, truth), float(scale), depth, len(common))


@dataclass(frozen=True)
class MoveCheck:
    translation_mm: float
    rotation_deg: float

    @property
    def moved(self) -> bool:
        return self.translation_mm > SETUP_MOVED_TRANSLATION_MM or self.rotation_deg > SETUP_MOVED_ROTATION_DEG


def compare_poses(saved: BoardPose, live: BoardPose, board_centre_m: Sequence[float]) -> MoveCheck:
    """How far a camera moved relative to the fixed board since the setup was saved.

    Both poses are compared in the CAMERA's frame: the angle between the two
    board rotations, and the shift of the board centre (R @ c + t, c in the
    board frame, e.g. BoardConfig.centre_m). The camera centre in the board frame
    would multiply the rotation noise of a single pose by the board distance (0.24
    degrees = 5 mm at 1.2 m), so a 5 mm limit would fire on noise alone; the
    board centre is where the fit is best pinned.
    """
    c = np.asarray(board_centre_m, dtype=float).reshape(3)
    shift = (saved.R @ c + saved.t) - (live.R @ c + live.t)
    return MoveCheck(
        translation_mm=float(np.linalg.norm(shift) * 1000.0),
        rotation_deg=rotation_angle_deg(saved.R, live.R),
    )


# --------------------------------------------------------------------------
# Reference boards (plan decision 11): has a camera moved since the setup?
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class ReferenceCheck:
    move: MoveCheck
    rms_px: float               # reprojection of the reference board with the stored intrinsics
    suspect: bool               # fits much worse than at setup: lens touched?

    @property
    def moved(self) -> bool:
        return self.move.moved


def check_reference(saved_R, saved_t, saved_rms_px: float, live: BoardPose, cfg: BoardConfig) -> ReferenceCheck:
    """Compare a camera's live pose to its reference board with the pose saved at setup.

    `live` should come from several averaged frames (average_detections): one
    frame's corner noise alone can exceed the rotation limit a few % of the time.
    "Suspect" uses the setup's own fit as the baseline, because a far or oblique
    reference board fits less tightly than the calibration board did.
    """
    saved = BoardPose(R=np.asarray(saved_R, dtype=float), t=np.asarray(saved_t, dtype=float).reshape(3),
                      rms_px=float(saved_rms_px), n_corners=0)
    limit = max(SUSPECT_REPROJECTION_RMS_PX, 2.0 * float(saved_rms_px))
    return ReferenceCheck(move=compare_poses(saved, live, cfg.centre_m), rms_px=live.rms_px,
                          suspect=live.rms_px > limit)


def diagnose_frame(frame: np.ndarray, boards: dict) -> dict:
    """Why a board is (not) detected in one frame: plain numbers for a support report.

    boards: name -> BoardConfig. Per board: markers of its dictionary found
    (all, and those that belong to this board), their median side in pixels,
    square-ish candidates the marker decoder rejected (many = markers seen but
    too small/blurred/oblique to read), and ChArUco corners found. Per frame:
    brightness and the fraction of saturated pixels (glare).
    """
    gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    out = {
        "image_size": [int(gray.shape[1]), int(gray.shape[0])],
        "mean_brightness": round(float(gray.mean()), 1),
        "saturated_fraction": round(float((gray >= 250).mean()), 4),
        "boards": {},
    }
    for name, cfg in boards.items():
        dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, cfg.dictionary))
        corners, ids, rejected = cv2.aruco.ArucoDetector(dictionary).detectMarkers(gray)
        ids = [] if ids is None else [int(i) for i in ids.reshape(-1)]
        own = set(range(cfg.first_marker_id, cfg.first_marker_id + cfg.marker_count))
        sides = [float(np.mean(np.linalg.norm(c.reshape(4, 2) - np.roll(c.reshape(4, 2), 1, axis=0), axis=1)))
                 for c, i in zip(corners, ids) if i in own]
        det = BoardDetector(cfg).detect(gray)
        out["boards"][name] = {
            "markers_of_this_dictionary": len(ids),
            "markers_of_this_board": sum(i in own for i in ids),
            "board_markers_total": cfg.marker_count,
            "median_marker_side_px": round(float(np.median(sides)), 1) if sides else None,
            "rejected_candidates": 0 if rejected is None else len(rejected),
            "charuco_corners": int(len(det.ids)),
            "usable": bool(det.ok),
        }
    return out


def best_detection(dets: dict) -> tuple[str, Detection] | None:
    """(name, detection) of the usable detection with the most corners, e.g. which reference board a camera sees."""
    usable = [(name, d) for name, d in dets.items() if d is not None and d.ok]
    if not usable:
        return None
    return max(usable, key=lambda item: len(item[1].ids))
