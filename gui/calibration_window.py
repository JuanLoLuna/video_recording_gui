"""Calibration window (plan step 13): step-by-step camera calibration.

Opened by the main window's Calibrate… button. Non-modal and separate from the
recording controls, which it never touches: it only reads the latest frame of
each camera (CameraController.get_latest_frame) and writes calibration records
through backend.calibration_store.

Board detection runs on a worker thread a few times per second; the GUI thread
only feeds results to the capture session and draws the overlay.

The per-session setup task (fixed board, plan step 14) is not built yet.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Callable, Sequence

import cv2
import numpy as np
from PySide6.QtCore import QMarginsF, QRectF, Qt, QTimer
from PySide6.QtGui import QFont, QImage, QPageLayout, QPageSize, QPainter, QPdfWriter, QPen, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from backend import calibration as cal
from backend.calibration_store import CalibrationStore, IntrinsicsRecord
from gui.camera_preview import frame_to_qimage

DETECT_INTERVAL_S = 0.2       # worker: ~5 detections per second
UI_INTERVAL_MS = 120
LIVE_VIEW_SIZE = (720, 540)

_BIG = "font-size: 16px; font-weight: 600;"
_HINT = "color: #555;"


class _Detector:
    """Worker thread: latest frame of one camera -> board detection, a few times per second."""

    def __init__(self, controller, detector: cal.BoardDetector) -> None:
        self.controller = controller
        self.detector = detector
        self._lock = threading.Lock()
        self._latest: tuple[int, np.ndarray, cal.Detection] | None = None
        self._seq = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="calibration-detector", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            started = time.monotonic()
            frame = self.controller.get_latest_frame()
            if frame is not None:
                try:
                    det = self.detector.detect(frame)
                except Exception as exc:  # never let a bad frame kill the worker
                    print(f"[calibration] detection failed: {exc}")
                else:
                    with self._lock:
                        self._seq += 1
                        self._latest = (self._seq, frame, det)
            self._stop.wait(max(0.0, DETECT_INTERVAL_S - (time.monotonic() - started)))

    def latest(self):
        with self._lock:
            return self._latest

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)


def draw_overlay(frame: np.ndarray, det: cal.Detection | None, max_size=LIVE_VIEW_SIZE) -> np.ndarray:
    """Downscaled BGR copy of the frame with the detected corners drawn."""
    h, w = frame.shape[:2]
    scale = min(max_size[0] / w, max_size[1] / h, 1.0)
    small = cv2.resize(frame, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    if small.ndim == 2:
        small = cv2.cvtColor(small, cv2.COLOR_GRAY2BGR)
    if det is not None and len(det.ids):
        colour = (60, 200, 60) if det.ok else (0, 160, 255)
        for x, y in det.corners * scale:
            cv2.circle(small, (int(x), int(y)), 4, colour, -1, cv2.LINE_AA)
    return small


def write_board_pdf(path: str, cfg: cal.BoardConfig, label: str, dpi: int = 600) -> None:
    """True-size printable board on A4 landscape (raises ValueError if it does not fit)."""
    w_mm, h_mm = (v * 1000 for v in cfg.size_m)
    if w_mm > 277 or h_mm > 180:
        raise ValueError(f"a {w_mm:.0f} x {h_mm:.0f} mm board does not fit on A4 with margins")
    px_per_m = dpi / 0.0254
    board = cal.render_board(cfg, px_per_m)
    image = QImage(board.data, board.shape[1], board.shape[0], board.strides[0], QImage.Format.Format_Grayscale8)
    writer = QPdfWriter(path)
    writer.setResolution(dpi)
    writer.setPageLayout(QPageLayout(QPageSize(QPageSize.PageSizeId.A4), QPageLayout.Orientation.Landscape,
                                     QMarginsF(0, 0, 0, 0)))
    mm = dpi / 25.4
    painter = QPainter(writer)
    try:
        x0 = (297 - w_mm) / 2 * mm
        y0 = 10 * mm
        painter.drawImage(QRectF(x0, y0, w_mm * mm, h_mm * mm), image)
        yb = y0 + h_mm * mm + 5 * mm
        painter.setPen(QPen(Qt.GlobalColor.black, 0.3 * mm))
        painter.drawLine(int(x0), int(yb), int(x0 + 100 * mm), int(yb))
        for k in range(11):
            x = int(x0 + k * 10 * mm)
            painter.drawLine(x, int(yb), x, int(yb + 3 * mm))
        font = QFont("Arial")
        font.setPointSizeF(7)
        painter.setFont(font)
        painter.drawText(int(x0 + 105 * mm), int(yb + 2.5 * mm),
                         f"{label}  {cfg.squares_x}x{cfg.squares_y} ChArUco {cfg.dictionary}  "
                         f"squares {cfg.square_length_m * 1000:g} mm  markers {cfg.marker_length_m * 1000:g} mm")
        painter.drawText(int(x0 + 105 * mm), int(yb + 6.5 * mm),
                         "Print at 100% / Actual size. Bar = 100 mm: measure it and one square before use.")
    finally:
        painter.end()


class CalibrationWindow(QDialog):
    """Non-modal. `slots`: the main window's camera slots (serial, model, label, controller)."""

    def __init__(self, parent, slots: Sequence, store: CalibrationStore,
                 ensure_preview: Callable[[], bool], on_saved: Callable[[], None]) -> None:
        super().__init__(parent)
        self.setWindowTitle("Calibration")
        self.setModal(False)
        self.resize(820, 760)
        self.slots = list(slots)
        self.store = store
        self.ensure_preview = ensure_preview
        self.on_saved = on_saved
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="calibration-compute")
        self._detector: _Detector | None = None
        self._session: cal.CaptureSession | None = None
        self._board_cfg: cal.BoardConfig | None = None
        self._last_seq = 0
        self._last_det: cal.Detection | None = None
        self._last_state = ""
        self._compute: Future | None = None
        self._result: cal.IntrinsicsResult | None = None

        self.stack = QStackedWidget()
        root = QVBoxLayout(self)
        root.addWidget(self.stack)
        self.page_home = self._build_home()
        self.page_check = self._build_checklist()
        self.page_capture = self._build_capture()
        self.page_result = self._build_result()
        for page in (self.page_home, self.page_check, self.page_capture, self.page_result):
            self.stack.addWidget(page)

        self.ui_timer = QTimer(self)
        self.ui_timer.setInterval(UI_INTERVAL_MS)
        self.ui_timer.timeout.connect(self._tick)
        self.compute_timer = QTimer(self)
        self.compute_timer.setInterval(100)
        self.compute_timer.timeout.connect(self._poll_compute)

    # ------------------------------------------------------------------ pages
    def _build_home(self) -> QWidget:
        page = QWidget()
        v = QVBoxLayout(page)
        title = QLabel("Calibration")
        title.setStyleSheet("font-size: 20px; font-weight: 700;")
        v.addWidget(title)
        intro = QLabel(
            "3D pose needs two things:\n"
            "1. Each camera calibrated (lens and sensor). Once per camera, and again after anyone "
            "touches the lens or changes the camera's resolution.\n"
            "2. The cameras set up for the session: where they are relative to the fixed board. "
            "Every time a camera is moved.")
        intro.setWordWrap(True)
        v.addWidget(intro)

        self.camera_state_label = QLabel()
        self.camera_state_label.setWordWrap(True)
        self.camera_state_label.setStyleSheet(_HINT)
        v.addWidget(self.camera_state_label)
        self.saved_label = QLabel("")
        self.saved_label.setWordWrap(True)
        self.saved_label.setStyleSheet("color: #2e7d32; font-weight: 600;")
        v.addWidget(self.saved_label)

        self.calibrate_camera_button = QPushButton("1. Calibrate a camera…")
        self.calibrate_camera_button.setMinimumHeight(40)
        self.calibrate_camera_button.clicked.connect(lambda: self._go(self.page_check))
        v.addWidget(self.calibrate_camera_button)
        self.setup_button = QPushButton("2. Set up for this session…")
        self.setup_button.setMinimumHeight(40)
        self.setup_button.setEnabled(False)
        self.setup_button.setToolTip("Coming in the next step (fixed board setup and verification).")
        v.addWidget(self.setup_button)

        row = QHBoxLayout()
        self.print_combo = QComboBox()
        self.print_combo.addItems(list(cal.BOARD_PRESETS))
        row.addWidget(self.print_combo, stretch=1)
        self.print_button = QPushButton("Save printable board (PDF)…")
        self.print_button.clicked.connect(self._on_print_board)
        row.addWidget(self.print_button)
        v.addLayout(row)
        where = QLabel(f"Calibrations are stored in {self.store.root}")
        where.setStyleSheet(_HINT)
        where.setWordWrap(True)
        v.addWidget(where)
        v.addStretch(1)
        close = QPushButton("Close")
        close.clicked.connect(self.close)
        v.addWidget(close, alignment=Qt.AlignmentFlag.AlignRight)
        self._refresh_camera_state()
        return page

    def _build_checklist(self) -> QWidget:
        page = QWidget()
        v = QVBoxLayout(page)
        head = QLabel("Calibrate a camera: step 1 of 3, before you start")
        head.setStyleSheet(_BIG)
        v.addWidget(head)

        grid = QGridLayout()
        grid.addWidget(QLabel("Camera"), 0, 0)
        self.camera_combo = QComboBox()
        for slot in self.slots:
            self.camera_combo.addItem(slot.label, slot.serial)
        grid.addWidget(self.camera_combo, 0, 1)
        grid.addWidget(QLabel("Board"), 1, 0)
        self.board_combo = QComboBox()
        self.board_combo.addItems(list(cal.BOARD_PRESETS))
        self.board_combo.setCurrentText(cal.HANDHELD_PRESET)
        self.board_combo.currentTextChanged.connect(self._on_board_preset_changed)
        grid.addWidget(self.board_combo, 1, 1)
        grid.addWidget(QLabel("Measured square size"), 2, 0)
        self.square_spin = QDoubleSpinBox()
        self.square_spin.setRange(5.0, 200.0)
        self.square_spin.setDecimals(2)
        self.square_spin.setSuffix(" mm")
        self.square_spin.setValue(cal.BOARD_PRESETS[cal.HANDHELD_PRESET].square_length_m * 1000)
        grid.addWidget(self.square_spin, 2, 1)
        v.addLayout(grid)

        self.checks = [
            QCheckBox("The board is printed at 100% and mounted flat and rigid (no bends, no glare)."),
            QCheckBox("I measured one square with a ruler and entered it above."),
            QCheckBox("The lens focus, zoom and aperture are where they will stay, and locked (screws tight). "
                      "Moving them later makes this calibration wrong."),
            QCheckBox("The camera is set to the resolution used for recording."),
            QCheckBox("The board is in focus at the distances you will hold it."),
        ]
        for box in self.checks:
            box.toggled.connect(self._update_checklist_next)
            v.addWidget(box)
        how = QLabel(
            "Next you will move the board around in front of the camera. Hold it still for a moment in "
            "each position; views are taken automatically. Cover the whole image (corners and edges too), "
            "near and far, and tilt the board about 30 degrees in different directions.")
        how.setWordWrap(True)
        how.setStyleSheet(_HINT)
        v.addWidget(how)
        v.addStretch(1)
        nav = QHBoxLayout()
        back = QPushButton("Back")
        back.clicked.connect(lambda: self._go(self.page_home))
        nav.addWidget(back)
        nav.addStretch(1)
        self.check_next = QPushButton("Next: capture")
        self.check_next.setEnabled(False)
        self.check_next.clicked.connect(self._start_capture)
        nav.addWidget(self.check_next)
        v.addLayout(nav)
        return page

    def _build_capture(self) -> QWidget:
        page = QWidget()
        v = QVBoxLayout(page)
        self.capture_head = QLabel("Calibrate a camera: step 2 of 3, move the board")
        self.capture_head.setStyleSheet(_BIG)
        v.addWidget(self.capture_head)
        self.live_view = QLabel("Waiting for frames…")
        self.live_view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.live_view.setMinimumSize(480, 360)
        self.live_view.setStyleSheet("background: #222; color: #ccc;")
        v.addWidget(self.live_view, stretch=1)
        self.capture_state = QLabel("")
        self.capture_state.setStyleSheet(_BIG)
        v.addWidget(self.capture_state)

        row = QHBoxLayout()
        self.coverage_cells = []
        grid = QGridLayout()
        grid.setSpacing(2)
        for r in range(cal.COVERAGE_GRID):
            for c in range(cal.COVERAGE_GRID):
                cell = QLabel()
                cell.setFixedSize(22, 16)
                grid.addWidget(cell, r, c)
                self.coverage_cells.append(((c, r), cell))
        row.addLayout(grid)
        self.coverage_label = QLabel("")
        self.coverage_label.setWordWrap(True)
        row.addWidget(self.coverage_label, stretch=1)
        v.addLayout(row)
        self.hint_label = QLabel("")
        self.hint_label.setWordWrap(True)
        self.hint_label.setStyleSheet("color: #8a4b00;")
        v.addWidget(self.hint_label)

        nav = QHBoxLayout()
        back = QPushButton("Back")
        back.clicked.connect(self._cancel_capture)
        nav.addWidget(back)
        self.capture_now_button = QPushButton("Take this view")
        self.capture_now_button.clicked.connect(self._capture_now)
        nav.addWidget(self.capture_now_button)
        self.undo_button = QPushButton("Undo last")
        self.undo_button.clicked.connect(self._undo)
        nav.addWidget(self.undo_button)
        nav.addStretch(1)
        self.compute_button = QPushButton("Compute")
        self.compute_button.setEnabled(False)
        self.compute_button.clicked.connect(self._start_compute)
        nav.addWidget(self.compute_button)
        v.addLayout(nav)
        return page

    def _build_result(self) -> QWidget:
        page = QWidget()
        v = QVBoxLayout(page)
        self.result_head = QLabel("Calibrate a camera: step 3 of 3, result")
        self.result_head.setStyleSheet(_BIG)
        v.addWidget(self.result_head)
        self.result_verdict = QLabel("")
        self.result_verdict.setStyleSheet("font-size: 18px; font-weight: 700;")
        v.addWidget(self.result_verdict)
        self.result_text = QLabel("")
        self.result_text.setWordWrap(True)
        self.result_text.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        v.addWidget(self.result_text)
        v.addStretch(1)
        nav = QHBoxLayout()
        self.more_views_button = QPushButton("Back: add more views")
        self.more_views_button.clicked.connect(self._back_to_capture)
        nav.addWidget(self.more_views_button)
        self.drop_outliers_button = QPushButton("Drop the worst views and recompute")
        self.drop_outliers_button.clicked.connect(self._drop_outliers)
        nav.addWidget(self.drop_outliers_button)
        nav.addStretch(1)
        self.save_button = QPushButton("Save")
        self.save_button.clicked.connect(lambda: self._save(loose=False))
        nav.addWidget(self.save_button)
        self.save_loose_button = QPushButton("Save anyway (marked as failed)")
        self.save_loose_button.clicked.connect(lambda: self._save(loose=True))
        nav.addWidget(self.save_loose_button)
        v.addLayout(nav)
        return page

    # ------------------------------------------------------------- navigation
    def _go(self, page: QWidget) -> None:
        if page is self.page_home:
            self._refresh_camera_state()
        self.stack.setCurrentWidget(page)

    def _refresh_camera_state(self) -> None:
        lines = []
        for slot in self.slots:
            rec = self.store.current_intrinsics(slot.serial)
            state = "not calibrated" if rec is None else (
                f"calibrated {rec.created_at.replace('T', ' ')}, {rec.rms_px:.2f} px"
                + (" (saved as failed)" if rec.loose else ""))
            lines.append(f"{slot.label}: {state}")
        self.camera_state_label.setText("\n".join(lines))

    def _on_board_preset_changed(self, name: str) -> None:
        self.square_spin.setValue(cal.BOARD_PRESETS[name].square_length_m * 1000)

    def _update_checklist_next(self) -> None:
        self.check_next.setEnabled(all(box.isChecked() for box in self.checks))

    def _selected_slot(self):
        serial = self.camera_combo.currentData()
        return next(slot for slot in self.slots if slot.serial == serial)

    # ---------------------------------------------------------------- capture
    def _start_capture(self) -> None:
        if not self.ensure_preview():
            QMessageBox.warning(self, "Calibration", "The cameras could not start. Check them in the main window.")
            return
        preset = cal.BOARD_PRESETS[self.board_combo.currentText()]
        self._board_cfg = cal.with_measured_square(preset, self.square_spin.value())
        detector = cal.BoardDetector(self._board_cfg)
        self._session = cal.CaptureSession(detector.board)
        self._stop_detector()
        self._detector = _Detector(self._selected_slot().controller, detector)
        self._last_seq = 0
        self._last_det = None
        self.capture_head.setText(f"Calibrate {self._selected_slot().label}: step 2 of 3, move the board")
        self._go(self.page_capture)
        self.ui_timer.start()
        self._update_capture_labels()

    def _back_to_capture(self) -> None:
        if self._session is None:
            return
        self._detector = self._detector or _Detector(self._selected_slot().controller,
                                                     cal.BoardDetector(self._board_cfg))
        self._go(self.page_capture)
        self.ui_timer.start()
        self._update_capture_labels()

    def _cancel_capture(self) -> None:
        self.ui_timer.stop()
        self._stop_detector()
        self._session = None
        self._go(self.page_check)

    def _stop_detector(self) -> None:
        if self._detector is not None:
            self._detector.stop()
            self._detector = None

    def _tick(self) -> None:
        """GUI thread: take the newest detection, feed the capture session, redraw."""
        if self._detector is None or self._session is None:
            return
        latest = self._detector.latest()
        if latest is None or latest[0] == self._last_seq:
            return
        seq, frame, det = latest
        self._last_seq, self._last_det = seq, det
        update = self._session.feed(det, time.monotonic())
        self._last_state = update.state
        image = frame_to_qimage(draw_overlay(frame, det))
        if image is not None:
            self.live_view.setPixmap(QPixmap.fromImage(image).scaled(
                self.live_view.size(), Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation))
        self._update_capture_labels()

    def _update_capture_labels(self) -> None:
        if self._session is None:
            return
        n = len(self._session.views)
        target = self._session.target_views
        text = {
            "no board": "Board not found: bring it into view, whole and in focus.",
            "moving": "Hold still…",
            "steady": "Hold still…",
            "captured": "Captured.",
            "seen already": "Already have this view: move to a new position or angle.",
        }.get(self._last_state, "")
        self.capture_state.setText(f"{n} / {target} views.  {text}")
        cov = self._session.coverage
        for (c, r), cell in self.coverage_cells:
            cell.setStyleSheet("background: #66bb6a;" if (c, r) in cov.cells else "background: #e0e0e0;")
        self.coverage_label.setText(
            f"Image regions covered: {len(cov.cells)} / {cal.COVERAGE_GRID ** 2}.  "
            f"Near {cov.near}, far {cov.far}, tilted {cov.tilted}, flat {cov.flat}.")
        hints = self._session.hints()
        self.hint_label.setText("Still to do: " + "; ".join(hints) + "." if hints else "Coverage is good. Press Compute.")
        self.compute_button.setEnabled(n >= cal.MIN_INTRINSIC_VIEWS)
        self.compute_button.setText(f"Compute ({n} views)")
        self.undo_button.setEnabled(n > 0)

    def _capture_now(self) -> None:
        if self._session is not None and self._last_det is not None and self._session.capture_now(self._last_det):
            self._last_state = "captured"
        self._update_capture_labels()

    def _undo(self) -> None:
        if self._session is not None:
            self._session.undo()
        self._update_capture_labels()

    # ---------------------------------------------------------------- compute
    def _start_compute(self) -> None:
        if self._session is None:
            return
        self.ui_timer.stop()
        self._stop_detector()
        views = list(self._session.views)
        board = self._session.board
        self._compute = self._pool.submit(cal.calibrate_intrinsics, views, board)
        self.result_verdict.setText("Computing…")
        self.result_text.setText(f"{len(views)} views.")
        for b in (self.save_button, self.save_loose_button, self.drop_outliers_button, self.more_views_button):
            b.setEnabled(False)
        self._go(self.page_result)
        self.compute_timer.start()

    def _poll_compute(self) -> None:
        if self._compute is None or not self._compute.done():
            return
        self.compute_timer.stop()
        future, self._compute = self._compute, None
        self.more_views_button.setEnabled(True)
        try:
            result = future.result()
        except Exception as exc:
            self._result = None
            self.result_verdict.setText("Could not compute")
            self.result_text.setText(f"{exc}\n\nGo back and add more varied views.")
            return
        self._result = result
        outliers = cal.outlier_views(result.per_view_rms_px)
        worst = sorted(range(len(result.per_view_rms_px)), key=lambda i: -result.per_view_rms_px[i])[:5]
        K = result.K
        self.result_verdict.setText(
            f"PASS: {result.rms_px:.3f} px" if result.passed
            else f"FAIL: {result.rms_px:.3f} px (needs < {cal.INTRINSIC_RMS_PX} px)")
        self.result_verdict.setStyleSheet(
            "font-size: 18px; font-weight: 700; color: " + ("#2e7d32;" if result.passed else "#c62828;"))
        cov = self._session.coverage
        self.result_text.setText(
            f"Camera: {self._selected_slot().label}\n"
            f"Views used: {result.n_views}, image {result.image_size[0]} x {result.image_size[1]}\n"
            f"Focal length: fx {K[0, 0]:.1f} px, fy {K[1, 1]:.1f} px; centre {K[0, 2]:.1f}, {K[1, 2]:.1f}\n"
            f"Distortion: {', '.join(f'{d:+.4f}' for d in result.D[:5])}\n"
            f"Coverage: {len(cov.cells)}/9 regions, near {cov.near}, far {cov.far}, tilted {cov.tilted}\n"
            f"Worst views: {', '.join(f'#{i + 1} {result.per_view_rms_px[i]:.2f} px' for i in worst)}\n"
            + (f"{len(outliers)} view(s) are much worse than the rest (blurred or bent board?)."
               if outliers else "No outlier views."))
        self.save_button.setEnabled(result.passed)
        self.save_loose_button.setEnabled(not result.passed)
        self.drop_outliers_button.setEnabled(bool(outliers))

    def _drop_outliers(self) -> None:
        if self._result is None or self._session is None:
            return
        self._session.drop(cal.outlier_views(self._result.per_view_rms_px))
        self._start_compute()

    def _save(self, loose: bool) -> None:
        if self._result is None or self._session is None:
            return
        slot = self._selected_slot()
        fingerprint = slot.controller.get_sensor_fingerprint() or {}
        w, h = self._result.image_size
        if fingerprint.get("Width") not in (None, w) or fingerprint.get("Height") not in (None, h):
            QMessageBox.warning(self, "Calibration", "The camera's resolution changed during calibration. "
                                                     "Calibrate again.")
            return
        cov = self._session.coverage
        record = IntrinsicsRecord(
            serial=slot.serial, model=slot.model,
            K=self._result.K.tolist(), D=[float(d) for d in self._result.D],
            image_size=[int(w), int(h)], fingerprint=fingerprint, board=self._board_cfg.to_dict(),
            rms_px=float(self._result.rms_px), n_views=int(self._result.n_views),
            per_view_rms_px=[float(e) for e in self._result.per_view_rms_px],
            coverage={"cells": sorted(list(c) for c in cov.cells), "near": cov.near, "far": cov.far,
                      "tilted": cov.tilted, "flat": cov.flat},
            loose=loose,
        )
        path = self.store.save_intrinsics(record)
        self.on_saved()
        self._session = None
        self._result = None
        self.saved_label.setText(f"Saved {slot.label}" + (" (marked as failed)" if loose else "") + f": {path}")
        self._go(self.page_home)

    # ------------------------------------------------------------------ board
    def _on_print_board(self) -> None:
        name = self.print_combo.currentText()
        default = name.split(" - ")[0].replace(" ", "_") + ".pdf"
        path, _ = QFileDialog.getSaveFileName(self, "Save printable board", default, "PDF (*.pdf)")
        if not path:
            return
        try:
            write_board_pdf(path, cal.BOARD_PRESETS[name], name)
        except Exception as exc:
            QMessageBox.warning(self, "Calibration", f"Could not write the board: {exc}")
            return
        QMessageBox.information(self, "Calibration", f"Saved {path}.\nPrint it at 100% / Actual size.")

    # --------------------------------------------------------------- shutdown
    def closeEvent(self, event) -> None:
        self.ui_timer.stop()
        self.compute_timer.stop()
        self._stop_detector()
        self._pool.shutdown(wait=False, cancel_futures=True)
        super().closeEvent(event)
