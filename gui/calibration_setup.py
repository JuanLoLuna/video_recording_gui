"""Calibration window, task 2: set up the cameras for this session (plan step 14, decision 11).

1. Before you start: both cameras calibrated, board A RESTING flat at its marked
   spot (static, so the unsynchronised cameras do not matter), reference boards
   reference boards mounted (B1/B2, or the big-marker G1/G2), measured sizes.
2. Capture: ~1 s of frames per camera, averaged. Each camera's pose to board A
   (board A's frame is the session's world frame), the camera-to-camera baseline
   and the triangulation check (backend.calibration.compute_setup); each
   camera's pose to whichever reference board it sees best.
3. Verify: board A raised >= 15 cm (on a box), triangulated with the SAVED poses
   and compared with its real size (verify_setup). PASS -> the setup is saved.

Mixed into CalibrationWindow; uses its store, slots, ensure_preview, on_saved,
_go() and _detector bookkeeping conventions.
"""

from __future__ import annotations

import json
import time
from datetime import datetime

import cv2
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from backend import calibration as cal
from backend.calibration_store import SetupRecord
from gui.calibration_common import _Detector, draw_overlay
from gui.camera_preview import frame_to_qimage

SETUP_CAPTURE_S = 1.2           # frames averaged per capture (board A and references are static)
SETUP_VIEW_SIZE = (400, 300)
_BIG = "font-size: 16px; font-weight: 600;"
_HINT = "color: #555;"
_ERROR = "color: #c62828; font-weight: 600;"
_OK = "color: #2e7d32; font-weight: 600;"
_A_COLOUR = (60, 200, 60)       # BGR: board A corners
_REF_COLOUR = (255, 140, 0)     # BGR: reference board corners


_REF_SHORT = dict(zip(cal.REFERENCE_PRESETS, ("B1", "B2", "G1", "G2")))


def _ref_label(name: str) -> str:
    return _REF_SHORT.get(name, name)


class SetupTaskMixin:
    # ------------------------------------------------------------------ pages
    def _build_setup_pages(self) -> list[QWidget]:
        self._setup_detectors: dict[str, _Detector] = {}
        self._setup_last_seq: dict[str, int] = {}
        self._setup_latest: dict[str, dict] = {}       # serial -> {name: Detection} of the newest frame
        self._setup_frames: dict = {}                   # serial -> newest full-resolution frame (diagnosis)
        self._setup_collect: dict[str, list] | None = None  # serial -> [{name: Detection}] while capturing
        self._setup_collect_until = 0.0
        self._setup_collect_for = ""                    # "setup" | "verify"
        self._setup_boards: dict[str, cal.BoardConfig] = {}
        self._setup_intrinsics: dict = {}               # serial -> IntrinsicsRecord
        self._setup_result: cal.SetupResult | None = None
        self._setup_dets_a: dict = {}
        self._setup_refs: dict = {}                     # serial -> (preset name, Detection, BoardPose) or None
        self._verify_result: cal.VerifyResult | None = None
        return [self._build_setup_check(), self._build_setup_live()]

    def _build_setup_check(self) -> QWidget:
        page = QWidget()
        self.page_setup_check = page
        v = QVBoxLayout(page)
        head = QLabel("Set up for this session: step 1 of 3, before you start")
        head.setStyleSheet(_BIG)
        v.addWidget(head)
        self.setup_cameras_label = QLabel("")
        self.setup_cameras_label.setWordWrap(True)
        v.addWidget(self.setup_cameras_label)

        grid = QGridLayout()
        grid.addWidget(QLabel("Board A measured square"), 0, 0)
        self.setup_square_a = QDoubleSpinBox()
        self.setup_square_a.setRange(5.0, 200.0)
        self.setup_square_a.setDecimals(2)
        self.setup_square_a.setSuffix(" mm")
        self.setup_square_a.setValue(cal.BOARD_PRESETS[cal.HANDHELD_PRESET].square_length_m * 1000)
        grid.addWidget(self.setup_square_a, 0, 1)
        grid.addWidget(QLabel("Boards B1/B2 measured square (if used)"), 1, 0)
        self.setup_square_b = QDoubleSpinBox()
        self.setup_square_b.setRange(5.0, 200.0)
        self.setup_square_b.setDecimals(2)
        self.setup_square_b.setSuffix(" mm")
        self.setup_square_b.setValue(cal.BOARD_PRESETS[cal.REFERENCE_PRESETS[0]].square_length_m * 1000)
        grid.addWidget(self.setup_square_b, 1, 1)
        grid.addWidget(QLabel("Boards G1/G2 measured marker side (if used)"), 2, 0)
        self.setup_marker_g = QDoubleSpinBox()
        self.setup_marker_g.setRange(5.0, 300.0)
        self.setup_marker_g.setDecimals(2)
        self.setup_marker_g.setSuffix(" mm")
        self.setup_marker_g.setValue(cal.measured_size_mm(cal.BOARD_PRESETS[cal.REFERENCE_PRESETS[2]]))
        self.setup_marker_g.setToolTip("Outer edge of one black marker on G1/G2 (nominal 70 mm)")
        grid.addWidget(self.setup_marker_g, 2, 1)
        v.addLayout(grid)

        self.setup_checks = [
            QCheckBox("Board A is lying flat at its marked spot, where both cameras see it, and nobody is touching it."),
            QCheckBox("The cameras are mounted where they will stay for this session."),
        ]
        for box in self.setup_checks:
            box.toggled.connect(self._update_setup_check_next)
            v.addWidget(box)
        refs = QLabel(
            "Reference boards: one where camera 1 sees it (B1 or G1), one where camera 2 sees it (B2 or G2); "
            "either camera may see more than one. The big-marker G boards are read from about 3x farther, so "
            "they can go on a wall away from the subject. They let the app notice later if a camera was "
            "moved, and reuse this setup in the next session if nothing moved. Without them the setup is "
            "only valid until the next Detect.")
        refs.setWordWrap(True)
        refs.setStyleSheet(_HINT)
        v.addWidget(refs)
        self.setup_check_error = QLabel("")
        self.setup_check_error.setWordWrap(True)
        self.setup_check_error.setStyleSheet(_ERROR)
        v.addWidget(self.setup_check_error)
        v.addStretch(1)
        nav = QHBoxLayout()
        back = QPushButton("Back")
        back.clicked.connect(lambda: self._go(self.page_home))
        nav.addWidget(back)
        nav.addStretch(1)
        self.setup_check_next = QPushButton("Next: capture the setup")
        self.setup_check_next.setEnabled(False)
        self.setup_check_next.clicked.connect(self._start_setup)
        nav.addWidget(self.setup_check_next)
        v.addLayout(nav)
        return page

    def _build_setup_live(self) -> QWidget:
        page = QWidget()
        self.page_setup_live = page
        v = QVBoxLayout(page)
        self.setup_head = QLabel("")
        self.setup_head.setStyleSheet(_BIG)
        v.addWidget(self.setup_head)
        self.setup_instructions = QLabel("")
        self.setup_instructions.setWordWrap(True)
        v.addWidget(self.setup_instructions)
        row = QHBoxLayout()
        self.setup_views: dict[str, QLabel] = {}
        self.setup_view_status: dict[str, QLabel] = {}
        for slot in self.slots:
            col = QVBoxLayout()
            caption = QLabel(slot.label)
            caption.setStyleSheet("font-weight: 600;")
            col.addWidget(caption)
            view = QLabel("Waiting for frames…")
            view.setAlignment(Qt.AlignmentFlag.AlignCenter)
            view.setMinimumSize(*SETUP_VIEW_SIZE)
            view.setStyleSheet("background: #222; color: #ccc;")
            col.addWidget(view, stretch=1)
            status = QLabel("")
            status.setWordWrap(True)
            col.addWidget(status)
            self.setup_views[slot.serial] = view
            self.setup_view_status[slot.serial] = status
            row.addLayout(col)
        v.addLayout(row, stretch=1)
        legend = QLabel("Green dots: board A. Orange dots: the reference board (B1/B2/G1/G2).")
        legend.setStyleSheet(_HINT)
        v.addWidget(legend)
        self.setup_verdict = QLabel("")
        self.setup_verdict.setStyleSheet("font-size: 16px; font-weight: 700;")
        v.addWidget(self.setup_verdict)
        self.setup_text = QLabel("")
        self.setup_text.setWordWrap(True)
        self.setup_text.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        v.addWidget(self.setup_text)

        nav = QHBoxLayout()
        self.setup_back = QPushButton("Back")
        self.setup_back.clicked.connect(self._setup_go_back)
        nav.addWidget(self.setup_back)
        self.setup_snapshot_button = QPushButton("Save snapshot for diagnosis")
        self.setup_snapshot_button.setToolTip(
            "Saves each camera's current frame and a report of what was detected, to send for help")
        self.setup_snapshot_button.clicked.connect(self._save_setup_snapshot)
        nav.addWidget(self.setup_snapshot_button)
        nav.addStretch(1)
        self.setup_capture_button = QPushButton("Capture setup")
        self.setup_capture_button.clicked.connect(lambda: self._setup_begin_collect("setup"))
        nav.addWidget(self.setup_capture_button)
        self.setup_next_button = QPushButton("Next: verify")
        self.setup_next_button.clicked.connect(self._setup_to_verify)
        nav.addWidget(self.setup_next_button)
        self.verify_button = QPushButton("Verify")
        self.verify_button.clicked.connect(lambda: self._setup_begin_collect("verify"))
        nav.addWidget(self.verify_button)
        self.setup_save_button = QPushButton("Save setup")
        self.setup_save_button.clicked.connect(self._save_setup)
        nav.addWidget(self.setup_save_button)
        v.addLayout(nav)
        return page

    # ------------------------------------------------------------- navigation
    def _open_setup(self) -> None:
        self.setup_check_error.setText("")
        lines, ok = [], True
        self._setup_intrinsics = {}
        for slot in self.slots:
            try:
                rec = self.store.current_intrinsics(slot.serial)
            except Exception as exc:
                rec, note = None, f"could not read the calibration ({exc})"
            else:
                note = "not calibrated: calibrate it first" if rec is None else (
                    f"calibrated, {rec.rms_px:.2f} px" + (" (saved as failed)" if rec.loose else ""))
            ok &= rec is not None
            self._setup_intrinsics[slot.serial] = rec
            lines.append(f"{slot.label}: {note}")
        if len(self.slots) != 2:
            ok = False
            lines.append(f"The setup needs exactly two cameras ({len(self.slots)} detected).")
        self._setup_cameras_ok = ok
        self.setup_cameras_label.setText("\n".join(lines))
        self.setup_cameras_label.setStyleSheet("" if ok else _ERROR)
        for box in self.setup_checks:
            box.setChecked(False)
        self._update_setup_check_next()
        self._go(self.page_setup_check)

    def _update_setup_check_next(self) -> None:
        self.setup_check_next.setEnabled(
            getattr(self, "_setup_cameras_ok", False) and all(b.isChecked() for b in self.setup_checks))

    def _start_setup(self) -> None:
        if not self.ensure_preview():
            self.setup_check_error.setText("The cameras could not start. Check them in the main window.")
            return
        self._setup_boards = {
            "A": cal.with_measured_square(cal.BOARD_PRESETS[cal.HANDHELD_PRESET], self.setup_square_a.value()),
            **{name: cal.with_measured_square(
                cal.BOARD_PRESETS[name],
                (self.setup_marker_g if cal.BOARD_PRESETS[name].kind == "grid" else self.setup_square_b).value())
               for name in cal.REFERENCE_PRESETS},
        }
        self._stop_setup_detectors()
        for slot in self.slots:
            detectors = {name: cal.BoardDetector(cfg) for name, cfg in self._setup_boards.items()}
            self._setup_detectors[slot.serial] = _Detector(slot.controller, detectors)
            self._setup_last_seq[slot.serial] = 0
        self._setup_latest = {}
        self._setup_result = None
        self._verify_result = None
        self._setup_show_phase("setup")
        self._go(self.page_setup_live)
        self.ui_timer.start()

    def _setup_show_phase(self, phase: str) -> None:
        self._setup_phase = phase
        self.setup_verdict.setText("")
        self.setup_text.setText("")
        if phase == "setup":
            self.setup_head.setText("Set up for this session: step 2 of 3, capture the setup")
            self.setup_instructions.setText(
                "Board A lies flat at its marked spot. Check that both cameras see it whole (green dots), "
                "and that each sees its reference board (orange). Nobody touches anything, then press "
                "Capture setup (about 1 second).")
        else:
            self.setup_head.setText("Set up for this session: step 3 of 3, verify")
            self.setup_instructions.setText(
                "Put board A on the box, at least 15 cm higher than where it was, where both cameras see "
                "it. Let go of it, then press Verify (about 1 second). The app measures it in 3D with the "
                "setup just captured and checks it comes out at its real size.")
        self._setup_update_buttons()

    def _setup_update_buttons(self) -> None:
        collecting = self._setup_collect is not None
        both_see_a = len(self.slots) == 2 and all(
            (self._setup_latest.get(s.serial) or {}).get("A") is not None
            and self._setup_latest[s.serial]["A"].ok for s in self.slots)
        setup_phase = self._setup_phase == "setup"
        self.setup_capture_button.setVisible(setup_phase)
        self.setup_capture_button.setEnabled(setup_phase and both_see_a and not collecting)
        self.setup_next_button.setVisible(setup_phase)
        self.setup_next_button.setEnabled(bool(self._setup_result and self._setup_result.passed) and not collecting)
        self.verify_button.setVisible(not setup_phase)
        self.verify_button.setEnabled(not setup_phase and both_see_a and not collecting)
        self.setup_save_button.setVisible(not setup_phase)
        self.setup_save_button.setEnabled(bool(self._verify_result and self._verify_result.passed)
                                          and not collecting)
        self.setup_back.setEnabled(not collecting)

    def _setup_go_back(self) -> None:
        if self._setup_phase == "verify":
            self._verify_result = None
            self._setup_show_phase("setup")
            return
        self._stop_setup_detectors()
        self._go(self.page_setup_check)

    def _setup_to_verify(self) -> None:
        self._verify_result = None
        self._setup_show_phase("verify")

    def _stop_setup_detectors(self) -> None:
        for det in self._setup_detectors.values():
            det.stop()
        self._setup_detectors = {}
        self._setup_collect = None

    # ------------------------------------------------------------------- live
    def _setup_tick(self) -> None:
        """GUI thread: newest detections per camera -> overlay, status, and collection while capturing."""
        changed = False
        for slot in self.slots:
            worker = self._setup_detectors.get(slot.serial)
            latest = None if worker is None else worker.latest()
            if latest is None or latest[0] == self._setup_last_seq.get(slot.serial):
                continue
            seq, frame, dets = latest
            self._setup_last_seq[slot.serial] = seq
            self._setup_latest[slot.serial] = dets
            self._setup_frames[slot.serial] = frame
            changed = True
            if self._setup_collect is not None:
                self._setup_collect[slot.serial].append(dets)
            ref = cal.best_detection({n: dets.get(n) for n in cal.REFERENCE_PRESETS})
            image = frame_to_qimage(draw_overlay(
                frame, dets.get("A"), max_size=SETUP_VIEW_SIZE,
                extra=[] if ref is None else [(ref[1], _REF_COLOUR)]))
            if image is not None:
                view = self.setup_views[slot.serial]
                view.setPixmap(QPixmap.fromImage(image).scaled(
                    view.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
            a = dets.get("A")
            a_text = (f"board A: {len(a.ids)}/{self._setup_boards['A'].corner_count} corners"
                      if a is not None and a.ok else "board A: not found")
            r_text = ("reference: none seen" if ref is None
                      else f"reference: {_ref_label(ref[0])}, {len(ref[1].ids)} corners")
            self.setup_view_status[slot.serial].setText(f"{a_text}; {r_text}")
        if self._setup_collect is not None and time.monotonic() >= self._setup_collect_until:
            self._setup_finish_collect()
            changed = True
        if changed:
            self._setup_update_buttons()

    def _setup_begin_collect(self, what: str) -> None:
        self._setup_collect = {slot.serial: [] for slot in self.slots}
        self._setup_collect_until = time.monotonic() + SETUP_CAPTURE_S
        self._setup_collect_for = what
        self.setup_verdict.setStyleSheet("font-size: 16px; font-weight: 700;")
        self.setup_verdict.setText("Capturing… keep still")
        self.setup_text.setText("")
        self._setup_update_buttons()

    def _setup_finish_collect(self) -> None:
        collected, self._setup_collect = self._setup_collect, None
        averaged = {serial: {name: cal.average_detections([f[name] for f in frames if f.get(name) is not None])
                             for name in self._setup_boards} for serial, frames in collected.items()}
        try:
            if self._setup_collect_for == "setup":
                self._compute_setup(averaged)
            else:
                self._compute_verify(averaged)
        except Exception as exc:  # show it; never leave the page stuck
            self.setup_verdict.setStyleSheet("font-size: 16px; font-weight: 700; color: #c62828;")
            self.setup_verdict.setText("Could not compute")
            self.setup_text.setText(str(exc))

    # ---------------------------------------------------------------- compute
    def _intr(self, serial: str):
        rec = self._setup_intrinsics[serial]
        return np.asarray(rec.K, dtype=float), np.asarray(rec.D, dtype=float)

    def _compute_setup(self, averaged: dict) -> None:
        board_a = cal.make_board(self._setup_boards["A"])
        dets_a = {serial: per["A"] for serial, per in averaged.items()}
        intrinsics = {serial: self._intr(serial) for serial in dets_a}
        result = cal.compute_setup(dets_a, intrinsics, board_a)
        refs = {}
        for serial, per in averaged.items():
            best = cal.best_detection({n: per.get(n) for n in cal.REFERENCE_PRESETS})
            pose = None if best is None else cal.solve_board_pose(
                best[1], cal.make_board(self._setup_boards[best[0]]), *intrinsics[serial])
            refs[serial] = None if pose is None else (best[0], best[1], pose)
        self._setup_result, self._setup_dets_a, self._setup_refs = result, dets_a, refs

        labels = {s.serial: s.label for s in self.slots}
        lines = []
        for serial in dets_a:
            pose = result.poses.get(serial)
            ref = refs[serial]
            lines.append(
                f"{labels[serial]}: " + ("board A not usable" if pose is None else
                                         f"board A fit {pose.rms_px:.2f} px, {pose.n_corners} corners")
                + "; reference " + ("none (this camera's moves cannot be checked later)" if ref is None else
                                    f"{_ref_label(ref[0])}, fit {ref[2].rms_px:.2f} px"))
        if result.baseline_mm is not None:
            lines.append(f"Distance between the cameras: {result.baseline_mm:.0f} mm")
        if result.triangulation_rms_mm is not None:
            lines.append(f"Board A measured in 3D: {result.triangulation_rms_mm:.2f} mm RMS off "
                         f"(limit {cal.SETUP_TRIANGULATION_RMS_MM} mm, {result.triangulated_corners} corners)")
        problems = []
        if result.missing:
            problems.append("board A not found by " + ", ".join(labels[s] for s in result.missing))
        problems += [f"{labels[s]}: board A fits {result.poses[s].rms_px:.2f} px (limit "
                     f"{cal.SETUP_REPROJECTION_RMS_PX} px)" for s, ok in result.per_camera_passed.items() if not ok]
        if (result.triangulation_rms_mm is not None
                and result.triangulation_rms_mm >= cal.SETUP_TRIANGULATION_RMS_MM):
            problems.append("board A does not come out at its real shape in 3D")
        passed = result.passed
        self.setup_verdict.setStyleSheet(
            "font-size: 16px; font-weight: 700; color: " + ("#2e7d32;" if passed else "#c62828;"))
        self.setup_verdict.setText("Setup captured: PASS. Next: verify." if passed else "Setup: FAIL")
        if problems:
            lines.append("Problems: " + "; ".join(problems) + ". Check board A is whole, flat, still and in "
                         "focus in both views, then capture again. If it keeps failing, recalibrate the camera.")
        self.setup_text.setText("\n".join(lines))

    def _compute_verify(self, averaged: dict) -> None:
        board_a = cal.make_board(self._setup_boards["A"])
        a, b = [s.serial for s in self.slots]
        det_a, det_b = averaged[a]["A"], averaged[b]["A"]
        if det_a is None or det_b is None:
            missing = [s.label for s in self.slots if averaged[s.serial]["A"] is None]
            raise ValueError("board A was not seen by " + ", ".join(missing) + " while verifying.")
        result = cal.verify_setup(det_a, self._setup_result.poses[a], *self._intr(a),
                                  det_b, self._setup_result.poses[b], *self._intr(b), board_a)
        self._verify_result = result
        passed = result.passed
        self.setup_verdict.setStyleSheet(
            "font-size: 16px; font-weight: 700; color: " + ("#2e7d32;" if passed else "#c62828;"))
        self.setup_verdict.setText("Verify: PASS. Save the setup." if passed else "Verify: FAIL")
        lines = []
        if result.rms_mm is not None:
            lines.append(f"Board A on the box: {result.scale_error_pct:+.2f} % off its real size "
                         f"(limit ±{cal.VERIFY_SCALE_ERROR_PCT} %), shape {result.rms_mm:.2f} mm RMS, "
                         f"{result.depth_change_m * 100:.0f} cm above where it was, {result.corners} corners")
        if result.problems:
            lines.append("Problems: " + "; ".join(result.problems) + ".")
            lines.append("If it is high enough and still fails: go Back and capture the setup again; if it "
                         "keeps failing, recalibrate the cameras (a lens may have been touched).")
        self.setup_text.setText("\n".join(lines))

    def _save_setup_snapshot(self) -> None:
        """Full-resolution frame + detection report per camera, in <calibration dir>/diagnostics/<time>/."""
        folder = self.store.root / "diagnostics" / datetime.now().strftime("%Y%m%d_%H%M%S")
        try:
            folder.mkdir(parents=True, exist_ok=True)
            report = {"created_at": datetime.now().isoformat(timespec="seconds"),
                      "boards": {name: cfg.to_dict() for name, cfg in self._setup_boards.items()},
                      "cameras": {}}
            lines = []
            for slot in self.slots:
                frame = self._setup_frames.get(slot.serial)
                if frame is None:
                    report["cameras"][slot.serial] = {"label": slot.label, "frame": None}
                    lines.append(f"{slot.label}: no frame yet")
                    continue
                image_name = f"{slot.serial}.png"
                cv2.imwrite(str(folder / image_name), frame)
                diag = cal.diagnose_frame(frame, self._setup_boards)
                report["cameras"][slot.serial] = {"label": slot.label, "frame": image_name, **diag}
                lines.append(f"{slot.label}: {diag['image_size'][0]}x{diag['image_size'][1]}, brightness "
                             f"{diag['mean_brightness']}, saturated {diag['saturated_fraction'] * 100:.1f} %")
                for name, b in diag["boards"].items():
                    lines.append(
                        f"  {_ref_label(name) if name != 'A' else 'A'}: markers {b['markers_of_this_board']}"
                        f"/{b['board_markers_total']}, side {b['median_marker_side_px']} px, "
                        f"rejected {b['rejected_candidates']}, corners {b['charuco_corners']}, "
                        f"{'usable' if b['usable'] else 'NOT usable'}")
            (folder / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            (folder / "report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        except Exception as exc:
            self.setup_text.setText(f"Could not save the snapshot: {exc}")
            return
        self.setup_text.setText(f"Snapshot saved to {folder} (send the whole folder).\n" + "\n".join(lines))

    # ------------------------------------------------------------------- save
    def _save_setup(self) -> None:
        result, verify = self._setup_result, self._verify_result
        if result is None or verify is None or not verify.passed:
            return
        cameras = []
        for slot in self.slots:
            pose = result.poses[slot.serial]
            ref = self._setup_refs.get(slot.serial)
            cameras.append({
                "serial": slot.serial,
                "R": pose.R.tolist(), "t": pose.t.tolist(), "rms_px": float(pose.rms_px),
                "intrinsics_id": self._setup_intrinsics[slot.serial].id,
                "reference": None if ref is None else {
                    "name": ref[0], "board": self._setup_boards[ref[0]].to_dict(),
                    "R": ref[2].R.tolist(), "t": ref[2].t.tolist(), "rms_px": float(ref[2].rms_px)},
            })
        record = SetupRecord(
            cameras=cameras, board=self._setup_boards["A"].to_dict(),
            baseline_mm=result.baseline_mm, triangulation_rms_mm=result.triangulation_rms_mm,
            passed=bool(result.passed),
            verify={"passed": True, "rms_mm": verify.rms_mm, "scale_error_pct": verify.scale_error_pct,
                    "depth_change_m": verify.depth_change_m, "corners": verify.corners},
        )
        try:
            path = self.store.save_setup(record)
        except Exception as exc:
            self.setup_verdict.setStyleSheet("font-size: 16px; font-weight: 700; color: #c62828;")
            self.setup_verdict.setText("Could not save the setup")
            self.setup_text.setText(f"{exc}\nPress Save setup to try again.")
            return
        self._stop_setup_detectors()
        self.ui_timer.stop()
        self.on_saved()
        no_ref = [s.label for s in self.slots if self._setup_refs.get(s.serial) is None]
        self.saved_label.setText(
            f"Setup saved: {path}" + ("" if not no_ref else
                                      f". No reference board for {', '.join(no_ref)}: redo the setup after the "
                                      "next Detect."))
        self._go(self.page_home)
