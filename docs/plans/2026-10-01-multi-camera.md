# Two-camera support — implementation plan

Branch: `feat/multi-camera`. Drafted 2026-10-01 from rig measurements (see "Evidence") and a planning pass over the code.
Target: **2 cameras at 30 fps**, ~1 h sessions, ~2 h/day. A one-camera setup must behave exactly as before.

## Decisions (user-confirmed)

| # | Decision |
|---|---|
| 1 | Codecs unchanged: uncompressed GREY default, MJPEG opt-in. Codec/quality work belongs to a separate fork — **do not touch `_open_cv2_writer`, codec choice, or the compression/quality controls' logic**; only fan them out per camera. |
| 2 | One output folder, per-camera filename **suffix** (not subfolders). |
| 3 | Cross-camera sync is **out of scope**. Sync-pulse / label calls just fan out to every controller and must never raise. |
| 4 | Primary (unsuffixed) camera: **does not matter**. Rule: first serial in `SLEEVE_VIDEO_GUI_CAMERA_SERIALS`, else the lowest serial. |
| 5 | A configured camera is missing: **manual start refuses**; unattended power-resume records with what is present and warns loudly. |
| 6 | Downstream ingestion (`smart_sleeve_data_processing`) is **not modified now**. The second camera is archive-only video until the first two-camera ingest. |
| 7 | Disk warning thresholds lowered: warn 8 h, critical 2 h (was 24 h / 6 h). |

## Evidence (scripts/multi_camera_probe.py, on the rig)

- Cameras: Firefly FFY-U3-04S2M #23227865 (720x540 Mono8), Blackfly S BFS-U3-13Y3M #26134271 (1280x1024 Mono8); SuperSpeed via a Thunderbolt dock. `cam_list[0]` is the Firefly — enumeration order is not stable, so bind by serial.
- 2 cameras together vs solo at 30 and 60 fps: 0 gaps, no contention, <=0.15 cores. Threads are enough; no multiprocessing.
- MJPEG (FFMPEG backend) both cameras: 30 fps write 8.7/2.0 ms per frame, ~13 GB/h; 60 fps 8.5/2.0 ms, 0.76 cores. Uncompressed ~184 GB/h.
- Fixed on main: `(IS_COLOR, QUALITY)` params made OpenCV fall to the ~80x slower CV_MJPEG backend.

## Design

- **One `CameraController` per camera**, bound by serial (`GetBySerial`); legacy no-serial path keeps `cam_list[0]` and pins the serial it finds.
- **`SharedSystemHolder`** (`backend/spinnaker_system.py`): owner-tracked reference set (`acquire(self)` / `release(self)` / `restart_if_sole_owner(self)`) over `PySpin.System`; controllers never call `GetInstance`/`ReleaseInstance` themselves. Releasing is idempotent per owner, so a repeated `stop()` cannot free the System under another camera's deferred teardown. Never take a controller lock while holding the holder's lock. Fault recovery: sole owner -> today's full rebuild (validated path); shared -> rebuild only this camera's `Camera`/`CameraList`, re-find by serial, never touch the System.
- **Naming in one place** (`SessionPaths.camera_tag`): tag `cam<serial>` goes *inside the stem*, before `-NNNN` / `_metadata`; the primary stays unsuffixed; the single WAV is untagged.
  `recording_20261001_101500-0000.avi` (primary) / `recording_20261001_101500_cam26134271-0000.avi` / `..._cam26134271_metadata.csv`.
  Known downstream gap (deliberately deferred): the video regexes in `smart_sleeve_data_processing` (`rules.yaml` video_raw/video_metadata, `router.py` RECORDING_RE) do not match tagged names; fix at the first two-camera ingest. The proposed pattern is `^recording_\d{8}_\d{6}(?:_cam[A-Za-z0-9]+)?(?:-\d{4})?\.avi$`.
- **`CameraGroup`** (`backend/camera_group.py`, duck-typed, no PySpin): start/stop fan-out with rollback; stop order = `stop_recording` on all first, then `stop()` each, System released last and never after a deferred teardown. Slot tags must be unique (at most one untagged); the group never invents a primary (`primary` may be `None`; `default_slot` is where the GUI points); recording starts the **primary last**; `best_effort=True` keeps whichever cameras start (power-resume path).
- **Disk guard**: `estimate_bytes_per_hour(streams)` from real per-camera frame sizes (~183.5 GB/h uncompressed).
- **GUI** (`gui/main.py`): `self.camera` becomes a property for the camera selected in the tuning panel (keeps ~20 tuning call sites); start/stop/record/notify/frame-rate/exposure-lock/compression go through the group. **Exposure lock must force `ExposureAuto=Off` on every camera**, not just the selected one. Preview tiles (downscaled, `render_ms` measured), one diagnostics CSV per camera, one warning banner with `[model serial]` prefixes when N>1.

## Steps (commit-sized)

Pure-logic steps 1-7 are unit-tested on macOS (`unittest`, no mocking, injected callables / fakes). Steps 8+ touch PySpin and need the rig.

1. `SessionPaths.camera_tag` + naming tests (legacy names byte-identical with no tag)
2. `backend/camera_registry.py` — env parsing, selection, tags, primary rule, range intersection
3. Disk-rate math + new thresholds
4. `backend/spinnaker_system.py` — `SharedSystemHolder`
5. Events header: optional `camera_serial`/`camera_model`/`session`
6. `backend/camera_group.py`
7. Preview scaling helper + `render_ms` diagnostics columns
8. ✅ `CameraController`: holder, serial binding, reinit by serial, two-phase recording, accessors (`serial`, `model`, `get_stream_rate()`), tagged thread names, preview `max_size` — done and unit/integration-tested on macOS against fake cameras; **rig: single-camera regression still to run**
9. ✅ `scripts/multi_controller_smoke.py` + `backend/session_verify.py` — two real controllers headless (`tests/test_two_camera_integration.py` runs the same flow against fake cameras on macOS) — **rig: 2 x 10 min, then unplug test**
10-14. ✅ GUI (written together, tested headless): `gui/main.py` builds a `CameraGroup` at Detect through the registry (`enumerate_cameras` + `select_cameras`); start/stop/record/sync/label/frame-rate/compression/exposure-lock all fan out; `self.camera` is a property for the camera the tuning panel points at (selector shown only with several cameras); per-camera preview tiles (`gui/camera_preview.py`, downscaled before reaching the GUI thread), per-camera diagnostics accumulator + CSV (`render_ms` included), per-camera captions and `[model #serial]` banner prefixes, a sustained-backlog warning, per-slot `SessionPaths` from one timestamp, disk estimate from real frame sizes, manual start refused when a configured camera is missing, power-resume records best-effort. `tests/test_gui_cameras.py` drives the real `MainWindow` (Qt offscreen) against fake cameras. **Rig: single-camera GUI regression, then the full two-camera checklist below.**
15. Validation report in `docs/reports/` (downstream changes deferred)

## Rig validation checklist (step 14)

- **Single camera**: names identical to today; 10 min with 0 gaps/incomplete/errors; unplug/replug -> 1 reinit, dense `record_frame_index`, timeline break logged.
- **Two cameras, 30 fps, 60 min in the real app** (uncompressed to D:, plus one MJPEG run), per camera: `camera_frame_gaps`, incomplete, acquisition_errors, append_failures, camera_reinits all 0; metadata rows = sum of segment `frame_count` = last `record_frame_index`; rows >= 0.999 x 3600 x fps; `append_queue_depth` max < 5 (uncompressed), <= 6 (MJPEG); `.incomplete/` empty after stop; `scripts/verify_avi.py` decoded frames = manifest.
  Uncompressed segments roll by the 3 GB byte ceiling long before 15 min (Blackfly ~80 s, Firefly ~260 s) — expected.
- **Resources over the hour**: process CPU avg <= 1.5 cores, working set growth <= 200 MB after minute 5, no thermal throttling.
- **Preview**: `preview_age_ms` p95 <= 250 ms per tile; no GUI stall; both tiles' `render_ms` p95 sum <= 10 ms.
- **Fault**: unplug one camera 20-30 s mid-recording, once per camera and once both: the other camera shows 0 gaps/errors/reinits; the unplugged one gets `timeline_break`, 1 reinit, a fault-forced roll, and resumes at its own resolution; replug into a different port re-binds the right serial.
- **Stop/teardown**: both final segments renamed, both `segments.csv` end `session_stop`; 5 start/stop cycles without restart; app closes in < 5 s; no "teardown deferred".
- **Power pause/resume**: both stop; resume uses one new shared stem. **Labels**: `label_start/label_end` in both metadata CSVs; Sync Pulse with no DAQ raises nothing.

## Step 8 requirements carried over from the audit

Verified against the real `CameraController` by an independent audit of steps 1-7. Step 8 (and 10/14) must do all of these:

- **Bind by serial everywhere**: use `select_cameras(...).bound[0].serial`, never `cam_list[0]` (enumeration order changes between boots). A controller with no explicit serial resolves one through the registry and pins it.
- **Always write camera identity**: events header gets `recording=<stem>`, `camera_serial`, `camera_model`, `session=<basename>` (camera_control.py ~1200 currently passes `session_paths.basename`). This is also how a silent primary swap is detected afterwards.
- **Quick Stop -> Start bug (exists on main)**: `start_recording()` returns `(True, "...already starting or in progress")` while a previous recording is still closing (`recording_active and record_stop_requested`), so the GUI shows RECORDING while nothing records. Return `(False, "previous recording still closing")` or make the caller wait.
- **Clear stale pending state in `start_recording()`**: `_pending_label_event` / sync window left over from the last session lands on the next session's first frame.
- **Two-phase start**: `stop_recording()` only sets a flag and cannot undo an accepted start. Open every sidecar first, raise the start flag only after all cameras succeed (the group already starts the primary last as a partial mitigation).
- **Reinit ordering**: set `self.cam = None` before `CameraList.Clear()` (validated by `scripts/reinit_spike.py`); find the camera again with `GetBySerial(self.serial)`; call `GetCameras()` and fall back to `UpdateCameras()`. Sole owner (`restart_if_sole_owner(self)`) keeps today's full rebuild; shared rebuilds only this camera.
- **Per-camera names**: thread names (`frame-metadata-writer`, `segment-closer`, `segment-appender`, acquisition thread), `[camera]` log prefixes, "Recording requested: ..." text.
- **Byte-ceiling rolls are never pre-armed** (`should_prepare` is frame-based): with uncompressed 30 fps every 3 GB roll opens its writer on the append thread. Check `append_queue_depth` around roll boundaries against the < 5 criterion.
- GUI (step 10/14): per-slot `SessionPaths` via `with_camera(slot.tag)` from one `datetime.now()`; per-camera `fps_of`; per-camera diagnostics logger; `estimate_bytes_per_hour` into `assess_disk` (until then the GUI still uses the 1-camera 100 fps default rate; `SLEEVE_VIDEO_GUI_PLANNED_HOURS` already adds the long-run prompt); `broadcast()` reports `ok=True` even when a setter returns `False`, so callers must inspect `.value`; delete the unused `metadata_csv_path()` helper in `frame_metadata.py`.

## Controller behaviour worth knowing (from the step 8-9 audit)

- A legacy controller (no `serial=`) pins the serial it finds only while it runs; a clean `stop()` releases the pin, so swapping a different camera in and pressing Preview works as before. A controller created with `serial=` keeps it.
- `start_recording()` can be accepted and still fail asynchronously (segment 0 cannot be opened on the acquisition thread). The sidecars are then closed and deleted and `last_start_error` is set; the GUI/group should poll it after a start. `stop()` also cancels a prepared-but-never-begun recording.
- Stop followed by an immediate Start is refused while the old recording is closing, including the window where a start was accepted but not yet acted on.
- Fault layout (pre-existing, validated in the phase 2/3 reports): the first frame after a reinit is appended to the OLD segment and the segment roll happens just after it, so a camera `FrameID` reset or forward jump sits inside one segment. `session_verify` therefore attributes one discontinuity to each `timeline_break` event instead of relying on the `segment` column.
- The acquisition loop drops its local camera/image handles at the top of every iteration and before any recovery, so `_reinitialize_camera` can actually let go of the old device (the spike dropped its references too). This part is inferred, not rig-verified: the unplug test in `multi_controller_smoke.py --fault-serial` is what proves it.
- `scripts/multi_controller_smoke.py` refuses to run (exit 2) when a configured camera is missing, treats the unplugged camera's grab errors as expected, and fails if an append queue peaks at 5 or more.

## Risks (inferred, not verified from code)

- ~~R1/R2~~ **Resolved on the rig (`scripts/reinit_spike.py`, 2026-10-01):** with the Firefly streaming, the Blackfly was unplugged, torn down on its own, re-found by serial with plain `GetCameras()` (one `CameraList` per camera) and restarted; the Firefly had 0 frame gaps and 0 errors throughout. The spike's PASS criterion was strengthened afterwards (grab errors, rate and longest-interval checks, FrameIDs readable) — re-run it once with the new criterion and also with the cameras swapped (unplug the Firefly).
- R3: PySpin `GetInstance`/`ReleaseInstance` ref-count semantics (the holder makes this moot).
- Primary-name hazard: without `SLEEVE_VIDEO_GUI_CAMERA_SERIALS`, the untagged names follow whoever is plugged in. The registry now warns when several cameras are seen unpinned; the stable setup is to set the variable (e.g. `23227865,26134271`). If the configured primary is missing, the session has no untagged video (nothing downstream-ingestible) and the GUI must say so.
- R4: preview cost on the i7-10510U (estimate 5-10 ms of each 33 ms tick) — measured by `render_ms`.
- R5: which nodes the Firefly exposes (Gamma, BlackLevel, DeviceLinkThroughputLimit) — handled as N/A.
- R6/R7: MJPEG size fraction and `main.py` compression lines will conflict with the codec fork — keep edits there to receiver renames only.
- Stream buffers: ~1.4 GB pinned for two cameras (5 s x fps ceiling); confirm `[camera] stream buffer count: applied=` on the rig.
