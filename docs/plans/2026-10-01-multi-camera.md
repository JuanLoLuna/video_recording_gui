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
- **`SharedSystemHolder`** (`backend/spinnaker_system.py`): ref-counted owner of `PySpin.System`; controllers never call `GetInstance`/`ReleaseInstance` themselves. Fault recovery: sole owner -> today's full rebuild (validated path); shared -> rebuild only this camera's `Camera`/`CameraList`, re-find by serial, never touch the System.
- **Naming in one place** (`SessionPaths.camera_tag`): tag `cam<serial>` goes *inside the stem*, before `-NNNN` / `_metadata`; the primary stays unsuffixed; the single WAV is untagged.
  `recording_20261001_101500-0000.avi` (primary) / `recording_20261001_101500_cam26134271-0000.avi` / `..._cam26134271_metadata.csv`.
  Known downstream gap (deliberately deferred): the video regexes in `smart_sleeve_data_processing` (`rules.yaml` video_raw/video_metadata, `router.py` RECORDING_RE) do not match tagged names; fix at the first two-camera ingest. The proposed pattern is `^recording_\d{8}_\d{6}(?:_cam[A-Za-z0-9]+)?(?:-\d{4})?\.avi$`.
- **`CameraGroup`** (`backend/camera_group.py`, duck-typed, no PySpin): start/stop fan-out with rollback; stop order = `stop_recording` on all first, then `stop()` each, System released last and never after a deferred teardown.
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
8. `CameraController`: holder, serial binding, reinit by serial, accessors (`serial`, `model`, `get_stream_rate()`), tagged thread names, preview `max_size` — **rig: single-camera regression**
9. `scripts/multi_controller_smoke.py` — two real controllers headless — **rig: 2 x 10 min, then unplug test**
10. GUI with the group at N=1 — **rig: single-camera GUI regression**
11. Detection/binding for N cameras + preview tiles — **rig**
12. Per-camera diagnostics, warnings, health captions, generic backlog warning (`append_queue_depth >= 30`) — **rig**
13. Tuning camera selector + frame-rate intersection — **rig**
14. Two-camera recording: per-camera `SessionPaths` from one shared `datetime.now()`, disk estimate wired in, missing-camera policy — **rig: full checklist**
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

## Risks (inferred, not verified from code)

- **R1 (highest)**: a camera-only reinit while the other camera streams can re-discover a replugged USB3 camera via `GetCameras()`. Fallbacks: `system.UpdateCameras()`, then a coordinated restart of both cameras (costs the healthy one a gap). Worth a small rig spike before step 8.
- R2: two simultaneous per-controller `CameraList`s / `GetCameras()` during another camera's acquisition are safe (the probe only ever had one list in use).
- R3: PySpin `GetInstance`/`ReleaseInstance` ref-count semantics (the holder makes this moot).
- R4: preview cost on the i7-10510U (estimate 5-10 ms of each 33 ms tick) — measured by `render_ms`.
- R5: which nodes the Firefly exposes (Gamma, BlackLevel, DeviceLinkThroughputLimit) — handled as N/A.
- R6/R7: MJPEG size fraction and `main.py` compression lines will conflict with the codec fork — keep edits there to receiver renames only.
- Stream buffers: ~1.4 GB pinned for two cameras (5 s x fps ceiling); confirm `[camera] stream buffer count: applied=` on the rig.
