# Camera sync + 3D calibration — implementation plan

Branch: `feat/sync-and-3d-calibration`. Drafted 2026-10-05 from the code on `main` (7d1849d) and the lab's existing calibration material (see "Lab references").
Target: **two cameras captured at the same instant, stable at 60 fps** (headroom for 30 fps experiments), and **per-camera calibration managed in the GUI** so 3D pose can be computed downstream. Recording must never be blocked by calibration state; a one-camera setup must behave exactly as today.

Two parts, done in this order: **A. Sync** (needed first; it also lets the per-session calibration use a handheld board), **B. Calibration in the GUI**.

## Decisions

| # | Decision | Status |
|---|---|---|
| 1 | Calibration never blocks recording. Missing/invalid calibration → amber "3D pose not available: <reason>" line + a `calibration_status` entry in the session files. | user |
| 2 | Calibration lives in the GUI behind a **Calibrate…** button that opens a **separate, non-modal Calibration window** (step-by-step), not a mode of the main window. Reasons under B. | proposed |
| 3 | Intrinsics stored **per camera serial**, with the camera settings that change them (the "sensor fingerprint"); the camera setup (extrinsics) stored per setup and copied into each session folder. | user |
| 4 | Sync has two separate jobs. **Alignment** (put every camera frame and the sleeve on one timeline): the **sync-service heartbeat LED**, seen by both cameras. **Simultaneity** (both cameras expose at the same instant): **hardware trigger**, implemented generically (per-camera role `free` / `primary` / `secondary`) so both wirings below use the same code. | alignment: user (2026-10-05); trigger wiring open (Q1) |
| 9 | **Part A paused after step 1** (2026-10-05). Simultaneity will come from an **external pulse (H2)**, done later. Until then recordings are free-running (gap up to ½ frame, cycling every 1–1.6 min) and the heartbeat LED aligns them. Part B goes ahead now: it does not depend on sync (intrinsics are per camera; the setup uses a fixed board, so the two views need not be simultaneous). | user (2026-10-05) |
| 10 | **Two A4 boards** (A4 is the largest print), both 7×5 ChArUco, 36 mm squares, 27 mm markers (24 corners, vs 16 on the lab board): **A — handheld** (`DICT_5X5_50`) for intrinsics and the verification step; **B — fixed** (`DICT_4X4_50`, bigger marker cells for distance) stays in the scene. Different dictionaries so A can be in view next to B without ids clashing. Printed at 100 %, squares measured and the measured size entered. Mounted flat and rigid (foam board / acrylic), matte. | proposed (2026-10-05) |
| 8 | ~~One fixed board seen by both cameras~~ → **revised 2026-10-05 (decision 11)**. | superseded |
| 11 | **Setup from a resting shared board, movement from per-camera reference boards** (one board in both views was hard to place). (a) *Setup:* handheld **board A rests** flat at a marked spot both cameras see (resting = static, so the unsynced cameras do not matter); its frame is the session's world frame. (b) *Verify:* board A on a box ≥ 15 cm higher. (c) *References:* **B1** (`DICT_4X4_50`, ids 0–16) and **B2** (same dictionary, ids 17–33), each fixed where ONE camera sees it; at setup each camera's pose to its reference is saved, and a live check compares it (averaged frames) to spot a moved camera. A camera seeing both B boards is fine: different ids. No reference visible = the setup cannot be confirmed in later sessions (redo it after Detect). **Big-marker references G1/G2** (added 2026-10-05: a ChArUco B board was only readable by the Firefly right where the subject sits): plain ArUco grids, 3×2 markers of 70 mm (4×4, ids 34–39 / 40–45), 258×176 mm on A4, read from ~3× farther (simulated 5–6 m vs 1.4–1.8 m), sub-pixel corners, 0 false "moved" in 30 noisy checks at 2 m; either B or G can serve as a camera's reference. **Board A-big** (added 2026-10-06: on the rig the Firefly saw board A's 5×5 markers at ~13 px and read only 3–8 of 17): 6×4 ChArUco, 45 mm squares, 4×4 markers of 36 mm (cells 6.0 mm vs 3.9 mm), `DICT_4X4_100` ids 50–61 — ≥ 4 bits from every B/G marker in any rotation, 0 misreads in 432 trials; 15 corners. Chosen per session in the setup page. Board A no longer has to lie flat: it rests anywhere both cameras see it (e.g. on the panel); verification = moved ≥ 15 cm toward the cameras. | user (2026-10-05/06) |
| 5 | Stability target: both cameras **60 fps, 60 min**, both codecs, with sync on. 30 fps is the production rate. | user |
| 6 | Lab conventions kept: ChArUco board (default 5×5, 40 mm squares, 30 mm markers, `DICT_5X5_50`), acceptance thresholds from `penncubed_analyze/calibrate_stereo.py` (intrinsics RMS < 0.5 px, stereo RMS < 1.0 px, triangulation < 0.5 mm), plus an export in that script's `.npz` layout for the SLEAP pipeline. | proposed |
| 7 | Downstream (`smart_sleeve_data_processing`) is not modified. New per-session files use names that the video regex (`^recording_\d{8}_\d{6}(?:_cam…)?(?:-\d{4})?\.avi$`) cannot match. | carried over |

### Open questions (need answers before the steps they block)

- **Q1 (blocks A3/A7):** wiring — primary/secondary cable (H1) or external pulse generator (H2)? Is the NI DAQ present in the kitchenette setup, and does it have a counter output? Would a small microcontroller (e.g. Pi Pico) be acceptable?
- ~~Q2~~ answered: fixed board. Still open: where it goes (must be in both views, out of the hands' way, LED beside it).
- **Q3 (blocks A1):** which GPIO cables do we have for the Firefly and the Blackfly S?
- **Q4 (blocks A2) — data in (see skew evidence):** is the LED-only result (frames aligned, exposures up to half a period apart, keypoints interpolated) good enough for the 3D hand analysis, or are simultaneous exposures required? Step 1 measures the numbers to decide.

## Facts from the code (relevant to this plan)

- One acquisition thread per camera, free-running at `AcquisitionFrameRate` with `AcquisitionFrameRateEnable=True`, `TriggerMode` never set (only read back in diagnostics). Threads are already concurrent — **concurrency does not align exposures; each sensor starts its frames on its own clock.**
- Per-frame metadata: `camera_frame_id` and `timestamp_us` come from chunk data (each camera's own clock, not related to the other camera's or to the host's). `system_time` is host wall time at *retrieval* (after USB transfer + driver queueing, so it carries ms-level jitter).
- **`timestamp_us` is actually nanoseconds** (test file: +10,375,569 per frame at ~96 fps). Do not rename the column (downstream reads it by name); document it and use ns in new code.
- `AcquisitionWatchdog`: grab timeout `max(0.2, 5/fps)` s, stall `max(5, 10/fps)` s → reinit. In triggered mode "no trigger arriving" looks exactly like a stall today and would cause a reinit loop.
- `_configure_camera_nodes()` runs at start **and every reinit** — the right place to apply trigger settings (same as the exposure lock).
- `CameraGroup` already starts recording in a fixed order (primary last) with a two-phase start.
- NI DAQ support exists for one digital-output line (`ni_control.NIDaqDO`, `PulseManager`, the "Sync Pulse" button), Windows only, opt-in via `SLEEVE_VIDEO_GUI_NI`. No counter/pulse-train output yet.
- Evidence from the multi-camera plan: both cameras free-run at 60 fps with 0 gaps (probe), MJPEG at 60 fps passed a 5 min smoke run; **no 60 min run at 60 fps yet**, and none with triggering.

## Rig evidence — `sync_probe.py nodes`, 2026-10-05

| | Firefly FFY-U3-04S2M #23227865 (fw 2101.0.19.0) | Blackfly S BFS-U3-13Y3M #26134271 (fw 1808.0.120.0) |
|---|---|---|
| GPIO | Line0–3, each Input **or** Output; `LineFormat` not reported (non-isolated) | Line0 opto-isolated **input**; Line1 opto-isolated **output**; Line2 non-isolated Input/Output (`V3_3Enable` available); Line3 non-isolated **input** only |
| Output sources | `ExposureActive`, `FrameTriggerWait`, line passthrough, `SerialPort0` | same + `UserOutput0–3`, counters, logic blocks |
| FrameStart `TriggerSource` | Software, Line0–3 | Software, Line0, Line2, Line3, UserOutput0–3, counters, logic blocks |
| `TriggerActivation` | LevelLow/High, Falling/RisingEdge | same + AnyEdge |
| `TriggerOverlap` | **not available** (trigger only after the previous readout) | Off / **ReadOut** |
| `TriggerDelay` | **not available** | 14–65 520 µs |
| Timestamp latch | yes (`TimestampIncrement` 480) | yes (`TimestampIncrement` 1000) |

Consequences:
- **Both wirings are possible.** The Blackfly is the better *secondary*: it has `TriggerOverlap=ReadOut` (higher triggered rates) and `TriggerDelay` (to line up exposure centres when exposures differ). Proposed H1: **Firefly primary** (`LineN` Output, `LineSource=ExposureActive`) → **Blackfly secondary** on **Line3** (non-isolated input) or Line0 (opto input), `RisingEdge`, overlap `ReadOut`. Common ground between the two GPIO connectors is required; output drive / pull-up of the Firefly's non-isolated lines to be confirmed from the FLIR Firefly GPIO wiring note before connecting.
- For H2 (external generator) both cameras can take an input: Blackfly Line0 (opto, tolerant of 5 V signals) or Line3; Firefly any of Line0–3 (check its input voltage range).
- Firefly as a secondary has no overlap: exposure + readout must fit the period (fine at 60 fps with exposure ≤ ~14 ms).
- `TimestampIncrement` differs (480 vs 1000); the `skew` run measures each camera's real tick length directly.

## Rig evidence — `sync_probe.py skew`, 2026-10-05

Both cameras free-running together, 10 min at 30 fps then 10 min at 60 fps. Raw data: Box `SmartSleeve/RawData/temp/sync_probe_20261005_131323/`. Numbers below are re-analysed with the corrected probe (achieved rate from camera timestamps, start burst excluded from jitter).

| | 30 fps | 60 fps |
|---|---|---|
| Frame gaps / incomplete / errors (both cameras) | 0 / 0 / 0 | 0 / 0 / 0 |
| Real frame rate, Firefly / Blackfly | 29.9887 / 29.9992 | 59.9774 / 59.9940 |
| Gap between the cameras, median / p99 / max | 7.96 / 16.50 / 16.66 ms | 4.15 / 8.25 / 8.33 ms |
| Gap drift | −20.9 ms/min: every possible gap every **1.6 min** | −16.6 ms/min: every gap every **1.0 min** |
| Camera clock vs laptop (latch fit residual) | +4.7 / +5.1 ppm (59 / 116 µs) | +5.1 / +5.7 ppm (41 / 62 µs) |
| `system_time` arrival jitter p50 / p99, Firefly / Blackfly | 1.9 / 3.8 ms, 1.7 / 2.2 ms | 1.8 / 3.7 ms, 1.8 / 2.3 ms |

Findings:
- **Acquisition is stable at 60 fps** for both cameras together (10 min, no gaps). The 60 min recorder runs with video writing are still to do.
- **Timestamps are nanoseconds** (1.000005 ns per tick from the latch fit, 1.0000 from frame steps), confirming `timestamp_us` holds ns.
- **The gap between the cameras is uniformly spread over the whole possible range** (median ≈ quarter period, max = half period). It is **not** crystal drift: the two camera clocks agree to < 1 ppm. It is the **Firefly's frame-rate setting**: it cannot hit 30.000 / 60.000 and runs ~0.035 % slow (29.988 / 59.976 applied), so the Blackfly slides through every phase every 1–1.6 min. Every recording therefore contains long stretches at the worst case.
- **`system_time` is a poor clock for pairing** (ms-level jitter plus a 170–300 ms burst at start while queued frames arrive). Camera timestamps mapped through the latch fit (or the LED fit) are 40–120 µs.
- **Q4 answer from the data:** without a trigger, half of all frame pairs are ≥ 8 ms apart at 30 fps (≥ 4 ms at 60 fps). Keypoints must be interpolated to common times, and fast hand motion between frames is the limit.

New option this suggests (software, no wiring) — **S3, software phase lock:** because the camera clocks agree to < 1 ppm, the drift comes only from unequal rate settings. Periodically nudging the Blackfly's `AcquisitionFrameRate` (it has finer steps) using latch-mapped timestamps could hold the gap near 0. Expected accuracy is bounded by the Blackfly's rate step size and how cleanly it accepts live rate changes. Both are unmeasured: a rate change might drop or stretch a frame. It is still not simultaneous capture, and it adds a control loop to the recorder. Worth a short rig test only if wiring a trigger turns out to be impractical.

## Rig evidence — heartbeat LED, 30 fps recording, 2026-10-05

Recorder (MJPEG, 30 fps) `recording_20261005_135059`, 5.75 min, heartbeat `heartbeat_20261005T175017Z` (2–4 s intervals, USB-6501 `Dev1/port0/line0`). Box `SmartSleeve/RawData/temp/60 fps/` (the folder names are swapped: this is the 30 fps run). Analysed per camera: LED ROI brightness → rising edges → matched to heartbeat pulses (laptop perf_counter) → fit `pulse time = a + b · camera_timestamp_ns`.

| | Firefly #23227865 | Blackfly #26134271 |
|---|---|---|
| Flashes detected / matched / pulses in window | 118 / 118 / 118 | 118 / 118 / 118 |
| Fit residual range | ±16.2 ms (= ± half a frame) | ±16.6 ms (= ± half a frame) |
| Clock vs laptop (LED fit) | +8.0 ppm | +5.9 ppm |
| Clock vs laptop (latch, probe run) | +4.7 ppm | +5.1 ppm |
| Arrival (`monotonic_s`) after exposure, median | 13.5 ms | 11.6 ms |

- **Every flash seen by both cameras, no false or missed detections.** Residuals are exactly the frame quantisation, with no outliers.
- **The LED mapping agrees with the latch mapping** within the fit's uncertainty (about ±9 ppm for 6 min of 2–4 s pulses; longer recordings tighten it).
- **The inter-camera gap measured through the LED** (median 8.0, p99 16.5, max 16.7 ms) matches the probe → two independent methods agree.
- The existing single-camera `video_fiducial_diagnose.py` REJECTs the same data: it fits on frame index at an assumed rate and full-frame brightness. Step 2 must fit on camera timestamps with an LED ROI, as done here.
- **60 fps run** (`recording_20261005_135859`, 12 min, uncompressed through the real recorder, 6 + 19 segments, every segment decoded = manifest; heartbeat `T175836Z`): Firefly / Blackfly **236 / 236 flashes matched of 236**, 0 unmatched; residuals ±8.4 / ±9.0 ms (≈ ± half a 60 fps frame); clock vs laptop +4.5 / +6.1 ppm (latch probe: +5.1 / +5.7); inter-camera gap median 4.15, p99 8.25, max 8.34 ms (probe: 4.15 / 8.25 / 8.34). Recording: 43 318 / 43 330 frames, 0 FrameID gaps, 0 reinits.
- Precision now: frame-level onsets (±½ frame each, averaged over the pulses). Sub-frame onsets from partially lit frames (9 and 21 such frames here) are the step 2 improvement.

---

# Part A — Camera sync

## The problem, in numbers

Free-running cameras expose at unrelated moments. The offset between a frame from camera 1 and the nearest frame from camera 2 is anywhere from 0 to half a frame period, and it **drifts** over a session because the two camera crystals differ slightly (tens of ppm). Triangulating a moving hand from two views taken at different moments gives a wrong 3D point.

| Rate | Worst-case offset | Average | Hand at 1 m/s moves |
|---|---|---|---|
| 30 fps | 16.7 ms | 8.3 ms | up to ~17 mm between views |
| 60 fps | 8.3 ms | 4.2 ms | up to ~8 mm |
| Hardware trigger | ~µs | ~µs | negligible |

Hand-washing motions are fast enough that this is comparable to the size of fingers, so it matters for 3D hands.

## Options

| | Option | How | Alignment | Cost / risk |
|---|---|---|---|---|
| **S0** | Leave as is | Pair frames afterwards by `system_time` | ≤ half a period + ms-level USB jitter, drifting | Free. Keeps today's validated fault isolation. Not good enough for fast 3D hands |
| **S1** | Software clock mapping | Periodically latch each camera's clock (`TimestampLatch` → `TimestampLatchValue`) bracketed by `perf_counter()`; fit offset + drift per camera; map every frame's chunk timestamp to host time | Knows *when* each frame was taken to ~0.1–1 ms, but exposures are still up to half a period apart; analysis interpolates keypoints between frames | No wiring. Small code. **Needed in every option** to pair frames and to measure sync |
| **S2** | Software trigger | One host thread fires `TriggerSoftware` on both cameras each period | ms-level jitter from USB command latency + Windows timer resolution (1–15.6 ms), unbounded under load | No wiring, but worst at exactly what we need (60 fps on a laptop). **Not recommended** |
| **H1** | Primary/secondary over a cable | One camera free-runs and outputs `ExposureActive` on a GPIO line; the other is set to `TriggerSource=Line…`, `TriggerActivation=RisingEdge` | ~µs | One GPIO cable (+ maybe a pull-up resistor). Couples the cameras: if the primary is unplugged or reinits, the secondary stops too. Frame rate set on the primary only |
| **H2** | External pulse generator | NI DAQ counter output (or a microcontroller) sends a pulse train to both cameras, both triggered | ~µs | Needs a counter output or a ~$5–25 microcontroller + cables. Symmetric: one camera unplugged does not stop the other. The same pulse can go to the sleeve/other devices → **one clock for video and EMG**. Generator stopping = both stop (must be detected) |
| — | PTP (IEEE 1588) | — | — | GigE cameras only; ours are USB3. Not applicable |

| **L** | **Heartbeat LED** (`~/repos/Sleeve/sync-service`) | The NI DAQ line that drives the sleeve heartbeat (jittered 8–12 s intervals, 100 ms pulses, uniquely lockable) also drives an LED (through a transistor driver) seen by both cameras. Per camera, fit LED onsets against the heartbeat log → camera hardware timestamp ↔ laptop monotonic time; the sleeve is fitted to the same log | Every frame of both cameras and the sleeve on **one timeline**, sub-ms after an hour of pulses (fit over hundreds of edges). **Does not make exposures simultaneous**: still up to half a period apart → 2D keypoints must be interpolated to common times before triangulation | LED + driver only; already planned for the sleeve. `video_fiducial_diagnose.py` already does this for one camera. LED must stay visible (not under the hands); an occluded stretch just has fewer edges |

**The LED and the trigger are complementary, not alternatives.** The LED answers *when* each frame was taken (camera ↔ camera ↔ sleeve). A trigger makes the two cameras take their frames *at the same moment*. With the LED alone, 3D is computed by interpolating each camera's 2D keypoints to shared time points — fine for slow motion, increasingly wrong for fast rubbing (error grows with hand speed × gap; at 60 fps the gap is ≤ 8.3 ms). The heartbeat itself cannot be the trigger: it is one pulse every 8–12 s, not a frame clock.

**Recommendation:** use the LED (L) for alignment — it replaces S1 as the primary method (S1 latching is kept only as a cheap per-session cross-check, built if the LED fit proves noisy). Add the generic trigger roles for simultaneity. Start the rig tests with **H1** (only a cable, fastest to validate). Move to **H2** if the NI DAQ or a microcontroller is available in the experiment setup, because it keeps cameras independent under faults and gives a common clock with the sleeve. The code difference between H1 and H2 is only the primary's role (`primary` vs `secondary`) and who sets the rate.

### Trade-offs vs. today

- **Gains:** frames paired by construction; 3D error from timing removed; per-session calibration can use a handheld board; frame pairing becomes a pass/fail test.
- **Costs:** wiring that can be knocked loose; a new failure mode ("no trigger") that must be shown clearly; with H1 the validated fault isolation between cameras is lost (one camera's fault stops both); the secondary's frame rate is no longer set in the GUI; exposure must fit the trigger period on both cameras (60 fps → ≤ ~15 ms exposure, more light needed in the kitchenette).
- **Exposure centres:** triggering aligns exposure *starts*. If the two cameras use different exposure times, their centres differ by half the difference; use equal exposures or `TriggerDelay` on the shorter one.

## Design

- **LED alignment, offline, in sync-service** (where the heartbeat log, `decode.py` lock/fit and `video_fiducial_diagnose.py` already live — not duplicated here): extend it to N cameras and to fit LED onsets against each camera's **hardware timestamp** (`timestamp_us`, actually ns — more regular than `system_time`, which carries USB/driver jitter), with sub-frame onset estimates from partial-exposure brightness (global-shutter sensors). Output per camera: `camera_ns → laptop monotonic` per timeline segment (a reinit resets the camera clock → new segment), fit uncertainty, flashes seen/missed; and a cross-camera pairing report.
- **LED in the recorder (live, light):** with the board pose known, the LED's position in each image is known; the setup step checks the LED is in view of every camera, and during preview/recording a cheap ROI brightness check shows "LED seen <n> s ago" per camera (warning if a camera has not seen it for > 3 intervals). The recorder does no fitting.
- **NI DAQ line ownership:** the heartbeat process owns the DAQ; the recorder's NI output is already off by default (`SLEEVE_VIDEO_GUI_NI`). Keep it off for every run in this plan. **Out of scope here (future plan):** remove the "Sync Pulse" button and integrate the heartbeat into the GUI.
- **`backend/clock_sync.py`** (pure, optional — S1 cross-check): fit `host_s = a + b·camera_ns` from latch samples (keep the sample with the smallest host bracket per window); map frame timestamps; pair two cameras' frames by mapped time (nearest within half a period) and report paired fraction, skew p50/p99/max, unpaired runs. The pairing code is shared with the LED report.
- **Clock sampler** (optional, with the above) in `CameraController`: every 2 s (and at start/stop/reinit) latch + read the timestamp, bracketed by `perf_counter()`; write `<stem>_cam<serial>_clock.csv` (primary unsuffixed as usual). Takes `_camera_lock` briefly; never blocks the grab loop.
- **`TriggerConfig`** per controller: `role ∈ {free, primary, secondary}`, `input_line`, `output_line`, `activation`, `delay_us`. Applied in `_configure_camera_nodes()` (start + every reinit), read back and reported in diagnostics (`trigger_mode`, `trigger_source`, `line_source`), refused with a clear message if the camera lacks the line. `free` = today's behaviour, byte-identical.
  - secondary: `TriggerMode=Off` → `TriggerSelector=FrameStart`, `TriggerSource=<line>`, `TriggerActivation=RisingEdge`, `TriggerOverlap=ReadOut`, `TriggerMode=On`, `AcquisitionFrameRateEnable=False`.
  - primary: free-run as today + `LineSelector=<line>`, `LineMode=Output`, `LineSource=ExposureActive`.
  - Firefly vs Blackfly S line names/electrical details differ — **taken from the A1 probe, not assumed**.
- **Watchdog in triggered mode:** a secondary with no frames for the stall time checks whether the device answers (read a cheap node). Answers → state `waiting_for_trigger` (warning "no trigger signal on <camera>", **no reinit**); does not answer → today's reinit path. Recording keeps running; the gap is logged as a timeline event, not a fault.
- **Start order:** secondaries `BeginAcquisition` first (armed), primary / generator last, so the first frames pair. With H2 the generator is paused during the two-phase recording start → the first recorded frame is the same trigger on both cameras.
- **Frame rate:** the GUI fps box drives the primary (H1) or the generator (H2); secondaries show "set by trigger". Exposure clamp uses the trigger rate.
- **Config:** `SLEEVE_VIDEO_GUI_SYNC` (e.g. `h1:primary=23227865:Line2,secondary=26134271:Line3` / `h2:...` / `free`) + a Setup & options control. Default `free` until validated.
- **Session record:** events header gets `sync_mode`, roles, lines; `session_verify` adds the pairing report.

## What the 60 fps stability test needs

1. **Exposure ≤ ~15 ms** on both cameras (the clamp does this) and enough light for that exposure.
2. **Disk:** uncompressed at 60 fps ≈ 23.3 MB/s (Firefly) + 78.6 MB/s (Blackfly) ≈ **102 MB/s, ~367 GB/h**; the disk guard will warn below 8 h of space (~2.9 TB). MJPEG ≈ 2× the 30 fps size (~26 GB/h) at ~0.76 cores.
3. **Smoke script** (`scripts/multi_controller_smoke.py`) extended with `--sync free|h1|h2` and the pairing report; pass/fail added for pairing.
4. **The heartbeat LED in view of both cameras** for every run (with `heartbeat.py` running; for the test runs a shorter interval, e.g. `--min-interval 2 --max-interval 4`, gives more edges per hour). With a trigger, the onset must land on the same paired frame in both videos; without one, the report gives the true inter-camera gap. (Optional: scope on both `ExposureActive` lines.)
5. Runs (each per sync mode used):
   - 60 min @ 60 fps uncompressed, 60 min @ 60 fps MJPEG, 60 min @ 30 fps MJPEG (production).
   - Fault: unplug each camera 20–30 s once; with H1 also the primary (secondary must show `waiting_for_trigger`, not reinit-loop); pull the trigger cable 20 s.
   - Stop/start 5 cycles; power pause/resume once.

**Pass criteria (per camera, plus pairs):** the existing multi-camera checklist (0 gaps/incomplete/errors/append failures; rows ≥ 0.999 × duration × fps; `append_queue_depth` max < 5 uncompressed / ≤ 6 MJPEG; CPU avg ≤ 1.5 cores; preview p95 ≤ 250 ms) **plus** paired frames ≥ 99.9 % outside fault windows, mapped skew p99 ≤ 0.1 ms (H1/H2; report-only for free-run), LED onset on the same pair in 100 % of pulses, no reinit while the trigger cable is pulled.

---

# Part B — Calibration in the GUI

## UX: separate Calibration window

A **Calibrate…** button (next to Detect / Preview; disabled while recording) opens a non-modal **Calibration** window. A separate window rather than changing the main one because:
- calibration needs a large live view with overlays (detected corners, coverage map) and step text that the main window has no room for;
- the main window keeps its recording controls and state untouched, so nothing in the validated recording path changes;
- it can be closed/cancelled at any step with no side effects, and the main window shows the result as a status line.

The window uses the cameras the main window already has (starts Preview if needed). Two tasks, each a short step list with **Back / Next** and the instructions shown on screen:

**1. Calibrate a camera (intrinsics — once per camera, after a lens/resolution change)**
1. *Before you start* (checklist the user ticks): board printed at 100 % — measure one square with a ruler and enter it; lens focus, zoom and aperture locked (these lenses are manual, so the app cannot detect a change); camera at the resolution used for recording.
2. *Pick the camera.*
3. *Move the board* through the view. Live overlay of detected corners; a coverage map (3×3 image regions × near/far × tilted/flat) fills in; frames are captured automatically when the board is still ~0.5 s and in a new pose. Target 30–40 views; text says what is missing ("tilt more", "cover the left edge").
4. *Compute* (worker thread): RMS, per-view errors, option to drop outliers.
5. *Result:* PASS/FAIL against 0.5 px. Save (a failed calibration can be saved only as "loose", and that flag is stored in the file and shown in the status).

**2. Set up for this session (extrinsics — every time the cameras are moved; decision 11)**
1. *Before you start:* both cameras calibrated (checked from the store), board A resting flat at its marked spot, B1/B2 mounted; measured square sizes for A and for the B boards.
2. *Live check:* per camera "board A: N corners", "reference: B1/B2 N corners" (whichever that camera sees best); Next enabled when both cameras see board A (a missing reference is allowed with a warning).
3. *Capture:* average ~1 s of frames per camera (board A and the references are static).
4. *Compute:* per-camera pose to board A, reprojection RMS, camera-to-camera baseline in mm, triangulated board corners vs real board (mm); each camera's pose to its reference board and that fit's RMS.
5. *Verify:* raise board A **≥ 15 cm** toward the cameras (e.g. on a box), hold it still; the app triangulates it with the *saved* camera poses and compares it with the real board: size error < 1 %, shape RMS < 2 mm. Needed because step 4 is self-consistent: it re-uses the board the poses came from, so even a 20 % focal-length error still passes it (synthetic test). A sideways move does not reveal scale errors; a height change does (10 % focal error → ~2 % size error at 20 cm).
6. *Result:* PASS/FAIL against the thresholds → becomes the current setup.

Also: **Print board…** writes a PDF/PNG of the configured board at true size.

## Status in the main window (never blocks)

One line under the previews: **"3D pose: ready"** or amber **"3D pose not available — <reason> [Calibrate…]"**. Reasons, per camera:

| State | Meaning |
|---|---|
| `missing` | no intrinsics for this serial |
| `mismatch` | stored sensor fingerprint differs from the camera now (Width, Height, OffsetX/Y, Binning, Decimation, ReverseX/Y, PixelFormat) |
| `loose` | saved despite failing the threshold |
| `old` | older than N days (note only) |
| `suspect` | board visible and the live reprojection error with stored intrinsics > 1 px (lens probably touched) |
| `no setup` / `setup moved` | no setup captured since the cameras were detected, or (fixed board) the live board pose moved more than ~5 mm / 0.5° from the saved setup |

At record start the current status is written to the session, and if a valid setup exists its full contents are copied to `<stem>_calibration.json` so the recording always carries the exact calibration it was taken with.

## Storage

`SLEEVE_VIDEO_GUI_CALIBRATION_DIR` (default `%LOCALAPPDATA%\SleeveVideoGUI\calibration`):

```
intrinsics/<serial>/<YYYYMMDD_HHMMSS>.json  K, D, image_size, fingerprint, board, rms, n_views, coverage, loose, app commit
intrinsics/<serial>/current.json            pointer to the active file (history is never overwritten)
setups/<YYYYMMDD_HHMMSS>.json               serials, per-camera R/t to the board frame, RMS, baseline, triangulation mm, board, sync mode
setups/current.json
```

Plus **Export for lab pipeline** → `<rig>_stereo_<date>.npz` in the `calibrate_stereo.py` layout (`K_a, D_a, K_b, D_b, R, T, E, F`, metrics).

## Design

- **`backend/calibration.py`** (pure, OpenCV only — `cv2.aruco.CharucoDetector` is in the installed OpenCV 5.0, no contrib package needed): board build/render, detection, view selection + coverage, `calibrateCamera`, `solvePnP`, relative pose, triangulation check, live drift check.
- **`backend/calibration_store.py`** (pure): read/write/history, fingerprint compare, `assess(record, fingerprint, now, live=None) -> status`.
- **Controller:** `get_sensor_fingerprint()` (read-only node reads).
- **Detection cost:** runs in a worker at 2–5 Hz on full-resolution frames (accuracy needs full resolution), never on the GUI thread; overlays drawn from the latest result. Must not move the preview `render_ms` budget.
- **Tests:** synthetic images rendered from a known K/D/pose (`projectPoints` + warp) → calibration recovers K within tolerance, `solvePnP` recovers pose, drift check fires; store round-trip and every status; the Calibration window driven headless (Qt offscreen) against `tests/fake_spinnaker.py` cameras that return rendered board frames.

---

# Steps (commit-sized)

Pure-logic steps are unit-tested without cameras; rig steps are marked **[rig]**.

**Part A**
1. **[rig]** `scripts/sync_probe.py`: per camera, list GPIO lines and allowed `LineMode`/`LineSource`/`TriggerSource` values, `TimestampLatch` support, timestamp tick rate (confirm ns); run both cameras free at 30 and 60 fps for 10 min with clock latching and report the measured skew and drift. Plus one free-running two-camera recording with the heartbeat LED in view. → evidence for Q1/Q3/Q4. ✅ written: `scripts/sync_probe.py` (`nodes`, `skew`; analysis unit-tested in `tests/test_sync_probe.py`), run instructions in `docs/runbooks/2026-10-05-sync-probe-step1.md` — **rig run pending**.
2. **(sync-service repo)** Multi-camera LED alignment: fit on camera hardware timestamps, sub-frame onsets, per-segment fits, cross-camera pairing report. Pairing code shared with `session_verify`. (S1 latch sampler only if the LED fit is not enough.)
3. `TriggerConfig` in `CameraController` (applied at start + reinit, read back, `free` identical to today) + fake-camera tests.
4. Watchdog `waiting_for_trigger` path + tests.
5. `CameraGroup` start order (secondaries armed first) + tests.
6. GUI: sync setting, "set by trigger" fps display, trigger warnings; events header fields; per-camera "LED seen" indicator.
7. (H2 only) pulse-train generator: NI DAQ counter output or microcontroller; rate from the GUI fps box.
8. Smoke script `--sync`, pairing pass/fail; LED-test procedure in `docs/`.
9. **[rig]** Validation runs above → report in `docs/reports/`.

**Part B**
10. ✅ `backend/calibration.py` + synthetic tests (`tests/test_calibration.py`: board images rendered through a known K/D/pose). Thresholds: intrinsics < 0.5 px (lab), setup per-camera reprojection < 1.0 px (lab stereo), setup/verify triangulation < 2.0 mm (lab's 0.5 mm assumes a close rig; at 1–1.5 m one pixel is ~1–2 mm, tune in step 16), verify size error < 1 % with the board ≥ 15 cm off the setup plane. Findings: with a 16-corner board the principal point and distortion trade off (±5–10 px between runs, no bias), so intrinsics are judged on views not used for fitting; the setup triangulation cannot catch wrong intrinsics → step 5 *Verify* above.
11. ✅ `backend/calibration_store.py` + tests (`tests/test_calibration_store.py`); `CameraController.get_sensor_fingerprint()`. Records are **JSON** (readable, diffable; the lab `.npz` is the step 15 export), written atomically, history never overwritten. `assess_session()` gives the one-line status. **Setup reuse:** an older setup stays valid when the live fixed-board check confirms neither camera moved; until that check has answered for every camera, a setup older than the last Detect counts as stale. Verification is required for "ready".
12. ✅ Main-window 3D status line (`gui/calibration_status.py`, under the previews; green = ready, amber = the reasons) + **Calibrate…** button (enabled after Detect, disabled while recording; shows a placeholder until step 13). Refreshed on Detect and Preview start (camera settings are only readable while acquiring). Every recording writes `<basename>_calibration.json` (status at start + the setup and intrinsics records), ready or not; a failure to read/write calibrations is printed and never stops the recorder. Tests: `tests/test_gui_calibration_status.py` (real MainWindow, Qt offscreen, fake cameras). Note: GUI tests need PySide6; without it they are skipped (40 skipped in a Qt-less env, 0 with Qt).
13. ✅ Calibration window (`gui/calibration_window.py`): start page (per-camera calibration state, **Save printable board (PDF)** for each preset via `QPdfWriter`, true size on A4), **Calibrate a camera**: checklist (camera, board preset, measured square size, lens locked, resolution, focus) → capture (live view with detected corners, 3×3 coverage map, near/far/tilted counts, plain-language hints; views taken automatically when the board is held still ≥ 0.5 s in a new pose, `CaptureSession`; manual take/undo) → compute in a worker (PASS/FAIL vs 0.5 px, focal/centre/distortion, worst views, "drop the worst views and recompute") → save (or save marked as failed). Detection runs on its own thread at ~5 Hz on full-resolution frames. The window starts Preview if needed, closes on Detect and when a recording starts, and never touches the recording controls. Tests: `tests/test_gui_calibration_window.py` clicks through a full calibration against a fake camera showing a rendered board (PASS, record saved, status updated). The setup task (step 14) is still disabled.
14. ✅ Calibration window: setup task (decision 11; `gui/calibration_setup.py`, shared pieces in `gui/calibration_common.py`; `BoardConfig.first_marker_id`, B1/B2 presets, `check_reference`, `best_detection`; `backend/reference_monitor.py`; tests `tests/test_reference_monitor.py`, `tests/test_gui_calibration_setup.py` — full setup → verify → save in the real windows on a rendered two-camera scene, then the live check with a camera turned 2°) + live reference check in the main window: every ~5 s, per camera, ~5 frames of its reference board are averaged, posed with the stored intrinsics and compared with the saved reference pose (`compare_poses`, board-centre metric); "moved" needs 2 consecutive checks (hysteresis), an occluded reference gives no answer; the same fit's RMS feeds "suspect" when it exceeds max(1 px, 2 × the RMS saved at setup). Runs in a worker, also while recording.
15. Export to the lab `.npz` layout.
16. **[rig]** Calibrate both cameras, set up, triangulate a ruler/known object at 30 fps with sync on; report.

# Risks

- **Firefly GPIO**: line count/electrical type not verified for FFY-U3-04S2M; if it cannot output or accept a trigger on a usable line, H1 must use the Blackfly as primary, or H2 is needed. Step 1 settles it.
- **Pull-ups / voltage levels**: non-isolated outputs typically need a pull-up; wrong wiring can give no trigger or double triggers. Follow the FLIR wiring note for each model; verify with the LED test.
- **Missed triggers** at 60 fps if exposure + readout exceed the period on the secondary (`TriggerOverlap=ReadOut` mitigates). The pairing report catches it.
- **USB bandwidth** for 2 × 60 fps uncompressed through the Thunderbolt dock was fine in the probe; the 60 min run confirms it.
- **LED occlusion / saturation**: hands or steam over the LED, or a saturated LED at long exposure, remove edges; the fit tolerates gaps (the heartbeat locks in ~5 pulses) but long blind stretches widen the uncertainty. Mount it out of the washing area; the "LED seen" indicator catches it live.
- **Heartbeat DO latency** is software-timed and one-sided; sync-service's envelope fit already handles it, and it is common to cameras and sleeve, so it cancels between them.
- **Latch timing** across USB adds ~0.1–1 ms uncertainty to S1; fine for pairing, not as a sync method by itself.
- **Manual lenses**: a touched focus ring silently invalidates intrinsics; only the live `suspect` check (board visible) can catch it.
- **Laptop load**: detection worker + 2 × 60 fps preview; measured via `render_ms` and CPU in step 13/16.

# Lab references

- PennCubed KB `pesaranlab:Procedures/aruco-camera-calibration.md` (2026-06-02) — per-recording ArUco extrinsics, ChArUco intrinsics, Thalamus ArUco node.
- Skill `rig-stereo-calibration` → `penncubed_analyze/calibrate_stereo.py` (installed at `/opt/penncubed-analyze/current`) — board defaults, thresholds, `.npz` layout; stereo only, intrinsics+extrinsics together, no "fixed intrinsics" mode.
- Meetings 2026-06-24 (OCD sleeve symptom provocation): two cameras + ArUco + calibration for the next experiment, kitchenette included.
