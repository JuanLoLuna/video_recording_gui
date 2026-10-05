# Step 1 — Sync probe: how to run it on the rig

Plan: [`docs/plans/2026-10-05-sync-and-3d-calibration.md`](../plans/2026-10-05-sync-and-3d-calibration.md), step 1.
Script: [`scripts/sync_probe.py`](../../scripts/sync_probe.py).

**What this answers**
- Which GPIO lines and trigger options each camera actually has, which decides how a trigger cable could be wired (plan Q1, Q3).
- How far apart the two cameras' frames are today at 30 and 60 fps, and how fast that gap drifts (plan Q4).
- Whether the heartbeat LED is seen cleanly by both cameras in a real recording (input for plan step 2).

**Time:** about 1 hour in total. Part 1 takes 1 min, Part 2 about 22 min unattended, Part 3 about 25 min.

**No screenshots needed.** Every run writes `report.txt` (exactly what was printed) plus JSON/CSV files into its own `probe_output\sync_probe_<date>_<time>\` folder. Send those folders back.

**Nothing in this step changes the recorder.** The probe opens the cameras directly, the same way `scripts/multi_camera_probe.py` does, and writes only CSV/JSON files.

---

## Before you start

- [ ] Both cameras plugged into the ports you use for experiments (SuperSpeed). Use the same cameras, cables and dock as in the experiment.
- [ ] Laptop on mains power, with sleep disabled for the next hour. The probe does not keep the laptop awake the way the recorder does.
- [ ] The recorder GUI and SpinView are **closed**. Either one holds the cameras open, and the probe then cannot open them.
- [ ] For Part 3 only:
  - [ ] the heartbeat LED is wired through its transistor driver (see `Sleeve/sync-service/README.md`, "Video timebase diagnostic");
  - [ ] the `sleeve-sync` conda environment exists;
  - [ ] the NI DAQ is connected.

### Get the code

In a terminal (PowerShell) in the recorder repo:

```powershell
git fetch
git switch feat/sync-and-3d-calibration
git pull
```

Activate the same Python environment you use to run the recorder (the one with PySpin). All `python` commands in Parts 1 and 2 run from the repo root in that environment.

---

## Part 1 — What the cameras support (about 1 min)

```powershell
python scripts\sync_probe.py nodes
```

**What it does:** opens each camera, reads its GPIO lines, trigger options, timestamp nodes and sensor settings, and closes it again. Nothing is changed except the line and trigger *selectors*, which are put back afterwards.

**You should see** one block per camera, for example:

```
=== Blackfly S BFS-U3-13Y3M #26134271 (SuperSpeed) ===
  timestamp latch: yes
  GPIO lines:
    Line0: format=OptoCoupled mode=Input (allowed ['Input'])
    Line1: ...
  FrameStart trigger:
    TriggerSource: Software allowed ['Software', 'Line0', 'Line2', 'Line3']
```

**Check:**
- [ ] Both cameras are listed, and both show `SuperSpeed`.
- [ ] Note whether each camera says `timestamp latch: yes`. A `NO` is not a failure, but tell me.

Saved to `probe_output\sync_probe_<date>_<time>\`: `report.txt` (what you saw) and `nodes.json` (full details).

---

## Part 2 — How far apart the frames are today (about 22 min, unattended)

```powershell
python scripts\sync_probe.py skew
```

**What it does:**
- Runs both cameras together, free-running exactly as the recorder runs them: 10 min at 30 fps, then 10 min at 60 fps.
- Exposure is set to manual and capped to fit the frame period, as the recorder does at these rates.
- Keeps every frame's camera timestamp and arrival time, and once per second reads each camera's clock against the laptop's.
- Does not save video.

**During the run:**
- Don't touch the cameras or cables.
- Normal room lighting is fine.
- A progress line appears every 30 s.

**At the end you get a report per frame rate**, for example:

```
--- 60 fps ---
  Firefly ... #23227865: 36000 frames, achieved 60.000 fps, gaps 0, incomplete 0, errors 0  -> OK
    tick = 1.000012 ns (latch), ...; clock drift vs laptop +12.0 ppm; latch fit residual 180 us
    host arrival jitter (system_time as a clock): p50 0.9 ms, p99 3.1 ms, max 7.4 ms
  ...
  Gap between cameras (#26134271 frame minus nearest #23227865 frame):
    start +3.10 ms, end -5.80 ms, drifting -0.890 ms/min (passes through every phase every 19 min)
    |gap| p50 4.10 ms, p99 8.20 ms, max 8.33 ms (worst possible: 8.33 ms)
```

**Check:**
- [ ] Every camera line ends in `OK`, meaning 0 gaps, 0 incomplete and 0 errors. If not, note which camera and fps.
- [ ] `tick` is about 1.0 ns. This confirms the metadata column `timestamp_us` is really nanoseconds.
- [ ] The **Gap between cameras** lines for 30 and 60 fps are what plan Q4 is decided on. They are in `report.txt`; no need to copy them by hand.

Saved to `probe_output\sync_probe_<date>_<time>\`: `report.txt`, `summary.json`, and the raw `*_frames.csv`, `*_latch.csv` and `*_phase.csv`.

For a quick 1-minute trial first:

```powershell
python scripts\sync_probe.py skew --fps 60 --seconds 60
```

---

## Part 3 — Heartbeat LED seen by both cameras (about 25 min)

This uses the **real recorder**, so the data looks exactly like experiment data.

### 3.1 Place the LED
- [ ] The LED is in view of **both** cameras: check it in the recorder's Preview.
- [ ] It is away from where hands will be, and not behind the board.
- [ ] It looks small and bright, without a big bloom. If it blooms, tilt it away a little or add a diffuser or tape.

### 3.2 Start the heartbeat
In a **second** terminal:

```powershell
conda activate sleeve-sync
cd <path to>\Sleeve\sync-service
python heartbeat.py --line Dev1/port0/line0 --out D:\sync-logs --min-interval 2 --max-interval 4
```

- Use the DAQ line the LED is actually wired to.
- `--min-interval 2 --max-interval 4` flashes more often than the production 8–12 s, which gives more flashes per minute for this test.
- [ ] The terminal prints one line per pulse, and the LED blinks every 2–4 s.

### 3.3 Record at 30 fps (10 min)
1. Start the recorder GUI as usual. Do **not** set `SLEEVE_VIDEO_GUI_NI=1`: the heartbeat owns the DAQ.
2. **Detect camera** → both cameras appear.
3. Set **30 fps**. Leave **MJPEG** ticked (the default at 30 fps).
4. **Start Recording**, wait **10 min**, then **Stop Recording**.

### 3.4 Record at 60 fps (5 min)
1. Set **60 fps**, and **tick MJPEG** by hand (at 60 fps the default is uncompressed).
2. **Start Recording**, wait **5 min**, then **Stop Recording**.

The lengths are chosen so each camera's video stays in one file; the analysis script reads one video file at a time.

### 3.5 Stop the heartbeat
Press **Ctrl-C** in the heartbeat terminal. It writes a `stop` record and sets the line back to idle.

### 3.6 Optional: quick check on site
In the `sleeve-sync` environment, for each camera of one recording:

```powershell
python video_fiducial_diagnose.py --video <rec folder>\recording_<stamp>-0000.avi --metadata <rec folder>\recording_<stamp>_metadata.csv --log "D:\sync-logs\heartbeat_*.jsonl"
python video_fiducial_diagnose.py --video <rec folder>\recording_<stamp>_cam<serial>-0000.avi --metadata <rec folder>\recording_<stamp>_cam<serial>_metadata.csv --log "D:\sync-logs\heartbeat_*.jsonl"
```

- Always pass `--metadata`. The script's automatic guess does not match the second camera's file names.
- If it finds too few flashes, add `--roi x,y,w,h` around the LED.
- [ ] Both cameras lock onto the heartbeat. This is a sanity check only; the two-camera analysis is step 2.

---

## What to bring back

| What | Where |
|---|---|
| Part 1 + Part 2 output: the whole folders, including `report.txt` | `probe_output\sync_probe_*\` in the recorder repo |
| Both recordings: all `.avi`, `_metadata.csv`, `_events.jsonl` and `_segments.csv` files of both cameras | the recorder's output folder |
| Heartbeat logs | `D:\sync-logs\heartbeat_*.jsonl` |
| Notes | anything odd: a camera not OK, latch `NO`, the LED hard to see, etc. |

Copy them to the server, or tell me where they are. For the optional check in 3.6, save its output too: add `| Tee-Object -FilePath fiducial_<serial>.txt` to each command.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Serials not found`, `run failed`, or an error from `Init()` | The recorder GUI or SpinView is still open; close it. Or a camera is unplugged |
| `No module named PySpin` | Wrong Python environment: use the recorder's |
| `No module named multi_camera_probe` | Run it as `python scripts\sync_probe.py` from the repo root, not from inside `scripts\` with a different path |
| A camera shows `High-Speed` / not SuperSpeed | Different port or cable; use the ones from the experiment and check the dock |
| Gaps or errors > 0 | Note the camera and fps, and run `python scripts\multi_camera_probe.py --fps 30 60` to compare with earlier results |
| Heartbeat says the line is busy | Another program owns the DAQ line (an old heartbeat, or the GUI with `SLEEVE_VIDEO_GUI_NI=1`) |
