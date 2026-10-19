# Two-camera recording: setup checklist

For ADL sessions on the two-camera rig (Firefly FFY-U3-04S2M #23227865 + Blackfly S
BFS-U3-13Y3M #26134271). The positions and settings come from the experiments in Agent HQ
Task "Two-camera setup #1" (sleeve_adl `docs/3d-setup-experiments.md` and
`docs/two-camera-setup-results.md`); this checklist is Task "Two-camera setup #2".

## 1. Once, when the rig is (re)built

**Camera placement** (heights are above the subject's shoulder line, standing at the table):

| | Firefly | Blackfly |
|---|---|---|
| Where | behind the subject, to the right | in front of the subject, to the left |
| Height / looking down | ~0.4 m above the shoulders, tilted down ~34° | ~0.15 m above the shoulders, tilted down ~16° |
| Horizontal distance to the subject | ~1.5 m | ~1.6 m |

- The two views are ~115° apart at the subject's wrist. Raising the Firefly further makes no
  difference; if it ever has to move sideways, move it **toward the Blackfly's side**, never
  away (20 cm away tripled how often the torso hides the hand).
- Never touch the lens focus or zoom once the camera is calibrated (that needs a new camera
  calibration). Tighten every clamp and mount screw: a turn of 0.1° is already visible.

**Framing** (check in Preview with the subject at the table):
- Both views: the table work area, the box behind the panel and the subject's sleeve arm,
  including the shoulder, with some margin at the image edges.
- Nothing that looks like a person in the background (the styrofoam head was detected as
  one). Keep clutter off the table edges.

**Reference boards** (fixed, never moved after the setup calibration):
- G2 where the Firefly sees it (now ~3.3 m away), G1 where the Blackfly sees it (~1.3 m).
- Taped flat. In the setup page each camera must read all 24 corners of its board.
- Not in anyone's shadow (see the light section).

**Light:**
- One battery-powered infrared (IR) illuminator, in front of and above the subject, pointing
  down at the work area. Battery power means no mains flicker (measured < 0.5%).
- Charge it before each session; the Auto setup button corrects a dimmer light, up to the
  cameras' gain limits.
- Optional later: a second light so the work area is lit from the front on both sides (~60°
  each side of the subject). After adding or moving a light, run the setup calibration again.

## 2. Every session

1. **Detect, then Preview.** Both cameras start at the chosen settings: exposure 8 ms, auto
   exposure and auto gain off, gain 2.5 dB (Firefly) / 3.4 dB (Blackfly).
2. **Read the calibration bar.**
   - "3D pose: ready": the saved setup still applies (the app checks every 5 s that neither
     camera moved, against the reference boards). Skip the setup calibration.
   - "camera moved", "no setup" or anything else: run the **setup calibration** (Calibrate →
     setup, board A at its marked spot). Keep "Set camera brightness from board A first"
     ticked: Capture setup then sets both cameras' brightness from board A before capturing,
     and Save setup stores each camera's reference-board brightness.
3. **Auto setup cameras** (main window), with board A removed: each camera's gain is set so its
   reference board reads as bright as at the last setup. Expect "OK" for both cameras.
4. While steps 2 and 3 run, **stand clear** of the light's path to board A and to the
   reference boards: a shadow on a board skews the gains for the whole session.
5. Optional, recommended on a new day or after any change: record **30 s** of normal movement
   (reaches, hand on the table, arm hanging) and run the framing check (section 4).
6. **During the ADLs the experimenter stays out of both views** (in view only between
   trials). The pose model mixes up two people who are close together.

## 3. What the messages mean

| Message | What to do |
|---|---|
| Auto setup: "OK, gain X dB, reference G2 white N" | Nothing. |
| "gain X dB (its limit), board A white N" | Fine: the gain is at the camera's limit but the board is well exposed. |
| "... not found" | Make the board fully visible (not covered, whole in view), press again. |
| "... reduce the light" / "add light" | The board is clipped / too dark even at the gain limit: aim the light away or closer, then press again. |
| Auto setup gain more than 2 dB away from the setup's gain | Something may shadow the reference board, or the light changed: check and press again; if the light really changed, rerun the setup. |
| "camera moved since setup (...)" | That camera turned or shifted: run the setup calibration. |
| "lens may have moved" | The camera calibration no longer fits: calibrate that camera again, then the setup. |

## 4. Framing check (sleeve_adl)

On cortex, after uploading the test recording to Box:

```bash
python3 scripts/two_camera_framing_check.py --raw <recording folder> --extract
```

It extracts the keypoints on a GPU (about a minute for 30 s), then checks the sleeve arm and
hand in both cameras, the image edges, exposure around the hand, the reference boards and the
calibration status, and writes `framing_check.md` with PASS or FAIL per item and overall.
