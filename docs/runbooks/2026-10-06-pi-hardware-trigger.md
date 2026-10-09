# Hardware trigger from a Raspberry Pi: setup, wiring, rig test

Branch `feat/hw-trigger`. Plan: `docs/plans/2026-10-05-sync-and-3d-calibration.md`, Part A, option H2.

**What this gets you:** both cameras expose on the same electrical edge (µs apart), at an exact rate from the Pi's hardware PWM clock.

**Already validated (2026-10-06):**
- Pi 4, Raspberry Pi OS trixie, hardware PWM on GPIO18.
- Scope: 60.001 Hz, 0.99998 ms pulses.
- Logic capture: no missing pulses, sub-µs period jitter.

**Every step below saves its output to a file. No screenshots needed.**

---

## 1. The Pi (done once)

Already done on `juan-rpi-01`:
- user account;
- USB serial gadget;
- PWM overlay;
- `pi-trigger` service via `pi_trigger/install.sh` (output in `pi_trigger/install-report.txt`).

The Pi's **USB-C** port goes to the **recording laptop**. It powers the Pi and carries the control link.

The laptop sees **two USB serial ports** from the Pi:

| Port | Use |
|---|---|
| first | Pi login console (`screen` / PuTTY) |
| second | trigger |

The probe finds the trigger port by itself.

**Quick manual check** from the Mac, on the second port:

```
screen /dev/cu.usbmodemXXXX 115200
```

Typing is not echoed. Then:

- `ping` + Enter
- `start 60`
- `status`
- `stop`

Quit with Ctrl-A, K, y.

## 2. Wiring (cameras unplugged from USB while wiring)

Pi header numbering: **physical pin 12 = BCM GPIO18** (the PWM output), physical pin 14 = GND.

```
Pi physical pin 12 (GPIO18) ──[220 Ω]── Blackfly S Hirose pin 1   (Line3, non-isolated input)
                            └─[220 Ω]── Firefly S JST pin 3, WHITE (Line2)
Pi physical pin 14 (GND)    ─────────── Blackfly S Hirose pin 6 (GND) + Firefly S JST pin 5, BROWN (GND)
```

**Firefly S, FLIR cable ACC-01-3015 (JST 6-pin).** These colours are FLIR's:

| Pin | Colour | Signal | Use |
|---|---|---|---|
| 1 | orange | Line0 / 1.8 V UART TX | **do not use** |
| 2 | black | Line1 / 1.8 V UART RX | **do not use** |
| 3 | **white** | **Line2** | **trigger input** |
| 4 | green | Line3 | spare input |
| 5 | **brown** | **GND** | **ground** |
| 6 | red | 3.3 V camera power OUT | **never connect** |

**Blackfly S, Hirose HR10A-7P-6S pigtail.** A third-party cable's colours will NOT match FLIR's. **Identify the pins by continuity** (multimeter beep mode) and note the colours:

| Pin | Signal | Use |
|---|---|---|
| 1 | **Line3**, non-isolated input (also VAUX power in) | **trigger input. Max 3.6 V** |
| 2 | Line0, opto-isolated input | do not use (needs 3.5–7 mA) |
| 3 | Line2 / 3.3 V power OUT | **never connect** |
| 4 | Line1, opto output | do not use |
| 5 | **opto** ground (isolated) | **not** this one for ground |
| 6 | **GND** | **ground** |

**Rules:**
- **Ground first.** Connect the Pi GND to both camera GNDs before the signal wire.
- Only the Pi's **3.3 V** output goes into the cameras. Never 5 V: the Blackfly's Line3 is rated 2.6–3.6 V for a high.
- The 220 Ω series resistors protect the Pi pin. FLIR says an external 3.3/5 V trigger needs no pull-up.
- Before plugging the cameras back in, check with the Analog Discovery that the junction still shows 0 / 3.3 V pulses at 60 Hz.

Electrical facts are from FLIR's GPIO pages for FFY-U3-04S2 and BFS-U3-13Y3, and app note TAN2016-008.

## 3. Rig test (recording laptop, recorder GUI and SpinView closed)

```powershell
git fetch
git switch feat/hw-trigger
pip install pyserial
python scripts\sync_probe.py trigger --fps 30 --seconds 60
```

That's a 1-minute trial. When it looks right, run the real test: 10 min at 30 Hz, then 10 min at 60 Hz.

```powershell
python scripts\sync_probe.py trigger
```

**What it does:**
- Stops the Pi.
- Puts both cameras in trigger mode: Firefly **Line2**, Blackfly **Line3**, rising edge.
  - Exposure is capped at half the period.
  - Override the lines with `--line 23227865=Line3` etc. if wired differently.
- Arms the cameras and waits 1 s with no pulses. **Any frame now is a stray trigger.**
- Starts the Pi, records, and stops the Pi.
- **Always puts both cameras back in free-running mode**, even if the run fails.

**Pass criteria, per rate:**

| | Pass |
|---|---|
| frames before start | **0** on both cameras (otherwise: noise or a loose ground) |
| frames vs pulses sent | equal on both (`missed 0`; ±1 at the start/stop edges is OK) |
| frame interval (camera clock) | std ≪ 10 µs; no intervals > 1.5 periods |
| same-pulse gap between the cameras | a **constant offset** (the two models' trigger-to-timestamp delays differ) with a spread p99 of a few tens of µs at most, versus 4–8 ms median free-running |

**Send back:** the whole `probe_output\sync_probe_<date>_<time>\` folder: `report.txt`, `summary.json`, and the `trigger_*` CSVs.

**If the recorder later seems to wait forever:** a camera was left in trigger mode, for example after a crash mid-run. Power-cycle the camera (unplug its USB for 5 s), or run any `sync_probe.py` command, which resets trigger mode at the end.
