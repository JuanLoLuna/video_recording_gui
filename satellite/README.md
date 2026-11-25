# Satellite camera streaming with Thalamus: runbook

Goal: a camera on a second computer (the "satellite") streams over a cable or
switch with no internet to the receiving laptop, and the laptop saves the video
with timestamps from both computers. This folder adds files only; it does not
touch the existing recording app.

Thalamus = acquisition software from the Cajigas lab (GitHub cajigaslab/Thalamus,
pip package `thalamus_neuro`, Python module `thalamus`). Everything below was
read from upstream commit a0b4c991 (source copy at /home/airflow/proc/Thalamus).

## What runs where

| Computer | Address | Config | Thalamus ports (gRPC / user interface / dotnet) |
|---|---|---|---|
| Satellite (camera, Linux, nsu-pesar-20) | 192.168.50.10 | `configs/satellite_thorcam.json` | 50050 / 50051 / 50052, opened to the network (`--open`), TLS |
| Receiving laptop | 192.168.50.20 | `configs/receiver.json` | 50070 / 50071 / 50072, local only |

The receiving laptop already runs the Sleeve laptop clock service on port 50050,
so its Thalamus must not use the default 50050-50052; that is why it uses 50070-50072.
(gRPC = the network protocol Thalamus uses between computers.)

## Current satellite: Thorlabs camera, Jarl's route (verified 2026-11-24)

The satellite (nsu-pesar-20, Ubuntu 24.04, user bijanadmin) has a Thorlabs DCC1240M camera, sold by IDS as
UI-124xSE-M. It is not a webcam (no /dev/video*) and has no GenTL driver, so the FFMPEG and GENICAM nodes cannot
open it. Jarl set it up on 2026-10-09 with:

- IDS uEye driver 4.96.1 (packages `ueye-*`, services `ueyeusbdrc`, `ueyeethdrc`).
- Python environment `~/venv-thalamus`: thalamus-neuro 1.0.53 + **thalamus-contrib 0.0.12** (pip). Contrib adds the
  **THORCAM** node (reads the camera through the uEye driver) and **MEDIA_CONVERTER** (H264 encoder), loaded with `-C`.
- TLS: `~/certs/` holds `ca.pem`, `thalamus.internal.pem`, `thalamus.internal-key.pem` (server name `thalamus.internal`,
  client certificates required). The receiver must use certificates from the same authority.

`launch_satellite.sh` now starts exactly that: Jarl's environment, `-C`, `--open`, the certificates and
`configs/satellite_thorcam.json` (copy of Jarl's `~/sink.json`; nodes renamed `thorcam1` and `thorcam1_h264`,
1280x1024, 15 frames per second, exposure 66.6 ms, H264 quality 5, quantization 23). `DRY_RUN=1` prints the command.
After the window opens, tick **Running** on `thorcam1` (Thalamus starts every node switched off).
Verified 2026-11-24 with Jarl's own command: live image in the node's view window. Verified 2026-11-24 16:00 with
this launcher and the renamed config: live image on `thorcam1`. Not yet verified: the H264 node producing output
(it showed no output bytes while no receiver was connected), and any receiver connecting.

Receiver: `configs/receiver.json` pulls `thorcam1_h264` from `192.168.50.10:50050`. `launch_receiver.*` add `-C` and the
certificates from `satellite/certs/` (copy the three .pem files from the satellite's `~/certs`; git ignores that
folder). The receiver needs `pip install thalamus-contrib` as well. Use the same Thalamus and contrib versions on both
computers (1.0.53 / 0.0.12 is the known-working pair).

Viewing on the receiver: Thalamus 1.0.53's image viewer cannot show H264 frames (ticking View on `sat_cam1`
crashes the preview with `UnboundLocalError: ... 'format'` in image_viewer.py; the stream itself keeps flowing and
STORAGE2 still records). For a picture, run `sat_cam1_preview`, a second REMOTE node that pulls the uncompressed
`thorcam1` (about 20 MB/s at 1280x1024, 15 frames per second, 8-bit), and tick View on it.

The webcam (FFMPEG) instructions below remain for other cameras (`configs/satellite.json`, `configs/satellite_linux.json`;
start with `THALAMUS_CONFIG=... launch_satellite.sh`). On the satellite, the firewall rule is for port 50050, not 50060.

## Where each config key came from

Thalamus source paths are relative to the repo root (upstream a0b4c991).

- Config shape `{"nodes": [{"name":..., "type":..., <properties>}]}`:
  `thalamus/pipeline/main_impl.py` (reads `config['nodes']`) and the example
  `SimpleUseCase/Heart Rate Variability/SimpleUseCase.json`.
- FFMPEG properties `Input Format`, `Input Name`, `Options`, `Time Source`, `Running`:
  `src/thalamus/image_node.cpp` (`start()` and `on_change`) and `docs/source/nodes/ffmpeg.rst`.
  `Options` is a dictionary of text values passed to FFmpeg (values must be text, hence `"30"`).
  `View` is from the node's widget list in `thalamus/pipeline/thalamus_window.py`.
- REMOTE properties `Address`, `Node`, `Probe Frequency`, `Probe Size`, `Running`:
  `src/thalamus/remote_node.cpp` (`on_change`), defaults in `thalamus_window.py`, `docs/source/nodes/remote.rst`.
- STORAGE2 properties `Output File`, `Compress Analog`, `Compress Video`, `Simple Copy`, `Running`,
  and the `Sources` list entries (`Node`, `Time Series`, `Image`, `Motion`, `Text`):
  `src/thalamus/storage2_node.cpp` (around lines 1456-1490), `thalamus/pipeline/storage2_widget.py`
  (`on_add`), `docs/source/nodes/storage2.rst`.
- Recording file layout and `time` / `remote_time` fields: `proto/thalamus.proto`
  (`StorageRecord`, `Image` fields 8 and 9), `thalamus/record_reader2.py`.
- Command-line flags `-c --open -p -u -d`: `thalamus/pipeline/main_impl.py`.

Important: on startup Thalamus sets every node's `Running` to false
(`main_impl.py`, loop over `config['nodes']`). So after launching you must tick
Running yourself in each window, even though the configs say false anyway.

`Compress Video` is false in `receiver.json` for the first test (the widget default is true) to keep
the first test simple; turn it on later to save disk space (it uses the OpenH264 library, see below).

## Offline preparation (do this while both computers still have internet)

1. Python 3.10 on both computers (matches the existing app). Create the virtual environment
   (folder where Thalamus lives): `python -m venv C:\thalamus_venv`.
2. On any online computer download the Thalamus wheel (the installable file, `thalamus_neuro-*.whl`)
   from the GitHub releases page of cajigaslab/Thalamus, plus the wheels it depends on:
   `pip download thalamus_neuro -d wheels` (or `pip download` the .whl file into the same folder).
   Copy the `wheels` folder by USB stick and install offline:
   `C:\thalamus_venv\Scripts\pip install --no-index --find-links wheels thalamus_neuro`.
   Use the same operating system and Python version for the download as for the install.
3. OpenH264 (Cisco's video encoder, only needed for `Compress Video`): Thalamus downloads it
   into `%USERPROFILE%\.thalamus` at startup (`thalamus/openh264.py`). With no internet it
   cannot. Either:
   - Download it while online and place it there. File names (version 2.6.0):
     Windows `openh264-2.6.0-win64.dll`; Linux `libopenh264-2.6.0-linux64.8.so`.
     Download address: `http://ciscobinary.openh264.org/<file name>.bz2`, then unpack the
     .bz2 (7-Zip) and put the unpacked file in the `.thalamus` folder, or
   - Opt out: create an empty file `%USERPROFILE%\.thalamus\.no_openh264`
     (Linux `~/.thalamus/.no_openh264`). Thalamus then skips the download; keep `Compress Video` off.
4. Copy this `satellite` folder to both computers (for example `C:\satellite`).
5. Edit `configs/satellite.json`: replace `video=EDIT_ME_CAMERA_NAME` with the real name.
   Find it on Windows with: `ffmpeg -list_devices true -f dshow -i dummy` (needs FFmpeg on the
   satellite; if it is not installed, the camera name is also shown in Windows Device Manager
   under Cameras). On Linux use `"Input Format": "v4l2"` and `"Input Name": "/dev/video0"`.
6. Edit `configs/receiver.json`: set `Output File` (folder must exist) and, if your addresses differ,
   `Address`.

## Static addresses (no router, no internet)

Windows: Settings > Network > Ethernet > IP assignment > Edit > Manual > IPv4 on.
Satellite: IP 192.168.50.10, subnet mask 255.255.255.0, no gateway. Receiving laptop:
192.168.50.20, same mask, no gateway. Check from the laptop: `ping 192.168.50.10`.
Windows may label this network "Public"; set it to "Private" so the firewall rule below applies.

## Windows firewall on the satellite (administrator command prompt)

    netsh advfirewall firewall add rule name="Thalamus satellite 50060" dir=in action=allow protocol=TCP localport=50060 profile=private,public

Remove later with: `netsh advfirewall firewall delete rule name="Thalamus satellite 50060"`.
Only port 50060 is needed (the receiving laptop connects out to it). Ports 50061 and 50062
stay closed.

## Unattended start on the satellite

Set Windows to sign in automatically (run `netplwiz`, untick "Users must enter a user name and
password"), put a shortcut to `launch_satellite.bat` in `shell:startup`, and set power settings to
never sleep. Thalamus is a desktop (Qt) program, so it needs a signed-in desktop session: it will not
start from a service or before login. Then tick Running on `cam1` by hand (see "Open items").

## 10-minute test (no patient)

1. Cable the two computers, set the addresses, run the firewall command, ping works.
2. Satellite: double-click `launch_satellite.bat`. A Thalamus window opens with node `cam1`.
   Tick `View` to see the picture, then tick `Running`. Expect live video.
3. Laptop: double-click `launch_receiver.bat`. Nodes `sat_cam1` and `storage` appear. Tick `Running`
   on `sat_cam1`; its status should say Connected and show Bytes Per Second and ping. Tick `View` to watch.
4. Tick `Running` on `storage`. Record 10 minutes (wave a hand in front of the camera at the start and end).
5. Untick `Running` on `storage`. The file is `<Output File>.YYYYMMDD.R`
   (YYYYMMDD = date, R = recording number) plus a `.json` with the config.
6. Optional stress: during a second recording unplug the cable for 5 seconds, then plug it back in.
7. Run the check (below). Share `report.txt` and `report.json`, not screenshots.

## Checking a recording

    python satellite\stream_check.py C:\thalamus_recordings\session.20260101.1 --expected-fps 30

Optional: `--node sat_cam1` (default), `--out DIR` (default: next to the recording). It needs only
Python 3.8 or newer, no Thalamus install, and writes `report.txt` and `report.json`. It reports frame count,
duration, frame interval (mean, median, 99th percentile) on both clocks, gaps longer than twice the
median interval (count and first 20), estimated dropped frames, and drift and jitter of
`time - remote_time`. The two clocks (receiving laptop and satellite) are unrelated, so that difference is
not a true latency; only its drift and scatter are meaningful. Tests: `python -m unittest tests.test_satellite_stream_check`.

## Storing the incoming video: receiver_service.py

Run inside the laptop's `thalamus` conda env while `launch_receiver.bat` is running and `sat_cam1` is Running:

    python satellite\receiver_service.py --serial 4103305000 start --output-dir D:\recordings
    python satellite\receiver_service.py stop   --output-dir D:\recordings
    python satellite\receiver_service.py status --output-dir D:\recordings

Options: `--address localhost:50070`, `--certs satellite\certs`, `--expected-fps 15`, `--bytes-per-hour 7.2e9`.
`start` checks free disk (backend/disk_guard; refuses when critical, `--force` overrides), points `storage` at
`D\.incomplete\sat<serial>\recording_YYYYMMDD_HHMMSS_sat<serial>` and sets `Running`. `stop` sets `Running`
false, waits for the file to stop growing, runs the stream check, converts the H264 to mp4 without re-encoding (`satellite/h264_to_mp4.py`, in a child process; needs `pip install av` in the Thalamus env), and
only then moves finished files into `D`: `<stem>-0000.mp4`, `<stem>.thalamus`, `<stem>.thalamus.json`,
`<stem>_stream_check.txt/.json`, and last `<stem>_satellite_manifest.json`. If the mp4 step fails the raw
recording is kept and the error is in the manifest. Each command writes `D\receiver_service_last.json`.
Tests (fake Thalamus): `python -m unittest tests.test_receiver_service`.

UNVERIFIED (never run against a real Thalamus): the `ThalamusControl` connection (ThalamusThread over TLS, setting
properties by `setitem` and waiting for the acknowledgement); that setting `Output File` then `Running` makes STORAGE2
open the new file; the name of the `.json` sidecar STORAGE2 writes (both `<file>.json` and `<stem>.json` are tried);
which keys a REMOTE
node exposes for Connected status; clean exit of the process (it ends with `os._exit`).

## Clock sync (plan step 3)

Maps each frame's satellite time (`remote_time`, the satellite's steady clock) to the laptop's steady and wall clocks.
The clock service is Jarl's `server.exe` on the laptop (port 50050, the one the phones use); `clock_server.py` is only a test tool.

On the satellite (its Thalamus 1.0.66 venv has grpc and numpy; `satellite_lan.sh up` opens TCP 50061):

    ~/venv-thalamus-new/bin/python satellite/clock_logger.py --server laptop=192.168.50.20:50050 --out ~/clock_log --duration-s 0 --serve-port 50061

It logs forever (one subfolder `~/clock_log/YYYYMMDD` per day) and serves the CSVs read-only on port 50061.
On the laptop, add the URL to the receiver service (global option, before `stop`):

    python satellite\receiver_service.py --clock-log-url http://192.168.50.10:50061/ stop --output-dir D:\recordings

`--clock-label LABEL` is needed only if the logger measures several servers. server.exe needs Windows Firewall to allow
inbound TCP 50050 from the satellite's network. Works with or without the `CLK1` ping trailer: with it the pings give the
mapping ("pings"), without it the analog stream plus half the ping round trip does ("analog+half-rtt", about 0.1 to 0.8 ms
off against a cable reference).

At stop, `<stem>_satellite_sync.csv` has one row per frame: `frame_index`, `satellite_steady_ns` (remote_time),
`laptop_steady_ns` (mapped), `laptop_wall_ns` (mapped plus the median Wall minus steady of the server; blank without clock
service), `arrival_laptop_steady_ns` (`time`, when the frame reached the laptop). The manifest `sync` block holds `synced`,
`method`, `drift_ppm` (satellite clock against laptop clock), `offset_at_start_ns` (laptop steady minus satellite steady at the
first frame), `uncertainty_ms`, `samples`, `span_s`, `clock_instance_ids`, `transfer_delay_min_ms` and `warnings`.
`transfer_delay_min_ms` = smallest (arrival time minus mapped time) over the frames: it should be small and not negative
(warning if below minus the uncertainty or above 200 ms). If the clock log cannot be fetched or used, stop still succeeds, `synced`
is false with a `reason`, and the CSV holds the fallback "arrival-min (no clock service)" (min-filtered `time - remote_time`
line: includes the smallest transfer delay, no wall time).
Offline: `python3 satellite/clock_logger.py analyze DIR` writes `report.txt/json` comparing CURRENT, CURRENT-STEADY, NEW and SERVER-EXE.
A recording that spans midnight still works (the server joins the last 3 day folders).

## Satellite window

One window, one click: open it and nothing else needs starting by hand.

    python satellite\satellite_window.py

On opening, a worker (the status line names each step) (1) starts the receiver Thalamus exactly like `launch_receiver.bat`, unless
something already answers on `receiver_address` (then it is reused); its config is generated from `receiver_config_template` with
`satellite_address` as the Address of every REMOTE node and written to `satellite/configs/receiver.window.local.json` (so cable or
router is just the setting; `receiver.local.json` is not read); (2) switches the camera node `sat_cam1` on (the preview node stays
off); (3) starts `clock_server.py` on `clock_server_port` unless something already answers there (for example Sleeve server.exe).
Children run with their working folder and logs in `satellite/logs/` (git-ignored). A child that dies turns the line red with the
last log lines; a later poll restarts it, at most once per 30 s. On close (after the "Stop recording?" question) only the children
this window started are stopped; anything it found running is left alone. Set `start_receiver` / `start_clock_server` to false to
turn the auto-start off.
The receiver Thalamus runs with no window of its own (Qt offscreen platform); its messages go to `satellite/logs/receiver.log`.
Set `show_thalamus_window` to true to see its window again for troubleshooting.

The status line has a coloured dot: **green** = synced (clock fit with its uncertainty and the clock-log label), receiver connected,
camera on, and while recording the file is growing; **orange** = a setup step running, starting/stopping, camera off, the clock log's
newest sample over 10 s old, or the recording file not growing for 3 s; **red** = not synced, clock log unreachable, receiver not
reachable, or a child process stopped (details in grey under it). One button starts or stops a recording (stop takes seconds: it
converts the video and syncs the clock); a line below shows the result (frames, `synced +-x ms` or `NOT synced: reason`, output
folder). The box shows a slow live picture of the satellite camera: the window subscribes to the image stream of node `sat_cam1`
(the H264 one the receiver already pulls) over its existing Thalamus connection, looks only for keyframes (about one per second),
decodes at most `preview_max_fps` of them (default 1; 0 = preview off; needs PyAV, `pip install av`) and shows them 640 px wide in grey.
Under the box: "picture 0.8 s old"; older than 5 s it greys out and says "no picture (reason)"; a dropped stream is resubscribed
every 5 s. The status line shows `frames N/s` (frames arriving over the last 5 s); orange if the camera is on but no frames came for 5 s.
It does not touch recording (a separate stream subscription).

Settings live in `satellite/configs/window.local.json` (git-ignored, created with defaults on first run, opened by the "Settings..."
button, applied after a restart); every key is described at the top of `satellite_window.py`. Logic without Qt is in
`window_logic.py` and `window_processes.py`; tests: `python -m unittest tests.test_window_logic tests.test_window_processes`.
Headless check: `QT_QPA_PLATFORM=offscreen python satellite/satellite_window.py --screenshot out.png --quit-after 12`.

## Open items (UNVERIFIED unless stated)

- UNVERIFIED: which Thalamus version Jarl's computer runs. Both ends should use the same version; the
  configs follow upstream commit a0b4c991, not necessarily the released wheel.
- UNVERIFIED: satellite operating system (Windows assumed; Linux scripts provided).
- UNVERIFIED: whether Windows `dshow` camera names and `Options` (`framerate`, `video_size`) suit the
  chosen webcam; list the camera's supported modes with `ffmpeg -f dshow -list_options true -i video="<name>"`.
- UNVERIFIED: time units in the recording are assumed nanoseconds (the check script has `--time-unit-seconds`).
  Not run against a real Thalamus recording yet; the parser was tested only on synthetic files built from `proto/thalamus.proto`.
- UNVERIFIED: the exact pip wheel name and the full list of offline dependencies; confirm on the releases page.
- Not tested: launching Thalamus itself (needs the Windows machines and a camera).
