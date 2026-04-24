# backend/audio_control.py
"""
Session audio recording to WAV (same base filename as video).

Uses sounddevice + soundfile. Requires PortAudio (often bundled with wheels).

Uses a PortAudio callback + CallbackStop for reliable stop on macOS/Windows.
Optional live monitoring sends the mic to the default output (duplex stream);
use headphones to avoid acoustic feedback.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

import numpy as np


def list_audio_input_devices() -> list[tuple[int, str]]:
    """
    Return (device_index, label) for each host input device.

    If sounddevice is unavailable or errors, returns an empty list.
    """
    try:
        import sounddevice as sd
    except Exception:
        return []

    try:
        devices = sd.query_devices()
        hostapis = sd.query_hostapis()

        def hostapi_name(hostapi_index: int | None) -> str:
            try:
                if hostapi_index is None:
                    return "Unknown"
                ha = hostapis[int(hostapi_index)]
                if isinstance(ha, dict):
                    return str(ha.get("name") or "Unknown")
                return "Unknown"
            except Exception:
                return "Unknown"

        # Windows commonly lists the same physical mic multiple times across host APIs.
        # Prefer WASAPI entries and deduplicate by normalized device name.
        prefer_order = [
            "Windows WASAPI",
            "WASAPI",
            "WDM-KS",
            "Windows WDM-KS",
            "DirectSound",
            "Windows DirectSound",
            "MME",
            "Windows MME",
        ]
        prefer_rank = {n: r for r, n in enumerate(prefer_order)}

        candidates: list[tuple[str, int, int, str]] = []
        # tuple: (norm_name, rank, device_index, label)
        for i, d in enumerate(devices):
            if not isinstance(d, dict):
                continue
            if int(d.get("max_input_channels", 0) or 0) < 1:
                continue
            name = str(d.get("name", f"device {i}")).strip()
            ha_name = hostapi_name(d.get("hostapi"))  # type: ignore[arg-type]
            rank = prefer_rank.get(ha_name, len(prefer_order))
            norm = " ".join(name.lower().split())
            label = f"{i}: {name} ({ha_name})"
            candidates.append((norm, rank, i, label))

        # Stable: sort by name, host-api preference, then index.
        candidates.sort(key=lambda t: (t[0], t[1], t[2]))

        out: list[tuple[int, str]] = []
        seen: set[str] = set()
        for norm, _rank, idx, label in candidates:
            if norm in seen:
                continue
            seen.add(norm)
            out.append((idx, label))
        return out
    except Exception:
        return []


def _open_duplex_stream(
    sd: Any,
    input_device: int,
    out_ch: int,
    in_samplerate: int,
    callback: Any,
) -> Any | None:
    """
    Open full-duplex mic -> default output. Tries explicit output device and
    several sample rates / block sizes (macOS often fails with (in, None) or SR mismatch).
    """
    try:
        _in_def, out_def = sd.default.device
        if out_def is None or int(out_def) < 0:
            return None
        out_dev = int(out_def)
        out_info = sd.query_devices(out_dev, "output")
        out_sr = int(float(out_info.get("default_samplerate") or in_samplerate))
        rates: list[int] = []
        for r in (in_samplerate, out_sr, 48_000, 44_100):
            if r not in rates:
                rates.append(r)
        for sr in rates:
            for extra in (
                {"blocksize": 1024},
                {"blocksize": 512, "latency": "high"},
            ):
                try:
                    return sd.Stream(
                        device=(input_device, out_dev),
                        samplerate=sr,
                        channels=(1, out_ch),
                        dtype="float32",
                        callback=callback,
                        **extra,
                    )
                except Exception:
                    continue
        return None
    except Exception:
        return None


class MicLevelPreview:
    """
    Opens the mic without recording: updates a smoothed peak level (0–1) for a UI meter
    and optionally passes audio to the default output (duplex), same as SessionAudioRecorder.
    """

    def __init__(self) -> None:
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._stream: Any = None
        self._stream_lock = threading.Lock()
        self._level_lock = threading.Lock()
        self._level_smoothed = 0.0
        self.had_duplex_output = False

    def level_0_100(self) -> int:
        with self._level_lock:
            return int(min(100, max(0, round(self._level_smoothed * 100.0))))

    def _feed_level(self, indata: Any) -> None:
        peak = float(np.max(np.abs(indata)))
        scaled = min(1.0, peak * 5.0)
        with self._level_lock:
            self._level_smoothed = 0.78 * self._level_smoothed + 0.22 * scaled

    def start(self, device: int, *, monitor: bool = True) -> None:
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("Mic preview already running.")
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

        import sounddevice as sd

        info = sd.query_devices(device, "input")
        if int(info.get("max_input_channels", 0) or 0) < 1:
            raise ValueError(f"Device {device} has no input channels.")

        samplerate = int(float(info.get("default_samplerate") or 48000))
        channels = 1

        self._stop.clear()
        self.had_duplex_output = False
        with self._level_lock:
            self._level_smoothed = 0.0

        def run() -> None:
            try:
                use_duplex = bool(monitor)
                out_ch = 1
                if use_duplex:
                    try:
                        _in_def, out_def = sd.default.device
                        if out_def is None or int(out_def) < 0:
                            use_duplex = False
                        else:
                            oinfo = sd.query_devices(out_def, "output")
                            out_ch = max(
                                1,
                                min(
                                    2,
                                    int(oinfo.get("max_output_channels", 1) or 1),
                                ),
                            )
                    except Exception:
                        use_duplex = False

                def input_cb(
                    indata: Any,
                    frames: int,
                    time_info: Any,
                    status: Any,
                ) -> None:
                    if self._stop.is_set():
                        raise sd.CallbackStop
                    self._feed_level(indata)

                stream: Any | None = None
                if use_duplex:

                    def duplex_cb(
                        indata: Any,
                        outdata: Any,
                        frames: int,
                        time_info: Any,
                        status: Any,
                    ) -> None:
                        if self._stop.is_set():
                            raise sd.CallbackStop
                        self._feed_level(indata)
                        n_out = int(outdata.shape[1])
                        if n_out == 1:
                            outdata[:] = indata
                        else:
                            for c in range(n_out):
                                outdata[:, c] = indata[:, 0]

                    stream = _open_duplex_stream(
                        sd, device, out_ch, samplerate, duplex_cb
                    )
                    if stream is not None:
                        self.had_duplex_output = True
                    else:
                        print(
                            "[audio] preview monitor unavailable "
                            "(duplex open failed); level meter only."
                        )

                if stream is None:
                    stream = sd.InputStream(
                        device=device,
                        channels=channels,
                        samplerate=samplerate,
                        dtype="float32",
                        blocksize=1024,
                        callback=input_cb,
                    )

                with stream:
                    with self._stream_lock:
                        self._stream = stream
                    try:
                        while stream.active:
                            time.sleep(0.05)
                    finally:
                        with self._stream_lock:
                            self._stream = None
            except Exception as exc:
                print(f"[audio] mic preview failed: {exc}")

        self._thread = threading.Thread(
            target=run,
            name="MicLevelPreview",
            daemon=True,
        )
        self._thread.start()

    def stop(self, timeout: float | None = None) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                print("[audio] warning: mic preview thread did not finish.")
            self._thread = None
        with self._level_lock:
            self._level_smoothed = 0.0


class SessionAudioRecorder:
    """
    Records mono PCM16 WAV in a background thread while start/stop bracket a session.
    """

    def __init__(self) -> None:
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._stream: Any = None
        self._stream_lock = threading.Lock()
        self._level_lock = threading.Lock()
        self._level_smoothed = 0.0

    def level_0_100(self) -> int:
        with self._level_lock:
            return int(min(100, max(0, round(self._level_smoothed * 100.0))))

    def _feed_level(self, indata: Any) -> None:
        peak = float(np.max(np.abs(indata)))
        scaled = min(1.0, peak * 5.0)
        with self._level_lock:
            self._level_smoothed = 0.78 * self._level_smoothed + 0.22 * scaled

    def start(
        self,
        wav_path: str | Path,
        device: int,
        *,
        monitor: bool = True,
    ) -> None:
        """
        Begin recording to wav_path using the given sounddevice input index.

        If monitor is True, also play input to the default output device (duplex).
        Raises if the device is invalid or the stream cannot start.
        """
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("Audio recorder already running.")
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

        import sounddevice as sd
        import soundfile as sf

        path = Path(wav_path)
        info = sd.query_devices(device, "input")
        if int(info.get("max_input_channels", 0) or 0) < 1:
            raise ValueError(f"Device {device} has no input channels.")

        samplerate = int(float(info.get("default_samplerate") or 48000))
        channels = 1

        self._stop.clear()
        with self._level_lock:
            self._level_smoothed = 0.0

        def run() -> None:
            try:
                with sf.SoundFile(
                    str(path),
                    "w",
                    samplerate,
                    channels,
                    subtype="PCM_16",
                ) as f:
                    use_duplex = bool(monitor)
                    out_ch = 1
                    if use_duplex:
                        try:
                            _in_def, out_def = sd.default.device
                            if out_def is None or int(out_def) < 0:
                                use_duplex = False
                            else:
                                oinfo = sd.query_devices(out_def, "output")
                                out_ch = max(
                                    1,
                                    min(
                                        2,
                                        int(oinfo.get("max_output_channels", 1) or 1),
                                    ),
                                )
                        except Exception:
                            use_duplex = False

                    def input_cb(
                        indata: Any,
                        frames: int,
                        time_info: Any,
                        status: Any,
                    ) -> None:
                        if self._stop.is_set():
                            raise sd.CallbackStop
                        self._feed_level(indata)
                        f.write(indata.copy())

                    stream: Any | None = None
                    if use_duplex:

                        def duplex_cb(
                            indata: Any,
                            outdata: Any,
                            frames: int,
                            time_info: Any,
                            status: Any,
                        ) -> None:
                            if self._stop.is_set():
                                raise sd.CallbackStop
                            self._feed_level(indata)
                            f.write(indata.copy())
                            n_out = int(outdata.shape[1])
                            if n_out == 1:
                                outdata[:] = indata
                            else:
                                for c in range(n_out):
                                    outdata[:, c] = indata[:, 0]

                        stream = _open_duplex_stream(
                            sd, device, out_ch, samplerate, duplex_cb
                        )
                        if stream is None:
                            print(
                                "[audio] live monitor unavailable "
                                "(duplex open failed); recording without monitor."
                            )

                    if stream is None:
                        stream = sd.InputStream(
                            device=device,
                            channels=channels,
                            samplerate=samplerate,
                            dtype="float32",
                            blocksize=1024,
                            callback=input_cb,
                        )

                    with stream:
                        with self._stream_lock:
                            self._stream = stream
                        try:
                            while stream.active:
                                time.sleep(0.05)
                        finally:
                            with self._stream_lock:
                                self._stream = None
            except Exception as exc:
                print(f"[audio] recording failed: {exc}")

        self._thread = threading.Thread(
            target=run,
            name="SessionAudioRecorder",
            daemon=True,
        )
        self._thread.start()

    def stop(self, timeout: float | None = None) -> None:
        """Stop recording and wait until the WAV file is fully closed.

        timeout=None waits as long as needed (avoid cutting sessions short).
        """
        # Signal the callback to raise CallbackStop; do not call stream.stop()
        # from this thread (can deadlock with PortAudio on some macOS setups).
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                print(
                    "[audio] warning: recorder thread did not finish; "
                    "WAV may be incomplete."
                )
            self._thread = None
        with self._level_lock:
            self._level_smoothed = 0.0
