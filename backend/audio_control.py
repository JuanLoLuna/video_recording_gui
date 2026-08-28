# backend/audio_control.py
"""
Session audio recording to WAV (same base filename as video).

Uses sounddevice + soundfile. Requires PortAudio (often bundled with wheels).

Uses a PortAudio callback + CallbackStop for reliable stop on macOS/Windows.
Optional live monitoring sends the mic to the default output (duplex stream);
use headphones to avoid acoustic feedback.
"""

from __future__ import annotations

import queue
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from backend.audio_format import AudioFormatChoice, AudioWriteBudget, choose_audio_format
from backend.audio_health import (
    AudioHealthCounters,
    AudioHealthSnapshot,
    plan_reconnect,
    resolve_device_index_by_name,
    silence_frames_for_gap,
)


def _resolve_audio_format(
    sf: Any, samplerate: int, channels: int, subtype: str = "PCM_16"
) -> AudioFormatChoice:
    """Probe this libsndfile build at runtime and pick a streaming-safe container."""
    return choose_audio_format(
        available_formats=sf.available_formats(),
        samplerate=samplerate,
        channels=channels,
        subtype=subtype,
        format_checker=sf.check_format,
    )


def _samplerate_candidates(
    sd: Any, input_device: int, output_device: int, primary: int
) -> list[int]:
    """Sample rates to try for duplex / split monitoring (shared input+output SR)."""
    rates: list[int] = []
    try:
        di = sd.query_devices(input_device, "input")
        do = sd.query_devices(output_device, "output")
        ri = int(float(di.get("default_samplerate") or 0))
        ro = int(float(do.get("default_samplerate") or 0))
        for r in (primary, ro, ri, 48_000, 44_100, 96_000, 32_000, 16_000):
            if r > 0 and r not in rates:
                rates.append(r)
    except Exception:
        for r in (primary, 48_000, 44_100):
            if r > 0 and r not in rates:
                rates.append(r)
    return rates


def _duplex_stream_kwargs_variants(sd: Any) -> list[dict[str, Any]]:
    """Keyword-argument sets to try when opening sd.Stream (duplex)."""
    variants: list[dict[str, Any]] = [
        {"latency": "high"},
        {"blocksize": 1024},
        {"blocksize": 512, "latency": "high"},
        {"blocksize": 2048, "latency": "high"},
        {"blocksize": 256, "latency": "high"},
        {"blocksize": 128, "latency": "high"},
        {"blocksize": 0},
        {"blocksize": 0, "latency": "high"},
    ]
    if sys.platform == "win32":
        try:
            ws_in = sd.WasapiSettings(exclusive=False)
            ws_out = sd.WasapiSettings(exclusive=False)
            variants.append({"extra_settings": (ws_in, ws_out), "latency": "high"})
            variants.append({"extra_settings": (ws_in, ws_out), "blocksize": 0})
        except Exception:
            pass
    return variants


def _run_split_monitor_streams(
    sd: Any,
    *,
    input_device: int,
    output_device: int,
    samplerates: list[int],
    out_ch: int,
    stop_event: threading.Event,
    on_input_mono: Callable[[Any], None],
    blocksize: int = 1024,
    on_monitor_started: Callable[[], None] | None = None,
    on_status: Callable[[Any], None] | None = None,
) -> bool:
    """
    Input-only + output-only streams with a small queue (Windows-friendly when
    full-duplex open fails across MME/WASAPI/etc.).
    """
    audio_q: queue.Queue = queue.Queue(maxsize=8)

    def in_cb(
        indata: Any,
        frames: int,
        time_info: Any,
        status: Any,
    ) -> None:
        if stop_event.is_set():
            raise sd.CallbackStop
        if on_status is not None:
            on_status(status)
        on_input_mono(indata)
        try:
            audio_q.put_nowait(indata.copy())
        except queue.Full:
            try:
                _ = audio_q.get_nowait()
            except queue.Empty:
                pass
            try:
                audio_q.put_nowait(indata.copy())
            except queue.Full:
                pass

    def make_out_cb(channels: int):
        def out_cb(
            outdata: Any,
            frames: int,
            time_info: Any,
            status: Any,
        ) -> None:
            if stop_event.is_set():
                raise sd.CallbackStop
            try:
                ind = audio_q.get_nowait()
            except queue.Empty:
                outdata[:] = 0
                return
            n_in = int(ind.shape[0])
            n_out_ch = int(outdata.shape[1])
            if n_in >= frames:
                mono = ind[:frames, 0]
            else:
                mono = np.zeros(frames, dtype=np.float32)
                mono[:n_in] = ind[:, 0]
            if n_out_ch == 1:
                outdata[:, 0] = mono
            else:
                for c in range(n_out_ch):
                    outdata[:, c] = mono

        return out_cb

    for sr in samplerates:
        for ch_try in (out_ch, 1) if out_ch != 1 else (1,):
            for bs in (blocksize, 512, 2048, 256, 0):
                try:
                    out_cb = make_out_cb(ch_try)
                    with sd.OutputStream(
                        device=output_device,
                        channels=ch_try,
                        samplerate=sr,
                        blocksize=bs,
                        dtype="float32",
                        latency="high",
                        callback=out_cb,
                    ), sd.InputStream(
                        device=input_device,
                        channels=1,
                        samplerate=sr,
                        blocksize=bs,
                        dtype="float32",
                        latency="high",
                        callback=in_cb,
                    ):
                        if on_monitor_started is not None:
                            try:
                                on_monitor_started()
                            except Exception:
                                pass
                        while not stop_event.is_set():
                            time.sleep(0.05)
                    return True
                except Exception:
                    continue
    return False


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


def list_audio_output_devices() -> list[tuple[int, str]]:
    """
    Return (device_index, label) for each host output device.

    Unlike inputs, we do **not** deduplicate by device name: on Windows several
    PortAudio indices can share the same driver string (e.g. Realtek jack vs
    internal speakers), and keeping only one entry hid the headphone endpoint.

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

        default_out_idx: int | None = None
        try:
            _din, dout = sd.default.device
            if dout is not None and int(dout) >= 0:
                default_out_idx = int(dout)
        except Exception:
            default_out_idx = None

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

        # Sort key: host API preference, then name, then index (stable, every index listed).
        rows: list[tuple[int, str, int, str]] = []
        for i, d in enumerate(devices):
            if not isinstance(d, dict):
                continue
            if int(d.get("max_output_channels", 0) or 0) < 1:
                continue
            name = str(d.get("name", f"device {i}")).strip()
            ha_name = hostapi_name(d.get("hostapi"))  # type: ignore[arg-type]
            rank = prefer_rank.get(ha_name, len(prefer_order))
            norm = " ".join(name.lower().split())
            label = f"{i}: {name} ({ha_name})"
            if default_out_idx is not None and i == default_out_idx:
                label += " — Windows default playback"
            rows.append((rank, norm, i, label))

        rows.sort(key=lambda t: (t[0], t[1], t[2]))
        return [(idx, label) for _rank, _norm, idx, label in rows]
    except Exception:
        return []


def _open_duplex_stream(
    sd: Any,
    input_device: int,
    out_ch: int,
    in_samplerate: int,
    callback: Any,
    *,
    output_device: int | None = None,
) -> Any | None:
    """
    Open full-duplex mic -> output. Tries sample rates, block sizes, output
    channel counts, and (on Windows) WASAPI shared settings. Returns None if
    no combination opens (caller may use split streams).
    """
    try:
        if output_device is None:
            _in_def, out_def = sd.default.device
            if out_def is None or int(out_def) < 0:
                return None
            out_dev = int(out_def)
        else:
            out_dev = int(output_device)
        rates = _samplerate_candidates(sd, input_device, out_dev, in_samplerate)
        ch_opts = (out_ch, 1) if out_ch != 1 else (1,)
        for ch in ch_opts:
            for sr in rates:
                for extra in _duplex_stream_kwargs_variants(sd):
                    try:
                        return sd.Stream(
                            device=(input_device, out_dev),
                            samplerate=sr,
                            channels=(1, ch),
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

    def start(
        self,
        device: int,
        *,
        monitor: bool = True,
        output_device: int | None = None,
    ) -> None:
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
                out_dev: int | None = None
                if use_duplex:
                    try:
                        out_dev = output_device
                        if out_dev is None:
                            _in_def, out_def = sd.default.device
                            if out_def is None or int(out_def) < 0:
                                use_duplex = False
                            else:
                                out_dev = int(out_def)
                        if use_duplex and out_dev is not None:
                            oinfo = sd.query_devices(out_dev, "output")
                            out_ch = max(
                                1,
                                min(
                                    2,
                                    int(oinfo.get("max_output_channels", 1) or 1),
                                ),
                            )
                    except Exception:
                        use_duplex = False
                        out_dev = None

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
                used_split_monitor = False
                if use_duplex and out_dev is not None:

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
                        sd,
                        device,
                        out_ch,
                        samplerate,
                        duplex_cb,
                        output_device=output_device,
                    )
                    if stream is not None:
                        self.had_duplex_output = True
                    else:
                        rates = _samplerate_candidates(
                            sd, device, out_dev, samplerate
                        )
                        used_split_monitor = _run_split_monitor_streams(
                            sd,
                            input_device=device,
                            output_device=out_dev,
                            samplerates=rates,
                            out_ch=out_ch,
                            stop_event=self._stop,
                            on_input_mono=self._feed_level,
                            on_monitor_started=lambda: setattr(
                                self, "had_duplex_output", True
                            ),
                        )
                        if not used_split_monitor:
                            print(
                                "[audio] preview monitor unavailable "
                                "(duplex and split-monitor open failed); "
                                "level meter only."
                            )

                if used_split_monitor:
                    return

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
        self.had_duplex_output = False
        self.audio_format_choice: AudioFormatChoice | None = None
        self.health = AudioHealthCounters()

    def level_0_100(self) -> int:
        with self._level_lock:
            return int(min(100, max(0, round(self._level_smoothed * 100.0))))

    def health_snapshot(self) -> AudioHealthSnapshot:
        return self.health.snapshot()

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
        output_device: int | None = None,
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

        choice = _resolve_audio_format(sf, samplerate, channels)
        self.audio_format_choice = choice
        budget = AudioWriteBudget.for_choice(choice, samplerate)

        self._stop.clear()
        with self._level_lock:
            self._level_smoothed = 0.0
        self.had_duplex_output = False

        def run() -> None:
            try:
                sf_kwargs: dict[str, Any] = {}
                if choice.format is not None:
                    sf_kwargs["format"] = choice.format
                with sf.SoundFile(
                    str(path),
                    "w",
                    samplerate,
                    channels,
                    subtype=choice.subtype,
                    **sf_kwargs,
                ) as f:
                    use_duplex = bool(monitor)
                    out_ch = 1
                    out_dev: int | None = None
                    if use_duplex:
                        try:
                            out_dev = output_device
                            if out_dev is None:
                                _in_def, out_def = sd.default.device
                                if out_def is None or int(out_def) < 0:
                                    use_duplex = False
                                else:
                                    out_dev = int(out_def)
                            if use_duplex and out_dev is not None:
                                oinfo = sd.query_devices(out_dev, "output")
                                out_ch = max(
                                    1,
                                    min(
                                        2,
                                        int(oinfo.get("max_output_channels", 1) or 1),
                                    ),
                                )
                        except Exception:
                            use_duplex = False
                            out_dev = None

                    def input_cb(
                        indata: Any,
                        frames: int,
                        time_info: Any,
                        status: Any,
                    ) -> None:
                        if self._stop.is_set():
                            raise sd.CallbackStop
                        self.health.note_status(status)
                        self._feed_level(indata)
                        f.write(indata.copy())
                        budget.note_frames(frames)
                        if budget.should_stop():
                            self._stop.set()

                    stream: Any | None = None
                    used_split_monitor = False
                    if use_duplex and out_dev is not None:

                        def duplex_cb(
                            indata: Any,
                            outdata: Any,
                            frames: int,
                            time_info: Any,
                            status: Any,
                        ) -> None:
                            if self._stop.is_set():
                                raise sd.CallbackStop
                            self.health.note_status(status)
                            self._feed_level(indata)
                            f.write(indata.copy())
                            budget.note_frames(frames)
                            if budget.should_stop():
                                self._stop.set()
                            n_out = int(outdata.shape[1])
                            if n_out == 1:
                                outdata[:] = indata
                            else:
                                for c in range(n_out):
                                    outdata[:, c] = indata[:, 0]

                        stream = _open_duplex_stream(
                            sd,
                            device,
                            out_ch,
                            samplerate,
                            duplex_cb,
                            output_device=output_device,
                        )
                        if stream is not None:
                            self.had_duplex_output = True
                        else:

                            def on_mono(indata: Any) -> None:
                                self._feed_level(indata)
                                f.write(indata.copy())
                                budget.note_frames(indata.shape[0])
                                if budget.should_stop():
                                    self._stop.set()

                            # Keep stream SR == WAV SR (split path does not resample).
                            used_split_monitor = _run_split_monitor_streams(
                                sd,
                                input_device=device,
                                output_device=out_dev,
                                samplerates=[samplerate],
                                out_ch=out_ch,
                                stop_event=self._stop,
                                on_input_mono=on_mono,
                                on_monitor_started=lambda: setattr(
                                    self, "had_duplex_output", True
                                ),
                                on_status=self.health.note_status,
                            )
                            if not used_split_monitor:
                                print(
                                    "[audio] live monitor unavailable "
                                    "(duplex and split-monitor open failed); "
                                    "recording without monitor."
                                )

                    if used_split_monitor:
                        # Split-monitor is a rarer Windows fallback that already
                        # blocks internally until stop; no reconnect support here.
                        return

                    if stream is None:
                        stream = sd.InputStream(
                            device=device,
                            channels=channels,
                            samplerate=samplerate,
                            dtype="float32",
                            blocksize=1024,
                            callback=input_cb,
                        )

                    device_name = str(info.get("name", "")).strip()
                    while not self._stop.is_set():
                        try:
                            with stream:
                                with self._stream_lock:
                                    self._stream = stream
                                try:
                                    while stream.active and not self._stop.is_set():
                                        time.sleep(0.05)
                                finally:
                                    with self._stream_lock:
                                        self._stream = None
                        except Exception as exc:
                            print(f"[audio] input stream dropped: {exc}; reconnecting")

                        if self._stop.is_set():
                            break

                        # Stream ended without a stop request: the device dropped
                        # (e.g. USB unplug). Keep the WAV open, re-resolve the
                        # device by name (its index may have shifted on
                        # re-enumeration), and pad the gap with silence so the
                        # sample count stays aligned with wall-clock time.
                        drop_time = time.monotonic()
                        self.had_duplex_output = False
                        stream = None
                        attempt = 0
                        while stream is None and not self._stop.is_set():
                            attempt += 1
                            plan = plan_reconnect(device_name, attempt)
                            self._stop.wait(plan.backoff_s)
                            if self._stop.is_set():
                                break
                            try:
                                devices = sd.query_devices()
                            except Exception:
                                devices = []
                            idx = resolve_device_index_by_name(device_name, devices)
                            if idx is None:
                                continue

                            def reconnect_cb(
                                indata: Any,
                                frames: int,
                                time_info: Any,
                                status: Any,
                            ) -> None:
                                if self._stop.is_set():
                                    raise sd.CallbackStop
                                self.health.note_status(status)
                                self._feed_level(indata)
                                f.write(indata.copy())
                                budget.note_frames(frames)
                                if budget.should_stop():
                                    self._stop.set()

                            try:
                                stream = sd.InputStream(
                                    device=idx,
                                    channels=channels,
                                    samplerate=samplerate,
                                    dtype="float32",
                                    blocksize=1024,
                                    callback=reconnect_cb,
                                )
                            except Exception:
                                stream = None

                        if stream is None:
                            # Stop was requested while still trying to reconnect.
                            break

                        gap_s = time.monotonic() - drop_time
                        n_silence = silence_frames_for_gap(gap_s, samplerate)
                        if n_silence > 0:
                            f.write(np.zeros((n_silence, channels), dtype="float32"))
                        self.health.note_reconnect()
                        print(
                            f"[audio] reconnected to '{device_name}' after "
                            f"{gap_s:.2f}s ({n_silence} silence frames inserted)"
                        )
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
