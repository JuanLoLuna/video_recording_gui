# backend/audio_control.py
"""
Session audio recording to WAV (same base filename as video).

Uses sounddevice + soundfile. Requires PortAudio (often bundled with wheels).

Uses a PortAudio callback + CallbackStop for reliable stop on macOS/Windows
(blocking read() + abort() from another thread can hang or truncate).
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any


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
        out: list[tuple[int, str]] = []
        for i, d in enumerate(devices):
            if not isinstance(d, dict):
                continue
            if int(d.get("max_input_channels", 0) or 0) < 1:
                continue
            name = str(d.get("name", f"device {i}"))
            out.append((i, f"{i}: {name}"))
        return out
    except Exception:
        return []


class SessionAudioRecorder:
    """
    Records mono PCM16 WAV in a background thread while start/stop bracket a session.
    """

    def __init__(self) -> None:
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._stream: Any = None
        self._stream_lock = threading.Lock()

    def start(self, wav_path: str | Path, device: int) -> None:
        """
        Begin recording to wav_path using the given sounddevice input index.

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

        def run() -> None:
            try:
                with sf.SoundFile(
                    str(path),
                    "w",
                    samplerate,
                    channels,
                    subtype="PCM_16",
                ) as f:

                    def callback(
                        indata: Any,
                        frames: int,
                        time_info: Any,
                        status: Any,
                    ) -> None:
                        if self._stop.is_set():
                            raise sd.CallbackStop
                        if status:
                            pass  # optional: log input overflow
                        f.write(indata.copy())

                    stream = sd.InputStream(
                        device=device,
                        channels=channels,
                        samplerate=samplerate,
                        dtype="float32",
                        blocksize=1024,
                        callback=callback,
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
