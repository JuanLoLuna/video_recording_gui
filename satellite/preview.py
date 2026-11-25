"""Slow live picture of the satellite camera for satellite_window.py.

Source: the node the receiver already pulls (sat_cam1 = the satellite's H264 node), through the SAME
gRPC connection ThalamusControl holds: stub.image(ImageRequest(node=NodeSelector(name=NODE), framerate=0)),
the call thalamus/image_viewer.py makes.  Each Image message of that stream is one H264 access unit
(Image.format == 12, Annex-B, data[0]).  Frames are only counted and scanned for an IDR NAL; at most
`max_fps` keyframes per second are decoded (PyAV, fresh decoder per keyframe, flushed), scaled to
`width` px, 8-bit grey.  Nothing here touches the storage node.

Threads: the coroutine that reads the stream runs on Thalamus' own asyncio thread (cheap work only);
decoding runs in a second thread; a supervisor thread (re)subscribes at most once per 5 s.
The GUI calls snapshot() (a dict copy under a lock)."""
from __future__ import annotations

import asyncio
import queue
import threading
import time
from typing import Callable, Optional

import preview_logic as pl

RESUBSCRIBE_S = 5.0


def decode_keyframe(data: bytes, width: int = 640):
    """One Annex-B IDR access unit -> (w, h, stride, grey bytes) scaled to at most `width` px wide,
    or None when nothing could be decoded. Needs PyAV and numpy."""
    import av
    import numpy as np
    ctx = av.CodecContext.create("h264", "r")
    frames = list(ctx.decode(av.Packet(data))) + list(ctx.decode(None))
    if not frames:
        return None
    f = frames[0]
    w = min(width, f.width)
    h = max(1, round(f.height * w / f.width))
    arr = f.reformat(width=w, height=h, format="gray").to_ndarray()
    stride = (w + 3) // 4 * 4            # QImage wants 32-bit aligned scanlines
    if stride != w:
        arr = np.pad(arr, ((0, 0), (0, stride - w)))
    return w, h, stride, arr.tobytes()


class PreviewWorker:
    def __init__(self, control, node: str, max_fps: float = 1.0, width: int = 640, *,
                 clock: Callable[[], float] = time.monotonic,
                 decode: Callable = decode_keyframe,
                 make_request: Optional[Callable] = None, enabled: bool = True):
        self.control, self.node, self.width = control, node, width
        self.clock, self.decode = clock, decode
        self.make_request = make_request or self._default_request
        self.limiter = pl.RateLimiter(max_fps)
        self.meter = pl.FpsMeter()
        self.lock = threading.Lock()
        self.error: Optional[str] = None       # why the stream failed (stream error, wrong format, decode)
        self.conn_error: Optional[str] = "receiver not connected"   # only while there is no connection
        self.enabled = enabled
        self._sub_t: Optional[float] = None    # when the current subscription was made
        self.picture = None            # (w, h, stride, bytes)
        self.picture_t: Optional[float] = None
        self.picture_seq = 0
        self.q: queue.Queue = queue.Queue(maxsize=1)
        self.stop_flag = threading.Event()
        self._fut = None
        self._sub_thread = None
        self._last_try = -1e9
        self.threads = []

    # -- lifecycle
    def start(self) -> None:
        for fn, name in ((self._supervise, "preview-supervisor"), (self._decode_loop, "preview-decoder")):
            t = threading.Thread(target=fn, daemon=True, name=name)
            t.start()
            self.threads.append(t)

    def stop(self) -> None:
        self.stop_flag.set()
        self._unsubscribe()

    def _unsubscribe(self) -> None:
        fut, self._fut = self._fut, None
        if fut is not None:
            fut.cancel()
        self._sub_t = None

    def set_enabled(self, on: bool) -> None:
        """Off = cancel the image subscription (no receive, no decode) and forget the picture;
        on = the supervisor subscribes again within a second."""
        with self.lock:
            self.enabled = bool(on)
            if not on:
                self.picture, self.picture_t, self.error = None, None, None
                self.meter = pl.FpsMeter()
        if not on:
            self._unsubscribe()
        else:
            self._last_try = -1e9

    # -- supervisor: (re)subscribe, rate limited
    def _supervise(self) -> None:
        while not self.stop_flag.wait(1.0):
            try:
                self.supervise_once(self.clock())
            except Exception as exc:  # never die silently
                self._set_error("preview supervisor: %s" % exc)

    def supervise_once(self, now: float) -> None:
        if not self.enabled:
            return
        c = self.control
        thread = getattr(c, "_thread", None)
        if not getattr(c, "connected", False) or thread is None or getattr(thread, "loop", None) is None:
            with self.lock:
                self.conn_error = "receiver not connected"
            return
        with self.lock:
            self.conn_error = None   # connected now: this text must never outlive the connection
        alive = self._fut is not None and not self._fut.done() and self._sub_thread is thread
        if alive:
            return
        if self._fut is not None and not self._fut.done():   # old connection: drop its subscription
            self._fut.cancel()
        if now - self._last_try < RESUBSCRIBE_S:
            return
        self._last_try = now
        self._sub_thread = thread
        with self.lock:
            self._sub_t = now
            self.error = None        # a new subscription starts without the old stream's error
        self._fut = asyncio.run_coroutine_threadsafe(self._consume(thread), thread.loop)

    def _default_request(self):
        from thalamus import thalamus_pb2  # type: ignore
        return thalamus_pb2.ImageRequest(node=thalamus_pb2.NodeSelector(name=self.node), framerate=0)

    async def _consume(self, thread) -> None:
        try:
            stream = thread.stub.image(self.make_request())
            async for img in stream:
                self.on_image(img.format, img.data[0] if img.data else b"", self.clock())
            self._set_error("stream ended (does node %r exist and is it running?)" % self.node)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            msg = getattr(exc, "details", lambda: None)() or str(exc) or type(exc).__name__
            self._set_error("stream error: %s" % str(msg)[:120])

    # -- per frame (runs on the Thalamus loop: keep it cheap)
    def on_image(self, fmt: int, data: bytes, now: float) -> None:
        with self.lock:
            if not self.enabled:
                return
            self.conn_error = None   # frames are flowing, so there is a connection
            self.meter.add(now)
            if fmt != pl.H264_FORMAT:
                self.error = "node %r delivers image format %s, not H264" % (self.node, fmt)
                return
            self.error = None
        if data and pl.is_keyframe(data) and self.limiter.allow(now):
            try:
                self.q.put_nowait(data)
            except queue.Full:   # decoder still busy: drop this one
                pass

    def _decode_loop(self) -> None:
        while not self.stop_flag.is_set():
            try:
                data = self.q.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                pic = self.decode(data, self.width)
            except Exception as exc:
                self._set_error("decode failed: %s: %s" % (type(exc).__name__, exc))
                continue
            if pic is None:
                self._set_error("keyframe produced no picture")
                continue
            with self.lock:
                self.picture, self.picture_t = pic, self.clock()
                self.picture_seq += 1

    def _set_error(self, msg: str) -> None:
        with self.lock:
            self.error = msg

    def reason(self, now: float) -> Optional[str]:
        """Why there is no (fresh) picture, from the real state; call with the lock held."""
        if not self.enabled:
            return "preview off"
        if self.conn_error:
            return self.conn_error
        if self.error:
            return self.error
        age = self.meter.age_s(now)
        if age is not None and age <= pl.NO_FRAMES_S:
            return None
        if self._sub_t is None:
            return "subscribing to %s" % self.node
        if now - self._sub_t > pl.NO_FRAMES_S:
            return "no frames from %s (satellite not reachable?)" % self.node
        return "waiting for frames from %s" % self.node

    def snapshot(self, now: Optional[float] = None) -> dict:
        now = self.clock() if now is None else now
        with self.lock:
            return {"enabled": self.enabled, "fps": self.meter.fps(now), "frame_age_s": self.meter.age_s(now),
                    "picture": self.picture, "picture_seq": self.picture_seq,
                    "picture_age_s": None if self.picture_t is None else now - self.picture_t,
                    "error": self.reason(now)}
