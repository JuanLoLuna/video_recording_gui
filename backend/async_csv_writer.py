"""Write CSV rows off the calling thread, without ever blocking the producer.

Generalises the queue + daemon-writer-thread shape that
backend/preview_diagnostics.py pioneered for the per-second diagnostics
log, so it can also serve the per-frame metadata log (30 rows/s for up to
10 days) without either dropping rows or blocking the acquisition thread
on disk I/O.

Two modes, selected by drop_when_full:
  - drop_when_full=True  (diagnostics): a full queue means falling behind
    is fine to lose a sample of -- drop and count it.
  - drop_when_full=False (frame metadata): downstream tooling requires a
    dense 1..N record_frame_index with no gaps, so a full queue instead
    spills into an unbounded producer-owned overflow deque and is drained
    (in submission order) the next time submit() is called. This is safe
    because the overflow rate is bounded by the camera's frame rate, not
    by however slow the disk is: at 30 rows/s x ~90 bytes, even a full
    hour of total disk stall costs only ~10 MB of RAM.
"""

from __future__ import annotations

import csv
import queue
import threading
import time
from collections import deque
from pathlib import Path
from typing import Mapping, Sequence


class AsyncCsvWriter:
    _STOP = object()

    def __init__(
        self,
        fieldnames: Sequence[str],
        *,
        max_pending_rows: int = 120,
        flush_every_rows: int = 1,
        flush_every_seconds: float | None = None,
        drop_when_full: bool = True,
        thread_name: str = "async-csv-writer",
    ) -> None:
        self._fieldnames = list(fieldnames)
        self._max_pending_rows = max_pending_rows
        self._flush_every_rows = max(1, int(flush_every_rows))
        self._flush_every_seconds = flush_every_seconds
        self._drop_when_full = drop_when_full
        self._thread_name = thread_name

        self._queue: queue.Queue | None = None
        self._thread: threading.Thread | None = None
        self._path: Path | None = None
        self._last_error: str | None = None
        self._dropped_rows = 0

        self._overflow_lock = threading.Lock()
        self._overflow: deque = deque()
        self._overflow_high_water = 0

        self._opened_event = threading.Event()
        self._open_ok = False

        self._rows_submitted = 0
        self._rows_written = 0
        self._rows_flushed = 0

    # -- status -----------------------------------------------------------

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def path(self) -> Path | None:
        return self._path

    @property
    def last_error(self) -> str | None:
        return self._last_error

    @property
    def dropped_rows(self) -> int:
        return self._dropped_rows

    @property
    def rows_submitted(self) -> int:
        return self._rows_submitted

    @property
    def rows_written(self) -> int:
        return self._rows_written

    @property
    def rows_flushed(self) -> int:
        return self._rows_flushed

    @property
    def overflow_rows(self) -> int:
        with self._overflow_lock:
            return len(self._overflow)

    @property
    def overflow_high_water(self) -> int:
        return self._overflow_high_water

    # -- lifecycle ----------------------------------------------------------

    def start(self, path: str | Path) -> None:
        self.stop()
        self._path = Path(path)
        self._last_error = None
        self._dropped_rows = 0
        self._rows_submitted = 0
        self._rows_written = 0
        self._rows_flushed = 0
        with self._overflow_lock:
            self._overflow.clear()
        self._overflow_high_water = 0
        self._opened_event = threading.Event()
        self._open_ok = False
        self._queue = queue.Queue(maxsize=self._max_pending_rows)
        self._thread = threading.Thread(
            target=self._writer_loop,
            name=self._thread_name,
            daemon=True,
        )
        self._thread.start()

    def wait_until_open(self, timeout: float = 2.0) -> bool:
        """Block until the file header is written, or the open failed.

        Converts an async open failure (bad path, permissions) into a
        synchronous result the caller can act on immediately, rather than
        discovering it only after silently losing rows for the whole
        session.
        """
        self._opened_event.wait(timeout)
        return self._open_ok

    def submit(self, row: Mapping[str, object]) -> bool:
        if self._queue is None or not self.is_running:
            return False
        self._rows_submitted += 1

        if not self._drop_when_full:
            self._drain_overflow_into_queue()
            try:
                self._queue.put_nowait(dict(row))
                return True
            except queue.Full:
                with self._overflow_lock:
                    self._overflow.append(dict(row))
                    self._overflow_high_water = max(
                        self._overflow_high_water, len(self._overflow)
                    )
                return True

        try:
            self._queue.put_nowait(dict(row))
            return True
        except queue.Full:
            self._dropped_rows += 1
            return False

    def _drain_overflow_into_queue(self) -> None:
        with self._overflow_lock:
            while self._overflow:
                try:
                    self._queue.put_nowait(self._overflow[0])
                except queue.Full:
                    return
                self._overflow.popleft()

    def stop(self, timeout: float = 3.0) -> None:
        work_queue = self._queue
        worker = self._thread
        if work_queue is not None and worker is not None and worker.is_alive():
            # Drain any overflow with a blocking put -- shutdown is the one
            # place blocking the caller is correct, since nothing else is
            # racing to fill the queue at this point.
            with self._overflow_lock:
                pending_overflow = list(self._overflow)
                self._overflow.clear()
            for row in pending_overflow:
                try:
                    work_queue.put(row, timeout=timeout)
                except queue.Full:
                    self._last_error = "Overflow rows did not drain during shutdown."
                    break
            try:
                work_queue.put(self._STOP, timeout=timeout)
            except queue.Full:
                self._last_error = "Diagnostics queue did not drain during shutdown."
            worker.join(timeout=timeout)
            if worker.is_alive():
                self._last_error = "CSV writer thread did not stop cleanly."
        self._thread = None
        self._queue = None

    # -- worker thread ------------------------------------------------------

    def _writer_loop(self) -> None:
        assert self._path is not None
        assert self._queue is not None
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=self._fieldnames)
                writer.writeheader()
                handle.flush()
                self._open_ok = True
                self._opened_event.set()

                rows_since_flush = 0
                last_flush_at = time.monotonic()
                while True:
                    remaining = None
                    if self._flush_every_seconds is not None and rows_since_flush > 0:
                        elapsed = time.monotonic() - last_flush_at
                        remaining = max(0.0, self._flush_every_seconds - elapsed)
                    try:
                        item = self._queue.get(timeout=remaining)
                    except queue.Empty:
                        handle.flush()
                        self._rows_flushed += rows_since_flush
                        rows_since_flush = 0
                        last_flush_at = time.monotonic()
                        continue

                    try:
                        if item is self._STOP:
                            break
                        writer.writerow(
                            {field: item.get(field, "") for field in self._fieldnames}
                        )
                        self._rows_written += 1
                        rows_since_flush += 1
                        elapsed = time.monotonic() - last_flush_at
                        should_flush = rows_since_flush >= self._flush_every_rows or (
                            self._flush_every_seconds is not None
                            and elapsed >= self._flush_every_seconds
                        )
                        if should_flush:
                            handle.flush()
                            self._rows_flushed += rows_since_flush
                            rows_since_flush = 0
                            last_flush_at = time.monotonic()
                    finally:
                        self._queue.task_done()

                if rows_since_flush:
                    handle.flush()
                    self._rows_flushed += rows_since_flush
        except Exception as exc:
            self._last_error = f"{exc.__class__.__name__}: {exc}"
            self._opened_event.set()
