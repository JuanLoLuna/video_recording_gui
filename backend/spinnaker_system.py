"""Shared ownership of the process-wide Spinnaker System.

PySpin.System is a process-wide singleton. With one camera, one controller
could safely GetInstance() on start and ReleaseInstance() on stop (and in the
middle of fault recovery). With two controllers, camera A releasing -- or
rebuilding -- the System during its own stop()/reinit pulls it out from under
camera B while B's acquisition thread is still inside native code: a
use-after-free, not a catchable Python exception.

SharedSystemHolder reference-counts it instead. Controllers acquire() once
when they start and release() once when they have genuinely torn down; the
real ReleaseInstance() happens only when the last owner lets go. A controller
whose teardown was deferred (a stuck acquisition thread -- see
backend/teardown.py) simply never releases, so the System is never freed
under a handle that might still be in use.

PySpin is imported lazily, so this module (and its tests) loads anywhere.
"""

from __future__ import annotations

import threading
from typing import Callable


def _pyspin_get_instance():
    import PySpin

    return PySpin.System.GetInstance()


class SharedSystemHolder:
    """Reference-counted System, with references tracked PER OWNER.

    Tracking owners (not just a count) matters because the real stop() path
    runs more than once on the same controller -- closeEvent, preview-stop,
    a group stopping a camera that never started, start()'s own failure
    cleanup. A bare counter would let a second stop() of camera B release the
    reference camera A is still holding (A's teardown was deferred), freeing
    the System under A's stuck thread. With owners, release() is idempotent
    per owner and can never drop someone else's reference.

    Lock rule for callers: never take a controller lock while holding this
    one -- acquire()/release()/restart_if_sole_owner() hold it across the
    native GetInstance()/ReleaseInstance() calls, so a GUI-thread read of
    .system / .count can block for that long.
    """

    def __init__(self, get_instance: Callable[[], object] | None = None) -> None:
        self._get_instance = get_instance or _pyspin_get_instance
        self._lock = threading.RLock()
        self._system = None
        self._owners: set[int] = set()
        self.last_release_error: str | None = None

    @property
    def count(self) -> int:
        with self._lock:
            return len(self._owners)

    @property
    def system(self):
        """The live System, or None when nobody holds it."""
        with self._lock:
            return self._system

    def holds(self, owner: object) -> bool:
        with self._lock:
            return id(owner) in self._owners

    def acquire(self, owner: object):
        """Take `owner`'s reference, creating the System if there isn't one.

        Idempotent: an owner that already holds a reference gets the System
        back without a second reference. If GetInstance() raises, nothing is
        changed. Also repairs the half-failed-restart state (an owner but no
        System) for whoever asks next.
        """
        with self._lock:
            if self._system is None:
                self._system = self._get_instance()
            self._owners.add(id(owner))
            return self._system

    def release(self, owner: object) -> bool:
        """Drop `owner`'s reference; ReleaseInstance() only when none remain.

        Returns True if this call actually released the System. Releasing for
        an owner that holds nothing is a no-op (False), so a repeated stop()
        is harmless. A ReleaseInstance() that raises is recorded in
        last_release_error and the System is still treated as gone, matching
        how the single-camera code always handled it (try/except: pass).
        """
        with self._lock:
            if id(owner) not in self._owners:
                return False
            self._owners.discard(id(owner))
            if self._owners:
                return False
            system, self._system = self._system, None
            if system is not None:
                try:
                    system.ReleaseInstance()
                except Exception as exc:
                    self.last_release_error = f"{exc.__class__.__name__}: {exc}"
            return True

    def restart_if_sole_owner(self, owner: object) -> bool:
        """Release and re-create the System, but only if `owner` is the ONLY holder.

        This is the full System -> CameraList -> Camera rebuild the single
        camera recovery path has always used (validated on the rig by real
        unplug/replug). With any other owner alive -- or if the caller holds
        no reference at all -- it refuses (False) and the caller must rebuild
        only its own camera instead.
        """
        with self._lock:
            if self._owners != {id(owner)}:
                return False
            old, self._system = self._system, None
            if old is not None:  # None = a previous restart failed half-way
                try:
                    old.ReleaseInstance()
                except Exception as exc:
                    self.last_release_error = f"{exc.__class__.__name__}: {exc}"
            # If this raises, the caller keeps its reference (still the only
            # owner, system None) and simply retries -- same as the backoff
            # loop that has always driven reinit.
            self._system = self._get_instance()
            return True


_default_holder: SharedSystemHolder | None = None
_default_lock = threading.Lock()


def default_holder() -> SharedSystemHolder:
    """The process-wide holder every CameraController shares by default."""
    global _default_holder
    with _default_lock:
        if _default_holder is None:
            _default_holder = SharedSystemHolder()
        return _default_holder
