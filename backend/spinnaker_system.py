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
    def __init__(self, get_instance: Callable[[], object] | None = None) -> None:
        self._get_instance = get_instance or _pyspin_get_instance
        self._lock = threading.RLock()
        self._system = None
        self._count = 0
        self.last_release_error: str | None = None

    @property
    def count(self) -> int:
        with self._lock:
            return self._count

    @property
    def system(self):
        """The live System, or None when nobody holds it."""
        with self._lock:
            return self._system

    def acquire(self):
        """Take a reference, creating the System on the first one.

        If GetInstance() raises, the count is left unchanged.
        """
        with self._lock:
            if self._count == 0:
                self._system = self._get_instance()
            self._count += 1
            return self._system

    def release(self) -> bool:
        """Drop a reference; ReleaseInstance() only when the last one goes.

        Returns True if this call actually released the System. A release with
        no matching acquire is ignored (returns False) rather than driving the
        count negative. A ReleaseInstance() that raises is recorded in
        last_release_error and the System is still treated as gone, matching
        how the single-camera code always handled it (try/except: pass).
        """
        with self._lock:
            if self._count == 0:
                return False
            self._count -= 1
            if self._count > 0:
                return False
            system, self._system = self._system, None
            if system is not None:
                try:
                    system.ReleaseInstance()
                except Exception as exc:
                    self.last_release_error = f"{exc.__class__.__name__}: {exc}"
            return True

    def restart_if_sole_owner(self) -> bool:
        """Release and re-create the System, but only if nobody else holds it.

        This is the full System -> CameraList -> Camera rebuild the single
        camera recovery path has always used (validated on the rig by real
        unplug/replug). With any other owner alive it refuses -- returns False
        -- and the caller must rebuild only its own camera instead.
        """
        with self._lock:
            if self._count != 1:
                return False
            old, self._system = self._system, None
            if old is not None:  # None = a previous restart failed half-way
                try:
                    old.ReleaseInstance()
                except Exception as exc:
                    self.last_release_error = f"{exc.__class__.__name__}: {exc}"
            # If this raises, the caller keeps its reference (count stays 1,
            # system stays None) and simply retries -- same as the backoff
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
