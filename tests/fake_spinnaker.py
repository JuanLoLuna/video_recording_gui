"""Fake Spinnaker objects shared by the integration and GUI tests.

install_pyspin_stub() makes `import PySpin` work on a machine without it (just
enough pointer helpers for the code paths the fakes drive) and reports whether
the REAL module is present, in which case tests that need the stub must skip.
"""
import sys
import time
import types

import numpy as np

FPS = 30.0


def install_pyspin_stub() -> bool:
    """Returns True if the real PySpin is installed (and then touches nothing)."""
    try:
        import PySpin
    except ImportError:
        PySpin = types.ModuleType("PySpin")
        sys.modules["PySpin"] = PySpin
    if hasattr(PySpin, "System"):
        return True
    PySpin.CStringPtr = PySpin.CIntegerPtr = PySpin.CFloatPtr = lambda node: node
    PySpin.CBooleanPtr = PySpin.CEnumerationPtr = lambda node: node
    PySpin.IsReadable = lambda node: node is not None
    PySpin.IsWritable = lambda node: node is not None
    return False


class Node:
    def __init__(self, value):
        self.value = value

    def GetValue(self):
        return self.value


class Entry:
    """An enumeration entry (e.g. ExposureAuto -> "Off")."""

    def __init__(self, name):
        self.name = name

    def GetValue(self):
        return abs(hash(self.name)) % 1000

    def GetSymbolic(self):
        return self.name


# (min, max, value) for the numeric settings the app reads or writes; anything
# else a permissive map hands out is a generic 0..1e9 number.
_RANGES = {
    "AcquisitionFrameRate": (1.0, 170.0, 30.0),
    "ExposureTime": (10.0, 20000.0, 5000.0),
    "Gain": (0.0, 24.0, 0.0),
    "StreamBufferCountManual": (1, 6000, 100),
}


class SettingNode:
    """A camera setting that accepts every read and write the app makes."""

    def __init__(self, name):
        self.name = name
        low, high, value = _RANGES.get(name, (0, 10**9, 0))
        self.low, self.high, self.value = low, high, value
        self.entry = Entry("Off")

    def GetValue(self):
        return self.value

    def SetValue(self, value):
        self.value = value

    def GetMin(self):
        return self.low

    def GetMax(self):
        return self.high

    def GetEntryByName(self, name):
        return Entry(name)

    def SetIntValue(self, value):
        pass

    def GetCurrentEntry(self):
        return self.entry

    def GetEntries(self):
        return [Entry("Off"), Entry("Continuous")]


class NodeMap:
    """Known nodes return their value; with permissive=True every other name is
    a generic writable setting, which is what the controller's real
    _configure_camera_nodes() needs to run end to end against a fake camera."""

    def __init__(self, permissive=False, **values):
        self.values = values
        self.permissive = permissive

    def GetNode(self, name):
        if name in self.values:
            return Node(self.values[name])
        return SettingNode(name) if self.permissive else None


class FakeImage:
    def __init__(self, frame_id, height, width):
        self.frame_id = frame_id
        self.height, self.width = height, width

    def IsIncomplete(self):
        return False

    def GetChunkData(self):
        outer = self

        class Chunk:
            def GetFrameID(self):
                return outer.frame_id

            def GetTimestamp(self):
                return outer.frame_id * 33_333

        return Chunk()

    def GetNDArray(self):
        return np.full((self.height, self.width), self.frame_id % 251, dtype=np.uint8)

    def Release(self):
        pass


class FakeCamera:
    def __init__(self, serial, model, width, height):
        self.serial, self.model, self.width, self.height = serial, model, width, height
        self.streaming = False
        self._next_at = 0.0
        self._frame_id = 0

    def IsValid(self):
        return True

    def Init(self):
        pass

    def DeInit(self):
        pass

    def GetNodeMap(self):
        return NodeMap(permissive=True, Width=self.width, Height=self.height)

    def GetTLStreamNodeMap(self):
        return NodeMap(permissive=True)

    def GetTLDeviceNodeMap(self):
        return NodeMap(DeviceSerialNumber=self.serial, DeviceModelName=self.model,
                       DeviceVendorName="FAKE")

    def BeginAcquisition(self):
        self.streaming = True
        self._next_at = time.monotonic()

    def EndAcquisition(self):
        self.streaming = False

    def GetNextImage(self, timeout_ms):
        if not self.streaming:
            raise RuntimeError("not acquiring")
        self._next_at += 1.0 / FPS
        delay = self._next_at - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        self._frame_id += 1
        return FakeImage(self._frame_id, self.height, self.width)


class FakeCamList:
    def __init__(self, cameras):
        self.cameras = cameras

    def GetSize(self):
        return len(self.cameras)

    def __getitem__(self, i):
        return self.cameras[i]

    def GetBySerial(self, serial):
        for cam in self.cameras:
            if cam.serial == serial:
                return cam
        raise RuntimeError("not found")

    def Clear(self):
        pass


class FakeSystem:
    def __init__(self, cameras):
        self.cameras = cameras

    def GetCameras(self):
        return FakeCamList(self.cameras)

    def UpdateCameras(self):
        pass

    def ReleaseInstance(self):
        pass
