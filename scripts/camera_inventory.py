#!/usr/bin/env python3
"""Read-only inventory of every Spinnaker camera visible to this machine.

Run ON THE RECORDING LAPTOP (needs PySpin):
    python scripts/camera_inventory.py

For each camera prints model/serial/interface and, for USB3 Vision cameras,
the negotiated link speed -- the thing Windows PnP can't tell you. A camera
that enumerates as HighSpeed (USB 2.0) instead of SuperSpeed (USB 3.x) has
~1/10 the bandwidth and cannot sustain 1280x1024 Mono8 at 30 fps reliably.

Does NOT call Init() or BeginAcquisition(): it only reads the transport-layer
nodemap, so it is safe to run while nothing else owns the cameras. Do not run
it while the recorder GUI is open (Spinnaker's System is a process singleton
and the cameras may be exclusively held).

Required throughput per camera = width * height * bytes_per_pixel * fps.
For this app (1280x1024 Mono8): 1.31 MB/frame -> ~39 MB/s at 30 fps,
~79 MB/s at 60 fps, ~131 MB/s at 100 fps.
"""
import sys

import PySpin

TL_NODES = [
    "DeviceVendorName",
    "DeviceModelName",
    "DeviceSerialNumber",
    "DeviceVersion",
    "DeviceType",
    "DeviceDisplayName",
    "DeviceCurrentSpeed",  # USB: LowSpeed/FullSpeed/HighSpeed/SuperSpeed
    "GevDeviceIPAddress",  # GigE only
    "GevDeviceMaxPacketSize",
]


def read_node(nodemap, name):
    node = nodemap.GetNode(name)
    if node is None:
        return None
    if not PySpin.IsReadable(node):
        return None
    try:
        ptr = PySpin.CEnumerationPtr(node)
        if PySpin.IsReadable(ptr):
            return ptr.GetCurrentEntry().GetSymbolic()
    except Exception:
        pass
    try:
        return PySpin.CStringPtr(node).GetValue()
    except Exception:
        pass
    try:
        return PySpin.CIntegerPtr(node).GetValue()
    except Exception:
        return "<unreadable>"


def main() -> int:
    system = PySpin.System.GetInstance()
    try:
        lib = system.GetLibraryVersion()
        print(f"Spinnaker {lib.major}.{lib.minor}.{lib.type}.{lib.build}")
        cams = system.GetCameras()
        try:
            print(f"Cameras found: {cams.GetSize()}\n")
            for i in range(cams.GetSize()):
                cam = cams[i]
                try:
                    tl = cam.GetTLDeviceNodeMap()
                    print(f"--- camera index {i} ---")
                    for name in TL_NODES:
                        value = read_node(tl, name)
                        if value is not None:
                            print(f"  {name:24s} {value}")
                finally:
                    del cam
        finally:
            cams.Clear()
    finally:
        system.ReleaseInstance()
    return 0


if __name__ == "__main__":
    sys.exit(main())
