"""Read Windows power state and assess whether recording is safe to start."""

from __future__ import annotations

import sys
import ctypes
from dataclasses import dataclass


LOW_BATTERY_CONFIRM_PERCENT = 20
CRITICAL_BATTERY_PERCENT = 5


@dataclass(frozen=True)
class PowerStatus:
    supported: bool
    ac_online: bool | None = None
    battery_percent: int | None = None
    energy_saver_on: bool | None = None
    battery_low: bool = False
    battery_critical: bool = False
    error: str | None = None


@dataclass(frozen=True)
class PowerSafetyAssessment:
    level: str
    summary: str
    recording_blocked: bool
    requires_confirmation: bool
    reason: str | None = None


class _SystemPowerStatus(ctypes.Structure):
    _fields_ = [
        ("ACLineStatus", ctypes.c_ubyte),
        ("BatteryFlag", ctypes.c_ubyte),
        ("BatteryLifePercent", ctypes.c_ubyte),
        ("SystemStatusFlag", ctypes.c_ubyte),
        ("BatteryLifeTime", ctypes.c_uint32),
        ("BatteryFullLifeTime", ctypes.c_uint32),
    ]


def read_power_status() -> PowerStatus:
    """Return AC, battery, and Energy Saver state without external packages."""
    if not sys.platform.startswith("win"):
        return PowerStatus(supported=False)

    raw = _SystemPowerStatus()
    try:
        success = bool(ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(raw)))
    except Exception as exc:
        return PowerStatus(supported=True, error=f"{exc.__class__.__name__}: {exc}")
    if not success:
        return PowerStatus(
            supported=True,
            error="Windows could not read the current power status.",
        )

    ac_online = None if raw.ACLineStatus == 255 else raw.ACLineStatus == 1
    battery_percent = (
        None if raw.BatteryLifePercent == 255 else int(raw.BatteryLifePercent)
    )
    battery_flags_known = raw.BatteryFlag != 255
    return PowerStatus(
        supported=True,
        ac_online=ac_online,
        battery_percent=battery_percent,
        energy_saver_on=(
            None if raw.SystemStatusFlag not in (0, 1) else raw.SystemStatusFlag == 1
        ),
        battery_low=battery_flags_known and bool(raw.BatteryFlag & 2),
        battery_critical=battery_flags_known and bool(raw.BatteryFlag & 4),
    )


def assess_power_safety(
    status: PowerStatus,
    *,
    low_battery_percent: int = LOW_BATTERY_CONFIRM_PERCENT,
    critical_battery_percent: int = CRITICAL_BATTERY_PERCENT,
) -> PowerSafetyAssessment:
    """Classify power state for recording without changing system settings."""
    if not status.supported:
        return PowerSafetyAssessment(
            level="neutral",
            summary="Power safety: monitoring is available on Windows only",
            recording_blocked=False,
            requires_confirmation=False,
        )
    if status.error:
        return PowerSafetyAssessment(
            level="warning",
            summary="Power safety: status unavailable",
            recording_blocked=False,
            requires_confirmation=False,
            reason=status.error,
        )

    percent_text = (
        "unknown charge"
        if status.battery_percent is None
        else f"{status.battery_percent}% battery"
    )
    if status.energy_saver_on:
        return PowerSafetyAssessment(
            level="danger",
            summary=f"Power unsafe: Energy Saver is ON ({percent_text})",
            recording_blocked=True,
            requires_confirmation=False,
            reason=(
                "Energy Saver caused camera-frame loss in testing. "
                "Turn it off before recording."
            ),
        )

    on_battery = status.ac_online is not True
    is_critical = (
        status.battery_critical
        if status.battery_percent is None
        else status.battery_percent <= critical_battery_percent
    )
    if on_battery and is_critical:
        return PowerSafetyAssessment(
            level="danger",
            summary=f"Power unsafe: critically low ({percent_text})",
            recording_blocked=True,
            requires_confirmation=False,
            reason="Connect AC power before recording.",
        )

    is_low = (
        status.battery_low
        if status.battery_percent is None
        else status.battery_percent <= low_battery_percent
    )
    if on_battery and is_low:
        return PowerSafetyAssessment(
            level="warning",
            summary=f"Power warning: unplugged with {percent_text}",
            recording_blocked=False,
            requires_confirmation=True,
            reason=(
                "The battery is low. Connect AC power when possible; continuing "
                "could trigger Energy Saver or an abrupt shutdown."
            ),
        )
    if on_battery:
        return PowerSafetyAssessment(
            level="warning",
            summary=f"Power warning: unplugged ({percent_text}), Energy Saver off",
            recording_blocked=False,
            requires_confirmation=False,
            reason="AC power is recommended for reliable recording.",
        )

    saver_text = (
        "Energy Saver off"
        if status.energy_saver_on is False
        else "Energy Saver status unknown"
    )
    return PowerSafetyAssessment(
        level="safe" if status.energy_saver_on is False else "warning",
        summary=f"Power ready: AC connected, {saver_text}",
        recording_blocked=False,
        requires_confirmation=False,
    )
