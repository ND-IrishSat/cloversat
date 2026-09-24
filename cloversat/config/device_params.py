"""Device parameters: pins, bus addresses, sample rates, calibration tables.

Purpose:  Hardware wiring and per-device settings, consumed by drivers.
Inputs:   n/a
Outputs:  ``DeviceParams`` and one sub-dataclass per device type.
Status:   [NEW] [BLOCKED: HW]
Port from: legacy/HardwareInterface/magnetorquers.py (hardcoded pins),
           legacy/HardwareInterface/reaction_wheels/pcb_config.py,
           legacy/HardwareInterface/vn100/vn100_interface.py (port, baud)

Calibration tables live here but are *applied* in ``estimation/``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Vn100Params:
    """Testbed VN-100 IMU."""

    port: Optional[str] = None
    baud: Optional[int] = None
    sample_rate_hz: Optional[float] = None


@dataclass
class TorquerDriverParams:
    """Torquer H-bridge wiring."""

    pwm_channels: Optional[list[int]] = None
    direction_pins: Optional[list[int]] = None
    pwm_frequency_hz: Optional[float] = None


@dataclass
class WheelDriverParams:
    """Wheel motor driver wiring. [BLOCKED: OQ-3]"""

    pwm_channels: Optional[list[int]] = None
    direction_pins: Optional[list[int]] = None
    hall_pins: Optional[list[int]] = None


@dataclass
class WatchdogParams:
    """Hardware watchdog + RTC."""

    device_path: str = "/dev/watchdog"
    timeout_s: Optional[float] = None


@dataclass
class StorageParams:
    """Triple-copy persistence storage."""

    copy_paths: Optional[list[str]] = None


@dataclass
class MagCalibration:
    """Magnetometer calibration (applied in estimation/mag_conditioning)."""

    hard_iron_body: Optional[list[float]] = None
    soft_iron: Optional[list[list[float]]] = None


@dataclass
class DeviceParams:
    """All device settings."""

    vn100: Vn100Params = field(default_factory=Vn100Params)
    torquers: TorquerDriverParams = field(default_factory=TorquerDriverParams)
    wheels: WheelDriverParams = field(default_factory=WheelDriverParams)
    watchdog: WatchdogParams = field(default_factory=WatchdogParams)
    storage: StorageParams = field(default_factory=StorageParams)
    mag_calibration: MagCalibration = field(default_factory=MagCalibration)
