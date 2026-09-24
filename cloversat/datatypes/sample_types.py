"""Device sample and status types.

Purpose:  The shape every driver returns: a raw SI value, a timestamp, and a
          validity flag. Also the device status enum and self-test result.
Inputs:   n/a
Outputs:  ``Sample``, ``DeviceStatus``, ``SelfTestResult``.
Status:   [NEW]
Port from: none

Drivers return raw SI values with timestamp and validity. Calibration happens
in ``estimation/``, not in drivers.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Generic, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class Sample(Generic[T]):
    """One timestamped device reading.

    Attributes:
        value: Reading in SI units (e.g. ``Vec3`` in rad/s or T).
        t: Monotonic time of the reading [s].
        valid: False if the driver knows the value is bad (timeout, CRC, range).
    """

    value: T
    t: float
    valid: bool


class DeviceStatus(Enum):
    """Coarse device state reported by drivers."""

    UNINITIALIZED = "uninitialized"
    OK = "ok"
    DEGRADED = "degraded"
    FAILED = "failed"
    SAFED = "safed"


@dataclass(frozen=True)
class SelfTestResult:
    """Result of ``Interface.self_test()``.

    Attributes:
        passed: True if the device passed.
        code: Device-specific error code, 0 when passed.
        detail: Short human-readable note for telemetry/logs.
    """

    passed: bool
    code: int = 0
    detail: str = ""
