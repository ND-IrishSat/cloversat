"""Accelerometer interface.

Purpose:  Abstract accelerometer, used only for shock / deployment detection.
Inputs:   n/a
Outputs:  ``AccelInterface`` ABC; ``read_accel() -> Sample[Vec3]`` in m/s^2, body frame.
Status:   [LATER]
Port from: none. Reference only: legacy/HardwareInterface/vn100/vn100_interface.py (``read_accel``).
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from cloversat.datatypes.sample_types import Sample, SelfTestResult
from cloversat.datatypes.vector_types import Vec3


class AccelInterface(ABC):
    """Accelerometer (shock/deploy detection only)."""

    @abstractmethod
    def init(self) -> None:
        """Bring the device up. Must be safe to call more than once."""
        ...

    @abstractmethod
    def self_test(self) -> SelfTestResult:
        """Run the device's self test."""
        ...

    @abstractmethod
    def read_accel(self) -> Sample[Vec3]:
        """Return specific force ``a_body`` [m/s^2]."""
        ...
