"""Magnetometer interface.

Purpose:  Abstract 3-axis magnetometer.
Inputs:   n/a
Outputs:  ``MagInterface`` ABC; ``read_mag() -> Sample[Vec3]`` in Tesla, body frame.
Status:   [NEW]
Port from: none. Reference only: legacy/HardwareInterface/mpu9250/
           (mag_calibration.py, mag_custom_cal.py), legacy/HardwareInterface/bn055/,
           legacy/HardwareInterface/new_sensor_tests/.

Drivers return Tesla (not µT). Hard/soft-iron calibration is applied in
``estimation/mag_conditioning.py``, which also only accepts samples the
actuation scheduler flags as clean (torquers off).
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from cloversat.datatypes.sample_types import Sample, SelfTestResult
from cloversat.datatypes.vector_types import Vec3


class MagInterface(ABC):
    """3-axis magnetometer."""

    @abstractmethod
    def init(self) -> None:
        """Bring the device up. Must be safe to call more than once."""
        ...

    @abstractmethod
    def self_test(self) -> SelfTestResult:
        """Run the device's self test."""
        ...

    @abstractmethod
    def read_mag(self) -> Sample[Vec3]:
        """Return the measured field ``B_body`` [T], raw (uncalibrated)."""
        ...
