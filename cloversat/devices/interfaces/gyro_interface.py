"""Gyroscope interface.

Purpose:  Abstract rate gyro. Real (``hardware/vn100_driver.py``) and sim
          implementations sit behind it.
Inputs:   n/a
Outputs:  ``GyroInterface`` ABC; ``read_gyro() -> Sample[Vec3]`` in rad/s, body frame.
Status:   [NEW]
Port from: none. Reference only: legacy/HardwareInterface/mpu9250/,
           legacy/HardwareInterface/bn055/ (older gyro/mag parts),
           legacy/HardwareInterface/new_sensor_tests/ (bench data).

Drivers return raw SI values (rad/s, not deg/s) with timestamp and validity.
Bias correction and filtering happen in ``estimation/gyro_conditioning.py``.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from cloversat.datatypes.sample_types import Sample, SelfTestResult
from cloversat.datatypes.vector_types import Vec3


class GyroInterface(ABC):
    """Rate gyro."""

    @abstractmethod
    def init(self) -> None:
        """Bring the device up. Must be safe to call more than once."""
        ...

    @abstractmethod
    def self_test(self) -> SelfTestResult:
        """Run the device's self test."""
        ...

    @abstractmethod
    def read_gyro(self) -> Sample[Vec3]:
        """Return body rates ``omega_body`` [rad/s], raw (uncalibrated)."""
        ...
