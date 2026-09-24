"""Magnetorquer interface.

Purpose:  Abstract set of magnetorquers driven by voltage per axis.
Inputs:   n/a
Outputs:  ``TorquerInterface`` ABC.
Status:   [NEW]
Port from: none (driver: legacy/HardwareInterface/magnetorquers.py)

Only ``control/actuation_scheduler.py`` calls this interface; it owns the
torquers so it can guarantee clean magnetometer windows.
``safe()`` must be callable at any time, from any state.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from cloversat.datatypes.sample_types import Sample, SelfTestResult


class TorquerInterface(ABC):
    """Magnetorquers, one per axis."""

    @abstractmethod
    def init(self) -> None:
        """Bring the drivers up with all coils off."""
        ...

    @abstractmethod
    def self_test(self) -> SelfTestResult:
        """Run the device's self test."""
        ...

    @abstractmethod
    def set_torquer(self, axis: int, volts: float) -> None:
        """Drive torquer ``axis`` (0, 1, 2) at signed ``volts`` [V]."""
        ...

    def read_current(self, axis: int) -> Optional[Sample[float]]:
        """Return coil current [A] if the hardware measures it, else None."""
        return None

    @abstractmethod
    def safe(self) -> None:
        """Turn all coils off. Callable anytime; must not raise."""
        ...
