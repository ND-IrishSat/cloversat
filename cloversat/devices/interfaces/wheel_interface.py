"""Reaction wheel interface.

Purpose:  Abstract set of reaction wheels.
Inputs:   n/a
Outputs:  ``WheelInterface`` ABC.
Status:   [NEW] [BLOCKED: OQ-3]
Port from: none (driver: legacy/HardwareInterface/reaction_wheels/motors.py)

The meaning of ``cmd`` in ``set_wheel`` (torque, current, or speed setpoint)
is OQ-3. ``safe()`` must be callable at any time.

Related open questions: OQ-3 (wheel command interface), HW (wheel driver).
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from cloversat.datatypes.sample_types import Sample, SelfTestResult


class WheelInterface(ABC):
    """Reaction wheels, indexed 0..n-1."""

    @abstractmethod
    def init(self) -> None:
        """Bring the drivers up with all wheels stopped."""
        ...

    @abstractmethod
    def self_test(self) -> SelfTestResult:
        """Run the device's self test."""
        ...

    @abstractmethod
    def set_wheel(self, i: int, cmd: float) -> None:
        """Command wheel ``i``. Units depend on OQ-3."""
        ...

    @abstractmethod
    def get_wheel_rpm(self, i: int) -> Sample[float]:
        """Return wheel ``i`` speed [rpm, signed]. Convert with ``geometry/units.py``."""
        ...

    @abstractmethod
    def safe(self) -> None:
        """Stop driving all wheels. Callable anytime; must not raise."""
        ...
