"""Health sensor interface.

Purpose:  Abstract power, temperature, and solar panel current sensors.
Inputs:   n/a
Outputs:  ``HealthInterface`` ABC.
Status:   [NEW] [BLOCKED: HW]
Port from: none

Consumed by ``executive/heartbeat.py`` to build ``HealthReport``.

Related open questions: HW (health sensors).
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from cloversat.datatypes.sample_types import Sample, SelfTestResult


class HealthInterface(ABC):
    """Power, temperature and panel-current sensing."""

    @abstractmethod
    def init(self) -> None:
        """Bring the device up. Must be safe to call more than once."""
        ...

    @abstractmethod
    def self_test(self) -> SelfTestResult:
        """Run the device's self test."""
        ...

    @abstractmethod
    def read_power(self) -> Sample[dict[str, float]]:
        """Return bus voltages [V] and currents [A], keyed by rail name."""
        ...

    @abstractmethod
    def read_temps(self) -> Sample[dict[str, float]]:
        """Return temperatures [K], keyed by sensor name."""
        ...

    @abstractmethod
    def read_panel_currents(self) -> Sample[dict[str, float]]:
        """Return solar panel currents [A], keyed by panel name."""
        ...
