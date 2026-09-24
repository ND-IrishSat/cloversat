"""Heartbeat.

Purpose:  Collect health once per second into a ``HealthReport``.
Inputs:   ``HealthInterface`` readings, watchdog supervisor subsystem health.
Outputs:  ``HealthReport``.
Status:   [NEW]
Port from: none

Related open questions: HW (health sensors).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cloversat.config.executive_params import ExecutiveParams
    from cloversat.datatypes.health_types import HealthReport
    from cloversat.devices.interfaces.health_interface import HealthInterface
    from cloversat.executive.watchdog_supervisor import WatchdogSupervisor


class Heartbeat:
    """Builds a ``HealthReport`` at ``heartbeat_period_s``."""

    def __init__(self, params: ExecutiveParams, health: HealthInterface, supervisor: WatchdogSupervisor) -> None:
        raise NotImplementedError

    def collect(self, t: float) -> HealthReport:
        """Read health sensors and subsystem states; return the report."""
        raise NotImplementedError
