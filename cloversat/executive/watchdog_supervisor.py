"""Software watchdog supervisor.

Purpose:  Track per-subsystem check-ins (HEALTHY -> LATE -> STALLED -> FAILED)
          and kick the hardware watchdog only when all critical subsystems
          are healthy.
Inputs:   ``check_in(name, t)`` calls, ``ExecutiveParams.watchdog``,
          ``WatchdogInterface``.
Outputs:  ``SubsystemHealth`` per subsystem; hardware kicks.
Status:   [NEW]
Port from: none

Later:
    - Extra degrade actions.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cloversat.config.executive_params import ExecutiveParams
    from cloversat.datatypes.health_types import SubsystemHealth
    from cloversat.devices.interfaces.watchdog_interface import WatchdogInterface


class WatchdogSupervisor:
    """Per-subsystem deadline tracking in front of the hardware watchdog."""

    def __init__(self, params: ExecutiveParams, watchdog: WatchdogInterface) -> None:
        raise NotImplementedError

    def check_in(self, subsystem: str, t: float) -> None:
        """Record that ``subsystem`` ran at time ``t``."""
        raise NotImplementedError

    def update(self, t: float) -> dict[str, SubsystemHealth]:
        """Advance each subsystem's state and return all of them."""
        raise NotImplementedError

    def kick_if_healthy(self, t: float) -> bool:
        """Kick the hardware watchdog if no critical subsystem is STALLED/FAILED."""
        raise NotImplementedError
