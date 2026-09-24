"""Hardware watchdog and RTC interface.

Purpose:  Kick the hardware watchdog; read real-time clock.
Inputs:   n/a
Outputs:  ``WatchdogInterface`` ABC.
Status:   [NEW]
Port from: none

Only ``executive/watchdog_supervisor.py`` calls ``kick()``, and only when all
critical subsystems are healthy.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from cloversat.datatypes.sample_types import Sample, SelfTestResult


class WatchdogInterface(ABC):
    """Hardware watchdog + RTC."""

    @abstractmethod
    def init(self) -> None:
        """Open and arm the watchdog."""
        ...

    @abstractmethod
    def self_test(self) -> SelfTestResult:
        """Check the watchdog and RTC are reachable (does not trigger a reset)."""
        ...

    @abstractmethod
    def kick(self) -> None:
        """Reset the watchdog timer."""
        ...

    @abstractmethod
    def get_rtc_time(self) -> Sample[float]:
        """Return RTC time [s, UTC epoch]."""
        ...
