"""Raspberry Pi hardware watchdog and RTC driver.

Purpose:  Kicks the Pi's bcm2835_wdt through ``/dev/watchdog`` and reads the RTC.
Inputs:   ``WatchdogParams`` (device path, timeout).
Outputs:  ``WatchdogDriver`` (``WatchdogInterface``).
Status:   [NEW]
Port from: none

Close behaviour: with the kernel's default ``nowayout=0``, closing
``/dev/watchdog`` *disables* the watchdog only if the magic character ``'V'``
was written just before close; closing without it leaves the watchdog armed
and the board resets on timeout. Flight code must never write ``'V'`` except
on a deliberate, commanded shutdown; ``close()`` must not do it implicitly.
Check ``nowayout`` on the flight image (``nowayout=1`` ignores ``'V'``).

Only ``executive/watchdog_supervisor.py`` should call ``kick()``.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from cloversat.datatypes.sample_types import Sample, SelfTestResult
from cloversat.devices.interfaces.watchdog_interface import WatchdogInterface

if TYPE_CHECKING:
    from cloversat.config.device_params import WatchdogParams


class WatchdogDriver(WatchdogInterface):
    """``/dev/watchdog`` + RTC."""

    def __init__(self, params: WatchdogParams) -> None:
        raise NotImplementedError

    def init(self) -> None:
        """Open ``/dev/watchdog`` (arms it) and set the timeout if supported."""
        raise NotImplementedError

    def self_test(self) -> SelfTestResult:
        """Check the device is open and the RTC is readable."""
        raise NotImplementedError

    def kick(self) -> None:
        """Write a keepalive to the watchdog."""
        raise NotImplementedError

    def get_rtc_time(self) -> Sample[float]:
        """Return RTC time [s, UTC epoch]."""
        raise NotImplementedError

    def close(self, disarm: bool = False) -> None:
        """Close the device. Writes magic ``'V'`` first only if ``disarm`` is True."""
        raise NotImplementedError
