"""Spacecraft clock: monotonic time, RTC, and GPS time sync.

Purpose:  Single source of time for flight code. Tracks the offset between the
          monotonic clock, the RTC, and GPS/UTC time.
Inputs:   Monotonic time [s], RTC time [s] (``devices/interfaces/watchdog_interface``),
          GPS time from ``GpsFix`` samples.
Outputs:  Current UTC time [s since J2000 or Unix epoch -- pick one and document],
          time-validity flag.
Status:   [NEW]
Port from: none (legacy used sim time only; see legacy/params.py ``curr_date_time``)

Related open questions: OQ-8 (no uplink -> no ground time correction),
HW (flight computer RTC).

Notes:
    - GPS time is ahead of UTC by the leap-second offset; keep that offset as a
      param, not a hardcoded constant.
    - Before first GPS sync, time comes from the RTC restored at boot and is
      flagged degraded.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from cloversat.config.estimation_params import EstimationParams


class Clock:
    """Maps monotonic time to UTC using RTC and GPS sync."""

    def __init__(self, params: EstimationParams) -> None:
        raise NotImplementedError

    def now_monotonic(self) -> float:
        """Monotonic time [s]. Never jumps."""
        raise NotImplementedError

    def now_utc(self) -> float:
        """Best estimate of UTC [s]."""
        raise NotImplementedError

    def set_from_rtc(self, t_rtc: float, t_mono: float) -> None:
        """Seed the UTC offset from the RTC reading taken at ``t_mono``."""
        raise NotImplementedError

    def sync_gps(self, t_gps: float, t_mono: float) -> None:
        """Correct the UTC offset from a GPS time taken at ``t_mono`` (GPS->UTC applied)."""
        raise NotImplementedError

    def gps_to_utc(self, t_gps: float) -> float:
        """Convert GPS time to UTC using the leap-second offset."""
        raise NotImplementedError

    def time_valid(self) -> bool:
        """True once time has been synced from GPS at least once."""
        raise NotImplementedError

    def last_sync_age(self, t_mono: float) -> Optional[float]:
        """Seconds since the last GPS sync, or None if never synced."""
        raise NotImplementedError
