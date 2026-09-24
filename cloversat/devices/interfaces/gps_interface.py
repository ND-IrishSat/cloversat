"""GPS receiver interface.

Purpose:  Abstract GPS receiver returning an ECEF fix and GPS time.
Inputs:   n/a
Outputs:  ``GpsFix`` dataclass, ``GpsInterface`` ABC;
          ``read_gps() -> Sample[GpsFix]``.
Status:   [NEW] [BLOCKED: HW]
Port from: none. Reference only: legacy/HardwareInterface/gps_interface.py (stub).

``GpsFix`` is defined here rather than in ``datatypes/`` because only the GPS
interface and ``estimation/clock.py`` / ``estimation/sgp4_propagator.py`` use it.

ECEF <-> geodetic conversion lives in ``geometry/frames.py`` (replaces pyproj).

# TODO(bug): legacy/HardwareInterface/gps_interface.py returns zeros from
#   ``generate_gps()`` and uses the deprecated ``pyproj.transform``. Do not port.

Related open questions: HW (GPS receiver).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from cloversat.datatypes.sample_types import Sample, SelfTestResult
from cloversat.datatypes.vector_types import Vec3


@dataclass(frozen=True)
class GpsFix:
    """One GPS solution.

    Attributes:
        r_ecef: Position [m].
        v_ecef: Velocity [m/s].
        gps_time: GPS time of the fix [s since GPS epoch].
        fix_quality: Receiver fix type (0 = none, higher is better).
        num_sats: Satellites used in the solution.
    """

    r_ecef: Vec3
    v_ecef: Vec3
    gps_time: float
    fix_quality: int
    num_sats: int = 0


class GpsInterface(ABC):
    """GPS receiver."""

    @abstractmethod
    def init(self) -> None:
        """Bring the device up. Must be safe to call more than once."""
        ...

    @abstractmethod
    def self_test(self) -> SelfTestResult:
        """Run the device's self test."""
        ...

    @abstractmethod
    def read_gps(self) -> Sample[GpsFix]:
        """Return the latest fix. ``valid`` is False without a fix."""
        ...
