"""Ground station direction and visibility.

Purpose:  Direction from spacecraft to a ground station and whether it is above
          the elevation mask.
Inputs:   Station geodetic position (lat [rad], lon [rad], alt [m]), ``r_eci``
          [m], UTC time [s], elevation mask [rad].
Outputs:  Unit direction ``d_eci`` to station, visibility flag, elevation [rad].
Status:   [NEW]
Port from: reference only -- legacy/params.py ``CURRENT_GPS_DATA`` (Stinson-Remick
           coordinates, degrees).

Related open questions: HW (comm patch placement), OQ-8 (comms).

Uses ``geometry/frames.py`` for geodetic -> ECEF -> ECI.
"""
from __future__ import annotations

from dataclasses import dataclass

from cloversat.datatypes.vector_types import Vec3


@dataclass(frozen=True)
class GroundStation:
    """A ground station.

    Attributes:
        name: Station name.
        lat: Geodetic latitude [rad].
        lon: Longitude [rad].
        alt: Height above ellipsoid [m].
        min_elevation: Elevation mask [rad].
    """

    name: str
    lat: float
    lon: float
    alt: float
    min_elevation: float


def station_direction_eci(station: GroundStation, r_eci: Vec3, t_utc: float) -> Vec3:
    """Unit vector from spacecraft to station in ECI."""
    raise NotImplementedError


def station_elevation(station: GroundStation, r_eci: Vec3, t_utc: float) -> float:
    """Elevation of the spacecraft as seen from the station [rad]."""
    raise NotImplementedError


def station_visible(station: GroundStation, r_eci: Vec3, t_utc: float) -> bool:
    """True if the spacecraft is above the station's elevation mask."""
    raise NotImplementedError
