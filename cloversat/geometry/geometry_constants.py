"""Physical constants for Earth geometry and orbits. Fixed forever.

Purpose:  Shared Earth constants in SI.
Inputs:   n/a
Outputs:  ``MU_EARTH``, ``R_EARTH``, ``OMEGA_EARTH``, ``J2``, WGS-84 shape.
Status:   [NEW]
Port from: legacy/params.py (``GRAVITY_EARTH``; ``EARTH_RADIUS`` was in km)

Values: WGS-84 / EGM96 (J2). Keep SI: m, s, rad.

Related open questions: none.
"""
from __future__ import annotations

MU_EARTH: float = 3.986004418e14
"""Earth gravitational parameter [m^3/s^2]."""

R_EARTH: float = 6378137.0
"""WGS-84 equatorial radius [m]."""

OMEGA_EARTH: float = 7.2921150e-5
"""Earth rotation rate [rad/s]."""

J2: float = 1.08262668e-3
"""Earth second zonal harmonic [-]."""

F_EARTH: float = 1.0 / 298.257223563
"""WGS-84 flattening [-]."""
