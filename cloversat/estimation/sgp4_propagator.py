"""Orbit propagation from a preloaded TLE, re-seeded by GPS.

Purpose:  Position/velocity in ECI at any time. SGP4 from a TLE loaded before
          launch; when a GPS fix is available, re-seed the propagation from it.
Inputs:   TLE lines, UTC time [s], optional GPS fix (r, v in ECEF, time).
Outputs:  ``r_eci`` [m], ``v_eci`` [m/s], validity/degraded flag.
Status:   [NEW]
Port from: none (legacy used PySOL truth orbit: legacy/Simulator/PySOL/sol_sim.py)

Related open questions: OQ-8 (no uplink -> TLE cannot be refreshed; TLE-only
propagation is *degraded* and accuracy decays with age).

Notes:
    - Uses the ``sgp4`` package (reference implementation). SGP4 outputs TEME
      in km; convert to ECI (document which) and meters here.
    - ECEF->ECI for GPS re-seeding uses ``geometry/frames.py``.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from cloversat.datatypes.vector_types import Vec3

if TYPE_CHECKING:
    from cloversat.config.estimation_params import EstimationParams


class Sgp4Propagator:
    """TLE-based orbit propagator with GPS re-seeding."""

    def __init__(self, params: EstimationParams, tle_line1: str, tle_line2: str) -> None:
        raise NotImplementedError

    def propagate(self, t_utc: float) -> tuple[Vec3, Vec3]:
        """Return ``(r_eci, v_eci)`` [m, m/s] at ``t_utc``."""
        raise NotImplementedError

    def reseed_from_gps(self, r_ecef: Vec3, v_ecef: Vec3, t_utc: float) -> None:
        """Replace the propagation seed with a GPS state vector."""
        raise NotImplementedError

    def is_degraded(self, t_utc: float) -> bool:
        """True when running on TLE only, or the last GPS seed is too old."""
        raise NotImplementedError
