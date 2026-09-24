"""Reference geomagnetic field model.

Purpose:  Earth's magnetic field at a position and time, for comparison with
          the magnetometer (attitude determination, B-cross/B-dot targets).
Inputs:   ``r_eci`` [m], UTC time [s].
Outputs:  ``B_eci`` [T].
Status:   [NEW] [BLOCKED: OQ-2]
Port from: reference only -- legacy/Simulator/PySOL/wmm.py (WMM, not IGRF;
           outputs nT/µT in NED, needs frame + unit conversion).

Related open questions: OQ-2 (IGRF degree ~8 vs tilted dipole).

Later:
    - Configurable degree (``EstimationParams.igrf_degree``).
"""
from __future__ import annotations

# TODO(bug): legacy field values are mixed µT/T (e.g. params.CONSTANT_B_FIELD_MAG
# in µT, EARTH_MAGNETIC_FIELD_LEO in T). This module returns Tesla only.

from typing import TYPE_CHECKING

from cloversat.datatypes.vector_types import Vec3

if TYPE_CHECKING:
    from cloversat.config.estimation_params import EstimationParams


class FieldModel:
    """Reference magnetic field (IGRF or tilted dipole, per OQ-2)."""

    def __init__(self, params: EstimationParams) -> None:
        raise NotImplementedError

    def field_eci(self, r_eci: Vec3, t_utc: float) -> Vec3:
        """Return the reference field ``B_eci`` [T] at ``r_eci`` and ``t_utc``."""
        raise NotImplementedError


def dipole_field_eci(r_eci: Vec3, t_utc: float) -> Vec3:
    """Tilted-dipole field ``B_eci`` [T]. Candidate fallback for OQ-2."""
    raise NotImplementedError
