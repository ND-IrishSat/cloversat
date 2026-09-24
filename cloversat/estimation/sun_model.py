"""Low-precision sun direction.

Purpose:  Predicted sun direction from time (e.g. Astronomical Almanac
          low-precision formula, ~0.01 deg).
Inputs:   UTC time [s].
Outputs:  ``s_eci`` unit vector (Earth -> Sun).
Status:   [NEW]
Port from: none

Used by ``sun_target`` when the measured sun vector is invalid, and by
``eclipse_model``.
"""
from __future__ import annotations

from cloversat.datatypes.vector_types import Vec3


def sun_direction_eci(t_utc: float) -> Vec3:
    """Unit vector from Earth to Sun in ECI at ``t_utc``."""
    raise NotImplementedError


def sun_position_eci(t_utc: float) -> Vec3:
    """Earth -> Sun position in ECI [m] (needed for eclipse geometry)."""
    raise NotImplementedError
