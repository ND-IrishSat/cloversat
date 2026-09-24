"""Cylindrical Earth-shadow model.

Purpose:  Decide whether the spacecraft is in eclipse, and predict the next
          entry/exit.
Inputs:   ``r_eci`` [m], sun direction ``s_eci`` (unit), orbit propagator for
          prediction.
Outputs:  In-eclipse flag; next entry/exit times [s].
Status:   [NEW]
Port from: none

Used by ``sun_conditioning`` (reject sun samples in eclipse), the estimator
(``StateEstimate.in_eclipse``), and ``operational_mode`` (eclipse hold).
"""
from __future__ import annotations

from typing import Callable, Optional

from cloversat.datatypes.vector_types import Vec3


def in_eclipse(r_eci: Vec3, s_eci: Vec3) -> bool:
    """True if ``r_eci`` is inside Earth's cylindrical shadow for sun direction ``s_eci``."""
    raise NotImplementedError


def next_transitions(
    t_utc: float,
    propagate: Callable[[float], tuple[Vec3, Vec3]],
    sun_direction: Callable[[float], Vec3],
    horizon_s: float,
    step_s: float,
) -> tuple[Optional[float], Optional[float]]:
    """Return ``(t_next_entry, t_next_exit)`` within ``horizon_s``, None if not found."""
    raise NotImplementedError
