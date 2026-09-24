"""TRIAD attitude determination.

Purpose:  Attitude from two vector pairs measured in body and known in the
          reference frame. The first (primary) pair is trusted more.
Inputs:   ``v1_body``, ``v1_eci``, ``v2_body``, ``v2_eci`` (any scale; normalized
          internally).
Outputs:  ``Quat`` body <- inertial, or None if the pairs are near-parallel.
Status:   [NEW]
Port from: none

Can share the core construction with ``geometry/pointing.align``.
"""
from __future__ import annotations

from typing import Optional

from cloversat.datatypes.vector_types import Quat, Vec3


def triad(
    v1_body: Vec3,
    v1_eci: Vec3,
    v2_body: Vec3,
    v2_eci: Vec3,
    min_angle: float = 0.0,
) -> Optional[Quat]:
    """Return q (body <- inertial), or None if the angle between pairs < ``min_angle`` [rad]."""
    raise NotImplementedError
