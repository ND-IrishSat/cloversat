"""Two-vector alignment.

Purpose:  Build the attitude that aligns a primary body axis with a primary
          reference direction, and gets a secondary body axis as close as
          possible to a secondary reference direction.
Inputs:   Unit-ish vectors: two in body frame, two in inertial frame.
Outputs:  ``Quat`` (body <- inertial).
Status:   [NEW]
Port from: none. See legacy/Utils/transformations.py ``vector_to_quaternion``
           for the old single-vector version (copied into
           ``geometry/quaternion.py``'s legacy block).

Used by every pointing target in ``control/*_target.py`` (e.g. panel normal ->
sun with a patch aimed at Earth in free roll).

Degenerate case: if the primary and secondary references (or body vectors)
are parallel, the secondary constraint is undefined. Fall back to any
perpendicular secondary and report it via the ``degenerate`` flag.

Related open questions: none.
"""
from __future__ import annotations

from cloversat.datatypes.vector_types import Quat, Vec3


def align(
    primary_body: Vec3,
    primary_ref: Vec3,
    secondary_body: Vec3,
    secondary_ref: Vec3,
) -> Quat:
    """Attitude that puts ``primary_body`` on ``primary_ref`` exactly and
    ``secondary_body`` as close as possible to ``secondary_ref``.

    Refs are in the inertial frame. Uses the degenerate fallback when the
    secondary pair is parallel to the primary pair.
    """
    raise NotImplementedError


def is_degenerate(primary: Vec3, secondary: Vec3, tol_rad: float = 1e-3) -> bool:
    """True if ``primary`` and ``secondary`` are within ``tol_rad`` of parallel."""
    raise NotImplementedError


def fallback_secondary(primary: Vec3) -> Vec3:
    """Any unit vector perpendicular to ``primary`` (deterministic choice)."""
    raise NotImplementedError
