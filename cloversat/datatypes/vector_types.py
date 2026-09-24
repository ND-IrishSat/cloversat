"""Vector and quaternion type aliases.

Purpose:  Common names for 3-vectors and quaternions so signatures read clearly.
Inputs:   n/a
Outputs:  ``Vec3``, ``Quat`` aliases.
Status:   [NEW]
Port from: none

Conventions (docs/conventions.md):
    - ``Vec3``: numpy float array, shape (3,), SI units. Name variables with a
      frame suffix (``B_body``, ``r_eci``).
    - ``Quat``: numpy float array, shape (4,), ``[w, x, y, z]`` scalar first,
      rotation body <- inertial.

Aliases (not dataclasses) so numpy math works directly on them. Shape is not
enforced by the type checker; validate at module boundaries.
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt

Vec3 = npt.NDArray[np.float64]
"""3-vector, shape (3,), SI units."""

Quat = npt.NDArray[np.float64]
"""Quaternion, shape (4,), [w, x, y, z], body <- inertial."""
