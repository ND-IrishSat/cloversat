"""Sun-pointing target.

Purpose:  Attitude that puts the solar panel normal on the sun; the free roll
          about that axis aims a patch antenna at Earth.
Inputs:   Measured sun vector ``s_body`` (if valid) or predicted sun direction
          ``s_eci`` (estimation/sun_model), position ``r_eci`` [m], panel
          and patch normals (``SpacecraftParams``), current attitude ``q``.
Outputs:  ``AttitudeTarget``.
Status:   [NEW]
Port from: none

Primary: panel_normal_body -> sun. Secondary: patch normal -> nadir
(-r_eci / |r_eci|). Built with ``geometry.pointing.align``. Uses the
measured sun vector when valid, else the predicted one. Eclipse hold is the
operational mode's job, not this module's.

Related open questions: OQ-6, HW (panel/patch placement).
"""
from __future__ import annotations

from typing import Optional

from cloversat.datatypes.state_types import AttitudeTarget
from cloversat.datatypes.vector_types import Quat, Vec3


def sun_target(
    s_eci: Vec3,
    r_eci: Vec3,
    panel_normal_body: Vec3,
    patch_normal_body: Vec3,
    q: Quat,
    s_body_measured: Optional[Vec3] = None,
) -> AttitudeTarget:
    """Target attitude: panel normal -> sun, patch normal -> Earth (secondary).

    Args:
        s_eci: Predicted sun unit vector, ECI.
        r_eci: Spacecraft position [m], ECI.
        panel_normal_body: Solar panel normal (unit), body.
        patch_normal_body: Patch boresight to aim at Earth with the free roll (unit), body.
        q: Current attitude estimate (used to rotate a measured sun vector to ECI).
        s_body_measured: Valid measured sun unit vector in body, or None.

    Returns:
        Target with ``omega_target_body`` zero (inertial hold).
    """
    raise NotImplementedError
