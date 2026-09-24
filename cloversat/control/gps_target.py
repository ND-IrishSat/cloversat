"""GPS-pointing target.

Purpose:  Attitude that points the GPS patch at zenith for a fix.
Inputs:   Position ``r_eci`` [m], GPS patch normal (``SpacecraftParams``),
          a secondary pair (e.g. panel normal -> sun) to fix the roll.
Outputs:  ``AttitudeTarget``.
Status:   [NEW]
Port from: none

Primary: gps_patch_normal_body -> zenith (r_eci / |r_eci|).

Related open questions: OQ-6, HW (GPS patch placement).
"""
from __future__ import annotations

from cloversat.datatypes.state_types import AttitudeTarget
from cloversat.datatypes.vector_types import Vec3


def gps_target(
    r_eci: Vec3,
    gps_patch_normal_body: Vec3,
    secondary_body: Vec3,
    secondary_eci: Vec3,
) -> AttitudeTarget:
    """Target attitude: GPS patch -> zenith, secondary pair fixes roll."""
    raise NotImplementedError
