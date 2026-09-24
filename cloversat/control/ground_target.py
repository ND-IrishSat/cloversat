"""Ground-station pointing target.

Purpose:  Attitude that points the comm patch at a ground station, with the
          tracking rate needed to follow it during a pass.
Inputs:   Position/velocity ``r_eci``, ``v_eci`` [m, m/s], station direction
          and visibility (estimation/station_model), comm patch normal
          (``SpacecraftParams``), secondary pair to fix roll.
Outputs:  ``AttitudeTarget`` including ``omega_target_body``.
Status:   [NEW] [BLOCKED: HW]
Port from: none

Related open questions: OQ-6, OQ-8 (comms), HW (comm patch placement).
"""
from __future__ import annotations

from cloversat.datatypes.state_types import AttitudeTarget
from cloversat.datatypes.vector_types import Vec3


def ground_target(
    station_dir_eci: Vec3,
    station_rate_eci: Vec3,
    comm_patch_normal_body: Vec3,
    secondary_body: Vec3,
    secondary_eci: Vec3,
) -> AttitudeTarget:
    """Target attitude: comm patch -> station, plus tracking rate.

    Args:
        station_dir_eci: Unit vector spacecraft -> station, ECI.
        station_rate_eci: Angular rate of that line of sight [rad/s], ECI.
        comm_patch_normal_body: Comm patch boresight (unit), body.
        secondary_body: Secondary body axis to constrain roll.
        secondary_eci: Where that secondary axis should point, ECI.
    """
    raise NotImplementedError
