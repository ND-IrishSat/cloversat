"""Physical state estimate, persisted executive state, and attitude targets.

Purpose:  Data shapes passed between estimation, executive, and control.
Inputs:   n/a
Outputs:  ``StateEstimate``, ``PersistedState``, ``AttitudeTarget``.
Status:   [NEW]
Port from: none

"State" vs "mode": ``StateEstimate`` is the *physical* state (attitude, rates,
position). Operating modes live in ``command_types.Mode``. Flight code never
sees the true physical state; only ``cloversat/sim/`` has that.

Related open questions: OQ-1 (filter choice sets covariance shape).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import numpy.typing as npt

from cloversat.datatypes.command_types import Mode
from cloversat.datatypes.vector_types import Quat, Vec3


@dataclass
class StateEstimate:
    """Best estimate of the spacecraft's physical state at time ``t``.

    Attributes:
        t: Estimate time [s, monotonic].
        q: Attitude, [w, x, y, z], body <- inertial.
        omega_body: Body rates [rad/s].
        gyro_bias_body: Estimated gyro bias [rad/s].
        r_eci: Position [m].
        v_eci: Velocity [m/s].
        h_wheels_body: Total wheel angular momentum [N*m*s].
        in_eclipse: True if the spacecraft is in Earth's shadow.
        covariance: Filter covariance; shape set by the filter (OQ-1).
        attitude_valid: q is usable for pointing.
        rates_valid: omega_body is usable.
        orbit_valid: r_eci / v_eci are usable.
        wheels_valid: h_wheels_body is usable.
    """

    t: float
    q: Quat
    omega_body: Vec3
    gyro_bias_body: Vec3
    r_eci: Vec3
    v_eci: Vec3
    h_wheels_body: Vec3
    in_eclipse: bool
    covariance: Optional[npt.NDArray[np.float64]] = None
    attitude_valid: bool = False
    rates_valid: bool = False
    orbit_valid: bool = False
    wheels_valid: bool = False


@dataclass
class PersistedState:
    """Executive state that must survive a reset (triple-copy on disk).

    Attributes:
        mode: Last operating mode.
        deployed: Deployables confirmed deployed.
        deploy_attempts: Number of deploy attempts so far.
        t_saved: Time of the save [s, RTC].
        reset_count: Total resets since first boot.
        last_reset_cause: Reset cause string from boot.
        disabled_subsystems: Subsystems FDIR has disabled.
    """

    mode: Mode
    deployed: bool = False
    deploy_attempts: int = 0
    t_saved: float = 0.0
    reset_count: int = 0
    last_reset_cause: str = ""
    disabled_subsystems: list[str] = field(default_factory=list)


@dataclass
class AttitudeTarget:
    """Desired attitude and rate from a pointing target.

    Attributes:
        q_target: Desired attitude, [w, x, y, z], body <- inertial.
        omega_target_body: Desired body rate [rad/s].
        valid: False if the target could not be computed.
    """

    q_target: Quat
    omega_target_body: Vec3
    valid: bool
