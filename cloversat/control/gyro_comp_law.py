"""Gyroscopic compensation torque.

Purpose:  Feed-forward term that cancels gyroscopic coupling.
Inputs:   Body rate ``omega_body`` [rad/s], inertia ``I_body`` [kg*m^2],
          wheel momentum ``h_wheels_body`` [N*m*s].
Outputs:  Torque ``tau_body`` [N*m] to add to the PD torque.
Status:   [LATER]
Port from: none

Law:  tau = omega x (I omega + h)
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt

from cloversat.datatypes.vector_types import Vec3


def gyro_comp_torque(omega_body: Vec3, I_body: npt.NDArray[np.float64], h_wheels_body: Vec3) -> Vec3:
    """Gyroscopic compensation ``omega x (I omega + h)`` [N*m]."""
    raise NotImplementedError
