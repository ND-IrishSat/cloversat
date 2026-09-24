"""Spacecraft physical parameters.

Purpose:  Mass properties, geometry, mounting normals, and actuator specs.
Inputs:   n/a (values filled in once hardware is fixed)
Outputs:  ``SpacecraftParams``, ``TorquerSpec``, ``WheelSpec``.
Status:   [NEW] [BLOCKED: HW]
Port from: legacy/params.py (PHYSICS, REACTION WHEELS, MAGNETORQUERS sections)

Related open questions: HW (inertia/tip-off rates, patch placement,
wheel driver), OQ-3 (wheel command interface).

Legacy values for reference (not final): ``CUBESAT_BODY_INERTIA``,
``TRANSFORMATION`` (4-wheel NASA pyramid), ``RW_SPIN_AXIS_INERTIA``,
``RESISTANCE_MAG``, ``MAX_CURRENT_MAG``, ``AIR_NUM_TURNS``, ``AIR_AREA``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import numpy.typing as npt

from cloversat.datatypes.vector_types import Vec3


@dataclass
class TorquerSpec:
    """One magnetorquer.

    Attributes:
        axis_body: Dipole direction (unit vector).
        dipole_per_amp: Dipole per unit current [A*m^2/A].
        resistance: Coil resistance [ohm].
        max_voltage: Drive limit [V].
        max_current: Current limit [A].
    """

    axis_body: Optional[Vec3] = None
    dipole_per_amp: Optional[float] = None
    resistance: Optional[float] = None
    max_voltage: Optional[float] = None
    max_current: Optional[float] = None


@dataclass
class WheelSpec:
    """One reaction wheel.

    Attributes:
        spin_axis_inertia: Rotor inertia about spin axis [kg*m^2].
        max_torque: Torque limit [N*m].
        max_speed: Speed limit [rad/s].
        torque_constant: Motor Kt [N*m/A].
    """

    spin_axis_inertia: Optional[float] = None
    max_torque: Optional[float] = None
    max_speed: Optional[float] = None
    torque_constant: Optional[float] = None


@dataclass
class SpacecraftParams:
    """Everything about the physical spacecraft.

    Attributes:
        inertia_body: Body inertia tensor, wheels excluded [kg*m^2], (3, 3).
        dimensions: Outer dimensions [m], (3,).
        panel_normal_body: Primary solar panel normal (unit).
        comm_patch_normal_body: Comm antenna patch boresight (unit).
        gps_patch_normal_body: GPS antenna patch boresight (unit).
        sun_sensor_normal_body: Sun sensor boresight (unit).
        torquers: One spec per torquer.
        wheels: One spec per wheel.
        wheel_matrix: Spin axes as columns, (3, n_wheels).
        residual_dipole_body: Known residual dipole [A*m^2].
    """

    inertia_body: Optional[npt.NDArray[np.float64]] = None
    dimensions: Optional[Vec3] = None
    panel_normal_body: Optional[Vec3] = None
    comm_patch_normal_body: Optional[Vec3] = None
    gps_patch_normal_body: Optional[Vec3] = None
    sun_sensor_normal_body: Optional[Vec3] = None
    torquers: Optional[list[TorquerSpec]] = None
    wheels: Optional[list[WheelSpec]] = None
    wheel_matrix: Optional[npt.NDArray[np.float64]] = None
    residual_dipole_body: Optional[Vec3] = None
