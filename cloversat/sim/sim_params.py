"""Simulation-only parameters.

Purpose:  Truth, noise, and scenario settings. Flight params stay in
          ``cloversat/config/``; this holds only what the physics sim needs.
Inputs:   n/a
Outputs:  ``SimParams``.
Status:   [LATER]
Port from: legacy/params.py (INITIAL VALUES, ORBITAL DYNAMICS, SIM OPTIONS,
           SENSORS sections)
Related open questions: none

Legacy values for reference: ``QUAT_INITIAL``, ``VELOCITY_INITIAL``,
``ORBITAL_ELEMENTS`` (km), ``DT`` (0.1 s), ``SOLVER_METHOD``, ``SENSOR_NOISE``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from cloversat.datatypes.vector_types import Quat, Vec3


@dataclass
class SimParams:
    """All sim settings.

    Attributes:
        dt: Physics step [s].
        duration_s: Run length [s].
        epoch_utc: Sim start time (ISO 8601).
        orbital_elements: Initial elements (a [m], e, i, RAAN, argp, nu) [rad].
        q0: Initial true attitude, [w, x, y, z], body <- inertial.
        omega0_body: Initial true body rate [rad/s] (tip-off).
        sensor_noise: Enable sensor noise.
        gyro_noise_sd: Gyro noise SD [rad/s].
        gyro_bias0_body: Initial true gyro bias [rad/s].
        mag_noise_sd: Magnetometer noise SD [T].
        seed: RNG seed.
    """

    dt: Optional[float] = None
    duration_s: Optional[float] = None
    epoch_utc: Optional[str] = None
    orbital_elements: Optional[list[float]] = None
    q0: Optional[Quat] = None
    omega0_body: Optional[Vec3] = None
    sensor_noise: bool = True
    gyro_noise_sd: Optional[float] = None
    gyro_bias0_body: Optional[Vec3] = None
    mag_noise_sd: Optional[float] = None
    seed: Optional[int] = None
