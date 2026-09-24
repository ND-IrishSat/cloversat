"""Sensor error models for simulation.

Purpose:  Turn true quantities into what a sensor would report: noise, bias,
          scale factor, misalignment, dropouts, injected faults. Feeds the
          ``devices/sim/*_sim.py`` implementations.
Inputs:   True vectors from ``environment_truth`` / ``dynamics``, ``SimParams``
          (noise and fault settings).
Outputs:  Corrupted SI values (rad/s, T, unit sun vector, GPS r/v).
Status:   [LATER]
Port from: legacy/Simulator/simulator.py (sensor noise via ``SENSOR_NOISE``),
           legacy/params.py (``SENSOR_MAGNETOMETER_SD``, ``SENSOR_GYROSCOPE_SD``)
Related open questions: HW (flight IMU, sun sensor, GPS receiver specs).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from cloversat.datatypes.vector_types import Vec3

if TYPE_CHECKING:
    from cloversat.sim.sim_params import SimParams


class SensorModels:
    """Stateful sensor corruption (biases drift, faults latch)."""

    def __init__(self, sim: SimParams) -> None:
        ...

    def gyro(self, omega_true_body: Vec3, t: float) -> Vec3:
        """Measured body rate [rad/s]."""
        raise NotImplementedError

    def mag(self, B_true_body: Vec3, t: float) -> Vec3:
        """Measured field [T], including torquer/wheel contamination if modeled."""
        raise NotImplementedError

    def sun(self, s_true_body: Vec3, in_eclipse: bool, t: float) -> Vec3:
        """Measured sun direction (unit)."""
        raise NotImplementedError

    def gps(self, r_true_ecef: Vec3, v_true_ecef: Vec3, t: float) -> tuple[Vec3, Vec3]:
        """Measured ``(r_ecef [m], v_ecef [m/s])``."""
        raise NotImplementedError

    def inject_fault(self, sensor: str, kind: str) -> None:
        """Latch a named fault on a sensor (for scenario tests)."""
        raise NotImplementedError
