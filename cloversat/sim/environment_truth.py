"""True space environment for simulation.

Purpose:  True magnetic field, sun direction, eclipse state, and disturbance
          torques at a given time/position. Used to drive sensor models and
          dynamics; never seen by flight code.
Inputs:   Sim time [s], true ``r_eci`` [m], true attitude, ``SimParams``.
Outputs:  ``B_eci`` [T], ``s_eci`` (unit), eclipse flag, ``tau_dist_body`` [N*m].
Status:   [LATER]
Port from: legacy/Simulator/PySOL/wmm.py, legacy/Simulator/PySOL/sol_sim.py
           (legacy B field is in microtesla; truth here is tesla)
Related open questions: OQ-2 (flight field model; truth may use higher fidelity).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from cloversat.datatypes.vector_types import Quat, Vec3

if TYPE_CHECKING:
    from cloversat.sim.sim_params import SimParams


class EnvironmentTruth:
    """True B field, sun, eclipse, and disturbances."""

    def __init__(self, sim: SimParams) -> None:
        ...

    def mag_field_eci(self, t: float, r_eci: Vec3) -> Vec3:
        """True geomagnetic field [T]."""
        raise NotImplementedError

    def sun_direction_eci(self, t: float) -> Vec3:
        """True unit vector spacecraft -> sun."""
        raise NotImplementedError

    def in_eclipse(self, t: float, r_eci: Vec3) -> bool:
        """True if the spacecraft is in Earth's shadow."""
        raise NotImplementedError

    def disturbance_torque_body(self, t: float, r_eci: Vec3, q: Quat) -> Vec3:
        """Sum of modeled disturbance torques [N*m] (gravity gradient, drag, residual dipole)."""
        raise NotImplementedError
