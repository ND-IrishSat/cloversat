"""True orbit for simulation.

Purpose:  Provide the true position/velocity of the spacecraft over time.
          Wraps legacy PySOL (``cloversat/sim/pysol/``) or a high-fidelity
          propagator; independent of the flight ``estimation/sgp4_propagator``
          so sim truth and flight estimate can disagree realistically.
Inputs:   Sim time [s], ``SimParams`` (initial orbital elements, epoch).
Outputs:  ``OrbitTruth.state_at(t) -> (r_eci, v_eci)`` [m, m/s].
Status:   [LATER]
Port from: legacy/Simulator/PySOL/sol_sim.py (``generate_orbit_data``),
           legacy/params.py (``ORBITAL_ELEMENTS``; note: legacy uses km)
Related open questions: none
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from cloversat.datatypes.vector_types import Vec3

if TYPE_CHECKING:
    from cloversat.sim.sim_params import SimParams


class OrbitTruth:
    """True orbit source for the sim."""

    def __init__(self, sim: SimParams) -> None:
        ...

    def state_at(self, t: float) -> tuple[Vec3, Vec3]:
        """True ``(r_eci [m], v_eci [m/s])`` at sim time ``t``."""
        raise NotImplementedError
