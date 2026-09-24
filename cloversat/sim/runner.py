"""Simulation runner: steps physics and runs the real flight code through sim devices.

Purpose:  Closed-loop sim. Each tick: advance truth (``dynamics``,
          ``orbit_truth``, ``environment_truth``), let ``devices/sim/*`` read
          truth through ``sensor_models``, run the unmodified flight tick
          (estimation -> state manager -> control), apply actuator outputs back
          to truth. Records truth and flight outputs for ``tools/analysis``.
Inputs:   ``FlightConfig`` (from ``config/sim_profile.py``), ``SimParams``.
Outputs:  ``SimResult`` (time series of truth + flight state).
Status:   [LATER]
Port from: legacy/Simulator/simulator.py, legacy/Main/SimScripts/space_sim.py
Related open questions: none

Known legacy bugs (do not carry over):
    TODO(bug): ``simulator.check_state`` shares one if/elif chain between the
        torquer timer block and state checks, so with ``ACCURATE_MAG_READINGS``
        on the state checks never run. Detumble exit also uses the *true*
        state; flight code must only see ``StateEstimate``.
    TODO(bug): ``simulator.determine_attitude`` uses undefined ``self.B_true``
        (and would be sim truth leaking into estimation anyway).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from cloversat.config.flight_profile import FlightConfig
    from cloversat.sim.sim_params import SimParams


@dataclass
class SimResult:
    """Logged time series from one run.

    Attributes:
        truth: Per-tick truth records.
        flight: Per-tick flight outputs (StateEstimate, ModeCommand, ActuatorCommand).
    """

    truth: list[dict[str, Any]] = field(default_factory=list)
    flight: list[dict[str, Any]] = field(default_factory=list)


class SimRunner:
    """Owns truth models and the flight software instance for one run."""

    def __init__(self, config: FlightConfig, sim: SimParams) -> None:
        ...

    def step(self) -> None:
        """Advance truth by one physics step and run a flight tick when due."""
        raise NotImplementedError

    def run(self) -> SimResult:
        """Run to ``sim.duration_s`` and return the log."""
        raise NotImplementedError
