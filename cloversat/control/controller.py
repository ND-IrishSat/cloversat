"""Per-mode control dispatch.

Purpose:  Pick the control law and pointing target for the active mode and
          produce one ``ActuatorCommand`` per tick.
Inputs:   ``ModeCommand`` (from executive/state_manager), ``StateEstimate``,
          reference field ``B_body`` (from estimation), ``ControlParams``,
          ``SpacecraftParams``.
Outputs:  ``ActuatorCommand`` (wheel torque + dipole, body frame, SI).
Status:   [NEW]
Port from: legacy/Simulator/simulator.py (``Simulator.controls`` mode branches)

Dispatch (v1):
    DETUMBLE            -> bcross_law
    ORIENT_TRANSITION   -> bcross_law / attitude_pd_law (OQ-5)
    OPERATIONAL         -> target (sun/gps/ground) + attitude_pd_law,
                           momentum_dump_law in MOMENTUM_DUMP
    SAFE                -> bcross_law, then sun_target hold (OQ-7)
    BOOT                -> zero command

Allocation (wheel_allocation, torquer_allocation) and torquer timing
(actuation_scheduler) happen after this, not here.

Related open questions: OQ-5, OQ-6, OQ-7.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from cloversat.datatypes.command_types import ActuatorCommand, ModeCommand
from cloversat.datatypes.state_types import AttitudeTarget, StateEstimate
from cloversat.datatypes.vector_types import Vec3

if TYPE_CHECKING:
    from cloversat.config.control_params import ControlParams
    from cloversat.config.spacecraft_params import SpacecraftParams


class Controller:
    """Chooses law + target from the ``ModeCommand`` and returns an ``ActuatorCommand``."""

    def __init__(self, params: ControlParams, spacecraft: SpacecraftParams) -> None:
        """Store params. No hardware access."""
        raise NotImplementedError

    def step(self, command: ModeCommand, estimate: StateEstimate, B_body: Vec3) -> ActuatorCommand:
        """Run one control tick.

        Args:
            command: Mode, sub-mode and target from the state manager.
            estimate: Current state estimate.
            B_body: Conditioned magnetic field in body frame [T].

        Returns:
            Wheel torque and dipole request for this tick.
        """
        raise NotImplementedError

    def zero_command(self, source: str) -> ActuatorCommand:
        """Return an all-zero ``ActuatorCommand`` tagged with ``source``."""
        raise NotImplementedError

    def resolve_target(self, command: ModeCommand, estimate: StateEstimate) -> AttitudeTarget:
        """Compute the pointing target for ``command.submode`` (sun/gps/ground)."""
        raise NotImplementedError
