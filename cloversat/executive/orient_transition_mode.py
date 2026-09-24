"""ORIENT_TRANSITION mode.

Purpose:  Deploy if not yet deployed, acquire initial attitude, get a GPS fix.
          Exit to OPERATIONAL once attitude and orbit are valid.
Inputs:   ``StateEstimate`` (validity flags), ``PersistedState.deployed``,
          ``DeploymentSequence``.
Outputs:  ``ModeCommand``; next mode OPERATIONAL (or SAFE on failure).
Status:   [NEW] [BLOCKED: OQ-5]
Port from: none

Related open questions: OQ-5 (initial attitude acquisition), HW (deployment
mechanism, GPS receiver).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from cloversat.datatypes.command_types import Mode, ModeCommand
from cloversat.executive.state_manager import ModeHandler

if TYPE_CHECKING:
    from cloversat.config.executive_params import ExecutiveParams
    from cloversat.datatypes.state_types import StateEstimate
    from cloversat.executive.deployment_sequence import DeploymentSequence
    from cloversat.executive.fdir_checks import FdirFlags


class OrientTransitionMode(ModeHandler):
    """ORIENT_TRANSITION: deploy, acquire attitude, get a GPS fix."""

    mode = Mode.ORIENT_TRANSITION

    def __init__(self, params: ExecutiveParams, deployment: DeploymentSequence) -> None:
        raise NotImplementedError

    def enter(self, t: float) -> None:
        raise NotImplementedError

    def step(self, estimate: StateEstimate, flags: FdirFlags, t: float) -> ModeCommand:
        raise NotImplementedError

    def next_mode(self, estimate: StateEstimate, flags: FdirFlags, t: float) -> Optional[Mode]:
        raise NotImplementedError

    def exit(self, t: float) -> None:
        raise NotImplementedError
