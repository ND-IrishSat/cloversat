"""DETUMBLE mode.

Purpose:  Torquers only (B-cross law). Exit when sensed rates stay below the
          threshold for N seconds.
Inputs:   ``StateEstimate`` (``omega_body``, ``rates_valid``), ``ExecutiveParams``.
Outputs:  ``ModeCommand`` (no target); next mode ORIENT_TRANSITION.
Status:   [NEW] [BLOCKED: OQ-4]
Port from: legacy/Simulator/simulator.py ``check_state`` (reference only)

# TODO(bug): legacy detumble exit uses the *true* rates. Exit here must use
#   StateEstimate.omega_body only.

Related open questions: OQ-4 (detumble threshold/time).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from cloversat.datatypes.command_types import Mode, ModeCommand
from cloversat.executive.state_manager import ModeHandler

if TYPE_CHECKING:
    from cloversat.config.executive_params import ExecutiveParams
    from cloversat.datatypes.state_types import StateEstimate
    from cloversat.executive.fdir_checks import FdirFlags


class DetumbleMode(ModeHandler):
    """DETUMBLE: B-cross until rates are below threshold for the hold time."""

    mode = Mode.DETUMBLE

    def __init__(self, params: ExecutiveParams) -> None:
        raise NotImplementedError

    def enter(self, t: float) -> None:
        raise NotImplementedError

    def step(self, estimate: StateEstimate, flags: FdirFlags, t: float) -> ModeCommand:
        raise NotImplementedError

    def next_mode(self, estimate: StateEstimate, flags: FdirFlags, t: float) -> Optional[Mode]:
        raise NotImplementedError

    def exit(self, t: float) -> None:
        raise NotImplementedError
