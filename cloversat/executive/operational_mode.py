"""OPERATIONAL mode.

Purpose:  SUN_POINT by default, holding attitude in eclipse. GPS_POINT,
          GROUND_POINT, and MOMENTUM_DUMP preempt it when needed.
Inputs:   ``StateEstimate``, station visibility, wheel momentum, ``ExecutiveParams``.
Outputs:  ``ModeCommand`` with ``submode`` set (target built in control/).
Status:   [NEW] [BLOCKED: OQ-6]
Port from: none

Related open questions: OQ-6 (priority scheduler vs fixed cycle), OQ-7.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from cloversat.datatypes.command_types import Mode, ModeCommand, SubMode
from cloversat.executive.state_manager import ModeHandler

if TYPE_CHECKING:
    from cloversat.config.executive_params import ExecutiveParams
    from cloversat.datatypes.state_types import StateEstimate
    from cloversat.executive.fdir_checks import FdirFlags


def select_submode(estimate: StateEstimate, current: SubMode, t: float) -> SubMode:
    """Choose this tick's sub-mode (OQ-6)."""
    raise NotImplementedError


class OperationalMode(ModeHandler):
    """OPERATIONAL: sub-mode selection and pointing."""

    mode = Mode.OPERATIONAL

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
