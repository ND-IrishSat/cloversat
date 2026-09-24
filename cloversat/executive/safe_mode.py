"""SAFE mode.

Purpose:  Detumble, find the sun, hold. Minimal loads, beacon only.
Inputs:   ``StateEstimate``, ``FdirFlags``, ``ExecutiveParams``.
Outputs:  ``ModeCommand``; next mode per SAFE exit criteria (OQ-7).
Status:   [NEW] [BLOCKED: OQ-7]
Port from: none

Related open questions: OQ-7 (SAFE exit criteria and comms), OQ-8.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from cloversat.datatypes.command_types import Mode, ModeCommand
from cloversat.executive.state_manager import ModeHandler

if TYPE_CHECKING:
    from cloversat.config.executive_params import ExecutiveParams
    from cloversat.datatypes.state_types import StateEstimate
    from cloversat.executive.fdir_checks import FdirFlags


class SafeMode(ModeHandler):
    """SAFE: detumble -> sun search -> hold."""

    mode = Mode.SAFE

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
