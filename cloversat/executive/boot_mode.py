"""BOOT mode.

Purpose:  Read the reset cause, load ``PersistedState``, verify params,
          restore time, and choose the re-entry mode. Boot-loop protection:
          too many resets within a window forces SAFE.
Inputs:   ``PersistedState`` (via persistence), watchdog/RTC, ``ExecutiveParams``.
Outputs:  ``ModeCommand`` (no pointing); next mode.
Status:   [NEW]
Port from: none

Related open questions: OQ-7 (SAFE entry on boot loop).

Later:
    - Scheduled reboots.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from cloversat.datatypes.command_types import Mode, ModeCommand
from cloversat.executive.state_manager import ModeHandler

if TYPE_CHECKING:
    from cloversat.config.executive_params import ExecutiveParams
    from cloversat.datatypes.state_types import PersistedState, StateEstimate
    from cloversat.executive.fdir_checks import FdirFlags
    from cloversat.executive.persistence import Persistence


def is_boot_loop(state: PersistedState, t: float, params: ExecutiveParams) -> bool:
    """True if recent resets exceed ``boot_loop_resets`` within ``boot_loop_window_s``."""
    raise NotImplementedError


def choose_reentry_mode(state: PersistedState, boot_loop: bool) -> Mode:
    """Pick the mode to enter after BOOT from persisted state."""
    raise NotImplementedError


class BootMode(ModeHandler):
    """BOOT: restore state and pick the next mode."""

    mode = Mode.BOOT

    def __init__(self, params: ExecutiveParams, persistence: Persistence) -> None:
        raise NotImplementedError

    def enter(self, t: float) -> None:
        raise NotImplementedError

    def step(self, estimate: StateEstimate, flags: FdirFlags, t: float) -> ModeCommand:
        raise NotImplementedError

    def next_mode(self, estimate: StateEstimate, flags: FdirFlags, t: float) -> Optional[Mode]:
        raise NotImplementedError

    def exit(self, t: float) -> None:
        raise NotImplementedError
