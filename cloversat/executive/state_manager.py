"""Operating-mode state machine.

Purpose:  Pick the active mode each tick. Checks the active mode's exit
          criteria against ``StateEstimate`` and FDIR flags only, and outputs
          a ``ModeCommand``.
Inputs:   ``StateEstimate``, ``FdirFlags``, time.
Outputs:  ``ModeCommand``.
Status:   [NEW]
Port from: legacy/Simulator/simulator.py ``check_state`` (reference only),
           legacy/params.py ``PROTOCOL_MAP`` (old mode list)

Modes (``datatypes.command_types.Mode``): BOOT, DETUMBLE, ORIENT_TRANSITION,
OPERATIONAL, SAFE. Each is implemented by one ``*_mode.py`` as a
``ModeHandler``. FDIR can force SAFE from any mode.

# TODO(bug): legacy check_state shares one if/elif chain between the torquer
#   timer and the state checks, so with ACCURATE_MAG_READINGS on the state
#   checks never run. Keep timing (control/actuation_scheduler) separate.
# TODO(bug): legacy detumble exit uses the *true* state. Exit criteria here
#   must only read StateEstimate.

Related open questions: OQ-4, OQ-5, OQ-6, OQ-7.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

from cloversat.datatypes.command_types import Mode, ModeCommand

if TYPE_CHECKING:
    from cloversat.config.executive_params import ExecutiveParams
    from cloversat.datatypes.state_types import StateEstimate
    from cloversat.executive.fdir_checks import FdirFlags


class ModeHandler(ABC):
    """One operating mode. Implemented by each ``*_mode.py``."""

    mode: Mode

    @abstractmethod
    def enter(self, t: float) -> None:
        """Called once when the mode becomes active."""
        ...

    @abstractmethod
    def step(self, estimate: StateEstimate, flags: FdirFlags, t: float) -> ModeCommand:
        """Return this tick's ``ModeCommand``."""
        ...

    @abstractmethod
    def next_mode(self, estimate: StateEstimate, flags: FdirFlags, t: float) -> Optional[Mode]:
        """Return the mode to transition to, or None to stay."""
        ...

    @abstractmethod
    def exit(self, t: float) -> None:
        """Called once when the mode stops being active."""
        ...


class StateManager:
    """Runs the mode state machine over a set of ``ModeHandler``s."""

    def __init__(self, params: ExecutiveParams, handlers: dict[Mode, ModeHandler], initial: Mode) -> None:
        raise NotImplementedError

    @property
    def mode(self) -> Mode:
        """Currently active mode."""
        raise NotImplementedError

    def step(self, estimate: StateEstimate, flags: FdirFlags, t: float) -> ModeCommand:
        """Check transitions (FDIR SAFE first), then step the active mode."""
        raise NotImplementedError

    def force_mode(self, mode: Mode, t: float, reason: str) -> None:
        """Transition immediately (used by FDIR and boot re-entry)."""
        raise NotImplementedError
