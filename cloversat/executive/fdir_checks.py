"""Fault detection, isolation, and recovery checks.

Purpose:  Check the ``HealthReport`` and ``StateEstimate`` against limits,
          decide degrade actions, and force SAFE when needed.
Inputs:   ``HealthReport``, ``StateEstimate``, ``ExecutiveParams.fdir_limits``.
Outputs:  ``FdirFlags`` (read by the state manager).
Status:   [NEW]
Port from: none

Related open questions: OQ-4 (limits), OQ-7 (SAFE).

Later:
    - South Atlantic Anomaly pass logging.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cloversat.config.executive_params import ExecutiveParams
    from cloversat.datatypes.health_types import HealthReport
    from cloversat.datatypes.state_types import StateEstimate


@dataclass
class FdirFlags:
    """FDIR output for this tick.

    Attributes:
        force_safe: State manager must enter SAFE.
        faults: Names of active faults.
        disabled_subsystems: Subsystems FDIR has disabled.
    """

    force_safe: bool = False
    faults: list[str] = field(default_factory=list)
    disabled_subsystems: list[str] = field(default_factory=list)


class FdirChecks:
    """Evaluates fault checks each tick."""

    def __init__(self, params: ExecutiveParams) -> None:
        raise NotImplementedError

    def evaluate(self, health: HealthReport, estimate: StateEstimate, t: float) -> FdirFlags:
        """Run all checks and return this tick's flags."""
        raise NotImplementedError
