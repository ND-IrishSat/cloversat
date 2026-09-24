"""Deployment sequence.

Purpose:  Preconditions -> prepare -> verify -> deploy -> verify -> persist ->
          retry. The deployable's ``safe()`` always runs in ``finally``.
Inputs:   ``DeployableInterface``, ``Persistence``, ``ExecutiveParams``
          (inhibit time, burn limit, max attempts).
Outputs:  ``DeployResult``; updated ``PersistedState.deployed`` / ``deploy_attempts``.
Status:   [NEW]
Port from: none

The driver's ``deploy()`` also refuses unless prepared, inhibit expired, and
not already deployed; this sequence checks the same things first.

Related open questions: HW (deployment mechanism).
"""
from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cloversat.config.executive_params import ExecutiveParams
    from cloversat.devices.interfaces.deployable_interface import DeployableInterface
    from cloversat.executive.persistence import Persistence


class DeployResult(Enum):
    """Outcome of one deployment attempt."""

    DEPLOYED = "deployed"
    ALREADY_DEPLOYED = "already_deployed"
    INHIBITED = "inhibited"
    FAILED_RETRY = "failed_retry"
    FAILED_GAVE_UP = "failed_gave_up"


class DeploymentSequence:
    """Runs deployment attempts with verification and persistence."""

    def __init__(self, params: ExecutiveParams, deployable: DeployableInterface, persistence: Persistence) -> None:
        raise NotImplementedError

    def preconditions_met(self, t_since_first_boot: float) -> bool:
        """True if not deployed, inhibit expired, and attempts remain."""
        raise NotImplementedError

    def attempt(self, t_since_first_boot: float) -> DeployResult:
        """Run one attempt; always calls ``deployable.safe()`` in ``finally``."""
        raise NotImplementedError
