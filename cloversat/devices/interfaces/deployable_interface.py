"""Deployable mechanism interface.

Purpose:  Abstract burn-wire (or similar) deployment mechanism.
Inputs:   n/a
Outputs:  ``DeployableInterface`` ABC.
Status:   [NEW] [BLOCKED: HW]
Port from: none

Implementations must enforce, independent of the caller:
    - ``deploy()`` refuses unless ``prepare()`` succeeded, the inhibit timer
      has expired, and the mechanism is not already deployed.
    - a hard burn-time limit per attempt.
The sequencing (verify, retry, persist) is ``executive/deployment_sequence.py``.

Related open questions: HW (deployment mechanism).
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from cloversat.datatypes.sample_types import Sample, SelfTestResult


class DeployableInterface(ABC):
    """One deployable (antenna, panel)."""

    @abstractmethod
    def init(self) -> None:
        """Bring the device up with the burn circuit off."""
        ...

    @abstractmethod
    def self_test(self) -> SelfTestResult:
        """Run the device's self test without firing."""
        ...

    @abstractmethod
    def prepare(self) -> bool:
        """Arm the burn circuit. Return True if armed."""
        ...

    @abstractmethod
    def deploy(self) -> bool:
        """Fire once, bounded by the hard burn limit. Return False if refused."""
        ...

    @abstractmethod
    def cancel(self) -> None:
        """Abort an in-progress burn and disarm."""
        ...

    @abstractmethod
    def read_deployed(self) -> Sample[bool]:
        """Return True if the deployed switch reads deployed."""
        ...

    @abstractmethod
    def safe(self) -> None:
        """Disarm and turn the burn circuit off. Callable anytime; must not raise."""
        ...
