"""Sun sensor interface.

Purpose:  Abstract sun sensor returning a sun direction plus raw channels.
Inputs:   n/a
Outputs:  ``SunSensorInterface`` ABC; ``read_sun() -> Sample[Vec3]`` (unit
          vector, body frame), ``read_sun_channels() -> Sample[list[float]]``.
Status:   [NEW] [BLOCKED: HW]
Port from: none

Validity and eclipse gating happen in ``estimation/sun_conditioning.py``.

Related open questions: HW (sun sensor part).
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from cloversat.datatypes.sample_types import Sample, SelfTestResult
from cloversat.datatypes.vector_types import Vec3


class SunSensorInterface(ABC):
    """Sun sensor."""

    @abstractmethod
    def init(self) -> None:
        """Bring the device up. Must be safe to call more than once."""
        ...

    @abstractmethod
    def self_test(self) -> SelfTestResult:
        """Run the device's self test."""
        ...

    @abstractmethod
    def read_sun(self) -> Sample[Vec3]:
        """Return the sun direction ``s_body`` (unit vector)."""
        ...

    @abstractmethod
    def read_sun_channels(self) -> Sample[list[float]]:
        """Return raw per-channel readings (units set by the part)."""
        ...
