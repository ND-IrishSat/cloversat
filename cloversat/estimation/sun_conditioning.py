"""Sun sensor conditioning.

Purpose:  Basic validity checks and eclipse gating on sun-vector samples.
Inputs:   ``Sample[Vec3]`` raw ``s_body`` (unit), eclipse flag from
          ``eclipse_model``, intensity gate (``EstimationParams.sun_min_intensity``).
Outputs:  ``Sample[Vec3]`` ``s_body`` (unit, valid=False if rejected).
Status:   [NEW] [BLOCKED: HW]
Port from: none

Related open questions: HW (sun sensor choice).

Later:
    - Albedo rejection.
    - Gyro consistency check (sun vector motion vs integrated rates).
    - Panel-current cross-check.
    - Bad-channel exclusion.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from cloversat.datatypes.sample_types import Sample
from cloversat.datatypes.vector_types import Vec3

if TYPE_CHECKING:
    from cloversat.config.estimation_params import EstimationParams


def condition_sun(
    sample: Sample[Vec3],
    intensity: float,
    in_eclipse: bool,
    params: EstimationParams,
) -> Sample[Vec3]:
    """Normalize and gate one sun sample; ``valid=False`` in eclipse or below intensity."""
    raise NotImplementedError
