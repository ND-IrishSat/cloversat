"""Initial attitude acquisition.

Purpose:  Get a first attitude good enough to start the fine attitude filter.
Inputs:   Conditioned mag / sun / gyro samples, reference field and sun
          direction, ``EstimationParams`` (method, timeout).
Outputs:  Initial ``Quat`` (body <- inertial) + covariance, or "not yet".
Status:   [NEW] [BLOCKED: OQ-5]
Port from: none

Related open questions: OQ-5 (sun search + TRIAD vs two-epoch mag TRIAD vs
filter convergence from an arbitrary start).
"""
from __future__ import annotations

# TODO(bug): legacy simulator.determine_attitude (legacy/Simulator/simulator.py)
# uses the undefined self.B_true, which is sim truth anyway. Flight code must
# only use the reference field model (igrf_model) and measurements.

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import numpy as np
import numpy.typing as npt

from cloversat.datatypes.sample_types import Sample
from cloversat.datatypes.vector_types import Quat, Vec3

if TYPE_CHECKING:
    from cloversat.config.estimation_params import EstimationParams


@dataclass
class AttitudeInitResult:
    """Output of a successful acquisition.

    Attributes:
        q: Attitude, body <- inertial.
        covariance: Initial attitude covariance.
        t: Time of the solution [s].
    """

    q: Quat
    covariance: npt.NDArray[np.float64]
    t: float


class AttitudeInitializer:
    """Accumulates measurements until an initial attitude can be computed."""

    def __init__(self, params: EstimationParams) -> None:
        raise NotImplementedError

    def update(
        self,
        mag: Sample[Vec3],
        sun: Sample[Vec3],
        gyro: Sample[Vec3],
        B_ref_eci: Vec3,
        s_ref_eci: Vec3,
    ) -> Optional[AttitudeInitResult]:
        """Feed one tick of data; return a result once acquired, else None."""
        raise NotImplementedError

    def timed_out(self, t: float) -> bool:
        """True if acquisition has run longer than the configured timeout."""
        raise NotImplementedError

    def reset(self) -> None:
        """Start acquisition over."""
        raise NotImplementedError
