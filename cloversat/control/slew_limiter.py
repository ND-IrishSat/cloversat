"""Slew limiter.

Purpose:  Rate-limit changes in the attitude target so a new target doesn't
          command a step the wheels can't follow.
Inputs:   Previous limited target, new raw target, max slew rate [rad/s], dt [s].
Outputs:  Limited ``AttitudeTarget``.
Status:   [LATER]
Port from: none
"""
from __future__ import annotations

from cloversat.datatypes.state_types import AttitudeTarget


def limit_slew(previous: AttitudeTarget, requested: AttitudeTarget, max_rate: float, dt: float) -> AttitudeTarget:
    """Move from ``previous`` toward ``requested`` by at most ``max_rate * dt`` [rad]."""
    raise NotImplementedError
