"""Estimator: runs the estimation layer each tick and produces ``StateEstimate``.

Purpose:  Orchestrate clock, orbit, environment models, sensor conditioning,
          attitude init, and the attitude filter into one ``StateEstimate``.
Inputs:   This tick's device samples (gyro, mag, sun, GPS, wheel speeds),
          mag-clean flag from ``control/actuation_scheduler``, ``EstimationParams``.
Outputs:  ``StateEstimate`` with validity flags set per component.
Status:   [NEW]
Port from: reference only -- legacy/Simulator/simulator.py
           (``determine_attitude``), legacy/Main/SimScripts/run_UKF.py

Order per tick: clock -> orbit (sgp4, GPS re-seed) -> sun/eclipse/field models
-> conditioning -> attitude init or filter -> ``StateEstimate``.

Flight code never sees true physical state. Everything here comes from
measurements and models only.
"""
from __future__ import annotations

# TODO(bug): legacy simulator.determine_attitude reads self.B_true (undefined,
# and sim truth). Do not reproduce; use igrf_model output.
# TODO(bug): mixed µT/T units across legacy estimation code. Tesla only here.

from typing import TYPE_CHECKING, Optional

from cloversat.datatypes.sample_types import Sample
from cloversat.datatypes.state_types import StateEstimate
from cloversat.datatypes.vector_types import Vec3

if TYPE_CHECKING:
    from cloversat.config.estimation_params import EstimationParams
    from cloversat.devices.interfaces.gps_interface import GpsFix


class Estimator:
    """Owns all estimation components and builds a ``StateEstimate`` each tick."""

    def __init__(self, params: EstimationParams) -> None:
        raise NotImplementedError

    def step(
        self,
        t_mono: float,
        gyro: Sample[Vec3],
        mag: Sample[Vec3],
        sun: Sample[Vec3],
        gps: Optional[Sample[GpsFix]],
        wheel_speeds: Optional[Sample[list[float]]],
        mag_clean: bool,
    ) -> StateEstimate:
        """Run one estimation tick and return the new ``StateEstimate``."""
        raise NotImplementedError

    def latest(self) -> Optional[StateEstimate]:
        """Most recent estimate, or None before the first tick."""
        raise NotImplementedError
