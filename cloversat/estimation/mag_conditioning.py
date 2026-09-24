"""Magnetometer conditioning.

Purpose:  Apply hard-iron bias and soft-iron calibration, reject bad samples,
          and only accept samples the actuation scheduler flags as clean
          (torquers off and settled).
Inputs:   ``Sample[Vec3]`` raw ``B_body`` [T], mag-clean flag from
          ``control/actuation_scheduler``, calibration (``DeviceParams.mag_calibration``),
          norm gates (``EstimationParams``).
Outputs:  ``Sample[Vec3]`` calibrated ``B_body`` [T] (valid=False if rejected).
Status:   [NEW]
Port from: reference only -- legacy/HardwareInterface/mpu9250/mag_calibration.py,
           legacy/HardwareInterface/mpu9250/mag_custom_cal.py,
           legacy/HardwareInterface/calibration_pics/

Calibration lives here (not in drivers) so real and sim devices are treated
identically.

Later:
    - Wheel-motor field compensation (field vs wheel speed).
    - Air-core torquer residual-field subtraction.
"""
from __future__ import annotations

# TODO(bug): legacy code mixes µT and T for magnetometer data. Input and output
# here are Tesla only; convert at the driver.

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from cloversat.datatypes.sample_types import Sample
from cloversat.datatypes.vector_types import Vec3

if TYPE_CHECKING:
    from cloversat.config.estimation_params import EstimationParams


def apply_calibration(
    B_raw_body: Vec3,
    hard_iron_body: Vec3,
    soft_iron: npt.NDArray[np.float64],
) -> Vec3:
    """Return ``soft_iron @ (B_raw_body - hard_iron_body)`` [T]."""
    raise NotImplementedError


def condition_mag(
    sample: Sample[Vec3],
    mag_clean: bool,
    hard_iron_body: Vec3,
    soft_iron: npt.NDArray[np.float64],
    params: EstimationParams,
) -> Sample[Vec3]:
    """Calibrate and gate one magnetometer sample.

    Returns a sample with ``valid=False`` if the input is invalid, the
    scheduler did not flag it clean, or the norm is outside the gates.
    """
    raise NotImplementedError
