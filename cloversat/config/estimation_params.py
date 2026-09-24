"""Estimation parameters.

Purpose:  Sensor noise, field model choice, gating limits, attitude-init settings.
Inputs:   n/a
Outputs:  ``EstimationParams``.
Status:   [NEW]
Port from: legacy/params.py (UKF and SENSORS sections)

Related open questions: OQ-1 (filter), OQ-2 (field model), OQ-5 (attitude init).

Legacy values for reference: ``PROCESS_NOISE_MAG``, ``MEASUREMENT_MAGNETOMETER_NOISE``,
``MEASUREMENT_GYROSCOPE_NOISE``, ``ALPHA``, ``BETA``, ``K_SCALING``,
``SENSOR_MAGNETOMETER_SD``, ``SENSOR_GYROSCOPE_SD``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class EstimationParams:
    """All estimation settings.

    Attributes:
        gyro_noise: Gyro white noise [rad/s/sqrt(Hz)].
        gyro_bias_walk: Gyro bias random walk [rad/s^2/sqrt(Hz)].
        mag_noise: Magnetometer noise [T].
        sun_noise: Sun sensor direction noise [rad].
        field_model: "igrf" or "dipole" (OQ-2).
        igrf_degree: Truncation degree if IGRF.
        gyro_lpf_cutoff_hz: Gyro low-pass cutoff [Hz].
        mag_max_norm: Reject mag samples above this norm [T].
        mag_min_norm: Reject mag samples below this norm [T].
        sun_min_intensity: Reject sun samples below this [sensor units].
        attitude_init_method: OQ-5 selection.
        attitude_init_timeout_s: Give up and retry after this [s].
    """

    gyro_noise: Optional[float] = None
    gyro_bias_walk: Optional[float] = None
    mag_noise: Optional[float] = None
    sun_noise: Optional[float] = None
    field_model: Optional[str] = None
    igrf_degree: Optional[int] = None
    gyro_lpf_cutoff_hz: Optional[float] = None
    mag_max_norm: Optional[float] = None
    mag_min_norm: Optional[float] = None
    sun_min_intensity: Optional[float] = None
    attitude_init_method: Optional[str] = None
    attitude_init_timeout_s: Optional[float] = None
