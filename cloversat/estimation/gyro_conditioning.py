"""Gyro conditioning.

Purpose:  Subtract estimated gyro bias, low-pass filter, and provide the
          detumble rate estimate (valid before the attitude filter converges).
Inputs:   ``Sample[Vec3]`` raw ``omega_body`` [rad/s], bias estimate [rad/s],
          filter settings (``EstimationParams.gyro_lpf_cutoff_hz``).
Outputs:  ``Sample[Vec3]`` conditioned ``omega_body`` [rad/s].
Status:   [MODIFIED]
Port from: legacy/ukf/low_pass_filter/lowpassfilter.py (first-order LPF, copied below).
           Reference only (scripts with file I/O / plotting, not copied):
           legacy/ukf/low_pass_filter/testing_LPF.py,
           legacy/ukf/low_pass_filter/plottingLFT.py.

Related open questions: OQ-1 (the filter may own bias estimation).

Notes:
    - Legacy LPF parameterizes by time constant ``tau``; params use a cutoff
      frequency (tau = 1 / (2*pi*f_c)).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from cloversat.datatypes.sample_types import Sample
from cloversat.datatypes.vector_types import Vec3

if TYPE_CHECKING:
    from cloversat.config.estimation_params import EstimationParams


class GyroConditioner:
    """Bias correction + first-order low-pass filter for gyro samples."""

    def __init__(self, params: EstimationParams, dt: float) -> None:
        raise NotImplementedError

    def condition(self, sample: Sample[Vec3], bias_body: Vec3) -> Sample[Vec3]:
        """Return bias-corrected, filtered ``omega_body`` [rad/s]."""
        raise NotImplementedError

    def reset(self) -> None:
        """Clear filter state (e.g. after a long gap in samples)."""
        raise NotImplementedError


def detumble_rate_estimate(sample: Sample[Vec3], bias_body: Vec3) -> Sample[Vec3]:
    """Rate estimate usable in DETUMBLE, before attitude is known [rad/s]."""
    raise NotImplementedError


# ---- LEGACY START: legacy/ukf/low_pass_filter/lowpassfilter.py (reference only, not wired up) ----
'''
applying a loss pass filter on magnetometer and gyroscope signals
3/5/2026
'''

import numpy as np

'''
equation for Low Pass Filter: y[i] = alpha * x[i] + (1 - alpha) * y[i-1]

x[n] = current sensor reading (magnetometer or gyro)
y[n] = filtered output
y[n−1] = previous filtered value
α = smoothing factor (between 0 and 1)
'''
class LowPassFilter:
    def __init__(self, dt, tau):
        self.dt = float(dt)
        self.tau = float(tau)
        self.alpha = self.dt / (self.tau + self.dt)
        self.prev_output = None

    def apply(self, input_signal):
        input_signal = np.asarray(input_signal, dtype=float)

        if self.prev_output is None:
            output = input_signal.copy()
        else:
            output = self.alpha * input_signal + (1 - self.alpha) * self.prev_output

        self.prev_output = output
        return output
    
# ---- LEGACY END ----
