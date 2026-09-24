"""Torquer allocation: dipole -> per-torquer voltages.

Purpose:  Convert a requested body dipole into per-torquer drive voltages,
          respecting the total power cap and per-torquer voltage/current clamps.
Inputs:   ``dipole_body`` [A*m^2], ``TorquerSpec`` list (axis, dipole per amp,
          resistance, limits), power cap [W].
Outputs:  Per-torquer voltages [V], shape (n_torquers,).
Status:   [MODIFIED]
Port from: legacy/Simulator/sat_model.py (``Magnetorquer_Sat.momentToVoltage``),
           legacy/Controllers/Pointing/nadir_point.py (power-cap block)

Steps: project dipole onto torquer axes -> current = m / (dipole per amp)
-> V = I R -> if sum(V^2 / R) exceeds the cap, scale all voltages uniformly
(preserves direction) -> clamp each to +/- max_voltage.

Changes from legacy: per-torquer resistance from params (legacy used one
global ``RESISTANCE_MAG`` in momentToVoltage but per-torquer
``mag_sat.resistances`` in the power cap); axes from params instead of
assuming orthogonal x/y/z; power cap is a parameter (legacy hardcoded 0.92
margin on ``MAX_POWER_NADIR``).

The actuation scheduler decides *when* these voltages are applied.
"""
from __future__ import annotations

from typing import Sequence, TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from cloversat.datatypes.vector_types import Vec3

if TYPE_CHECKING:
    from cloversat.config.spacecraft_params import TorquerSpec


def dipole_to_voltage(dipole_body: Vec3, torquers: Sequence[TorquerSpec]) -> npt.NDArray[np.float64]:
    """Unclamped voltage per torquer that produces ``dipole_body``.

    Args:
        dipole_body: Requested dipole [A*m^2], body frame.
        torquers: Torquer specs (axis, dipole per amp, resistance).

    Returns:
        Voltage per torquer [V].
    """
    raise NotImplementedError


def apply_power_cap(
    voltages: npt.NDArray[np.float64],
    resistances: npt.NDArray[np.float64],
    power_cap: float,
) -> npt.NDArray[np.float64]:
    """Scale all voltages uniformly so ``sum(V^2 / R) <= power_cap`` [W]."""
    raise NotImplementedError


def clamp_voltages(voltages: npt.NDArray[np.float64], torquers: Sequence[TorquerSpec]) -> npt.NDArray[np.float64]:
    """Clamp each voltage to its torquer's voltage and current limits [V]."""
    raise NotImplementedError


def allocate_torquers(
    dipole_body: Vec3,
    torquers: Sequence[TorquerSpec],
    power_cap: float,
) -> npt.NDArray[np.float64]:
    """Full pipeline: dipole -> voltages -> power cap -> clamps."""
    raise NotImplementedError


# ---- LEGACY START: legacy/Simulator/sat_model.py (reference only, not wired up) ----
import math
import numpy as np
import sys
import os
# [scaffold: disabled] from params import *


class Magnetorquer_Sat():  # scaffold: class shell only; momentToVoltage copied verbatim
    def momentToVoltage(self, moment: np.ndarray) -> np.ndarray:
        '''
        Converts a desired magnetic moment to the voltage needed to generate that moment
        NOTE: all in body frame, which is why area can be treated as [0, 0 , A]

        @params:
            moment (np.ndarray, (1x3)): desired magnetic moment (Amps * m^2)
        @returns:
            voltage (np.ndarray, (1x3)): voltage needed to generate that moment (Volts)
        '''

        # calculate the voltage needed to generate the desired magnetic moment
        current = np.zeros((3))
        # find current for 2 ferro and 1 air core magnetorquers using dipole / nA*epsilon
        for i in range(3):
            current[i] = moment[i] / (self.mags[i].n * self.mags[i].area * self.mags[i].epsilon)

        # convert current to voltage using Ohm's Law
        voltage = np.array(current) * RESISTANCE_MAG
        # voltage = np.array(current) * self.resistances

        return np.array(voltage)
# ---- LEGACY END ----


# ---- LEGACY START: legacy/Controllers/Pointing/nadir_point.py (reference only, not wired up) ----
# scaffold: power-cap block from nadir_point(); a fragment of a function body, so kept as comments.
#
#     # Ensure that we don't exceed power constraint from Nearspace bus (or we get power cycled)
#     # P = V^2 / R (watts)
#     P_total = np.sum((voltage * voltage) / mag_sat.resistances)
#     # Giving some leeway to not exceed in any situation
#     if P_total > (MAX_POWER_NADIR * 0.92):
#         scale_factor = math.sqrt((MAX_POWER_NADIR * 0.92) / P_total)
#         # BRING EVERYTHING DOWN TO FIT IN THE POWER
#         voltage *= scale_factor
#
#     return voltage
# ---- LEGACY END ----
