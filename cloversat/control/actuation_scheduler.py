"""Actuation scheduler: owns the magnetorquers and their duty cycle.

Purpose:  Run the torquer cycle ACTUATE -> DEMAG -> SETTLE -> SAMPLE -> COMPUTE
          so the magnetometer is only read while the torquers are off and
          their field has decayed. Publishes a mag-clean flag that
          estimation/mag_conditioning uses to accept or reject samples.
Inputs:   Torquer voltages from torquer_allocation, monotonic time [s],
          ``ControlParams`` phase lengths and demag voltage, a
          ``TorquerInterface`` to drive.
Outputs:  Torquer drive commands (via the interface), current ``Phase``,
          ``mag_clean`` flag.
Status:   [MODIFIED]
Port from: legacy/Simulator/simulator.py (torquer timer logic in
           ``Simulator.check_state``; demag branch in ``Simulator.controls``)

Phases:
    ACTUATE  torquers driven with the latest voltages
    DEMAG    short reverse burst to demagnetize cores (demag_voltage)
    SETTLE   torquers off, wait for residual field to decay
    SAMPLE   torquers off, magnetometer read; mag_clean = True
    COMPUTE  estimation + control run on the clean sample; new voltages latched

Changes from legacy: the timer is its own module, separate from the mode
logic (legacy shared one if/elif chain with the state checks, see bug below);
time-based instead of step-count-based; always calls ``safe()`` on the
torquers when stopped.

Related open questions: OQ-4 (duty cycle vs detumble time).
"""
from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from cloversat.config.control_params import ControlParams
    from cloversat.devices.interfaces.torquer_interface import TorquerInterface


class Phase(Enum):
    """Torquer duty-cycle phase."""

    ACTUATE = "actuate"
    DEMAG = "demag"
    SETTLE = "settle"
    SAMPLE = "sample"
    COMPUTE = "compute"


class ActuationScheduler:
    """Time-driven torquer cycle. The only module that drives the torquers."""

    def __init__(self, params: ControlParams, torquers: TorquerInterface) -> None:
        """Store params and the torquer device; start in SETTLE with torquers safed."""
        raise NotImplementedError

    def set_voltages(self, voltages: npt.NDArray[np.float64]) -> None:
        """Latch voltages to apply at the next ACTUATE phase [V]."""
        raise NotImplementedError

    def step(self, t: float) -> Phase:
        """Advance the cycle to time ``t`` [s] and drive the torquers for that phase.

        Returns:
            The phase now active.
        """
        raise NotImplementedError

    @property
    def phase(self) -> Phase:
        """Current phase."""
        raise NotImplementedError

    @property
    def mag_clean(self) -> bool:
        """True only during SAMPLE: a magnetometer read now is torquer-free."""
        raise NotImplementedError

    def stop(self) -> None:
        """Stop cycling and call ``safe()`` on the torquers."""
        raise NotImplementedError


# ---- LEGACY START: legacy/Simulator/simulator.py (reference only, not wired up) ----
# scaffold: fragments of Simulator.check_state and Simulator.controls methods,
# kept as comments because they are not valid Python on their own.
#
# --- Simulator.check_state: torquer timer block (+ detumble branch it chains into) ---
#     def check_state(self, i):
#         '''
#         Checks current state of our mag_sat and updates if requirements are met
#             Must check self.states[i - 1] because current state is not updated yet
#         If detumbling, check threshold speed
#         If searching, check if horizon is found
#         If pointing, check if horizon is lost
#             Also always check if speed is too high??
#
#         For accurate mag readings, also handles all timers (torquersOffTimer, magnetometerReadingTimer)
#
#         '''
#
#         if ACCURATE_MAG_READINGS:
#         # TODO(bug): the timer block and the state checks below share one if/elif chain, so with
#         # ACCURATE_MAG_READINGS on, the state checks never run.
#
#             if self.torquersOffTimer * self.dt >= TORQUER_OFF_TIME and self.magnetometerReadingTimer * self.dt >= MAG_READING_INTERVAL:
#                 # if torquers have been off for a while and we took our reading (in generateData_step), reset timers
#                 # print(i, " reading took: ", self.magnetometerReadingTimer * self.dt, ", ", self.torquersOffTimer * self.dt)
#                 self.magnetometerReadingTimer = 0.0
#                 self.torquersOffTimer = 0.0
#
#             elif self.magnetometerReadingTimer * self.dt >= MAG_READING_INTERVAL - TORQUER_OFF_TIME:
#                 # if we're approaching a magnetometer reading, start updating the torquer off timer
#                 # print(i, " mags off", self.magnetometerReadingTimer * self.dt, ", ", self.torquersOffTimer * self.dt)
#                 self.torquersOffTimer += 1.0
#                 self.magnetometerReadingTimer += 1.0
#             else:
#                 # otherwise, update time since we've taken reading
#                 # print(i, " mags on", self.magnetometerReadingTimer * self.dt, ", ", self.torquersOffTimer * self.dt)
#                 self.magnetometerReadingTimer += 1.0
#
#         # if RUNNING_1D:
#         #     # extract current quaternion and convert to euler angles
#         #     q = self.states[i - 1][:4]
#         #     x, y, z = quaternion_to_euler(q)
#
#         #     # check if we've reached our desired angle for the first time
#         #     if (self.finishedTime == -1):
#         #         if (x<=DESIRED_ANGLE[0]):
#         #             print("REACHED DESIRED ANGLE OF " + str(DESIRED_ANGLE[0]) + " DEGREES AFTER " + str(i*self.dt) + " SECONDS!")
#         #             self.finishedTime = i*self.dt
#
#         # TODO(bug): detumble exit uses self.states, i.e. the TRUE sim state, not the estimate.
#         elif self.mag_sat.state == "detumble":
#             # threshold 0.5-1 degress per second per axis
#             thresholdLow = 0
#             thresholdHigh = DETUMBLE_THRESHOLD
#
#             angularX = abs(self.states[i - 1][4])
#             angularY = abs(self.states[i - 1][5])
#             angularZ = abs(self.states[i - 1][6])
#
#             if(self.finishedTime == -1) and i > 0:
#                 if (thresholdLow <= angularX <= thresholdHigh) and (thresholdLow <= angularY <= thresholdHigh) and (thresholdLow <= angularZ <= thresholdHigh):
#                     print("Successfully detumbled after " + str(i*self.dt) + " seconds!")
#                     # record first time we hit "detumbled" threshold (seconds)
#                     self.finishedTime = i*self.dt
#
#                     # When the "detumbled" threshold is hit, calculate total Energy
#                     # Total Energy is calculated as a "Rieman Sum" of the total power used at each time step multiplied by the time step
#                     for step in range(i):
#                         self.energy = self.energy + self.totalPower[step]*self.dt
#
#                     # move to nadir pointing protocol
#                     # return "point"
#                     return "idle"
#
#             return "detumble"
#
# --- Simulator.controls: demag branch ---
#     def controls(self, i):
#         '''
#         Based on saved sensor data and current protocol state, generate correct controls voltages
#             Also set voltage to zero if we're approaching a magnetometer reading
#         Voltage for next step is stored in self.mag_voltages[i]
#         Info about what mode we're in is stored in self.mode[i]
#         '''
#         if ACCURATE_MAG_READINGS and self.torquersOffTimer == 1:
#
#             # if we just started turning off our torquers, send burst of voltage in opposite direction
#             # this simulates demagnitizing the magnetorquer core (theoritically)
#             if np.linalg.norm(self.mag_voltages[i - 1]) > 0:
#                 self.mag_voltages[i] = np.array([- DEMAGNITIZING_VOLTAGE * (v / abs(v)) for v in self.mag_voltages[i - 1]])
#             self.mode[i] = PROTOCOL_MAP['demagnetize']
#             self.errorQuats[i] = self.errorQuats[i - 1]
#             self.nadirError[i] = self.nadirError[i - 1]
#
#         elif ACCURATE_MAG_READINGS and self.torquersOffTimer * self.dt > 0:
#
#             # if torquers are off as they demagnitize, set voltage to 0
#             self.mag_voltages[i] = np.zeros((3))
#             self.mode[i] = PROTOCOL_MAP['demagnetize']
#             self.errorQuats[i] = self.errorQuats[i - 1]
#             self.nadirError[i] = self.nadirError[i - 1]
#
# ---- LEGACY END ----
