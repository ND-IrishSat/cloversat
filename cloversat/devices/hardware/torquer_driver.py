"""Magnetorquer driver.

Purpose:  Drives three magnetorquers through H-bridges on a PWM expander.
Inputs:   ``TorquerDriverParams`` (PWM channels, direction pins, PWM frequency),
          ``TorquerSpec`` list (max voltage for duty scaling), a PWM bus from
          ``buses/pwm.py``.
Outputs:  ``TorquerDriver`` (``TorquerInterface``).
Status:   [MODIFIED]
Port from: legacy/HardwareInterface/magnetorquers.py

Changes from legacy:
    - 3 axes, not 2; pins come from params, not module constants.
    - Signed volts in, duty cycle computed internally.
    - ``safe()`` sets all coils to zero current (coast), not brake;
      confirm intended off-state with the hardware team.
    - No ``print``; no import-time hardware setup.

Reference only: legacy/HardwareInterface/init.py (bus setup, see ``buses/``).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from cloversat.datatypes.sample_types import Sample, SelfTestResult
from cloversat.devices.interfaces.torquer_interface import TorquerInterface

if TYPE_CHECKING:
    from cloversat.buses.pwm import PwmBackend
    from cloversat.config.device_params import TorquerDriverParams
    from cloversat.config.spacecraft_params import TorquerSpec


class TorquerDriver(TorquerInterface):
    """Three H-bridge-driven magnetorquers."""

    def __init__(
        self,
        params: TorquerDriverParams,
        specs: list[TorquerSpec],
        pwm: PwmBackend,
    ) -> None:
        raise NotImplementedError

    def init(self) -> None:
        """Configure PWM channels and turn all coils off."""
        raise NotImplementedError

    def self_test(self) -> SelfTestResult:
        """Pulse each coil briefly and check current if measurable."""
        raise NotImplementedError

    def set_torquer(self, axis: int, volts: float) -> None:
        """Drive ``axis`` at signed ``volts`` [V], clamped to the spec limit."""
        raise NotImplementedError

    def read_current(self, axis: int) -> Optional[Sample[float]]:
        """Return coil current [A] if a current sense exists, else None."""
        raise NotImplementedError

    def safe(self) -> None:
        """All coils off. Callable anytime; must not raise."""
        raise NotImplementedError


# ---- LEGACY START: legacy/HardwareInterface/magnetorquers.py (reference only, not wired up) ----
# TODO(bug): only 2 torquers; flight needs 3.
# TODO(bug): pins hardcoded at import (PWM* constants, module-level `mag = Mag(...)`); move to params.
'''
magnetorquers.py
Authors: Tim Roberts & Sarah Kopfer

mag interface class
checks hall sensors readings and sets reaction wheel speeds??
initialization must be done through init.py??

MUST GET PINS BEFORE USING

'''
# [scaffold: disabled] from board import SCL, SDA
# [scaffold: disabled] import busio
# [scaffold: disabled] from adafruit_pca9685 import PCA9685
# [scaffold: disabled] import RPi.GPIO as GPIO
import time
import numpy as np
# [scaffold: disabled] from sklearn.linear_model import LinearRegression
# TODO(bug): broken import; no `hall` module on this path (it lives in reaction_wheels/old_scripts).
# [scaffold: disabled] from hall import checkHall
import random


# Define actual GPIO pin numbers
PWM4=4
PWM5=5
PWM7=7
PWM6=6
PWM0=0
PWM1=1
PWM3=3
PWM2=2
MAX_DUTY=65535


class Mag():
    def __init__(self, pins):
        self.pins = pins  # List of PWM channel numbers for AIN1, AIN2, BIN1, BIN2

    def magOn(self, duty_cycle, pindex, pca):
        print("Magnetorquer On")
        if duty_cycle > 0:
            pca.channels[self.pins[pindex][0]].duty_cycle = min(abs(duty_cycle), MAX_DUTY)
            pca.channels[self.pins[pindex][1]].duty_cycle = 0
        elif duty_cycle < 0:
            pca.channels[self.pins[pindex][0]].duty_cycle = 0
            pca.channels[self.pins[pindex][1]].duty_cycle = min(abs(duty_cycle), MAX_DUTY)
        else:
            for pin in self.pins[pindex]:
                pca.channels[pin].duty_cycle = MAX_DUTY

    # TODO(bug): magOff sets both H-bridge inputs high (brake) instead of low (coast); confirm intent.
    def magOff(self, pindex, pca):
        print("Magnetorquer Off")
        for pin in self.pins[pindex]:
            pca.channels[pin].duty_cycle = MAX_DUTY

# Initialize magnetorquers with their respective PWM channels
mag = Mag([
    [PWM4, PWM5, PWM7, PWM6],  # PWM channels for the first magnetorquer
    [PWM0, PWM1, PWM3, PWM2]   # PWM channels for the second magnetorquer
])

"""
# Example usage
try:
    mag.magOn(50, 0)
    time.sleep(10)
    mag.magOn(-50, 0)
    time.sleep(10)
    mag.magOff(0) # one magnetorquer
    time.sleep(10)
    mag.magOn(50, 1)
    time.sleep(10)
    mag.magOn(-50, 1)
    time.sleep(10)
    mag.magOff(1) # other magnetorquer
except KeyboardInterrupt:
    print("Interrupted by user")
finally:
    # Cleanup code if needed
    print("Cleanup complete.")
"""


# ---- LEGACY END ----
