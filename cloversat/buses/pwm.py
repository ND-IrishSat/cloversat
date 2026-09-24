"""PWM output wrapper.

Purpose:  Set duty cycle on a PWM channel, independent of the backend.
Inputs:   Channel index, duty in [0, 1].
Outputs:  None; raises ``i2c.BusError`` on failure.
Status:   [NEW]
Port from: none. Reference only: legacy/HardwareInterface/init.py
           (PCA9685 over I2C at 1500 Hz), legacy/HardwareInterface/magnetorquers.py
           (pigpio hardware PWM).

Backends today: PCA9685 (I2C) and pigpio. Duty is a fraction, not raw
counts; each backend scales to its own resolution.

Related open questions: HW (wheel driver, torquer driver), OQ-3.
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class PwmBackend(ABC):
    """A PWM source (PCA9685, pigpio, ...)."""

    @abstractmethod
    def set_duty(self, channel: int, duty: float) -> None:
        """Set ``channel`` to ``duty`` in [0, 1]."""
        ...

    @abstractmethod
    def set_frequency(self, hz: float) -> None:
        """Set PWM frequency [Hz] (per chip or per channel, backend-specific)."""
        ...


class Pca9685Pwm(PwmBackend):
    """PCA9685 16-channel PWM over I2C."""

    def __init__(self, i2c_bus: object, addr: int = 0x40) -> None:
        """Bind to a PCA9685 at ``addr`` on an ``I2CBus``."""
        raise NotImplementedError

    def set_duty(self, channel: int, duty: float) -> None:
        raise NotImplementedError

    def set_frequency(self, hz: float) -> None:
        raise NotImplementedError


class PigpioPwm(PwmBackend):
    """Pi hardware/software PWM via the pigpio daemon."""

    def __init__(self) -> None:
        """Connect to the local pigpio daemon."""
        raise NotImplementedError

    def set_duty(self, channel: int, duty: float) -> None:
        """``channel`` is the BCM GPIO number."""
        raise NotImplementedError

    def set_frequency(self, hz: float) -> None:
        raise NotImplementedError
