"""GPIO wrapper.

Purpose:  Read and write digital pins.
Inputs:   BCM pin number, value.
Outputs:  Pin level.
Status:   [NEW]
Port from: none. Reference only: legacy/HardwareInterface/init.py
           (``RPi.GPIO`` setup, enable pin, ``GPIO.cleanup()``).

Implementation hint: ``RPi.GPIO``, ``gpiozero`` or pigpio on the Pi; add
the dependency when ported. Pins come from ``config/device_params.py`` via
the driver, never hardcoded here.

Related open questions: HW (flight computer), OQ-10.
"""
from __future__ import annotations


class Gpio:
    """Digital pin access."""

    def __init__(self) -> None:
        """Initialize the GPIO backend (BCM numbering)."""
        raise NotImplementedError

    def setup_output(self, pin: int, initial: bool = False) -> None:
        """Configure ``pin`` as an output."""
        raise NotImplementedError

    def setup_input(self, pin: int, pull_up: bool = False) -> None:
        """Configure ``pin`` as an input."""
        raise NotImplementedError

    def read(self, pin: int) -> bool:
        """Read ``pin`` level."""
        raise NotImplementedError

    def write(self, pin: int, value: bool) -> None:
        """Drive ``pin`` to ``value``."""
        raise NotImplementedError

    def cleanup(self) -> None:
        """Release all pins."""
        raise NotImplementedError
