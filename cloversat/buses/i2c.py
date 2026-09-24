"""I2C bus wrapper.

Purpose:  Register reads/writes with locking, timeouts, and stuck-bus recovery.
Inputs:   7-bit device address, register, bytes.
Outputs:  Bytes read; raises ``BusError`` on failure (drivers turn that into
          ``Sample.valid = False``).
Status:   [NEW]
Port from: none. Reference only: legacy/HardwareInterface/init.py
           (``busio.I2C(SCL, SDA)`` setup for the PCA9685).

Implementation hint: ``smbus2`` or ``busio`` on the Pi; add the dependency
when this is ported. One lock per bus so concurrent tasks don't interleave.

Stuck-bus recovery: if SDA is held low, clock SCL up to 9 times via GPIO and
issue a STOP before retrying.

Related open questions: HW (flight computer), OQ-10 (language/platform).
"""
from __future__ import annotations

from typing import Optional


class BusError(Exception):
    """Raised when a bus transaction fails or times out."""


class I2CBus:
    """One physical I2C bus."""

    def __init__(self, bus_id: int, timeout_s: float = 0.05) -> None:
        """Open bus ``bus_id`` (e.g. 1 for ``/dev/i2c-1``)."""
        raise NotImplementedError

    def read(self, addr: int, reg: int, n: int) -> bytes:
        """Read ``n`` bytes starting at register ``reg`` of device ``addr``."""
        raise NotImplementedError

    def write(self, addr: int, reg: int, data: bytes) -> None:
        """Write ``data`` starting at register ``reg`` of device ``addr``."""
        raise NotImplementedError

    def recover(self) -> bool:
        """Attempt stuck-bus recovery. Returns True if the bus is free."""
        raise NotImplementedError

    def close(self) -> None:
        """Release the bus."""
        raise NotImplementedError


# TODO(bug): legacy/HardwareInterface/init.py (reference only) imports
# ``from motors import *`` / ``from hall import checkHall`` which don't match
# the HardwareInterface/ layout, and relies on ``busio``/``SCL``/``SDA``/
# ``PCA9685`` names it never imports. Don't carry that pattern over.
