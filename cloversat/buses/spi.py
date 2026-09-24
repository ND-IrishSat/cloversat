"""SPI bus wrapper.

Purpose:  Full-duplex transfers on one SPI device.
Inputs:   Bytes to send.
Outputs:  Bytes received (same length); raises ``i2c.BusError`` on failure.
Status:   [NEW]
Port from: none. Reference only: legacy/HardwareInterface/init.py.

Implementation hint: ``spidev`` on the Pi; add the dependency when ported.

Related open questions: HW (which parts use SPI), OQ-10.
"""
from __future__ import annotations


class SpiDevice:
    """One SPI device (bus + chip select)."""

    def __init__(self, bus: int, cs: int, max_speed_hz: int, mode: int = 0) -> None:
        """Open SPI ``bus`` with chip select ``cs``."""
        raise NotImplementedError

    def transfer(self, data: bytes) -> bytes:
        """Clock out ``data`` and return the bytes clocked in."""
        raise NotImplementedError

    def close(self) -> None:
        """Release the device."""
        raise NotImplementedError
