"""UART (serial) wrapper.

Purpose:  Open/read/write/close a serial port with timeouts.
Inputs:   Port path, baud rate, bytes.
Outputs:  Bytes read; raises ``i2c.BusError`` on failure.
Status:   [NEW]
Port from: none. Reference only: legacy/HardwareInterface/vn100/vn100_interface.py
           (VN-100 over serial via vnpy), legacy/HardwareInterface/gps_interface.py.

Implementation hint: ``pyserial``; add the dependency when ported.

Related open questions: HW (GPS receiver, flight IMU), OQ-10.
"""
from __future__ import annotations


class Uart:
    """One serial port."""

    def __init__(self, port: str, baud: int, timeout_s: float = 0.1) -> None:
        """Store settings; call ``open()`` to connect."""
        raise NotImplementedError

    def open(self) -> None:
        """Open the port."""
        raise NotImplementedError

    def read(self, n: int) -> bytes:
        """Read up to ``n`` bytes (may return fewer on timeout)."""
        raise NotImplementedError

    def write(self, data: bytes) -> int:
        """Write ``data``; returns bytes written."""
        raise NotImplementedError

    def close(self) -> None:
        """Close the port."""
        raise NotImplementedError
