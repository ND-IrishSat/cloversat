"""File-backed triple-copy storage driver.

Purpose:  Stores each persistence copy as a separate file on a dedicated
          partition, written atomically (temp file + fsync + rename).
Inputs:   ``StorageParams`` (one path per copy).
Outputs:  ``StorageDriver`` (``StorageInterface``).
Status:   [NEW]
Port from: none

Copies should live on a partition separate from the OS/root filesystem so a
corrupted root does not take the state with it. Majority vote on read is done
in ``executive/persistence.py``, not here.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from cloversat.datatypes.sample_types import SelfTestResult
from cloversat.devices.interfaces.storage_interface import StorageInterface

if TYPE_CHECKING:
    from cloversat.config.device_params import StorageParams


class StorageDriver(StorageInterface):
    """One file per copy."""

    def __init__(self, params: StorageParams) -> None:
        raise NotImplementedError

    def init(self) -> None:
        """Check the partition is mounted and the copy directories exist."""
        raise NotImplementedError

    def self_test(self) -> SelfTestResult:
        """Round-trip a test blob through a scratch file on the partition."""
        raise NotImplementedError

    def write_copy(self, i: int, blob: bytes) -> bool:
        """Atomically replace copy ``i``. Return True on success."""
        raise NotImplementedError

    def read_copy(self, i: int) -> Optional[bytes]:
        """Return copy ``i``, or None if missing/unreadable."""
        raise NotImplementedError
