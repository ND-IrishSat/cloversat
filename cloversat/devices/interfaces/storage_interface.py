"""Persistent storage interface.

Purpose:  Raw blob storage with independent copies, for triple-copy persistence.
Inputs:   n/a
Outputs:  ``StorageInterface`` ABC.
Status:   [NEW]
Port from: none

Serialization and majority voting live in ``executive/persistence.py``; this
interface only stores and returns bytes.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from cloversat.datatypes.sample_types import SelfTestResult


class StorageInterface(ABC):
    """N independent blob slots (N = 3 for triple copy)."""

    @abstractmethod
    def init(self) -> None:
        """Open or create the storage slots."""
        ...

    @abstractmethod
    def self_test(self) -> SelfTestResult:
        """Check each slot is readable and writable."""
        ...

    @abstractmethod
    def write_copy(self, i: int, blob: bytes) -> bool:
        """Write ``blob`` to slot ``i`` atomically. Return True on success."""
        ...

    @abstractmethod
    def read_copy(self, i: int) -> Optional[bytes]:
        """Return slot ``i`` contents, or None if missing/unreadable."""
        ...
