"""Persistence of ``PersistedState``.

Purpose:  Continuous triple-copy saves of ``PersistedState``; majority vote on
          read so one corrupted copy is outvoted.
Inputs:   ``PersistedState``, ``StorageInterface``.
Outputs:  Saved copies; voted ``PersistedState`` on load.
Status:   [NEW]
Port from: none
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from cloversat.datatypes.state_types import PersistedState
    from cloversat.devices.interfaces.storage_interface import StorageInterface


def encode(state: PersistedState) -> bytes:
    """Serialize with an integrity check (e.g. CRC)."""
    raise NotImplementedError


def decode(blob: bytes) -> Optional[PersistedState]:
    """Deserialize; None if the integrity check fails."""
    raise NotImplementedError


def majority_vote(copies: list[Optional[PersistedState]]) -> Optional[PersistedState]:
    """Return the state at least two valid copies agree on, else best effort or None."""
    raise NotImplementedError


class Persistence:
    """Triple-copy save/load over a ``StorageInterface``."""

    N_COPIES = 3

    def __init__(self, storage: StorageInterface) -> None:
        raise NotImplementedError

    def save(self, state: PersistedState) -> None:
        """Write all copies."""
        raise NotImplementedError

    def load(self) -> Optional[PersistedState]:
        """Read all copies and majority-vote."""
        raise NotImplementedError
