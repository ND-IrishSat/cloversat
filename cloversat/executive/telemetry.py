"""Telemetry.

Purpose:  Build ``TelemetryRecord``s from health, estimates, and events and
          queue them for comms.
Inputs:   ``HealthReport``, ``StateEstimate``, ``ModeCommand``, event strings.
Outputs:  Queued ``TelemetryRecord``s.
Status:   [NEW] [BLOCKED: OQ-9]
Port from: legacy/Main/HardwareScripts/bertha.py (CSV logging, reference only)

# TODO(bug): (reference note) legacy bertha.py builds csv_headers with
#   ``csv_headers.append(csv_raw_headers)``; should be ``extend``.
# TODO(bug): (reference note) legacy bertha.py recreates the PID every loop,
#   so the integral resets each iteration.

Related open questions: OQ-9 (record format), OQ-8 (no uplink / licensing).

Later:
    - Priority levels.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from cloversat.datatypes.command_types import ModeCommand
    from cloversat.datatypes.health_types import HealthReport, TelemetryRecord
    from cloversat.datatypes.state_types import StateEstimate


def health_record(report: HealthReport) -> TelemetryRecord:
    """Build a "health" record."""
    raise NotImplementedError


def adcs_record(estimate: StateEstimate, command: ModeCommand, t: float) -> TelemetryRecord:
    """Build an "adcs" record."""
    raise NotImplementedError


def event_record(t: float, name: str, detail: Optional[dict[str, Any]] = None) -> TelemetryRecord:
    """Build an "event" record."""
    raise NotImplementedError


class TelemetryQueue:
    """Bounded queue of records waiting for comms."""

    def __init__(self, max_len: int) -> None:
        raise NotImplementedError

    def push(self, record: TelemetryRecord) -> None:
        """Add a record, dropping the oldest if full."""
        raise NotImplementedError

    def pop(self) -> Optional[TelemetryRecord]:
        """Remove and return the oldest record, or None."""
        raise NotImplementedError
