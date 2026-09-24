"""Health and telemetry shapes.

Purpose:  Heartbeat output, telemetry records, and watchdog subsystem health.
Inputs:   n/a
Outputs:  ``HealthReport``, ``TelemetryRecord``, ``SubsystemHealth``.
Status:   [NEW]
Port from: none

Related open questions: OQ-9 (telemetry record format), HW (health sensors).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class SubsystemHealth(Enum):
    """Watchdog supervisor state for one subsystem's check-ins."""

    HEALTHY = "healthy"
    LATE = "late"
    STALLED = "stalled"
    FAILED = "failed"


@dataclass
class HealthReport:
    """1 Hz snapshot built by ``executive/heartbeat.py``.

    Fields are placeholders until health sensors are chosen (HW).

    Attributes:
        t: Report time [s].
        bus_voltage: Main bus voltage [V].
        bus_current: Main bus current [A].
        temperatures: Named temperatures [K].
        panel_currents: Named solar panel currents [A].
        subsystems: Watchdog health per subsystem name.
    """

    t: float
    bus_voltage: float = float("nan")
    bus_current: float = float("nan")
    temperatures: dict[str, float] = field(default_factory=dict)
    panel_currents: dict[str, float] = field(default_factory=dict)
    subsystems: dict[str, SubsystemHealth] = field(default_factory=dict)


@dataclass
class TelemetryRecord:
    """One telemetry item queued for comms. Format blocked on OQ-9.

    Attributes:
        t: Record time [s].
        kind: Record type tag (e.g. "health", "adcs", "event").
        payload: Record contents.
    """

    t: float
    kind: str
    payload: dict[str, Any]
