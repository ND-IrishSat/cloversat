"""Operating modes and commands.

Purpose:  Shapes that flow state manager -> controller -> allocation.
Inputs:   n/a
Outputs:  ``Mode``, ``SubMode``, ``ModeCommand``, ``ActuatorCommand``.
Status:   [NEW]
Port from: legacy/params.py (``PROTOCOL_MAP`` is the old mode list)

Related open questions: OQ-6 (sub-mode selection).
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Optional

from cloversat.datatypes.vector_types import Vec3

if TYPE_CHECKING:
    from cloversat.datatypes.state_types import AttitudeTarget


class Mode(Enum):
    """Operating modes of the state manager."""

    BOOT = "boot"
    DETUMBLE = "detumble"
    ORIENT_TRANSITION = "orient_transition"
    OPERATIONAL = "operational"
    SAFE = "safe"


class SubMode(Enum):
    """Sub-modes of OPERATIONAL."""

    SUN_POINT = "sun_point"
    GPS_POINT = "gps_point"
    GROUND_POINT = "ground_point"
    MOMENTUM_DUMP = "momentum_dump"


@dataclass
class ModeCommand:
    """State manager output for this tick.

    Attributes:
        mode: Active operating mode.
        submode: Active sub-mode, or None outside OPERATIONAL.
        target: Pointing target, or None if the mode does not point.
    """

    mode: Mode
    submode: Optional[SubMode] = None
    target: Optional["AttitudeTarget"] = None


@dataclass
class ActuatorCommand:
    """Controller output for this tick, before allocation.

    Attributes:
        wheel_torque_body: Requested wheel torque [N*m].
        dipole_body: Requested magnetic dipole [A*m^2].
        source: Name of the law that produced it (for telemetry).
    """

    wheel_torque_body: Vec3
    dipole_body: Vec3
    source: str
