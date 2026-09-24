"""Control parameters.

Purpose:  Gains, thresholds, power caps, and actuation scheduler timings.
Inputs:   n/a
Outputs:  ``ControlParams``.
Status:   [NEW]
Port from: legacy/params.py (CONTROLS, MAGNETORQUERS, SIM OPTIONS sections)

Related open questions: OQ-4 (numeric requirements).

Legacy values for reference: ``K`` (1.25e-4, detumble gain on firmware),
``KP``/``KD`` (1e-3), ``MAX_POWER_NADIR`` (0.5 W), ``MAG_READING_INTERVAL``
(0.45 s), ``TORQUER_OFF_TIME`` (0.25 s), ``DEMAGNITIZING_VOLTAGE``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class ControlParams:
    """All control settings.

    Attributes:
        bcross_gain: Detumble gain k.
        dump_gain: Momentum dump gain k.
        dump_h_start: Start dumping above this wheel momentum [N*m*s].
        dump_h_stop: Stop dumping below this (hysteresis) [N*m*s].
        kp: Attitude proportional gain.
        kd: Attitude derivative gain.
        max_wheel_torque: Per-wheel torque clamp [N*m].
        torquer_power_cap: Total torquer power cap [W].
        actuate_s: ACTUATE phase length [s].
        demag_s: DEMAG phase length [s].
        settle_s: SETTLE phase length [s].
        sample_s: SAMPLE phase length [s].
        demag_voltage: Demag burst voltage [V].
    """

    bcross_gain: Optional[float] = None
    dump_gain: Optional[float] = None
    dump_h_start: Optional[float] = None
    dump_h_stop: Optional[float] = None
    kp: Optional[float] = None
    kd: Optional[float] = None
    max_wheel_torque: Optional[float] = None
    torquer_power_cap: Optional[float] = None
    actuate_s: Optional[float] = None
    demag_s: Optional[float] = None
    settle_s: Optional[float] = None
    sample_s: Optional[float] = None
    demag_voltage: Optional[float] = None
