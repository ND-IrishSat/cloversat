"""Executive parameters.

Purpose:  Mode exit criteria, FDIR limits, per-subsystem watchdog settings,
          deployment inhibit/burn limits, boot-loop thresholds.
Inputs:   n/a
Outputs:  ``ExecutiveParams``, ``WatchdogSubsystemParams``.
Status:   [NEW]
Port from: legacy/params.py (``DETUMBLE_THRESHOLD``, ``QUAT_ERROR_TOLERANCE``,
           ``ANGULAR_RATE_TOLERANCE``)

Related open questions: OQ-4 (thresholds), OQ-7 (SAFE exit).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class WatchdogSubsystemParams:
    """Watchdog supervision for one subsystem.

    Attributes:
        deadline_s: Expected check-in period [s].
        late_after_n: Missed deadlines before LATE.
        failed_after_m: Missed deadlines before FAILED.
        critical: If True, FAILED stops kicking the hardware watchdog.
        degrade_action: Action name FDIR takes on FAILED.
    """

    deadline_s: Optional[float] = None
    late_after_n: Optional[int] = None
    failed_after_m: Optional[int] = None
    critical: bool = False
    degrade_action: Optional[str] = None


@dataclass
class ExecutiveParams:
    """All executive settings.

    Attributes:
        detumble_rate_threshold: Exit DETUMBLE below this |omega| [rad/s].
        detumble_hold_s: ... for this long [s].
        pointing_error_tolerance: "On target" attitude error [rad].
        rate_tolerance: "Settled" rate [rad/s].
        fdir_limits: Named FDIR limits.
        watchdog: Per-subsystem watchdog settings.
        deploy_inhibit_s: No deploy before this time since first boot [s].
        deploy_max_burn_s: Hard burn-time limit per attempt [s].
        deploy_max_attempts: Give up after this many attempts.
        boot_loop_resets: Resets within the window that count as a boot loop.
        boot_loop_window_s: Boot-loop window [s].
        control_period_s: Control tick period [s].
        heartbeat_period_s: Heartbeat period [s].
    """

    detumble_rate_threshold: Optional[float] = None
    detumble_hold_s: Optional[float] = None
    pointing_error_tolerance: Optional[float] = None
    rate_tolerance: Optional[float] = None
    fdir_limits: dict[str, float] = field(default_factory=dict)
    watchdog: dict[str, WatchdogSubsystemParams] = field(default_factory=dict)
    deploy_inhibit_s: Optional[float] = None
    deploy_max_burn_s: Optional[float] = None
    deploy_max_attempts: Optional[int] = None
    boot_loop_resets: Optional[int] = None
    boot_loop_window_s: Optional[float] = None
    control_period_s: Optional[float] = None
    heartbeat_period_s: float = 1.0
