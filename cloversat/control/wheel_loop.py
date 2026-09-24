"""Per-wheel inner control loop.

Purpose:  Turn a per-wheel torque command into a PWM duty (or speed setpoint)
          using measured wheel speed.
Inputs:   Per-wheel torque command [N*m] (wheel_allocation), measured wheel
          speed [rad/s] (``WheelInterface.get_wheel_rpm``, converted), dt [s],
          ``WheelSpec``.
Outputs:  Per-wheel drive command sent via ``WheelInterface.set_wheel``.
Status:   [BLOCKED: OQ-3]
Port from: legacy/HardwareInterface/reaction_wheels/skeleton.py, skeleton.ino,
           old_scripts/ (reference only); legacy PWM mapping in
           legacy/Controllers/PID_controller.py ``torque_to_pwm``

The shape of this module depends on OQ-3: torque/current command (open-loop
or current loop) vs speed setpoint (integrate torque to a speed target).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from cloversat.config.spacecraft_params import WheelSpec
    from cloversat.devices.interfaces.wheel_interface import WheelInterface


class WheelLoop:
    """Inner loop for all wheels."""

    def __init__(self, specs: list[WheelSpec], wheels: WheelInterface) -> None:
        """Store wheel specs and the wheel device."""
        raise NotImplementedError

    def step(self, torque_cmd: npt.NDArray[np.float64], dt: float) -> None:
        """Read wheel speeds, compute drive commands, send them to the wheels."""
        raise NotImplementedError

    def safe(self) -> None:
        """Command all wheels to their safe state."""
        raise NotImplementedError
