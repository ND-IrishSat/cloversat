"""Wheel torque allocation.

Purpose:  Map a body torque request onto per-wheel commands using the wheel
          matrix from params (3- or 4-wheel), then clamp to wheel limits.
Inputs:   ``tau_body`` [N*m], wheel matrix W (3 x n, spin axes as columns),
          per-wheel torque limits.
Outputs:  Per-wheel torque commands [N*m], shape (n,).
Status:   [MODIFIED]
Port from: legacy/Controllers/PID_controller.py (``PIDController.torque_to_pwm``)

Allocation: tau_wheels = pinv(W) @ tau_body, then scale uniformly if any wheel
exceeds its limit (preserves torque direction).

Changes from legacy: uses W from ``SpacecraftParams`` instead of a hardcoded
NASA pyramid; outputs torque, not PWM (the torque -> PWM / speed step is
wheel_loop, blocked on OQ-3); fixes the W_inv sign bug below.

Related open questions: OQ-3 (wheel command interface).

Later:
    - 4-wheel null-space biasing (keep wheels away from zero speed).
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt

from cloversat.datatypes.vector_types import Vec3


def allocation_matrix(wheel_matrix: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Pseudoinverse of the (3 x n) wheel matrix, shape (n x 3)."""
    raise NotImplementedError


def allocate_wheel_torque(
    tau_body: Vec3,
    wheel_matrix: npt.NDArray[np.float64],
    max_wheel_torque: float,
) -> npt.NDArray[np.float64]:
    """Split a body torque request across wheels.

    Args:
        tau_body: Requested torque [N*m], body frame.
        wheel_matrix: Spin axes as columns, (3, n).
        max_wheel_torque: Per-wheel torque limit [N*m].

    Returns:
        Per-wheel torque [N*m], shape (n,), scaled so no wheel exceeds its limit.
    """
    raise NotImplementedError


# ---- LEGACY START: legacy/Controllers/PID_controller.py (reference only, not wired up) ----
import numpy as np
# [scaffold: disabled] from params import *


class PIDController:  # scaffold: class shell only; torque_to_pwm copied verbatim
    def torque_to_pwm(self, L):
        # Reaction wheel transformation matrix for the NASA configuration
        alpha = 1 / np.sqrt(3)
        beta = 1 / np.sqrt(3)
        gamma = 1 / np.sqrt(3)

        # If fourth reaction wheel is not mounted exactly how we want we can adjust alpha, beta, gamma
        # TODO: what is this for??
        # W =  [[1, 0, 0, alpha],
        #      [0, 1, 0, beta],
        #      [0, 0, 1, gamma]]

        # Transformation matrix that maps 3-axis torque to the 4 reaction wheels
        # TODO(bug): W_inv row 3 uses +alpha*gamma; the pseudoinverse needs -alpha*gamma.
        W_inv = np.array([[(1 + beta**2 + gamma**2), -alpha * beta, -alpha * gamma],
                          [-alpha * beta, (1 + alpha**2 + gamma**2), -beta * gamma],
                          [alpha * gamma, -beta * gamma, (1 + alpha**2 + beta**2)],
                          [alpha, beta, gamma]]) / (1 + alpha**2 + beta**2 + gamma**2)

        # Convert torque (L) to reaction wheel space (4 motors)
        motor_torques = np.matmul(W_inv, L)
        # temporary fix to remove some variability
        # motor_torques = np.append(L, np.array([0]))

        # Map the torque output to PWM range
        pwm = (motor_torques / MAX_RW_TORQUE) * MAX_PWM

        # Convert to integer values for actual PWM signals
        pwm = np.array([int(p) for p in pwm])

        # Ensure PWM is within bounds of motor constraints
        pwm = np.clip(pwm, -MAX_PWM * 1.0, MAX_PWM * 1.0)

        # pwm = np.array([0, 3000, 0, 0])

        return pwm
# ---- LEGACY END ----
