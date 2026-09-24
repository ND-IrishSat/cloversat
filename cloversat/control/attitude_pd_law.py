"""Attitude PD control law.

Purpose:  Wheel torque that drives attitude error and rate error to zero.
Inputs:   Attitude ``q`` and target ``q_target`` ([w, x, y, z], body <- inertial),
          body rate ``omega_body`` and target rate [rad/s], gains Kp, Kd.
Outputs:  Wheel torque request ``tau_body`` [N*m].
Status:   [MODIFIED]
Port from: legacy/Controllers/PID_controller.py (``PIDController.pid_controller``,
           ``PIDController.pd_velocity_controller``)

Law:
    q_err = error quaternion (geometry/quaternion.py), sign-flipped so
            q_err.w >= 0 (shortest path)
    tau   = -Kp * q_err[1:4] - Kd * (omega - omega_target)

Changes from legacy: PD only (no integral term, so no windup); returns torque,
not PWM (PWM mapping is in wheel_allocation / wheel_loop); adds the
shortest-path sign flip; pure function, no stored state.

Archived, not ported: legacy/Controllers/eigen_axis.py, stuPIDexample.py,
test-PID-quat.py. Related: legacy/Controllers/adc_pd_controller.py (older PD
used by Main/HardwareScripts/banquet_demo.py and 1d_test_pi.py).

Related open questions: OQ-4 (pointing accuracy sets gains).

Later:
    - Eigenaxis slews.
"""
from __future__ import annotations

from cloversat.datatypes.vector_types import Quat, Vec3


def attitude_pd_torque(
    q: Quat,
    q_target: Quat,
    omega_body: Vec3,
    omega_target_body: Vec3,
    kp: float,
    kd: float,
) -> Vec3:
    """PD attitude torque ``tau = -Kp q_err_vec - Kd (omega - omega_target)``.

    Args:
        q: Current attitude, [w, x, y, z], body <- inertial.
        q_target: Target attitude, same convention.
        omega_body: Body rate [rad/s].
        omega_target_body: Target body rate [rad/s].
        kp: Proportional gain [N*m].
        kd: Derivative gain [N*m*s/rad].

    Returns:
        Torque request [N*m], body frame.
    """
    raise NotImplementedError


def rate_pd_torque(omega_body: Vec3, omega_target_body: Vec3, kp: float, kd: float) -> Vec3:
    """Rate-tracking torque (legacy ``pd_velocity_controller``), for testbed spin-up.

    Args:
        omega_body: Body rate [rad/s].
        omega_target_body: Target body rate [rad/s].
        kp: Rate-error gain.
        kd: Damping gain.

    Returns:
        Torque request [N*m], body frame.
    """
    raise NotImplementedError


# ---- LEGACY START: legacy/Controllers/PID_controller.py (reference only, not wired up) ----
import numpy as np
import os
import sys

# To import module that is in the parent directory of your current module:
# [scaffold: disabled] sys.path.insert(1, os.path.join(sys.path[0], '..'))
# [scaffold: disabled] from params import *
# [scaffold: disabled] from Utils.transformations import normalize, quaternion_multiply, delta_q


class PIDController:
    def __init__(self, kp, ki, kd, dt):
        '''
        PID controller class with gains for proportional, integral, and derivative control.

        @params:
            kp: Proportional gain
            ki: Integral gain
            kd: Derivative gain
            dt: Time step (sampling time)
        '''
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt
        self.integral_error = np.zeros(3)

    def pid_controller(self, state, target, omega, old_pwm):
        '''
        PID controller to compute PWM signals for reaction wheels.

        @params:
            state: Current normalized quaternion of cubesat (4 x 1) [q0 ; q1:3]
            target: Target normalized quaternion of cubesat (4 x 1) [q0 ; q1:3]
            omega: Current angular velocity of cubesat (3 x 1)
            old_pwm: Previous PWM values for each motor (4 x 1)

        @returns:
            pwm: PWM signals for each motor (4 x 1)
        '''

        # Find the error quaternion (delta_q_out) between current and target quaternion
        # represents the difference in orientation; [1, 0, 0, 0] meaning that they're aligned
        delta_q_out = delta_q(state, target)

        # Quaternion error (vector part only) and its proportional control component
        proportional = self.kp * delta_q_out[1:4]

        # TODO(bug): error quaternion is not sign-flipped (q vs -q), so it can take the long way around.
        # Update the integral error (accumulate error over time)
        # if there's small, consistent error, this corrects for it
        # TODO: introduce limit on this term to combat integral windup
        # TODO(bug): integral has no anti-windup.
        self.integral_error += delta_q_out[1:4] * self.dt
        # print(self.integral_error)
        integral = self.ki * self.integral_error

        # Derivative control component based on angular velocity (omega)
        # derivative term responds to how fast the error quaternion is changing over time (which is related to how fast we're spinning)
        # this allows us to anticipate and dampen rapid changes, opposing quick changes and preventing overshooting
        # alternatively, we could use the derivative of the error quaternion (using last error)
        derivative = -self.kd * omega

        # integral is striclty increasing. Are the error quat signs off or is that correct behavior
        # Total control output (torque command)
        L = proportional + integral + derivative
        return self.torque_to_pwm(L)


    def pd_velocity_controller(self, target_speed, current_speed, kp, kd):
        '''
        PD controller to compute PWM signals for reaction wheels. No integral term.

        @params:
            target_speed: Desired angular velocity of cubesat (3 x 1)
            current_speed: Current angular velocity of cubesat (3 x 1)
        '''
        # derivative term relates to the rate of change of the error, which can be thought of as the current speed
        L = kp * (current_speed - target_speed) - kd * current_speed
        return self.torque_to_pwm(L)

    # (torque_to_pwm moved to cloversat/control/wheel_allocation.py)
# ---- LEGACY END ----
