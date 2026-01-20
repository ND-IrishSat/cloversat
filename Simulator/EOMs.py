'''
EOMs.py
Authors: Michael Paulucci, Payton Reynolds, Kris, Michael Cukzun, other controls members

Mother file for all equations of motion for our cubesat systems
Returns the derivative of the next state, doesn't actually propagate the state

'''

import numpy as np
import math
from scipy.spatial.transform import Rotation as R
import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from params import *
from Utils.transformations import *


def eoms(quaternion: np.ndarray, w_sat: np.ndarray, w_rw: np.ndarray, tau_sat: np.ndarray, alpha_rw: np.ndarray, dt: float, I_body: np.ndarray, rw_off: float):
    '''
    Uses the Equations of Motion (EOMs) to yield the next values of quaternion and angular velocity of satellite
    The EOMs are based upon the current state, current reaction wheel speeds, and external torque being applied by the magnetorquers
    Propogates our state vector through time using physics predictions

    @params:
        quaternion (np.ndarray, (1x4)): quaternion describing orientation of satellite with respect to given reference frame
        w_sat (np.ndarray, (1x3)): angular velocity of whole satellite (w/ reaction wheel) (degrees/s. Really?)
        w_rw (np.ndarray, (1x4)): angular velocities of wheels (in respective wheel frame) (rad/s??)
        tau_sat (np.ndarray, (1x3)): external torque applied on the satellite, including magnetorquers and disturbances (N*m?)
        alpha_rw (nd.ndarray, (1x3)): angular acceleration of the reaction wheels in their respective wheel frames
        dt (float): timestep (seconds)
        I_body (np.ndarray, (3x3)): inertia tensor of satellite body
        rw_off (float): set to zero if using only magnetorquers, one if using reaction wheels

    @returns:
        (1x7) state vector containing derivatives of quaternion and angular velocity of satellite
            quaternion_dot (np.ndarray, (1x4)): first derivative of quaternion
            w_sat_dot (np.ndarray, (1x3)): first derivative of angular velocity of satellite
    '''

    # including a model of the force from the string (it just acts as a torsional spring w very low stiffness)
    # au = -k*delta(theta) where delta(theta) is just difference in angle from equilibrium. We just need to measure k

    # store components of angular velocity of satellite
    w_x = w_sat[0]
    w_y = w_sat[1]
    w_z = w_sat[2]

    # Quaternion product matrix for angular velocity of satellite
    ### THIS IS THE BIG OMEGA that should work for our scalar-component of quaternion first notation (based on https://ahrs.readthedocs.io/en/latest/filters/angular.html)
    w_sat_skew_mat = np.array([[0, -w_x, -w_y, -w_z],
                                [w_x, 0, w_z, -w_y],
                                [w_y, -w_z, 0, w_x],
                                [w_z, w_y, -w_x, 0]])


    # find reaction wheel angular momentum in the body frame (based on inertia and rw angular velocity)
    rw_angular_momentum_body = RW_SPIN_AXIS_INERTIA * np.matmul(RW_CONFIG_INERTIA, w_rw)

    # Similarly find vector describing torque caused by reaction wheel in the body frame
    # torque = derivative of angular momentum
    rw_torque_body = RW_SPIN_AXIS_INERTIA * np.matmul(RW_CONFIG_INERTIA, alpha_rw)

    # First derivative of quaternion
    quaternion_dot = 0.5 * np.matmul(w_sat_skew_mat, quaternion)

    # First derivative of angular velocity in body frame
    # Subtract gyroscopic term from all external and rw torquers: inverse_I * (T_mags + T_disturbances - T_rw - w_sat x (I * w_sat + rw_momentum))
    #       Cross product term accounts for change in coordinate frame of body relative to inertial frame
    # Reaction wheel torque and angular momentum are multiplied by TRANSFORMATION to convert from 4 to 3 axes
    w_sat_dot = np.matmul( CUBESAT_BODY_INERTIA_INVERSE, (tau_sat - np.matmul(TRANSFORMATION, rw_torque_body) - np.cross(w_sat, np.matmul(I_body, w_sat) + np.matmul(TRANSFORMATION, rw_angular_momentum_body))))
    # NOTE: total vel magnitude steadily increases over time due to euler's method (i think). Smaller timestep = less increase
    # NOTE: equivalent to multiply by w_skew_matrix instead of taking cross product
    # w_skew_matrix = np.array([[0, -w_z, w_y],
                        # [w_z, 0, -w_x],
                        # [-w_y, w_x, 0]])
    # w_sat_dot = np.matmul( np.linalg.inv(I_body), (tau_sat - np.matmul(w_skew_matrix, np.matmul(I_body, w_sat))))

    # or just take external torque into account (ignore gyroscopice coupling/nutation term)
    # w_sat_dot = np.matmul( np.linalg.inv(I_body), (tau_sat))

    return quaternion_dot, w_sat_dot

    # 1D version for testing:
    '''
    NOTE: This implementation uses simplified rotational dynamics, i.e. 2D dynamics
    Simplifications: quaternion (4x1) -> psi (angle, +psi_dot points out of the page)
                w_sat (3x1) -> psi_dot
        -written by juwan for 2024 banquet at 4 am the night before
    # Grab moment of inertia about z axis
    Izz = self.I_body[2, 2]

    # Calculate psi_ddot, i.e. angular acceleration
    psi_ddot = -(self.I_w_spin/Izz) * alpha_rw

    # Propagate
    new_psi = psi + psi_dot * dt
    new_psi_dot = psi_dot + psi_ddot * dt

    #print(new_psi)
    #print(new_psi_dot)

    new_state = np.array([new_psi, new_psi_dot])

    return new_state
    '''


def f_q(q, w, tau, dt, I):
    return eoms(q, w, 0, tau, 0, dt, I, 0)[0]
def f_w(q, w, tau, dt, I):
    return eoms(q, w, 0, tau, 0, dt, I, 0)[1]

def rk4(state, torque, dt, I_body):
    '''
    Find the next state of the satellite using the RK4 method
    Supposedly more stable (but slower) than Euler's method

    @params:
        state (1x7 ndarray): current state of the satellite (quaternion and angular velocity)
        torque (1x3 ndarray): torque being applied to the satellite
        dt (float): time step for RK4 integration (seconds)
        I_body (3x3 ndarray): inertia tensor of the satellite body
    @returns:
        quaternion_new (1x4 ndarray): quaternion after dt seconds
        w_sat_new (1x3 ndarray): angular velocity after dt seconds

    '''

    q_n, w_n = state[:4], state[4:]

    k1_q = f_q(q_n, w_n, torque, dt, I_body) * dt
    k1_w = f_w(q_n, w_n, torque, dt, I_body) * dt
    k2_q = f_q(q_n + k1_q/2, w_n + k1_w/2, torque, dt, I_body) * dt
    k2_w = f_w(q_n + k1_q/2, w_n + k1_w/2, torque, dt, I_body) * dt
    k3_q = f_q(q_n + k2_q/2, w_n + k2_w/2, torque, dt, I_body) * dt
    k3_w = f_w(q_n + k2_q/2, w_n + k2_w/2, torque, dt, I_body) * dt
    k4_q = f_q(q_n + k3_q, w_n + k3_w, torque, dt, I_body) * dt
    k4_w = f_w(q_n + k3_q, w_n + k3_w, torque, dt, I_body) * dt

    quaternion_new = normalize(q_n + (k1_q + 2*k2_q + 2*k3_q + k4_q) / 6)
    w_sat_new = w_n + (k1_w + 2*k2_w + 2*k3_w + k4_w) / 6

    return quaternion_new, w_sat_new


def ukf_propagator(quaternion: np.ndarray, w_sat: np.ndarray, w_rw: np.ndarray, tau_sat: np.ndarray, alpha_rw: np.ndarray, dt: float, I_body: np.ndarray, rw_off: float):
    '''
    Wrapper for EOMs for UKF prediction step
    '''
    if SOLVER_METHOD == "euler":
        # calculate quaternion and angular velocity derivatives from Equations of Motion
        q_dot, w_dot = eoms(quaternion, w_sat, w_rw, tau_sat, alpha_rw, dt, I_body, rw_off)

        # propagate state using Euler's method
        # find next state in time (based on current state, not sensor data)
        quaternion_new = normalize(quaternion + q_dot*dt)
        w_sat_new = w_sat + w_dot*dt

    elif SOLVER_METHOD == "rk4":
        quaternion_new, w_sat_new = rk4(np.concatenate((quaternion, w_sat)), tau_sat, dt, I_body)

    return np.concatenate((quaternion_new, w_sat_new))