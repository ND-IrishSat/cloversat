'''
nadir_point.py
Authors: Peyton Reynolds, Lauren Catalano, David Scully, Michael Kuczun, Andrew Gaylord, Sean Wilson

Controller for nadir pointing given an error quaterion
Outputs the voltage necissary to orient us towards nadir
Attitude Control acutation (step 2 of our nadir pointing process)

NOTE: removed because it's prioprietary
See https://github.com/ND-IrishSat/NearSpace to replace

'''

import math
import numpy as np
import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
# if RUNNING_MAYA:
    # from NearSpace.Pointing.image_processing import *
    # from Horizon_Sensor_Sim.Simulator.all_EOMs import normalize, delta_q

from Simulator.EOMs import *
from Simulator.sat_model import Magnetorquer_Sat
from Simulator.camera import Camera
from params import *


unsupported_case_count = 0

def determine_attitude(mag_sat: Magnetorquer_Sat):
    return None, None, None, None

def nadir_point(error_quat, mag_sat):
    '''
    Proportional-derivative controller to orient us towards a target quaternion
    Seeks to minimize the mangetude of the error quaternion

    @params:
        error_quat (1x4 np.ndarray): error quaternion between current and target nadir
            represents the difference in orientation; [1, 0, 0, 0] meaning that they're aligned
        mag_sat (Magnetorquer_satellite object): encapsulates current state of our cubesat
    @returns:
        voltage_in (1x3 np.ndarray): vector representing voltage we are sending along all three magnetorquers (Volts)
    '''

    # Define the torque using the error quaternion and angular velocity (equation 7.7 from Fund of Spacecraft Att Det)
    #   Derivative term  responds to how fast the error quaternion is changing over time (which is related to how fast we're spinning)
    #   this allows us to anticipate and dampen rapid changes, opposing quick changes and preventing overshooting
    torque = - mag_sat.kp * error_quat[1:4] - mag_sat.kd * mag_sat.w_sat

    # Define the magnetic moment by taking a cross product of the magnetic field with the previously defined torque
    # TODO: only works if they're all orthogonal
    # by doing this, our actual torque will not be aligned with desired torque if they're not orthogonal
    # https://math.stackexchange.com/questions/32600/whats-the-opposite-of-a-cross-product
    m = np.cross( mag_sat.B_body, torque )

    # According to Attitude Control by Magnetic Torquers paper (page 1048)
    m = m / math.pow(np.linalg.norm(mag_sat.B_body), 2)

    # convert magnetic moment to voltage
    voltage = mag_sat.momentToVoltage(m)

    # OR, just scale our max voltage according to the magnitude of the magnetic moment
    # voltage = MAX_VOLTAGE * m_unit

    # OR, compute the scaling factor to make the largest component equal to MAX_VOLTAGE
    # largest_component = np.max(np.abs(m))
    # scaling_factor = MAX_VOLTAGE / largest_component

    # voltage = m * scaling_factor

    # Ensure that we don't exceed power constraint from Nearspace bus (or we get power cycled)
    # P = V^2 / R (watts)
    P_total = np.sum((voltage * voltage) / mag_sat.resistances)
    # Giving some leeway to not exceed in any situation
    if P_total > (MAX_POWER_NADIR * 0.92):
        scale_factor = math.sqrt((MAX_POWER_NADIR * 0.92) / P_total)
        # BRING EVERYTHING DOWN TO FIT IN THE POWER
        voltage *= scale_factor

    return voltage