'''
B_dot.py
Author: Peyton Reynolds, Michael Paulucci, Andrew Gaylord

Contains the B dot algorithm used for detumbling our satellite with magnetorquers
References the Magnetorquer_Sat class in mag_EOMs.py

'''

# if RUNNING_MAYA:
    # from NearSpace.Simulator.sat_model import Magnetorquer_Sat

import numpy as np
import math
import os
import sys
# import params module from parent directory
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from params import *
from Simulator.sat_model import Magnetorquer_Sat

def B_dot(sat: Magnetorquer_Sat):
    '''
    B dot algorithm for detumbling our satellite

    @params:
        sat: object that represents our satellite and its magnetorquers. Includes:
            w_sat: angular velocity vector from sensor (rad/s)
            B_body: Magnetic field of body from sensor (Teslas)
            gyro_working: whether gyroscope is working or not (bool)
            prevB: previous magnetic field of body (Teslas)
            dt: timestep (s)
            3 magnetorquer objects, each with:
                n_sat: Number of coils for magnetorquer (int)
                A_sat: Area of the three magnetorquers resepctively (m^2)
                k: proportional factor/gain constant

    @returns:
        voltage_in: vector representing voltage we are sending along all three magnetorquers (Volts)
    '''

    # find desired magentic dipole/moment needed to counteract angular velocity
    B_magnitude_squared = np.linalg.norm(sat.B_body)**2
    if B_magnitude_squared == 0:
        print("Magnetic field vector magnitude cannot be zero.")

    # use b-cross equation when we have access to magnetic field data
    if sat.gyro_working:
        # compute magnetic moment using the b-cross control law: - (k / ||B||^2) * (B x w)
        # units: Amps * m^2 (B_body must be in teslas)
        b_dot_term = np.cross(sat.B_body, sat.w_sat)
        desiredMagneticMoment = np.array([-(sat.mags[i].k / B_magnitude_squared) * b_dot_term[i] for i in range(len(b_dot_term))])
    else:
        # if gyroscope is off, estimate derivative of B field
        # Computer magnetic moment using the control law without w: - k * B'
        # TODO: research methods to smooth noise out: Savitzky-Golay Filter, low pass filter, kalman, 5 point stencil, etc
        '''
        b-dot is perpendicular to b (supposedly) and proportional to w (so can decrease energy of system)
            dipole output should also be perpendicular to b (so no cross product needed)
            b-dot depends on w and the variation of the b field in the orbital frame over time (orbital rate)
            provides less info about angular rates (controller degrades) when the two terms are similar in magnitude
                 steady state of rougly twice orbital rate around axis of maximal interia
            ~= w x B
        can't simply just send voltages to whichever axis have largest rate of change
            moment_x = -k * B_dot_x activates x axis torquer to dampen y and z velocity
        downloaded time optimal paper (and similar from newBDot paper)
        to account for noise: take weighted average of measurements over period for several torquers?
        sensor polling: must turn off (nice graphic)
            derivative error is proportional to sampling period T ^ x - 1 (x = num of points for estimate)
        Want acceptable measure of velocity (Nyquist frequency): max frequency of changes in field that can be sensed without experiencing aliasing = 1/(2T)
            Max velocity that can be measured from 2 consecutive samples = pi / T rad/s
        Duty cycle (sigma): max activiation time of torquer for every T
            T <= pi / abs(w_max)
            Also T <= pi / (2 * sigma * abs(w_max))
        1D 5-point stencil method
            b_dot = (3*B[t - 4*T] - 16*B[t - 3*T] + 36*B[t - 2*T] - 48*B[t - T] + 25*B[t]) / (12*T)
        '''
        # changing negative seems to control whether it strictly decreases/increases
        # doesn't account for sign (or change thereof) of angular velocity
        if len(sat.prevB) >= MAG_READINGS_STORED:
            # # arrange x axis values
            # times = np.arange(-len(sat.prevB) + 1, 1) * sat.dt
            # dB = np.zeros_like(sat.B_body)
            # for i in range(3):
            #     # take linear regression of last few readings (slope = derivative)
            #     B_component = np.array([B[i] for B in sat.prevB])
            #     coeffs = np.polyfit(times, B_component, 1)
            #     dB[i] = - coeffs[0]

            # 1D 5-point stencil method (finite difference)
            # negative makes it so that z axis can detumble
            dB = - (3*sat.prevB[-5] - 16*sat.prevB[-4] + 36*sat.prevB[-3] - 48*sat.prevB[-2] + 25*sat.prevB[-1]) / (12 * MAG_READING_INTERVAL)
            # print("regression dB: ", dB)
        else:
            # Fall back to basic finite difference if not enough data
            # if top is larger than bottom, dB shoots to infinity??
            dB = (sat.B_body - sat.prevB[0]) / sat.dt

        # time-varying gain (depends on magnitude of B field)
        desiredMagneticMoment = -(sat.mags[0].k / B_magnitude_squared) * dB

        # bang-bang control low: if dB is positive, we want to apply a maximum negative magnetic moment
        # desiredMagneticMoment = np.zeros((3))
        # signs = np.sign(dB)
        # for i in range(len(dB)):
        #     if sat.mags[i].epsilon == 1:
        #         max_moment = AIR_MAX_MAGNETIC_MOMENT
        #     else:
        #         max_moment = FERRO_MAX_MAGNETIC_MOMENT

        #     # oppose the sign of the magnetic field change
        #     desiredMagneticMoment[i] = - max_moment * signs[i]
    # print("magnetic moment: ", magneticMoment)

    # convert from desired magnetic moment to voltage required to generate that moment
    voltage_in = sat.momentToVoltage(desiredMagneticMoment)

    return voltage_in


def maxPowerController(moments, sat):
    '''
    Generates voltages that will create desired magnetic moment
    Used for simple 1D test

    @params:
        moments: magnetic moments
        sat: object that represents our satellite and its magnetorquers

    @return:
        voltage that acheive the desired magnetic moments

    '''
    # return np.array([MAX_VOLTAGE, 0.0, 0.0])
    voltage_in = sat.momentToVoltage(moments)
    return voltage_in
