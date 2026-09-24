'''
ideal_test_cases.py
Authors: Andrew Gaylord
Last modified 2/4/24

Tests two ideal test cases for UKF algorithm: nonmoving and constant angular velocity

'''

import numpy as np
import random
import pygame
from pygame.locals import * 
from OpenGL.GL import *
from OpenGL.GLU import *
import datetime as dt

from pyquaternion import Quaternion

import UKF_algorithm
import hfunc

import os
import sys

# import params module from parent directory
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from sim.visualizer import *

# from PySOL import spacecraft as sp
# from PySol.spacecraft import *
from PySOL.sol_sim import *
import PySOL.spacecraft as sp
import PySOL.orb_tools as ot



def run_basic_test():
    '''
    ideal test case for UKF: zeroed out, nonmoving cubesat
    0 reaction wheel speed and angular velocity, constant magnetic field
    no frame of reference transformation or gps implementation

    THINGS CHANGED FOR TEST CASE ONE:
        removed hfunc and passed gps_data as magnetic field
        
        removed normalization

        removed zCov
        
        need to find optimal starting r, q for each case

        got rid of different dimensionality for m: instead, everything uses 7

        switched covariance calculation to option 1

        fixed crosscovariance calculation bug

        created diagonal starting covariance

        what is control input vector in EOMs?
    '''

    n = 7
    m = n - 1
    dt = .1

    # quaternion and angular velocity should be zero
    start = np.array([0, 0, 1, 0, 0, 0, 0])

    # start with diagonal covariance
    cov = np.identity(n) * 5e-10
    # print("Starting cov: ", cov)

    # we want magnetomer reading to be constant, rest to be 0
    data = [0, 1, 0, 0, 0, 0]

    # r: measurement noise (m x m)
    noiseMagnitude = 0.02
    r = np.diag([noiseMagnitude] * m)

    # q: process noise (n x n)
    noiseMagnitude = 0.005
    q = np.diag([noiseMagnitude] * n)

    # edit code so that lat/long/hieght are not needed 
    # make gps_data = magnetic field for this test only 
    # not even used in this test case bc hfunc is disabled
    gps_data = np.zeros(3)

    # control input vector for eoms, zero for this test
    reaction_speeds = np.zeros(4)
    
    i = 0
    while(1):
        start, cov, innov, innovCov = UKF_algorithm.UKF(start, cov, q, r, dt, gps_data, reaction_speeds, reaction_speeds, data)

        game_visualize(np.array([start[:4]]), i)
        # print(start[:4])
        
        # uncomment for cube to move
        # if 100 > i > 25:
        #     data[1] += .05
        # elif i > 100 and data[1] > 0:
        #     data[1] -= .05
        i += 1


def run_moving_test():
    '''
    Second ideal test for ukf: nonmoving cubesat spinning with constant velocity
    Constant magnetic field, 0 reaction wheel speed
    Before running algorithm, calculate ideal true quaternion used to find magnetometer reading for every step
    Frame of reference transition is utilized in hfunc and to find magnetometer reading based on true quaternion and b field
    Still incorrect r and q and no gps data (gps_data is passed as b field, not gps data)
    '''

    # number of steps to calculate
    n = 1000
    dt = 0.1
    # dimensionality of state space
    dim = 7
    # dimensionality of measurement space
    dimMes = dim - 1
    speed = 1
    # starting quaternion for propogator
    initQ = np.array([1, 0, 0, 0])
    # starting state estimate (should match initQ and w)
    start = [1, 0, 0, 0, 0, speed, speed/2]
    # angular velocity
    w = np.array([0, speed, speed/2])
    # starting covariance
    cov = np.identity(dim) * 5e-10
    # constant B field
    B_true = np.array([0, 0, 1])
    # reaction wheel speeds (0 for this test)
    reaction_speeds = np.zeros(4)
    old_reaction_speeds = np.zeros(4)

    # note: if model is less reliable/changes quickly, then q > r
    # r: measurement noise (m x m)
    noiseMagnitude = 0.01
    r = np.diag([noiseMagnitude] * dimMes)

    # q: process noise (n x n)
    noiseMagnitude = 0.005
    q = np.diag([noiseMagnitude] * dim)

    t0 = 0
    tf = n * dt
    i = 0

    # uncomment to store results and visualize after calculating
    # results = [[]]

    # initialize propogator object with inital quaternion and angular velocity
    propagator = AttitudePropagator(q_init=initQ, w_init=w)

    # use attitude propagator to find actual ideal quaternion for n steps
    states = propagator.propagate_states(t0, tf, n)

    # calculate sensor b field for every time step
    # rotation matrix(q) * true B field + noise
    # first value, then all the otheres
    B_sens = np.array([np.matmul(hfunc.quaternion_rotation_matrix(states[0]), B_true)])
    for a in range(1, n):
        B_sens = np.append(B_sens, np.array([np.matmul(hfunc.quaternion_rotation_matrix(states[a]), B_true)]), axis=0)

    # need to add small sensor noise
    
    while i < 1000:

        # create sensor data matrix of magnetomer reading and angular velocity
        data = [0] * dimMes
        data[0] = B_sens[i][0]
        data[1] = B_sens[i][1]
        data[2] = B_sens[i][2]
        data[3] = w[0]
        data[4] = w[1]
        data[5] = w[2]

        # run ukf algorithm for each iteration
        # note: for this test, b field is passed as gps_data instead of gps data
        start, cov, innov, innovCov = UKF_algorithm.UKF(start, cov, q, r, dt, list(B_true), reaction_speeds, old_reaction_speeds, data)

        # uncomment to run fully and visualize after
        # results.append(list(start[:4]))

        # debug print statements
        # print("Data: ", data)
        print("Data: ", ["{:0.4f}".format(x) for x in data])
        # print("State: ", start[:4])
        print("State quaternion: ", ["{:0.4f}".format(x) for x in start[:4]])
        # print("Ideal quaternion: ", states[i])
        print("Ideal quaternion: ", ["{:0.4f}".format(x) for x in states[i]])

        print("")

        # draw our estimate's quaternion
        game_visualize(np.array([start[:4]]), i)

        # draw ideal state quaternion
        # game_visualize(np.array([states[i]]), i)
        i += 1

    # uncomment to run fully and visualize after
    # results = results[1:]
    # print(results[:10])
    # game_visualize(np.array(results), 0)

    


if __name__ == "__main__":

    # run_basic_test()
    run_moving_test()