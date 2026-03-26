'''
hfunc.py
Authors: Micheal Paulucci, Andrew Gaylord

Transformation function hfunc for IrishSat Unscented Kalman Filter. Requires wmm.py and associated files.
'''

import numpy as np
import matplotlib.pyplot as plt
from PySOL.wmm import WMM
from mpl_toolkits.mplot3d import Axes3D
import math
from scipy.spatial.transform import Rotation
import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from params import *
from Utils.transformations import *


def hfunc(state, Bfield):
    '''
    hfunc
        transformation from state space to measurement space using magnetic field with respect to the earth at given time
        goes from earth orientation to CubeSat orientation so that it aligns with what our sensors will be giving us

    @params
        state: state estimate of system-quaternion, angular velocity, reaction wheel speed (1 x n)
        Bfield: B field in ECI frame of earth (1 x 3) (microteslas)
            used to be controls: gps and time data needed to calculate magnetic field with respect to the earth
            (latitude, longitude, height, time arrays)
            but now we calculate that separately

    @returns
        state array in measurement space (1 x n, with first element of quaternion becoming 0)
    '''

    # find rotation matrix of state quaternion
    quaternion = state[:4]
    rotationMatrix = quaternion_rotation_matrix(quaternion)

    # should we normalize?

    # TODO: Get b_earth from GPS, then cross with magnetometer data (b_body). Then convert 3d vector to quaternion
    # Need to go from simulation-aware to flight software aware

    # combine rotation matrix and b field of earth
    # other elements of state have 1 to 1 conversion, so add back before returning
    return np.concatenate((np.matmul(rotationMatrix, Bfield).ravel(), np.array(state[4:])))

    # For 1D use:
    '''
    # find angle in xy plane
    psi = state[0] # rad
    rotationMatrix = np.array([[np.cos(psi), -np.sin(psi)], [np.sin(psi), np.cos(psi)]])

    # calculate measured B-field
    #Btrue = np.array([Bx_true, By_true])
    Bmeas = np.matmul(rotationMatrix, B_true)

    return np.append(Bmeas, state[1])
    '''


if __name__ == '__main__':

    # example quaternion that we want to represent in measurement space
    # converts local frame of 1 to global frame dictated by earth's B field
    q = np.array([1, 0, 1, 1])
    q = normalize(q)
    rotationMatrix = quaternion_rotation_matrix(q)

    print('quaternion: ',q,'\nrotation matrix: ', rotationMatrix)

    # in our H func, original is the B field, q is our current quaternion
    original = [1,0,0]
    rotated = np.matmul(rotationMatrix,original)
    print("rotated: ", rotated)

    #PLOTTING (DOESNT MATTER)
    original = np.concatenate(([0, 0, 0], original))
    rotated = np.concatenate(([0, 0, 0], rotated))

    # print(original)

    soa = np.array([original, rotated])

    X, Y, Z, U, V, W = zip(*soa)

    print("vectors to graph: ", X, Y, Z, U, V, W)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.quiver(X, Y, Z, U, V, W)
    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])
    ax.set_zlim([-2, 2])
    plt.show()




        # n = 10
    # state = np.random.rand(n)
    # transformed = np.array()
    # need some kind of generator for random long/lat coordinates, height, and time
    #   arrays needed for controls:
        # lat_gd (np.array): array holding the geodesic latitude associated with a state
        # lon (np.array): array holding the longtitude associated with a state
        # h_ellp (np.array): array holding the estimated heights above the ellipsoid in m
        # t (np.array): array of times associated with an array of states, given in decimal years
    # controls = [[]]

    # Perform observation function, only needed for quaternion components. The rest have 1 to 1 mapping
    # transformed.append(transformed, np.array(hfunc(state, controls)))

    # transformed.append(transformed, np.array(state[4:]))

    # transformed = np.array([np.array(hfunc(state, q_wmm)), np.array(state[4:])])