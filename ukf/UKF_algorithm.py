'''
UKF_algorithm.py
Authors: Andrew Gaylord, Claudia Kuczun, Michael Paulucci, Alex Casillas, Anna Arnett
Last modified 3/3/24

Unscented Kalman Filter algorithm for IrishSat based on following resource:
The Unscented Kalman Filter for Nonlinear Estimation
Eric A. Wan and Rudolph van der Merwe
Oregon Graduate Institute of Science & Technology

Variables needed throughout UKF process:
  n = dimensionality of model (7)
  m = dimension of measurement space. Frame of reference of the satellite/sensors (6)
  q: process noise covariance matrix (n x n)
  r: measurement noise covariance matrix (m x m)
  scaling = parameter for sigma point generation (equal to alpha^2 * (n + k) - n)
  means = estimated current state (1 x n)
  cov = covariance matrix of current state (n x n)
  predMeans = matrix of predicted means based on physics EOMs (1 x n)
  predCov = matrix of predicted covariance (n x n)
  f = matrix of predicted sigma points (state space using EOMs) (2*n+1 x n)
  h = matrix of transformed sigma points (in the measurement space using hfunc) (2*n+1 x m)
  mesMeans = means in the measurement space after nonlinear transformation (hfunc) (1 x m)
  mesCov = covariance matrix of points in measurement space (m x m)
  data: magnetometer (magnetic field) and gyroscope (angular velocity) data reading from sensor at each step (1 x m)
  kalman = kalman gain for each step, measures how much we should change based on our trust of sensors vs our model (n x m)
'''

import numpy as np
import math
import scipy
import scipy.linalg
from hfunc import *
from typing import Optional
import os
import sys

# To import module that is in the parent directory of your current module:
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from params import *


def sigma(means, cov, n, scaling):
    '''
    sigma
        creates sigma point matrix that is a representative sampling of the mean and covariance of the system (eq 5-7)
        this allows for efficient transferal of information through our nonlinear function as we estimate our measurement space

    @params
        means: mean of estimated states so far. Also first column of sigma matrix (eq 5) (1 x n)
        cov: covariance matrix of state (n x n)
        n: dimensionality of model
        scaling: how far from mean we distribute our points, used in sigma point formula

    @returns
        sigmaMatrix: matrix of sigma points (2 * n + 1, n)
    '''
    # intialize 2N + 1 sigma points to zeroes
    sigmaMatrix = np.zeros((2*n+1,n))
    temp = np.zeros((n, n))

    # 1) sigma point generation
    # first column of sigma matrix is means
    sigmaMatrix[0] = means

    # take the square root of the inside
    temp = scipy.linalg.sqrtm(np.multiply(cov, (n + scaling)))

    # traverse n dimensions, calculating all other sigma points
    # means + sqrt for 1 to n
    sigmaMatrix[1:(n+1)] = np.add(means, temp)
    # means - sqrt for n + 1 to 2*n
    sigmaMatrix[(n+1):(2*n+1)] = np.subtract(means, temp)

    # return the sigma matrix (2 * n + 1 columns)
    return sigmaMatrix


class TEST1EOMS():
    '''
    Equations of motion class cubesat model
    '''
    def __init__(self, I_body: np.ndarray, I_w_spin: float, I_w_trans: float):

        # Initially store moment of inertia tensor w/o reaction wheel inertias!
        self.I_body = I_body
        self.I_body_inv = np.linalg.inv(I_body)

        # Store principal moment of inertia for reaction wheels about spin axis and about axis transverse to spin axis respectively
        self.I_w_spin = I_w_spin
        self.I_w_trans = I_w_trans

        # Transformation matrix for NASA config given in Fundamentals pg 153-154
        # converts from 4 to 3 reaction wheels
        self.W = TRANSFORMATION

        # Update: self.rw_config_inertia is almost certainly wrong here, need to rewrite! Looks like it's a rotation about Z
        # by theta_1D = 135 degrees
        #theta_1D = 135*np.pi/180.0
        #self.rw_config_inertia = np.array([[np.cos(theta_1D), np.sin(theta_1D), 0], [-np.sin(theta_1D), np.cos(theta_1D), 0], [1/np.sqrt(2), 0, 1/np.sqrt(2)]])

        # Moment of inertia tensor of rxn wheels [kg m^2]
        self.rw_config_inertia = RW_CONFIG_INERTIA

        # Calculate contributions of reaction wheel to moment of inertia tensor due to principal moment transverse to the spin axis
        # wtf this mean?????
        for i in np.arange(self.rw_config_inertia.shape[1]):
            self.I_body = self.I_body + I_w_trans*(np.identity(3) - np.matmul(self.rw_config_inertia[:, i], np.transpose(self.rw_config_inertia[:, i])))


    def eoms(self, quaternion: np.ndarray, w_sat: np.ndarray, w_rw: np.ndarray, tau_sat: np.ndarray, w_rw_dot: np.ndarray, dt: float):
        '''
        Uses the Equations of Motion (EOMs) to yield the next values of quaternion and angular velocity of satellite
        The EOMs are based upon the current state, current reaction wheel speeds, and external torque being applied by the magnetorquers
        Propogates our state vector through time using physics predictions

        @params:
            quaternion (np.ndarray, (1x4)): quaternion describing orientation of satellite with respect to given reference frame
            w_sat (np.ndarray, (1x3)): angular velocity of whole satellite (rad/s)
            w_rw (np.ndarray, (1x4)): angular velocities of wheels (in respective wheel frame) (rad/s)
            tau_sat (np.ndarray, (1x3)): external torque applied on the satellite, such as magnetorquers (N*m)
            w_rw_dot (nd.ndarray, (1x4)): angular acceleration of the reaction wheels in their respective wheel frames (derivative of w_rw)
            dt (float): timestep (seconds)

        @returns:
            (1x7) state vector containing new quaternion and angular velocity of satellite
                quaternion_new (np.ndarray, (1x4)): new orientation after dt seconds
                w_sat_new (np.ndarray, (1x3)): new angular velocity of satellite (rad/s)
        '''

        # store components of angular velocity of satellite
        w_x = w_sat[0]
        w_y = w_sat[1]
        w_z = w_sat[2]

        # Quaternion product matrix for angular velocity of satellite
        ### THIS IS THE BIG OMEGA that should work for our scalar-component of quaternion first notation (based on https://ahrs.readthedocs.io/en/latest/filters/angular.html)
        w_sat_skew_matrix = np.array([[0, -w_x, -w_y, -w_z],
                                    [w_x, 0, w_z, -w_y],
                                    [w_y, -w_z, 0, w_x],
                                    [w_z, w_y, -w_x, 0]])

        # find reaction wheel angular momentum in the body frame (based on inertia and rw angular velocity)
        rw_angular_momentum_body = self.I_w_spin * np.matmul(self.rw_config_inertia, w_rw)

        # Similarly find vector describing torque caused by reaction wheel in the body frame
        # torque = derivative of angular momentum
        rw_torque_body = self.I_w_spin * np.matmul(self.rw_config_inertia, w_rw_dot)

        # First derivative of quaternion
        quaternion_dot = (1/2) * np.matmul(w_sat_skew_matrix, quaternion)

        # First derivative of angular velocity
        # Depends on external torque, reaction wheel torque, and angular momentum
        #   Torque and angular momentum are multiplied by self.W to convert from 4 to 3 axes
        #   When taking derivative of angular momentum in the body frame, the time rate of change of
        #       the rotation of the coordinate frame itself relative to the inertial frame must be accounted for, which
        #       is why the cross product with w_sat is introduced
        # TODO: should w_sat be negative? (like so: np.cross(-w_sat, np.matmul(self.I_body, w_sat))
        w_sat_dot = np.matmul(self.I_body_inv, (tau_sat - np.matmul(self.W, rw_torque_body) - np.cross(w_sat, np.matmul(self.I_body, w_sat) + np.matmul(self.W, rw_angular_momentum_body))))

        # propagate quaternion and angular velocity through time using Euler's method
        quaternion_new = normalize(quaternion + quaternion_dot*dt)
        w_sat_new = w_sat + w_sat_dot*dt

        new_state = np.append(quaternion_new, w_sat_new)

        return new_state

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


def generatePredMeans(eomsClass, sigmaPoints, w0, w1, dt, reaction_speeds, old_reaction_speeds, n):
    '''
    generatePredMeans
        generate mean (eq 9) after passing sigma point distribution through a transformation function (eq 8)
        also stores and returns all transformed sigma points

    @params
        eomsClass: EOMs class to pass our sigma points through
        sigmaPoints: sigma point matrix (2xn+1 x n)
        w0, w1: weight for first and all other sigma points, respectively
        reaction_speeds/old_reaction_speeds: reaction wheel speeds for current and last time step (1 x 3 for 1d test)
        n: dimensionality of state space

    @returns
        means: mean of distribution in state space (1 x n)
        transformedSigma: sigma matrix of transformed points (n*2+1 x n)
    '''
    # initialize means and new sigma matrix with correct dimensionality
    means = np.zeros(n)
    transformedSigma = np.zeros((2 * n + 1, n))

    # calculate angular acceleration using old and current reaction wheel speeds
    alpha = (reaction_speeds - old_reaction_speeds) / dt

    # pass all sigma points to the transformation function
    for i in range(1, n * 2 + 1):
        # 3a) and 4a)
        # TODO: add external torques (tau_sat)
        x = eomsClass.eoms(sigmaPoints[i][:4], sigmaPoints[i][4:], reaction_speeds, 0, alpha, dt)

        # store calculated sigma point in transformed sigma matrix
        transformedSigma[i] = x
        # update mean with point
        means = np.add(means, x)

    # apply weight to mean without first point
    means *= w1

    # pass first sigma point through transformation function
    x = eomsClass.eoms(sigmaPoints[0][:4], sigmaPoints[0][4:], reaction_speeds, 0, alpha, dt)

    # store new point as first element in transformed sigma matrix
    transformedSigma[0] = x

    # adjust the means for first value and multiply by correct weight
    means = np.add(means, x*w0)

    return means, transformedSigma


def generateMesMeans(func, controlVector, sigmaPoints, w0, w1, n, dimensionality):
    '''
    generateMesMeans
        generate mean (eq 12) after passing sigma point distribution through non-linear transformation function (eq 11)
        also stores and returns all transformed sigma points

    @params
        func: transformation function we are passing sigma points through (H_func)
        controlVector: additional input needed for func: true magnetic field (1 x 3)
        sigmaPoints: sigma point matrix (2xn+1 x n)
        w0, w1: weight for first and all other sigma points, respectively
        n: dimensionality of model
        dimensionality: dimensionality of what state we are generating for (measurement space: m)

    @returns
        means: mean of distribution in measurement space (1 x m)
        transformedSigma: sigma matrix of transformed points (n*2+1 x m)
    '''
    # initialize means and new sigma matrix with correct dimensionality
    means = np.zeros(dimensionality)
    transformedSigma = np.zeros((2 * n + 1, dimensionality))

    # pass all sigma points to the transformation function
    for i in range(1, n * 2 + 1):
        # 3a) and 4a)
        x = func(sigmaPoints[i], controlVector)
        # store calculated sigma point in transformed sigma matrix
        transformedSigma[i] = x
        # update mean with point
        means = np.add(means, x)

    # apply weight to mean without first point
    means *= w1

    # pass first sigma point through transformation function
    x = func(sigmaPoints[0], controlVector)

    # store new point as first element in transformed sigma matrix
    transformedSigma[0] = x

    # adjust the means for first value and multiply by correct weight
    means = np.add(means, x*w0)

    return means, transformedSigma


def generateCov(means, transformedSigma, w0, w1, n, noise):
    '''
    generateCov
        generates covariance matrix from eq 10 and 13 based on means and sigma points

    @params
        means: means in state or measurement space (1 x n or 1 x m)
        transformedSigma: stored result of passing sigma points through the EOMs or H_func (n*2+1 x m or n*2+1 x n)
        w0, w1: weight for first and all other sigma points, respectively
        n: dimensionality of model
        noise: noise value array to apply to our cov matrix (r or q)

    @returns
        cov: covariance matrix in state or measurement space (n x n or m x m)
    '''
    # find dimension of cov by looking at size of sigma point array
    # prediction points will have n columns, measurement points will have m columns
    covDimension = transformedSigma.shape[1]

    # initialize cov with proper dimensionality
    cov = np.zeros((covDimension, covDimension))

    # for all transformed sigma points, apply covariance formula
    for i in range(1, n * 2 + 1):
        # subtract mean from sigma point and multiply by itself transposed
        arr = np.subtract(transformedSigma[i], means)[np.newaxis]
        arr = np.matmul(arr.transpose(), arr)
        cov = np.add(cov, arr)

    # separate out first value and update with correct weight
    arr = np.subtract(transformedSigma[0], means)[np.newaxis]
    d = np.matmul(arr.transpose(), arr) * w0

    # use other weight for remaining values
    cov *= w1

    # add back first element
    cov = np.add(cov, d)

    # add noise to covariance matrix
    cov = np.add(cov, noise)

    return cov


def generateCrossCov(predMeans, mesMeans, f, h, w0, w1, n):
    '''
    generateCrossCov
        use equation 14 to generate cross covariance between our means and sigma points in our state and measurement space

    @params
        predMeans: predicted means based on EOMs (1 x n)
        mesMeans: predicted means in measurement space (1 x m)
        f: sigma point matrix that has passed through the EOMs (n*2+1 x n)
        h: sigma point matrix propogated through non-linear transformation h func (n*2+1 x m)
        w0: weight for first value
        w1: weight for other values
        n: dimensionality of model

    @returns
        crossCov: represents uncertainty between our state and measurement space estimates (n x m)
    '''
    m = len(mesMeans)
    crossCov = np.zeros((n,m))

    for i in range(1, n * 2 + 1):
        arr1 = np.subtract(f[i], predMeans)[np.newaxis]
        arr2 = np.subtract(h[i], mesMeans)[np.newaxis]
        arr1 = np.matmul(arr1.transpose(), arr2)  # ordering?
        crossCov = np.add(crossCov, arr1)

    arr1 = np.subtract(f[0], predMeans)[np.newaxis]
    arr2 = np.subtract(h[0], mesMeans)[np.newaxis]

    # seperate out first element
    d = np.matmul(arr1.transpose(), arr2)

    # multiply by weights for first and other values
    crossCov = np.multiply(crossCov, w1)
    d = np.multiply(d, w0)

    # add first value back into cross covariance
    crossCov = np.add(crossCov, d)

    return crossCov


def UKF(means, cov, q, r, dt, b_true, reaction_speeds, old_reaction_speeds, data):
    '''
    UKF
        estimates state at time step based on sensor data, noise, and equations of motion

    @params
        means: means of previous states (1 x n)
        cov: covariance matrix of state (n x n)
        q: process noise covariance matrix (n x n)
        r: measurement noise covariance matrix (m x m)
        b_true: true magnetic field of satelitte with respect to earth in eci frame (microteslas) (1 x 3)
        reaction_speeds: control input for EOMs (1 x 4)
        old_reaction_speeds: speeds for past step, used to find angular acceleration (1 x 4)
        data: magnetometer (magnetic field) and gyroscope (angular velocity) data reading from sensor (1 x m)

    @returns
        means: calculated state estimate at current time (1 x n)
        cov: covariance matrix (n x n)
        innovation: difference between measurement and prediction (1 x m)
        innovationCov: covariance matrix of innovation (m x m)
    '''

    # dimensionality of state space = dimension of means
    n = len(means)
    # dimensionality of measurement space = dimension of measurement noise
    m = len(r)

    # scaling parameters
    # alpha and k scale points around the mean. To capture the kurtosis of a gaussian distribution, a=1 and k=3-n should be used
    #   If a decrease in the spread of the SPs is desired, use κ = 0 and alpha < 1
    #   If an increase in the spread of the SPs is desired, use κ > 0 and alpha = 1
    alpha = ALPHA
    k = K
    # beta minimizes higher order errors in covariance estimation
    beta = BETA
    # eq 1: scaling factor lambda
    scaling = alpha * alpha * (n + k) - n

    # eq 2-4: weights calculation
    w0_m = scaling / (n + scaling) # weight for first value for means
    w0_c = scaling / (n + scaling) + (1 - alpha * alpha + beta) # weight for first value for covariance
    w1 = 1 / (2 * (n + scaling)) # weight for all other values


    # eq 5-7: sigma point generation
    sigmaPoints = sigma(means, cov, n, scaling)


    # prediction step
    I_body = CUBESAT_BODY_INERTIA
    I_spin = SPIN_AXIS_INERTIA
    # used to edit our body's moment of inertia tensor somehow (0 for none)
    I_trans = TRANSVERSE_AXIS_INERTIA
    # intialize 1D EOMs using intertia measurements of cubeSat
    EOMS = TEST1EOMS(I_body, I_spin, I_trans)

    # eq 8-9: pass sigma points through EOMs (f) and generate mean in state space
    predMeans, f = generatePredMeans(EOMS, sigmaPoints, w0_m, w1, dt, reaction_speeds, old_reaction_speeds, n)

    # print("PREDICTED MEANS: ", predMeans)

    # eq 10: generate predicted covariance + process noise q
    predCov = generateCov(predMeans, f, w0_c, w1, n, q)

    # print("PRED COVID: ", predCov)


    # non linear transformation
    # eq 11-12: non linear transformation of predicted sigma points f into measurement space (h), and mean generation
    mesMeans, h = generateMesMeans(hfunc, b_true, f, w0_m, w1, n, m)

    # print("MEAN IN MEASUREMENT: ", mesMeans)

    # eq 13: measurement covariance + measurement noise r
    mesCov = generateCov(mesMeans, h, w0_c, w1, n, r)


    # measurement updates
    # eq 14: cross covariance. compare our different sets of sigma points and our predicted/measurement means
    crossCov = generateCrossCov(predMeans, mesMeans, f, h, w0_c, w1, n)

    # print("covariance in measurement: ", mesCov)
    # print("cross covariance: ", crossCov)

    # eq 15: calculate kalman gain (n x m) by multiplying cross covariance matrix and transposed predicted covariance
    kalman = np.matmul(crossCov, np.linalg.inv(mesCov))

    # print("KALMAN: ", kalman)

    # eq 16: updated final mean = predicted + kalman(measurement data - predicted means in measurement space)
    means = np.add(predMeans, np.matmul(kalman, np.subtract(data, mesMeans)))

    # normalize the quaternion to reduce small calculation errors over time
    # see note in eoms about normalizing
    means[0:4] = normalize(means[:4])

    # eq 17: updated covariance = predicted covariance - kalman * measurement cov * transposed kalman
    cov = np.subtract(predCov, np.matmul(np.matmul(kalman, mesCov), kalman.transpose()))

    # innovation testing

    # difference between measurement and prediction
    # V_k+1 = Z_k+1 - Z~_k+1|k
    # V_k+1 = Z_k+1 - H_k+1 * X_k+1|k

    innovation = np.subtract(data, mesMeans)

    # inovation cov
    # measurement noise + transition matrix * cov matrix
    # S_k+1 = R_k+1 + H_K+1 * P_k+1|k * H^T_k+1
    # which literally equals cov in measurement haha
    innovationCov = mesCov

    # print("MEANS AT END: ", means)
    # print("COV AT END: ", cov)
    return [means, cov, innovation, innovationCov]
