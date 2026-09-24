"""Fine attitude filter.

Purpose:  Estimate attitude q, body rate omega, and gyro bias from gyro, mag,
          and (when valid) sun measurements.
Inputs:   Conditioned ``Sample[Vec3]`` gyro [rad/s], mag [T], sun (unit);
          reference ``B_eci`` [T] and ``s_eci``; wheel momentum; dt [s].
Outputs:  q (body <- inertial), ``omega_body``, ``gyro_bias_body``, covariance.
Status:   [MODIFIED] [BLOCKED: OQ-1]
Port from: legacy/ukf/UKF_algorithm.py, legacy/ukf/hfunc.py (copied below).
           Reference only: legacy/ukf/kalman_ideal_test_cases.py,
           legacy/ukf/ukfDocs/report.tex, tests/filter/kalman_tests.py.

Related open questions: OQ-1 (MEKF vs extending the UKF).

Differences from legacy to resolve while porting:
    - Legacy state is [q(4), omega(3)] (n=7) with no gyro bias; target state
      adds bias.
    - Legacy UKF propagates with the full sim EOMs (``Simulator/EOMs.py``) and
      reaction-wheel speeds; flight filter should use gyro-driven kinematics.
    - Legacy ``UKF`` takes ``b_true`` (sim truth) as the reference field; flight
      must use ``igrf_model``.
    - Legacy reads module globals from ``params.py``; take ``EstimationParams``.
"""
from __future__ import annotations

# TODO(bug): legacy hfunc takes Bfield in microtesla while other code uses Tesla
# (mixed µT/T units). Use Tesla throughout.

from typing import TYPE_CHECKING, Optional

import numpy as np
import numpy.typing as npt

from cloversat.datatypes.sample_types import Sample
from cloversat.datatypes.vector_types import Quat, Vec3

if TYPE_CHECKING:
    from cloversat.config.estimation_params import EstimationParams


class AttitudeFilter:
    """Attitude + rate + gyro-bias filter (type decided by OQ-1)."""

    def __init__(self, params: EstimationParams) -> None:
        raise NotImplementedError

    def initialize(self, q0: Quat, covariance0: npt.NDArray[np.float64], t: float) -> None:
        """Start the filter from an initial attitude (from ``attitude_init``)."""
        raise NotImplementedError

    def predict(self, gyro: Sample[Vec3], h_wheels_body: Vec3, dt: float) -> None:
        """Propagate the state by ``dt`` [s] using gyro rates."""
        raise NotImplementedError

    def update_mag(self, mag: Sample[Vec3], B_ref_eci: Vec3) -> None:
        """Measurement update with a clean magnetometer sample."""
        raise NotImplementedError

    def update_sun(self, sun: Sample[Vec3], s_ref_eci: Vec3) -> None:
        """Measurement update with a valid sun sample."""
        raise NotImplementedError

    def attitude(self) -> Quat:
        """Current q, body <- inertial."""
        raise NotImplementedError

    def rates(self) -> Vec3:
        """Current bias-corrected ``omega_body`` [rad/s]."""
        raise NotImplementedError

    def gyro_bias(self) -> Vec3:
        """Current gyro bias estimate [rad/s]."""
        raise NotImplementedError

    def covariance(self) -> Optional[npt.NDArray[np.float64]]:
        """Current covariance, or None before initialization."""
        raise NotImplementedError

    def is_converged(self) -> bool:
        """True when covariance is below the convergence threshold."""
        raise NotImplementedError


# ---- LEGACY START: legacy/ukf/UKF_algorithm.py (reference only, not wired up) ----
'''
UKF_algorithm.py
Authors: Andrew Gaylord, Claudia Kuczun, Michael Paulucci, Alex Casillas, Anna Arnett

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
# [scaffold: disabled] from hfunc import *
from typing import Optional
import os
import sys

# To import module that is in the parent directory of your current module:
# [scaffold: disabled] sys.path.insert(1, os.path.join(sys.path[0], '..'))

# [scaffold: disabled] from params import *
# [scaffold: disabled] from Simulator.EOMs import *


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


def generatePredMeans(eomsFunc, sigmaPoints, w0, w1, dt, reaction_speeds, old_reaction_speeds, n):
    '''
    generatePredMeans
        generate mean (eq 9) after passing sigma point distribution through a transformation function (eq 8)
        also stores and returns all transformed sigma points

    @params
        eomsFunc: EOMs function to pass our sigma points through
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
        x = eomsFunc(sigmaPoints[i][:4], sigmaPoints[i][4:], reaction_speeds, 0, alpha, dt)

        # store calculated sigma point in transformed sigma matrix
        transformedSigma[i] = x
        # update mean with point
        means = np.add(means, x)

    # apply weight to mean without first point
    means *= w1

    # pass first sigma point through transformation function
    x = eomsFunc(sigmaPoints[0][:4], sigmaPoints[0][4:], reaction_speeds, 0, alpha, dt)

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
    k = K_SCALING
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
    # eq 8-9: pass sigma points through EOMs (f) and generate mean in state space
    predMeans, f = generatePredMeans(ukf_propagator, sigmaPoints, w0_m, w1, dt, reaction_speeds, old_reaction_speeds, n)

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
# ---- LEGACY END ----


# ---- LEGACY START: legacy/ukf/hfunc.py (reference only, not wired up) ----
'''
hfunc.py
Authors: Micheal Paulucci, Andrew Gaylord

Transformation function hfunc for IrishSat Unscented Kalman Filter. Requires wmm.py and associated files.
'''

import numpy as np
# [scaffold: disabled] import matplotlib.pyplot as plt
# [scaffold: disabled] from PySOL.wmm import WMM
# [scaffold: disabled] from mpl_toolkits.mplot3d import Axes3D
import math
from scipy.spatial.transform import Rotation
import sys, os
# [scaffold: disabled] sys.path.insert(1, os.path.join(sys.path[0], '..'))
# [scaffold: disabled] from params import *
# [scaffold: disabled] from Utils.transformations import *


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
# ---- LEGACY END ----
