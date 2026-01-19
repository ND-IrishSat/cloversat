'''
main.py
Author: Andrew Gaylord

Main file from Kalman_Testing repo

Primarily interfaces with the Simulator class from simulator.py to represent a state estimation model
Sets up fake models to simulate CubeSat and surrounding sensor suites
    This allows us to compare results of different kalman filters and controllers

'''


import sys, os
sys.path.extend([f'./{name}' for name in os.listdir(".") if os.path.isdir(name)])

from sim.PySOL.wmm import *
from sim.visualizer import *
from ukf.UKF_algorithm import *
from ukf.hfunc import *
from sim.simulator import *
from sim.graphing import *
from sim.tests import *
from sim.saving import *
from params import *

import matplotlib.pyplot as plt
import signal


'''

PySOL tells us the B field, ECI, ECEF, LLA
https://kieranwynn.github.io/pyquaternion/#normalisation
https://csimn.com/CSI_pages/PID.html

resources used:
State estimation II by Ian Reed
https://github.com/FrancoisCarouge/Kalman
https://www.researchgate.net/post/How_can_I_validate_the_Kalman_Filter_result
https://stats.stackexchange.com/questions/40466/how-can-i-debug-and-check-the-consistency-of-a-kalman-filter
WikibookonKalmanFilter.pdf

TODO: impliment PySol and print B field (and globe?)
TODO: clean up params.py + get access everywhere
TODO: more statistical tests, test data reading w/ wheels

which method is correct for normalized innovation covariance (test #2)? (and which CI?) (see tests.py)
    should interval bound be added to measurement, 0, or average?

'''


def signal_handler(sig, frame):
    '''
    closes all pyplot tabs when CTRL+C is entered
    '''
    plt.close('all')
    plt.clf()
    plt.cla()
    plt.close()


if __name__ == "__main__":
    
    # set up signal handler to shut down pyplot tabs
    signal.signal(signal.SIGINT, lambda sig, frame: signal_handler(sig, frame))

    ukf = Simulator(UKF, PIDController(KP, KI, KD, DT))

    # ukf.run_filter_sim()
    ukf.run_controls_sim()

    # # plot3DVectors(np.array([ukf.B_true, ukf.data[50][:3], ukf.data[100][:3], ukf.data[150][:3]]), 121)
    # plot3DVectors(result, 111)
    # plotData3D(ukf.data, 5, 111)
    # ideal_xyz = [np.matmul(quaternion_rotation_matrix(x), np.array([1, 0, 0])) for x in ukf.ideal_states]
    # plotData3D(ideal_xyz, 3, 111)