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

        # def run_filter_sim(self):
        # '''
        # Generates ideal states and sensor data, allowing us to benchmark our kalman filter against simulated "truth".
        # Can also be run with pre-existing sensor data (ideal_known = False and SENSOR_DATA_FILE != None)
        # '''

        # # text file with data values
        # dataFile = SENSOR_DATA_FILE

        # if IDEAL_KNOWN:
        #     # decide how we want our reaction wheels to spin at each time step
        #     # parameters: max speed, min speed, number of steps to flip speed after, step, bitset of which wheels to activate
        #     self.generateSpeeds(400, -400, self.n, 40, np.array([0, 1, 0, 0]))

        #     # find ideal state of cubesat through physics equations of motion
        #     self.propagate()

        # # generate data reading for each step
        # self.populateData()

        # # run our data through the specified kalman function
        # self.simulate()

        # # if true, run statistical tests outlined in Estimation II by Ian Reed
        # # these tests allow us to see how well our filter is performing
        # runTests = RUN_STATISTICAL_TESTS
        # sum = 0
        # if runTests:
        #     sum = self.runTests()

        # # plot our results and create pdf output + 3D visualization
        # self.plot_and_viz_results(sum=sum)



        # def run_controls_sim(self):
        # '''
        # Combines motor dynamics and PID controller to orient towards a target
        # Propogates our state step by step, as we want to dynamically change our "ideal" state based on our control output
        # '''

        # # generate data for first step so we can start at i = 1
        # self.generateData_step(0)

        # # define our target orientation and whether we want to reverse it halfway through
        # # TODO: x axis is bugged (or just different moments of inertia). Wants to go sideways
        # target = normalize(TARGET)
        # flip = False

        # for i in range(1, self.n):

        #     # get ideal next state based on current state and reaction wheel speeds of this step
        #     # NOTE: this "ideal" state is not super based on truth because it is not generated beforehand.
        #     #       it basically follows what our filter does, so it is not a good representation of the truth
        #     ideal = self.propagate_step(i)

        #     # create fake magnetometer data by rotating B field by ideal quaternion, and gyro by adding noise to angular velocity
        #     self.generateData_step(i)

        #     # filter our data and get next state
        #     # also run through our controls to get pwm => voltage => current => speed of reaction wheels
        #     filtered = self.simulate_step(i, target)
        #     # game_visualize(np.array([filtered]), i-1)

        #     # optionally return to starting orientation halfway through
        #     if i > self.n / 2 and flip == True:
        #         target = normalize(QUAT_INITIAL)

        # # plot our results and create pdf output + 3D visualization
        # self.plot_and_viz_results(controller=self.controller, target=target)

    # # plot3DVectors(np.array([ukf.B_true, ukf.data[50][:3], ukf.data[100][:3], ukf.data[150][:3]]), 121)
    # plot3DVectors(result, 111)
    # plotData3D(ukf.data, 5, 111)
    # ideal_xyz = [np.matmul(quaternion_rotation_matrix(x), np.array([1, 0, 0])) for x in ukf.ideal_states]
    # plotData3D(ideal_xyz, 3, 111)