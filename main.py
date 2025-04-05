'''
main.py
Author: Andrew Gaylord

Main file from Kalman_Testing repo

Sets up fake models to simulate CubeSat and surrounding sensor suites
    This allows us to compare results of different kalman filters and controllers
Utilizes graphing.py for vizualizations of state and statistical tests
Uses the Simulator class from simulator.py to represent a state estimation model for a certain kalman filter

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
TODO: move runfilter into Filter, make different getData funcs, put step into propogate
    **put all params in filter_init**
TODO: clean up params.py + get access everywhere
TODO: more statistical tests, test data reading w/ wheels
in main: run filter option should have 3 get data options: real (with viz), ideal, text file 

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


def run_filter_sim(filter, magNoises, gyroNoises):
    '''
    Generates ideal states and sensor data, allowing us to benchmark our kalman filter against simulated "truth". 
    Can also be run with pre-existing sensor data (ideal_known = False)

    @params:
        filter: kalman Filter object to run simulation with
        magNoises: magnetometer noise at each step
        gyroNoises: gyroscope noise at each step
    '''

    # text file with data values
    dataFile = DATA_FILE

    # if we aren't using real data, we need to make up reaction wheel speeds, find ideal state, and generate fake data
    if not filter.ideal_known:
        # this populates ukf.data and ukf.reaction_speeds
        filter.loadData(dataFile)
    else:
        # decide how we want our reaction wheels to spin at each time step
        # parameters: max speed, min speed, number of steps to flip speed after, step, bitset of which wheels to activate
        filter.generateSpeeds(400, -400, filter.n, 40, np.array([0, 1, 0, 0]))

        # find ideal state of cubesat through physics equations of motion
        filter.propagate()

        # generate data reading for each step 
        filter.generateData(magNoises, gyroNoises, 0)


    # run our data through the specified kalman function
    filter.simulate()

    # if true, run statistical tests outlined in Estimation II by Ian Reed
    # these tests allow us to see how well our filter is performing
    runTests = RUN_STATISTICAL_TESTS
    sum = 0
    if runTests:
        sum = filter.runTests()

    # plot our results and create pdf output + 3D visualization
    plot_and_viz_results(filter, sum=sum)


def run_controls_sim(filter, magNoises, gyroNoises):
    '''
    Combines motor dynamics and PID controller to orient towards a target
    Propogates our state step by step, as we want to dynamically change our "ideal" state based on our control output

    @params:
        filter: Filter object to run simulation with
        magNoises: magnetometer noise at each step
        gyroNoises: gyroscope noise at each step
    '''

    # generate data for first step so we can start at i = 1
    filter.generateData_step(0, magNoises[0], gyroNoises[0])

    # Initialize PID controller
    kp = KP   # Proportional gain
    # kp = MAX_PWM * 7e-8
    # close to kp allows for narrowing in on target, but not too close
    # smaller = oscillating more frequently, larger = overshooting more
    ki = KI     # Integral gain
    # ki = MAX_PWM * 5e-9
    # if this is too high, it overrotates
    kd = KD  # Derivative gain
    # kd = MAX_PWM * 1e-8
    pid = PIDController(kp, ki, kd, filter.dt)

    # define our target orientation and whether we want to reverse it halfway through
    # TODO: x axis is bugged (or just different moments of inertia). Wants to go sideways
    target = normalize(TARGET)
    flip = False

    for i in range(1, filter.n):

        # get ideal next state based on current state and reaction wheel speeds of this step
        # NOTE: this "ideal" state is not super based on truth because it is not generated beforehand. 
        #       it basically follows what our filter does, so it is not a good representation of the truth
        ideal = filter.propagate_step(i)
        
        # create fake magnetometer data by rotating B field by ideal quaternion, and gyro by adding noise to angular velocity
        filter.generateData_step(i, magNoises[i], gyroNoises[i])

        # filter our data and get next state
        # also run through our controls to get pwm => voltage => current => speed of reaction wheels
        filtered = filter.simulate_step(i, target, pid)
        # game_visualize(np.array([filtered]), i-1)

        if i > filter.n / 2 and flip == True:
            target = normalize(QUAT_INITIAL)

    # plot our results and create pdf output + 3D visualization
    plot_and_viz_results(filter, controller=pid, target=target)


def plot_and_viz_results(filter, controller=None, target=np.array([1, 0, 0, 0]), sum=0):
    '''
    Plots out filter states, data, and reaction wheel speeds, and creates pdf output + 3D visualization
    Allows us to visualize results of our filter/controls sim
    Based upon RESULT variable in params.py

    @params:
        filter: Filter object to plot and visualize results of
        controller: PIDController object (for controls sim)
        target: target quaternion (for controls sim)
        sum: sum of statistical tests if they were run
    '''
    # plot mag and gyro data
    filter.plotData()
    # plots filtered states (and ideal states if ideal_known = True)
    filter.plotStates()
    # plot reaction wheel speeds
    filter.plotWheelInfo()

    # 0 = only create pdf output, 1 = show 3D animation visualization, 2 = both, 3 = none
    visualize = RESULT

    if visualize == 1:
        filter.visualizeResults(filter.filtered_states)

    elif visualize == 0:

        filter.saveFile(OUTPUT_FILE, controller, target, sum, RUN_STATISTICAL_TESTS)

    elif visualize == 2:

        filter.saveFile(OUTPUT_FILE, controller, target, sum, RUN_STATISTICAL_TESTS)

        filter.visualizeResults(filter.filtered_states)

    # only show plot at end so they all show up
    plt.show()


if __name__ == "__main__":
    
    # set up signal handler to shut down pyplot tabs
    signal.signal(signal.SIGINT, lambda sig, frame: signal_handler(sig, frame))

    ukf = Simulator(UKF, PIDController(KP, KI, KD, DT))

    # clear output directory from last simulation
    clearDir(outputDir)

    run_filter_sim(ukf, ukf.magNoises, ukf.gyroNoises)
    # run_controls_sim(ukf, ukf.magNoises, ukf.gyroNoises)


    # # plot3DVectors(np.array([ukf.B_true, ukf.data[50][:3], ukf.data[100][:3], ukf.data[150][:3]]), 121)
    # plot3DVectors(result, 111)
    # plotData3D(ukf.data, 5, 111)
    # ideal_xyz = [np.matmul(quaternion_rotation_matrix(x), np.array([1, 0, 0])) for x in ukf.ideal_states]
    # plotData3D(ideal_xyz, 3, 111)