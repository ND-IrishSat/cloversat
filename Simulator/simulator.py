'''
simulator.py
Author: Andrew Gaylord

Contains simulator class for an arbitrary kalman filter and control system
Object contains system info, initialized values, state values, filter specifications, and all outputs
Class functions allow for easy initialization, propagation, data generation, simulation, and visualization

All parameters (variables in caps) are stored in *params.py*

'''


from Simulator.PySOL.wmm import *
from visualizer import *
import time
from graphing import *
from Simulator.kalman_tests import *
from saving import *

import os
import sys

# import params module from parent directory
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from params import *
from Controllers.PID_controller import *
from ukf.UKF_algorithm import *
from ukf.hfunc import *


class Simulator():
    def __init__ (self, kalmanMethod, controller):
        # number of steps to simulate
        self.n = int(TF / DT)
        # timestep between steps
        self.dt = DT
        # dimension of state and measurement space
        self.dim = STATE_SPACE_DIMENSION
        self.dim_mes = MEASUREMENT_SPACE_DIMENSION

        # set process noise and update starting cov guess
        # parameters: noise magnitude, k (see Estimation II article by Ian Reed)
        self.ukf_setQ(PROCESS_NOISE_MAG, PROCESS_NOISE_K)

        # set measurement noise
        # parameters: magnetometer noise, gyroscope noise
        self.ukf_setR(MEASUREMENT_MAGNETOMETER_NOISE, MEASUREMENT_GYROSCOPE_NOISE)

        # starting state (default is standard quaternion and no angular velocity)
        self.state = np.concatenate((normalize(QUAT_INITIAL), VELOCITY_INITIAL))
        # starting covariance (overrid by ukf_setQ)
        self.cov = np.identity(self.dim) * COVARIANCE_INITIAL_MAG

        # 2D array of n innovations and covariances (populated by filter.simulate)
        self.innovations = np.zeros((self.n, self.dim_mes))
        self.innovationCovs = np.zeros((self.n, self.dim_mes, self.dim_mes))

        # true magnetic field for every timestep in simulation
        if CONSTANT_B_FIELD:
            self.B_true = np.full((self.n, 3), CONSTANT_B_FIELD_MAG)
        else:
            # TODO: add PySOL propogation
            pass

        # Motor states
        self.current = np.array([0.0, 0.0, 0.0, 0.0]) # Current to each motor
        self.Th_Ta = np.array([0.0, 0.0, 0.0, 0.0]) # diff in temp between housing and ambient
        self.Tw_Ta = np.array([0.0, 0.0, 0.0, 0.0]) # diff in temp between winding and ambient
        # current for all n steps
        self.currents = np.zeros((self.n, 4))
        self.currents[0] = self.current

        # set sensor noises (if we're using ideal states to simulate sensor data)
        if IDEAL_KNOWN:
            magSD = SENSOR_MAGNETOMETER_SD
            self.magNoises = np.random.normal(0, magSD, (self.n, 3))

            gyroSD = SENSOR_GYROSCOPE_SD
            self.gyroNoises = np.random.normal(0, gyroSD, (self.n, 3))

        # 1x4 array of current reaction wheel speeds
        self.curr_reaction_speeds = RW_INITIAL
        # reaction wheel speed of last time step
        self.last_reaction_speeds = RW_INITIAL

        # reaction wheel speeds for all n steps
        self.reaction_speeds = np.zeros((self.n, 4))
        self.reaction_speeds[0] = RW_INITIAL

        # data values for all n steps
        self.data = np.zeros((self.n, self.dim_mes))

        # ideal states from EOMs for all n steps
        self.ideal_states = np.zeros((self.n, self.dim))
        self.ideal_states[0] = self.state

        # indicates whether we know our ideal states or not (i.e. if we are simulating or running with real data from text file or live)
        self.ideal_known = IDEAL_KNOWN

        # kalman filtered states for all n steps
        self.filtered_states = np.zeros((self.n, self.dim))
        self.filtered_states[0] = self.state

        # pwm values (motor signals) for all n steps
        self.pwms = np.zeros((self.n, 4))
        self.pwms[0] = np.array([0, 0, 0, 0])

        # covariance of system for all n steps
        self.covs = np.zeros((self.n, self.dim, self.dim))
        self.covs[0] = self.cov

        # what kalman filter to apply to this system
        self.kalmanMethod = kalmanMethod

        # controller object to use for this system (stores gain constants and provides pwm signal generation function)
        self.controller = controller

        # filter times for each step (for efficiency testing)
        self.times = np.zeros(self.n)


    def ukf_setR(self, magNoise, gyroNoise):
        '''
        set measurement noise R (dim_mes x dim_mes)

        @params:
             magNoise: noise for magnetometer
             gyroNoise: noise for gyroscope
        '''

        self.R = np.array([[magNoise, 0, 0, 0, 0, 0],
                        [0, magNoise, 0, 0, 0, 0],
                        [0, 0, magNoise, 0, 0, 0],
                        [0, 0, 0, gyroNoise, 0, 0],
                        [0, 0, 0, 0, gyroNoise, 0],
                        [0, 0, 0, 0, 0, gyroNoise]])


    def ukf_setQ(self, noiseMagnitude, R = 10):
        '''
        set process noise Q (dim x dim) and update initial covariance
        Q is based on dt (according to research) and initial cov = Q * R according to Estimation II by Ian Reed

        @params:
            noiseMagnitude: magnitude of Q
            R: parameter for initial covariance (10 is optimal)
        '''

        self.Q = np.array([[self.dt, 3*self.dt/4, self.dt/2, self.dt/4, 0, 0, 0],
                        [3*self.dt/4, self.dt, 3*self.dt/4, self.dt/2, 0, 0, 0],
                        [self.dt/2, 3*self.dt/4, self.dt, 3*self.dt/4, 0, 0, 0],
                        [self.dt/4, self.dt/2, 3*self.dt/4, self.dt, 0, 0, 0],
                        [0, 0, 0, 0, self.dt, 2*self.dt/3, self.dt/3],
                        [0, 0, 0, 0, 2*self.dt/3, self.dt, 2*self.dt/3],
                        [0, 0, 0, 0, self.dt/3, 2*self.dt/3, self.dt]
        ])
        self.Q = self.Q * noiseMagnitude

        # update starting cov guess
        self.cov = R * self.Q


    def generateSpeeds(self, max, min, flipSteps, step, indices):
        '''
        generates ideal/actual reaction wheel speeds for n steps
        goes to max for flipSteps and then decreases by step until min is reached
        populates self.reaction_speeds

        @params:
            max, min: max and min speeds
            flipSteps: how many stepts until speed is reversed
            step: how much to change speed by for each time step
            indices: bitset of sorts to signify which axis you want movement about (which reaction wheels to activate)
                speed on x and skew wheel would equal [1, 0, 0, 1]
        '''

        # start with 0 speed on all axices
        ideal_reaction_speeds = [self.curr_reaction_speeds]
        thing = 0

        for a in range(self.n):
            # increase/decrease by step if max/min is not reached
            # also check if inflection point (flipSteps) has been reached
            if (a < flipSteps and thing < max):
                thing += step
            elif thing > min and a > flipSteps:
                thing -= step

            result = np.array([thing, thing, thing, thing])
            # multiply by bitset to only get speed on proper axis
            result = indices * result
            ideal_reaction_speeds.append(result)


        # store in simulator object
        self.reaction_speeds = np.array(ideal_reaction_speeds[:self.n])

        return np.array(ideal_reaction_speeds[:self.n])


    def propagate(self):
        '''
        Generates ideal/actual states of cubesat for all n time steps
        Uses starting state and reaction wheel speeds at each step to progate through our EOMs (equations of motion)
            These equations simulate how our satellite would respond to our chosen conditions
        From this physics-based ideal state, we can generate fake data to pass through our filter
        '''

        # First state is already set in initialization
        for i in range(1, self.n):
            # populate ideal_states array starting at i = 1
            self.propagate_step(i)

            # set filtered states to ideal states if we're not simulating controls (will be overwritten by simulate)
            self.filtered_states[i] = self.ideal_states[i]

        return self.ideal_states


    def propagate_step(self, i):
        '''
        Based on our last filtered state, progate through our EOMs to get the next ideal state
        Populates self.ideal_states[i], allowing us to generate realistic data
        '''

        # use filtered states if we're simulating controls to get better results
        currQuat = self.filtered_states[i - 1][:4]
        currVel = self.filtered_states[i - 1][4:]

        # calculate reaction wheel acceleration
        alpha = (self.reaction_speeds[i] - self.reaction_speeds[i-1]) / self.dt

        # progate through our EOMs to get next ideal state
        currState = self.EOMS.eoms(currQuat, currVel, self.reaction_speeds[i], np.zeros(3), alpha, self.dt)

        # update next state
        self.ideal_states[i] = currState

        # print("ideal: ", currState)

        return currState


    def populateData(self):
        '''
        Populates self.data (n x dim_mes) from either ideal states, live data, or data from a file
            Ideal states: Adds noise to the ideal states to mimic what our sensors would be giving us
            Live data: Reads data from our imu and reaction wheels
            File: reads data from SENSOR_DATA_FILE if non-null
        '''

        # if we're not using ideal states and we have a data file, load data from file
        if not self.ideal_known and SENSOR_DATA_FILE != None:
            self.loadData(SENSOR_DATA_FILE)
        else:
            # if we're not using a file, populate data array from correct source
            for i in range(self.n):

                if self.ideal_known:
                    # generate fake data
                    self.generateData_step(i)

                elif SENSOR_DATA_FILE == None:
                    # if no data file is specified, read from our sensors
                    self.liveData_step(i)

        return self.data


    def generateData_step(self, i):
        '''
        Generates fake data for a single time step and populates self.data[i]
        Adds noise to the ideal states to mimic what our sensors would be giving us

        @returns:
            data: array of sensor fake data (1 x dim_mes)
        '''

        data = np.zeros(self.dim_mes)

        # calculate sensor b field for current time step (see h func for more info on state to measurement space conversion)
        # use current B field of earth to transform ideal state to measurement space + add noise
        # rotation matrix(q) * true B field + noise
        B_sens = np.array([np.matmul(quaternion_rotation_matrix(self.ideal_states[i]), self.B_true[i])]) + self.magNoises[i]

        data[:3] = B_sens

        # get predicted speed of this state + noise to mimic gyro reading
        data[3] = self.ideal_states[i][4] + self.gyroNoises[i][0]
        data[4] = self.ideal_states[i][5] + self.gyroNoises[i][1]
        data[5] = self.ideal_states[i][6] + self.gyroNoises[i][2]

        # store in data array
        self.data[i] = data

        return data


    def liveData_step(self, i):
        '''
        Reads one iteration of our sensors and stores in self.data[i]
        Returns magnetometer, gyroscope reading
        Populates self.reaction_speeds[i] with hall sensor reading
        '''

        # TODO: import interface libraries
        pass


    def loadData(self, fileName):
        '''
        Loads pre-made data from a txt file, used when ideal_known = False
        Populates self.data with sensor data from file
        Populates self.reaction_speeds with reaction wheel speeds from file

        @params:
            fileName: name of file to load data from
        '''
        try:
            # data is in the format a, b, c, x, y, z, e, f, g
            # a, b, c are magnetic field in state space readings, x, y, z are angular velocity, e, f, g are reaction wheel speeds
            # each line is a new time step
            # read in file line by line and store data and reaction wheel speeds in self.data and self.reaction_speeds
            data = []
            speeds = []
            with open(fileName, 'r') as file:
                for line in file:
                    data.append(np.array([float(x) for x in line.split(",")[:6]]))
                    speeds.append(np.array([float(x) for x in line.split(",")[6:]]))

            self.data = np.array(data)
            self.reaction_speeds = np.array(speeds)
            return data

        except FileNotFoundError:
            print(f"Error: Data file {fileName} not found")
            return 1
        except Exception as e:
            print(f"Error loading data: {e}")
            return 1


    def simulate(self):
        '''
        Simulates the state estimation process for n time steps
        Runs the specified kalman filter upon the the object's initial state and data/reaction wheel speeds for each time step
            Uses self.reaction_speeds: reaction wheel speed for each time step (n x 3) and self.data: data reading for each time step (n x dim_mes)

        Stores 2D array of estimated states (quaternions, angular velocity) in self.filter_states, covariances in self.covs, and innovation values and covariances in self.innovations/self.innovationCovs
        Also stores time taken for each estimation in self.times

        '''

        states = []

        # initialize current reaction wheel speed
        self.curr_reaction_speeds = self.reaction_speeds[0]

        # run each of n steps through the filter
        for i in range(self.n):
            # store old reaction wheel speed
            self.old_reaction_speeds = self.curr_reaction_speeds
            self.curr_reaction_speeds = self.reaction_speeds[i]

            start = time.time()
            # propagate current state through kalman filter and store estimated state and innovation
            self.state, self.cov, self.innovations[i], self.innovationCovs[i] = self.kalmanMethod(self.state, self.cov, self.Q, self.R, self.dt, self.B_true[i], self.curr_reaction_speeds, self.old_reaction_speeds, self.data[i])
            end = time.time()

            # store time taken for each step
            self.times[i] = end - start

            states.append(self.state)
            self.covs[i] = self.cov

        self.filtered_states = states
        return states


    def simulate_step(self, i, target):
        '''
        Runs one iteration of propogation for our controls simulation
        Finds the next state (self.filtered_states[i]), then generates uses our PID controller to find how our system wants to move
            From the PWM signal output, simulate how our reaction wheel motors would respond for next step (updates self.reaction_speeds[i+1])
        '''

        start = time.time()

        # run last state, reaction wheel speed, and data through filter to get a more accurate state estimate
        self.filtered_states[i], self.covs[i], self.innovations[i], self.innovationCovs[i] = self.kalmanMethod(
                self.filtered_states[i-1], self.covs[i-1],         # last state and covariance
                self.Q, self.R, self.dt,                           # process and measurement noise, dt
                self.B_true[i],                                    # true magnetic field at this timestep
                self.reaction_speeds[i], self.reaction_speeds[i-1],# current and last reaction wheel speeds
                self.data[i])                                      # data reading at this timestep (already generated/filled)

        end = time.time()
        self.times[i] = end - start

        # run state through our control script to get pwm signals for motors

        # Get current quaternion and angular velocity of cubesat
        quaternion = np.array(self.filtered_states[i][:4])
        omega = np.array(self.filtered_states[i][4:])

        # Run PD controller to generate output for reaction wheels based on target orientation
        self.pwms[i] = self.controller.pid_controller(quaternion, target, omega, self.pwms[i-1])

        # print("wheel speed: ", self.reaction_speeds[i])
        # print("PWM: ", self.pwms[i])
        # print("old current: ", self.currents[i-1])

        # update our temperature and current variables
        # TODO: find and enforce limits for current and temp (that match with max pwm)

        # convert from pwm to voltage
        voltage = (9/MAX_PWM) * self.pwms[i]
        # find the updated winding resistance based on ambient * our current temp
        Rw = RWA *(1 + ALPHA_CU * self.Tw_Ta)

        # update our current and ambient temperature difference variables
        # magic = 1
        # current_dot = (voltage - self.currents[i-1]*Rw - KV*self.reaction_speeds[i])/LW * magic
        # i_dot = (Vin - i*Rw - KV*omega_w)/LW
        # Th_Ta_dot = ((self.Th_Ta - self.Tw_Ta)/Rwh - self.Th_Ta/Rha)/Cha
        # Tw_ta_dot = (self.currents[i-1]**2*Rw - (self.Th_Ta - self.Tw_Ta)/Rwh)/Cwa

        # print("current_dot: ", current_dot)
        # print("speed_dot: ", omega_w_dot)

        # Simplified current calculation based on voltage and reaction wheel speed
        self.currents[i] = self.currents[i-1] + ((voltage - KV * self.reaction_speeds[i]) / Rw) * self.dt

        # external torque is 0 for now
        external_torque_on_wheel = RW_EXTERNAL_TORQUE

        # find the predicted next speed of our reaction wheels based on current speed, current, and external torque
        # Calculate angular acceleration: ω_dot = (motor torque - external torque - damping) / moment of inertia
        # TODO: should this be new or old current?
        omega_w_dot = (KT*self.currents[i] - external_torque_on_wheel - BM*self.reaction_speeds[i])/JM

        # Simplified temperature model: temperature increases based on current squared, and has a linear cooling term
        temp_increase_rate = self.currents[i]**2 * THERMAL_RESISTANCE
        temp_cooling_rate = COOLING_CONSTANT * (self.Th_Ta - self.Tw_Ta)

        # Update temperature variables
        self.Th_Ta += (temp_increase_rate - temp_cooling_rate) * self.dt

        # Assume the reaction wheel temperature adjusts similarly, with some coupling to the ambient temperature
        self.Tw_Ta += (temp_increase_rate * WHEEL_COUPLING_FACTOR - temp_cooling_rate) * self.dt

        # update our variables with Euler's method of propagation
        # self.currents[i] = self.currents[i-1] + current_dot * self.dt
        self.currents[i] = np.clip(self.currents[i], MIN_CURRENT_RW, MAX_CURRENT_RW)
        # self.Th_Ta += Th_Ta_dot * self.dt
        # self.Tw_Ta += Tw_ta_dot * self.dt
        next_speeds = self.reaction_speeds[i] + omega_w_dot * self.dt

        # print("current: ", self.currents[i])

        # print("next speeds: ", next_speeds)
        # print("")

        # update the next reaction wheel speed with our predicted rpm
        if i < self.n - 1:
            self.reaction_speeds[i + 1] = next_speeds

        return self.filtered_states[i]


    def run_controls_sim(self):
        '''
        Combines motor dynamics and PID controller to orient towards a target
        Propogates our state step by step, as we want to dynamically change our "ideal" state based on our control output
        '''

        # generate data for first step so we can start at i = 1
        self.generateData_step(0)

        # define our target orientation and whether we want to reverse it halfway through
        # TODO: x axis is bugged (or just different moments of inertia). Wants to go sideways
        target = normalize(TARGET)
        flip = False

        for i in range(1, self.n):

            # get ideal next state based on current state and reaction wheel speeds of this step
            # NOTE: this "ideal" state is not super based on truth because it is not generated beforehand.
            #       it basically follows what our filter does, so it is not a good representation of the truth
            ideal = self.propagate_step(i)

            # create fake magnetometer data by rotating B field by ideal quaternion, and gyro by adding noise to angular velocity
            self.generateData_step(i)

            # filter our data and get next state
            # also run through our controls to get pwm => voltage => current => speed of reaction wheels
            filtered = self.simulate_step(i, target)
            # game_visualize(np.array([filtered]), i-1)

            # optionally return to starting orientation halfway through
            if i > self.n / 2 and flip == True:
                target = normalize(QUAT_INITIAL)

        # plot our results and create pdf output + 3D visualization
        self.plot_and_viz_results(controller=self.controller, target=target)


    def run_filter_sim(self):
        '''
        Generates ideal states and sensor data, allowing us to benchmark our kalman filter against simulated "truth".
        Can also be run with pre-existing sensor data (ideal_known = False and SENSOR_DATA_FILE != None)
        '''

        # text file with data values
        dataFile = SENSOR_DATA_FILE

        if self.ideal_known:
            # decide how we want our reaction wheels to spin at each time step
            # parameters: max speed, min speed, number of steps to flip speed after, step, bitset of which wheels to activate
            self.generateSpeeds(400, -400, self.n, 40, np.array([0, 1, 0, 0]))

            # find ideal state of cubesat through physics equations of motion
            self.propagate()

        # generate data reading for each step
        self.populateData()

        # run our data through the specified kalman function
        self.simulate()

        # if true, run statistical tests outlined in Estimation II by Ian Reed
        # these tests allow us to see how well our filter is performing
        runTests = RUN_STATISTICAL_TESTS
        sum = 0
        if runTests:
            sum = self.runTests()

        # plot our results and create pdf output + 3D visualization
        self.plot_and_viz_results(sum=sum)


    def plotData(self):
        '''
        plots the magnetometer (magData.png) and gyroscope data (magData.png) found in self.data
        '''
        plotData_xyz(self.data)


    def plotStates(self):
        '''
        plots the filtered states (filteredQuaternion.png, filteredVelocity.png) found in self.filtered_states
        also plots ideal states (idealQuaternion.png, idealVelocity.png) found in self.ideal_states if self.ideal_known = True
        also also plots the euler angle of our ideal state (with respect to our starting state)
        '''
        if self.ideal_known:
            plotState_xyz(self.ideal_states, self.ideal_known)

        plotState_xyz(self.filtered_states, False)
        # unpack the filtered quaternion and convert it to euler angles
        # use the error quaternion between our starting state and current state to base angle off of starting point
        plotAngles(np.array([quaternion_to_euler(*delta_q(a[:4], QUAT_INITIAL)) for a in self.filtered_states]), "Euler angles", fileName="Euler.png")
        # plotAngles(np.array([quaternion_to_euler(*a[:4]) for a in self.filtered_states]), "Euler angles", fileName="Euler.png")


    def plotWheelInfo(self):
        '''
        Plot 3 graphs relating to reaction wheel simulation:
        Angular velocity of the wheels (ReactionSpeeds.png), PWM (pulse width modulation) signals (PWM.png), and current to the motors (Current.png)
        The simulated current is determined by the PWM signal output by our PID controller
        '''
        # angular velocity of our 4 wheels at every time step
        plot_xyz(self.reaction_speeds, "Reaction Wheel Speeds", fileName="ReactionSpeeds.png")

        # PWM signal output by our controller
        plot_xyz(self.pwms, "PWMs", fileName="PWM.png")

        # simulated current to our 4 wheels
        plot_multiple_lines([self.currents], ["Motor Current"], "Motor Current", fileName="Current.png")


    def plot_and_viz_results(self, controller=None, target=np.array([1, 0, 0, 0]), sum=0):
        '''
        Plots out filter states, data, and reaction wheel speeds, and creates pdf output + 3D visualization
        Allows us to visualize results of our filter/controls sim
        Based upon RESULT variable in params.py

        @params:
            controller: PIDController object (for controls sim)
            target: target quaternion (for controls sim)
            sum: sum of statistical tests if they were run
        '''

        # clear output directory from last simulation
        clearDir(OUTPUT_DIR)

        # plot mag and gyro data
        self.plotData()
        # plots filtered states (and ideal states if ideal_known = True)
        self.plotStates()
        # plot reaction wheel speeds
        self.plotWheelInfo()

        # 0 = only create pdf output, 1 = show 3D animation visualization, 2 = both, 3 = none
        visualize = RESULT

        if visualize == 1:
            self.visualizeResults(self.filtered_states)

        elif visualize == 0:

            self.saveFile(OUTPUT_FILE, controller, target, sum, RUN_STATISTICAL_TESTS)

        elif visualize == 2:

            self.saveFile(OUTPUT_FILE, controller, target, sum, RUN_STATISTICAL_TESTS)

            self.visualizeResults(self.filtered_states)

        # only show plot at end so they all show up
        plt.show()


    def runTests(self):
        '''
        runs 3 statistical tests on filter results according to Estimation II by Ian Reed:
            1. innovation test
            2. innovation squared test
            3. autocorrelation test

        creates approriate plots, prints info to command line, and returns the sum of innovations squared
        '''
        # test 1, 2, 3 respectively (see tests.py)
        plotInnovations(self.innovations, self.innovationCovs)
        sum = plotInnovationSquared(self.innovations, self.innovationCovs)
        plotAutocorrelation(self.innovations)
        return sum


    def saveFile(self, fileName, controller=None, target=[1, 0, 0, 0], sum=0, printTests=False):
        '''
        takes all saved pngs and compiles a pdf with the given fileName
        uses the formating function found within saving.py
        stores in OUTPUT_DIR variable declared in params.py and opens completed file
        only prints tests results of printsTests is True
        '''

        # savePNGs(OUTPUT_DIR)

        savePDF(fileName, OUTPUT_DIR, self, controller, target, sum, printTests)

        openFile(fileName)


    def visualizeResults(self, states):
        '''
        Given an array of states, visualize the cubesat moving in 3D
        '''
        # TODO: rewrite functions that visualize different data sets: ideal, filtered, data
        #   with plotting, cubesat, etc

        # or visualize 3 things: raw, filtered, ideal

        game_visualize(np.array(states), 0)

