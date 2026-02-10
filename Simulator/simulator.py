'''
simulator.py
Author: Andrew Gaylord, Sophie, Rawan, Rene, Chau, Daniel, Brian

Contains simulator class for an arbitrary kalman filter and control system
Object contains system info, initialized values, state values, filter specifications, and all outputs
Class functions allow for easy initialization, propagation, data generation, simulation, and visualization

All parameters (variables in caps) are stored in *params.py*

'''

import os
import sys
import time

from Simulator.PySOL.wmm import *
from Simulator.kalman_tests import *
from Simulator.visualizer import *
from Simulator.graphing import *
from Simulator.saving import *
from Simulator.EOMs import *

# import params module from parent directory
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from params import *
from Controllers.PID_controller import *
from ukf.UKF_algorithm import *
from ukf.hfunc import *
from Controllers.B_dot import B_dot, maxPowerController
from Controllers.Pointing.nadir_point import nadir_point, determine_attitude
from Controllers.Pointing.image_processing import processImage, firmwareImageProcessing

class Simulator():
    def __init__ (self, mag_sat = None, b_earth=[0, 0, 0], kalmanMethod = None, controller=None):
        # number of steps to simulate
        self.n = int(TF / DT)
        # timestep between steps
        self.dt = DT
        # dimension of state and measurement space
        self.dim = STATE_SPACE_DIMENSION
        self.dim_mes = MEASUREMENT_SPACE_DIMENSION

        # satellite model to use for simulation
        # current velocity is stored in mag_sat
        self.mag_sat = mag_sat

        # set process noise and update starting cov guess
        # parameters: noise magnitude, k (see Estimation II article by Ian Reed)
        self.ukf_setQ(PROCESS_NOISE_MAG, PROCESS_NOISE_K)

        # set measurement noise
        # parameters: magnetometer noise, gyroscope noise
        self.ukf_setR(MEASUREMENT_MAGNETOMETER_NOISE, MEASUREMENT_GYROSCOPE_NOISE)

        # starting state (default is standard quaternion and no angular velocity)
        # self.state = np.concatenate((normalize(QUAT_INITIAL), VELOCITY_INITIAL))
        # starting covariance (overrid by ukf_setQ)
        self.cov = np.identity(self.dim) * COVARIANCE_INITIAL_MAG

        # 2D array of n innovations and covariances (populated by filter.simulate)
        self.innovations = np.zeros((self.n, self.dim_mes))
        self.innovationCovs = np.zeros((self.n, self.dim_mes, self.dim_mes))

        # noise arrays for magnetometer and gyroscope (if we're using ideal states to simulate sensor data)
        if SENSOR_NOISE:
            magSD = SENSOR_MAGNETOMETER_SD
            self.magNoises = np.random.normal(0, magSD, (self.n, 3))

            gyroSD = SENSOR_GYROSCOPE_SD
            self.gyroNoises = np.random.normal(0, gyroSD, (self.n, 3))
        else:
            self.magNoises = np.zeros((self.n, 3))
            self.gyroNoises = np.zeros((self.n, 3))

        # true magnetic field for every timestep in simulation
        if CONSTANT_B_FIELD:
            # magnetic field in earth and body frame in microTeslas
            constant_field = np.asarray(CONSTANT_B_FIELD_MAG, dtype=float)
            self.B_earth = np.tile(constant_field, (self.n, 1))
            initial_body_field = np.matmul(quaternion_rotation_matrix(normalize(QUAT_INITIAL)), constant_field)
            self.B_body = np.tile(initial_body_field, (self.n, 1))
        else:
            if b_earth is None:
                raise ValueError("b_earth must be provided when CONSTANT_B_FIELD is False.")
            # record 2D array of magnetic field of earth for every timestep, ensuring numeric dtype
            self.B_earth = np.asarray(b_earth, dtype=float)
            if self.B_earth.shape != (self.n, 3):
                raise ValueError(f"Expected b_earth shape ({self.n}, 3), got {self.B_earth.shape}")
            # initialize array of magnetic field in the body frame and find first element
            # microteslas, noisy
            self.B_body = np.zeros((self.n, 3), dtype=float)
            # first element of B_body is set by generate_data_step(0)

        # Motor states
        # self.rw_current = RW_CURRENTS_INITIAL # Current to each motor
        self.Th_Ta = np.array([0.0, 0.0, 0.0, 0.0]) # diff in temp between housing and ambient
        self.Tw_Ta = np.array([0.0, 0.0, 0.0, 0.0]) # diff in temp between winding and ambient
        # current for all n steps
        self.rw_currents = np.zeros((self.n, 4))
        self.rw_currents[0] = RW_CURRENTS_INITIAL

        # 1x4 array of current reaction wheel speeds
        # self.curr_reaction_speeds = RW_INITIAL
        # reaction wheel speed of last time step
        # self.last_reaction_speeds = RW_INITIAL

        # reaction wheel speeds for all n steps
        self.rw_speeds = np.zeros((self.n, 4))
        self.rw_speeds[0] = RW_INITIAL

        # pwm values (motor signals) for all n steps
        self.pwms = np.zeros((self.n, 4))

        # current to magnetorquers for all n steps
        self.mag_currents = np.zeros((self.n, 3))
        self.mag_currents[0] = MAG_CURRENT_INITIAL
        self.mag_voltages = np.zeros((self.n, 3))
        self.mag_voltages[0] = MAG_VOLTAGE_INITIAL

        # simulated torque generated by magnetorquers for all n steps
        self.torques = np.zeros((self.n, 3))

        # store total power output
        self.power_output = np.zeros((self.n, 3))
        init_power = [dimension*dimension/RESISTANCE_MAG for dimension in MAG_VOLTAGE_INITIAL] # power = Watts being used at this particular time = V^2 / R
        self.power_output[0] = init_power # set initial power output based on initial current output, voltage
        self.totalPower = np.zeros((self.n)) # set total power per time step to 0

        # data values for all n steps
        self.data = np.zeros((self.n, self.dim_mes))

        if ACCURATE_MAG_READINGS:
            # track how many timesteps torquers have been off for
            # self.torquersOffTimer * dt is between 0 and TORQUER_OFF_TIME
            self.torquersOffTimer = TORQUER_OFF_TIME / self.dt
            # track how many timesteps since we've taken a mag reading
            # self.magnetometerReadingTimer * dt is between 0 and MAG_READING_INTERVAL
            self.magnetometerReadingTimer = MAG_READING_INTERVAL / self.dt

        # true state (orentiation + angular velocity) for all n steps
        self.states = np.zeros((self.n, self.dim))
        self.states[0] = np.concatenate((normalize(QUAT_INITIAL), VELOCITY_INITIAL))

        # kalman filtered states for all n steps
        self.filtered_states = np.zeros((self.n, self.dim))
        self.filtered_states[0] = self.states[0]

        # covariance of system for all n steps
        self.covs = np.zeros((self.n, self.dim, self.dim))
        self.covs[0] = self.cov

        # what kalman filter to apply to this system
        self.kalmanMethod = kalmanMethod

        # controller object to use for this system (stores gain constants and provides pwm signal generation function)
        self.controller = controller

        # filter times for each step (for efficiency testing)
        self.times = np.zeros(self.n)

        # error quaternion between target nadir (0, 0, -1) and current nadir (in body frame)
        self.trueErrorQuats = np.zeros((self.n, 4))
        # Determined (expiremental) error
        self.errorQuats = np.zeros((self.n, 4))

        # magnitude of nadir error quaternion
        self.trueNadirError = np.zeros((self.n))
        self.nadirError = np.zeros((self.n))

        # time for when angular velocity slowed down to 0-0.5 degrees per axis in seconds
        self.finishedTime = -1
        # self.finishedTime/3600 is time in hours

        if RUNNING_MAYA:
            # pitch and roll for cam 1 and 2, respectively
            self.pitches1 = np.zeros((self.n))
            self.rolls1 = np.zeros((self.n))
            self.pitches2 = np.zeros((self.n))
            self.rolls2 = np.zeros((self.n))
            # edge arrays for cam 1 and 2
            self.edges1 = np.zeros((self.n, 4))
            self.edges2 = np.zeros((self.n, 4))

        # what mode we're in for each time step
        self.mode = np.zeros((self.n))
        self.mode[0] = PROTOCOL_MAP[STARTING_PROTOCOL]

        # Set the total energy to 0 at the start (joules)
        self.energy = 0

        # generate data for first step
        # populates self.B_body[0], self.data[0], self.mag_sat.B_body, and self.mag_sat.w_sat
        self.generateData_step(self.states[0], 0)
        # update state and timers
        self.mag_sat.state = self.check_state(0)


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
        populates self.rw_speeds

        @params:
            max, min: max and min speeds
            flipSteps: how many stepts until speed is reversed
            step: how much to change speed by for each time step
            indices: bitset of sorts to signify which axis you want movement about (which reaction wheels to activate)
                speed on x and skew wheel would equal [1, 0, 0, 1]
        '''

        # start with 0 speed on all axices
        ideal_reaction_speeds = self.rw_speeds[0]
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
        self.rw_speeds = np.array(ideal_reaction_speeds[:self.n])

        return np.array(ideal_reaction_speeds[:self.n])


    def find_ideal(self, i):
        '''
        Find the "ideal" next state using physics EOMs based on our last state
        Used so we can generate more occurate data for that step

        TODO: switch to rk4, or remove entirely
        '''
        q_dot, w_dot = eoms(self.states[i-1][:4], self.states[i-1][4:], 0, self.mag_torques[i-1], 0, self.dt, self.mag_sat.I_body, 0)

        # propagate state using Euler's method
        quat = normalize(self.states[i-1][:4] + q_dot*self.dt)
        velocity = self.states[i-1][4:] + w_dot*self.dt

        return np.concatenate((quat, velocity))


    def generateData_step(self, state, i):
        '''
        Generates fake data for a single time step and populates self.data[i], self.B_body[i]
            Sets mag_sat.B_body and mag_sat.w_sat to simulated sensor readings
        Adds noise to the states to mimic what our sensors would be giving us

        @returns:
            data: array of sensor fake data (1 x dim_mes)
        '''
        data = np.zeros(self.dim_mes)

        if ACCURATE_MAG_READINGS and (self.magnetometerReadingTimer * self.dt < MAG_READING_INTERVAL):
            # if it hasn't been long enough since last reading, take last reading as an estimate
            data[:3] = self.data[i - 1][:3]
            self.B_body[i] = self.B_body[i - 1]

            # keep the mag_sat reading to be the same, and update prev value to be the same
            # self.mag_sat.prevB.append(self.mag_sat.B_body)
            # if (len(self.mag_sat.prevB) > 2):
                # self.mag_sat.prevB.pop(0)
        else:

            # calculate sensor b field for current time step (see h func for more info on state to measurement space conversion)
            # use current B field of earth to transform state to measurement space + add noise
            # rotation matrix(q) * true B field + noise (in microteslas)
            self.B_body[i] = np.matmul(quaternion_rotation_matrix(state[:4]), self.B_earth[i]) + self.magNoises[i]

            # Set the previous B field to current B field before future B field is calculated
            # Only set previous B field when step is greater than 0 (edge case)
            if i > 0:
                # self.mag_sat.prevB = self.mag_sat.B_body
                self.mag_sat.prevB.append(self.mag_sat.B_body)
                if (len(self.mag_sat.prevB) > MAG_READINGS_STORED):
                    self.mag_sat.prevB.pop(0)

            # store new reading in our sat object
            self.mag_sat.B_body = self.B_body[i].astype(np.float64)

            # convert from microteslas to teslas
            self.mag_sat.B_body = np.array([self.mag_sat.B_body[0] * 1e-6, self.mag_sat.B_body[1] * 1e-6, self.mag_sat.B_body[2] * 1e-6])

            data[:3] = self.B_body[i]

        # get predicted speed of this state + noise (in rad/s) to mimic gyro reading
        data[3] = state[4] + self.gyroNoises[i][0]
        data[4] = state[5] + self.gyroNoises[i][1]
        data[5] = state[6] + self.gyroNoises[i][2]
        # set the angular velocity reading of the satellite
        self.mag_sat.w_sat = np.array([data[3], data[4], data[5]])

        # store in data array
        self.data[i] = data

        return data


    def loadData(self, fileName):
        '''
        Loads pre-made data from a txt file, used when ideal_known = False
        Populates self.data with sensor data from file
        Populates self.rw_speeds with reaction wheel speeds from file

        @params:
            fileName: name of file to load data from
        '''
        try:
            # data is in the format a, b, c, x, y, z, e, f, g
            # a, b, c are magnetic field in state space readings, x, y, z are angular velocity, e, f, g are reaction wheel speeds
            # each line is a new time step
            # read in file line by line and store data and reaction wheel speeds in self.data and self.rw_speeds
            data = []
            speeds = []
            with open(fileName, 'r') as file:
                for line in file:
                    data.append(np.array([float(x) for x in line.split(",")[:6]]))
                    speeds.append(np.array([float(x) for x in line.split(",")[6:]]))

            self.data = np.array(data)
            self.rw_speeds = np.array(speeds)
            return data

        except FileNotFoundError:
            print(f"Error: Data file {fileName} not found")
            return 1
        except Exception as e:
            print(f"Error loading data: {e}")
            return 1


    def determine_attitude(self, i):
        '''
        Use Kalman filter to determine current attitude based on last state and sensor readings
        Populates self.filtered_states[i]

        TODO: check i indices (where in the process is this done)
        TODO: ensure only data from mag_sat is passed to kalmanMethod (and store current filtered_state in mag_sat?)
        TODO: don't pass b_true, pass GPS instead. Add GPS generation to generateData_step too?
        '''
        start = time.time()

        # run last state, reaction wheel speed, and data through filter to get a more accurate state estimate
        self.filtered_states[i], self.covs[i], self.innovations[i], self.innovationCovs[i] = self.kalmanMethod(
                self.filtered_states[i-1], self.covs[i-1],         # last state and covariance
                self.Q, self.R, self.dt,                           # process and measurement noise, dt
                self.B_true[i],                                    # true magnetic field at this timestep
                self.rw_speeds[i], self.rw_speeds[i-1],# current and last reaction wheel speeds
                self.data[i])                                      # data reading at this timestep (already generated/filled)

        end = time.time()
        self.times[i] = end - start


    def check_state(self, i):
        '''
        Checks current state of our mag_sat and updates if requirements are met
            Must check self.states[i - 1] because current state is not updated yet
        If detumbling, check threshold speed
        If searching, check if horizon is found
        If pointing, check if horizon is lost
            Also always check if speed is too high??

        For accurate mag readings, also handles all timers (torquersOffTimer, magnetometerReadingTimer)

        '''

        if ACCURATE_MAG_READINGS:

            if self.torquersOffTimer * self.dt >= TORQUER_OFF_TIME and self.magnetometerReadingTimer * self.dt >= MAG_READING_INTERVAL:
                # if torquers have been off for a while and we took our reading (in generateData_step), reset timers
                # print(i, " reading took: ", self.magnetometerReadingTimer * self.dt, ", ", self.torquersOffTimer * self.dt)
                self.magnetometerReadingTimer = 0.0
                self.torquersOffTimer = 0.0

            elif self.magnetometerReadingTimer * self.dt >= MAG_READING_INTERVAL - TORQUER_OFF_TIME:
                # if we're approaching a magnetometer reading, start updating the torquer off timer
                # print(i, " mags off", self.magnetometerReadingTimer * self.dt, ", ", self.torquersOffTimer * self.dt)
                self.torquersOffTimer += 1.0
                self.magnetometerReadingTimer += 1.0
            else:
                # otherwise, update time since we've taken reading
                # print(i, " mags on", self.magnetometerReadingTimer * self.dt, ", ", self.torquersOffTimer * self.dt)
                self.magnetometerReadingTimer += 1.0

        if RUNNING_1D and not DETUMBLE_1D:
            # extract current quaternion and convert to euler angles
            q = self.states[i - 1][:4]
            x, y, z = quaternion_to_euler(q)

            # check if we've reached our desired angle for the first time
            if (self.finishedTime == -1):
                if (x<=DESIRED_ANGLE[0]):
                    print("REACHED DESIRED ANGLE OF 90")
                    self.finishedTime = i*self.dt
                    print("Time needed is " + str(self.finishedTime))

        elif self.mag_sat.state == "detumble":
            # threshold 0.5-1 degress per second per axis
            thresholdLow = 0
            if RUNNING_1D and DETUMBLE_1D:
                thresholdHigh = DETUMBLE_THRESHOLD_1D
            else:
                thresholdHigh = DETUMBLE_THRESHOLD

            angularX = abs(self.states[i - 1][4])
            angularY = abs(self.states[i - 1][5])
            angularZ = abs(self.states[i - 1][6])

            if(self.finishedTime == -1) and i > 0:
                if (thresholdLow <= angularX <= thresholdHigh) and (thresholdLow <= angularY <= thresholdHigh) and (thresholdLow <= angularZ <= thresholdHigh):
                    print("Successfully detumbled after " + str(i*self.dt) + " seconds!")
                    # record first time we hit "detumbled" threshold (seconds)
                    self.finishedTime = i*self.dt

                    # When the "detumbled" threshold is hit, calculate total Energy
                    # Total Energy is calculated as a "Rieman Sum" of the total power used at each time step multiplied by the time step
                    for step in range(i):
                        self.energy = self.energy + self.totalPower[step]*self.dt

                    # move to nadir pointing protocol
                    # return "point"
                    return "idle"

            return "detumble"

        # We didn't actually implement this state
        # elif self.mag_sat.state == "search":
        #     # if both see earth in some capacity (even if one if full), move to pointing
        #     if RUNNING_MAYA and self.mag_sat.cam1.alpha >= 0.05 and self.mag_sat.cam2.alpha >= 0.05:
        #         # print("SWITCH TO POINT")
        #         return "point"
        #     else:
        #         # if we don't see the earth, check that we're below detumble threshold
        #         thresholdLow = 0
        #         thresholdHigh = DETUMBLE_THRESHOLD # 0.5-1 degress per second per axis
        #         angularX = abs(self.states[i - 1][4])
        #         angularY = abs(self.states[i - 1][5])
        #         angularZ = abs(self.states[i - 1][6])
        #         # if we're below threshold, move to pointing (for step 2 (AC) solving only)
        #         if (thresholdLow <= angularX <= thresholdHigh) and (thresholdLow <= angularY <= thresholdHigh) and (thresholdLow <= angularZ <= thresholdHigh):
        #             return "point"
        #         else:
        #             return "detumble"

        elif self.mag_sat.state == "point":
            return "point"
            # # check if we lost horizon--move to search
            # if RUNNING_MAYA and ((0.0 <= self.mag_sat.cam1.alpha <= 0.05) or (0.0 <= self.mag_sat.cam2.alpha <= 0.05)):
            #     return "search"
            # else:
            #     return "point"
        elif self.mag_sat.state == "target_point":
            return "target_point"
        elif self.mag_sat.state == "idle":
            return "idle"
        else:
            return "INVALID"


    def controls(self, i):
        '''
        Based on saved sensor data and current protocol state, generate correct controls voltages
            Also set voltage to zero if we're approaching a magnetometer reading
        Voltage for next step is stored in self.mag_voltages[i]
        Info about what mode we're in is stored in self.mode[i]
        '''
        if ACCURATE_MAG_READINGS and self.torquersOffTimer == 1:

            # if we just started turning off our torquers, send burst of voltage in opposite direction
            # this simulates demagnitizing the magnetorquer core (theoritically)
            if np.linalg.norm(self.mag_voltages[i - 1]) > 0:
                self.mag_voltages[i] = np.array([- DEMAGNITIZING_VOLTAGE * (v / abs(v)) for v in self.mag_voltages[i - 1]])
            self.mode[i] = PROTOCOL_MAP['demagnetize']
            self.errorQuats[i] = self.errorQuats[i - 1]
            self.nadirError[i] = self.nadirError[i - 1]

        elif ACCURATE_MAG_READINGS and self.torquersOffTimer * self.dt > 0:

            # if torquers are off as they demagnitize, set voltage to 0
            self.mag_voltages[i] = np.zeros((3))
            self.mode[i] = PROTOCOL_MAP['demagnetize']
            self.errorQuats[i] = self.errorQuats[i - 1]
            self.nadirError[i] = self.nadirError[i - 1]

        elif RUNNING_1D and not DETUMBLE_1D:

            # for 1D test, use simple custom controller
            self.mag_voltages[i] = maxPowerController(DESIRED_MAGNETIC_MOMENTS, self.mag_sat)
            self.mag_voltages[i] = np.clip(self.mag_voltages[i], -MAX_VOLTAGE_MAG, MAX_VOLTAGE_MAG)
            self.mode[i] = PROTOCOL_MAP['detumble']

        elif self.mag_sat.state == "detumble":

            # if running b-dot instead of b-cross, don't run until we have proper data
            if GYRO_WORKING or len(self.mag_sat.prevB) >= MAG_READINGS_STORED:
                # oppose angular velocity
                self.mag_voltages[i] = B_dot(self.mag_sat)
                self.mag_voltages[i] = np.clip(self.mag_voltages[i], -MAX_VOLTAGE_MAG, MAX_VOLTAGE_MAG)

            self.mode[i] = PROTOCOL_MAP['detumble']

        # elif self.mag_sat.state == "search":

        #     # do nothing while horizon searching, for now
        #     self.mag_voltages[i] = np.zeros((3))
        #     self.mode[i] = -1

        elif self.mag_sat.state == "point":

            # Run every NADIR_INTERVAL seconds (where there is DT steps per second)
            if i % int(NADIR_INTERVAL / DT) == 0:
                # only need to update our voltages when image updates
                # based on errorQuats[i], calculate voltage output to get to nadir

                if ADCS_TYPE == "AD":
                    # Use our image processing results to find where earth is
                    AD_results = determine_attitude(self.mag_sat)
                    self.errorQuats[i] = AD_results[2]
                    self.nadirError[i] = np.linalg.norm(self.errorQuats[i][1:])

                self.mag_voltages[i] = nadir_point(self.errorQuats[i], self.mag_sat)
                self.mode[i] = PROTOCOL_MAP['point']
                # We still clamp voltages on firmware
                self.mag_voltages[i] = np.clip(self.mag_voltages[i], -MAX_VOLTAGE_MAG, MAX_VOLTAGE_MAG)
            else:
                # If we don't have a new image, don't bother running controls loop
                self.mag_voltages[i] = self.mag_voltages[i - 1]
                self.mode[i] = self.mode[i - 1]
                self.errorQuats[i] = self.errorQuats[i - 1]
                self.nadirError[i] = self.nadirError[i - 1]

        elif self.mag_sat.state == "target_point":
            # run state through our control script to get pwm signals for motors

            # Get current quaternion and angular velocity of cubesat
            # quaternion = np.array(self.filtered_states[i][:4])
            quaternion = np.array(self.states[i-1][:4])  # Use true state for now
            # omega = np.array(self.filtered_states[i][4:])
            omega = np.array(self.states[i-1][4:])  # Use true state for now

            # Run PD controller to generate output for reaction wheels based on target orientation
            self.pwms[i] = self.controller.pid_controller(quaternion, TARGET, omega, self.pwms[i-1])
            self.mode[i] = PROTOCOL_MAP['target_point']

        elif self.mag_sat.state == "idle":
            self.mode[i] = PROTOCOL_MAP['idle']

        # Clamp with bitmask
        self.pwms[i] *= RW_AXES
        self.mag_voltages[i] *= MAG_AXES


    def propagate_step(self, i):
        '''
        Based on our last state and voltage output from our controls (voltages[i]), progate through our EOMs to get the next state
        Populates self.states[i], self.torques[i], and self.mag_currents[i] the end of the timestep
        '''

        mag_currents = self.mag_currents[i - 1]

        moment = mag_currents * np.array([mag.n * mag.area * mag.epsilon for mag in self.mag_sat.mags])
        # note: we want to use true body magnetic field of last iteration, not noisy sensor reading
        true_b_body = self.mag_sat.B_body - (self.magNoises[i] * 1e-6)
        # torque (N*m) generated by magnetorquers: cross-product of moment with magnetic field (of body? wrong)
        mag_torque = np.cross(moment, true_b_body)

        # power = Watts being used at this particular time
        # calculate power output of magnetorquers = current output * operational voltage
        # power = np.abs(MAX_VOLTAGE * currents)
        # P = V * I
        # power = np.abs(self.mag_voltages[i] * currents)
        # TODO: or V^2 / R?
        power = np.abs((self.mag_voltages[i] * self.mag_voltages[i]) / self.mag_sat.resistances)
        if sum(power) > MAX_POWER_NADIR and self.mag_sat.state == "point":
            print(f"Total nadir power limit exceeded at step {i}: {sum(power)} W (array = {power})")

        # store sim data for this iteration
        self.power_output[i] = np.array(power)
        self.torques[i] = np.array(mag_torque)
        self.mag_currents[i] = mag_current_propagator(mag_currents, self.mag_voltages[i], self.mag_sat, self.dt)

        # Based on PWM outputs from controls(), update reaction wheel speeds
        self.rw_currents[i] = rw_current_propagator(self.pwms[i], self.Tw_Ta, self.rw_speeds[i-1], self.rw_currents[i-1], self.dt)
        self.rw_speeds[i], self.Th_Ta, self.Tw_Ta = rw_speed_propagator(self.rw_speeds[i-1], self.rw_currents[i], self.Th_Ta, self.Tw_Ta, self.dt)

        # Use bitmask to ensure that only desired hardware is turned on/simulated
        self.mag_currents[i] *= MAG_AXES
        self.torques[i] *= FREEDOM_OF_MOVEMENT_AXES
        self.rw_currents[i] *= RW_AXES
        self.rw_speeds[i] *= RW_AXES
        # Control signals are clamped in .controls()
        # self.mag_voltages[i] *= MAG_AXES
        # self.pwms[i] *= RW_AXES

        # calculate reaction wheel acceleration
        alpha = (self.rw_speeds[i] - self.rw_speeds[i-1]) / self.dt

        if SOLVER_METHOD == "euler":
            # calculate quaternion and angular velocity derivatives from Equations of Motion
            q_dot, w_dot = eoms(self.states[i-1][:4], self.states[i-1][4:], self.rw_speeds[i], self.torques[i], alpha, self.dt, self.mag_sat.I_body)

            # propagate state using Euler's method
            # find next state in time (based on current state, not sensor data)
            quaternion_new = normalize(self.states[i-1][:4] + q_dot*self.dt)
            w_sat_new = self.states[i-1][4:] + w_dot*self.dt

        elif SOLVER_METHOD == "rk4":
            quaternion_new, w_sat_new = rk4(self.states[i-1], self.rw_speeds[i], alpha, self.torques[i], self.dt, self.mag_sat.I_body)

        self.states[i] = np.concatenate((quaternion_new, w_sat_new))
        return self.states[i]


    def findTrueNadir(self, ideal, gps, i):
        '''
        Given the current (estimated) state, find error quaternion between the current orientation and nadir
        Stores error quaternion in self.trueErrorQuats[i]
        Stores magnitude of x,y,z component of error quaternion in self.trueNadirError[i]

        @params:
            ideal (1x7 np.array): The current state of the satellite (quaternion and angular velocity)
            gps (1x3 np.array): The GPS position of the satellite in ECEF coordinates [X, Y, Z] in meters

        @returns:
            nadir_body (1x3 np.array): The Earth-pointing vector in the satellite's body frame (unit vector).
            error_quat (1x4 np.array): rotation quaternion from current orientation to nadir (w x y z)
        '''

        # ECEF gps points towards satellite in earth body frame
        nadir_ecef = -gps / np.linalg.norm(gps)

        # convert current orientation to rotation matrix (we use the form w,x,y,z)
        rotation = R.from_quat([ideal[1], ideal[2], ideal[3], ideal[0]])

        # convert from ECEF to the satellite's body frame
        nadir_body = rotation.inv().apply(nadir_ecef)

        # target nadir in body frame
        # -z is down for our satellite
        nadir_target = np.array(BODY_DOWN)

        error_quat = np.zeros((4))

        # find the error quaternion between the target nadir and the current nadir
        # this finds shortest arc (https://stackoverflow.com/questions/1171849/finding-quaternion-representing-the-rotation-from-one-vector-to-another/1171995#1171995)
        # Doesn't handle parallel vectors
        error_quat[1:] = np.cross(nadir_body, nadir_target)
        error_quat[0] = math.sqrt((np.linalg.norm(nadir_body) ** 2) * (np.linalg.norm(nadir_target) ** 2)) + np.dot(nadir_body, nadir_target)
        error_quat = normalize(error_quat)

        # print(f"Nadir body (step {i}): {nadir_body[0]:.3f}, {nadir_body[1]:.3f}, {nadir_body[2]:.3f}")
        # print(f"    error_quat: {error_quat[0]:.3f}, {error_quat[1]:.3f}, {error_quat[2]:.3f}, {error_quat[3]:.3f}")
        # print(f"    Angle error (deg): {quaternion_to_angle(error_quat):.3f}")
        # print(f"    Off-nadir angle (deg): {off_nadir_angle(nadir_body):.3f}")

        if ADCS_TYPE != "AD":
            self.errorQuats[i] = error_quat
            self.nadirError[i] = np.linalg.norm(error_quat[1:])

        self.trueErrorQuats[i] = error_quat
        self.trueNadirError[i] = np.linalg.norm(error_quat[1:])

        return nadir_body, error_quat


    def process_images(self, image1, image2, i):
        '''
        Takes in our two simulated EHS images and stores the info in mag_sat object
        '''
        # image1_results = processImage(image1, degree=1)
        image1_results = firmwareImageProcessing(image1)
        self.mag_sat.cam1.roll = image1_results[0]
        self.mag_sat.cam1.pitch = image1_results[1]
        self.mag_sat.cam1.alpha = image1_results[2]
        self.mag_sat.cam1.edges = image1_results[3]
        self.mag_sat.cam1.midpoint = image1_results[4]
        self.mag_sat.cam1.ends = image1_results[5]
        # image2_results = processImage(image2, degree=1)
        image2_results = firmwareImageProcessing(image2)
        self.mag_sat.cam2.roll = image2_results[0]
        self.mag_sat.cam2.pitch = image2_results[1]
        self.mag_sat.cam2.alpha = image2_results[2]
        self.mag_sat.cam2.edges = image2_results[3]
        self.mag_sat.cam2.midpoint = image2_results[4]
        self.mag_sat.cam2.ends = image2_results[5]
        self.mag_sat.updateAlphaReadings()

        # # if we don't see horizon, differentiate between seeing all earth and all space
        if self.mag_sat.cam1.roll == -1:
            if self.mag_sat.cam1_alpha_increasing:
                # if the last few images saw increasingly more earth, we assume we're now seeing all earth
                self.mag_sat.cam1.alpha = 1.0
            else:
                # if we were seeing less and less earth, assume we're seeing space now
                self.mag_sat.cam1.alpha = 0.0

        # same for cam2
        if self.mag_sat.cam2.roll == -1:
            if self.mag_sat.cam2_alpha_increasing:
                # if the last few images saw increasingly more earth, we assume we're now seeing all earth
                self.mag_sat.cam2.alpha = 1.0
            else:
                # if we were seeing less and less earth, assume we're seeing space now
                self.mag_sat.cam2.alpha = 0.0

        # for loop that controls how many in advance we need to save
        # ex: if we're simulating every other cam, set same measurements for next time step
        for a in range(PIC_INTERVAL):
            if i + a < self.n:
                self.pitches1[i + a] = self.mag_sat.cam1.pitch
                self.rolls1[i + a] = self.mag_sat.cam1.roll
                self.edges1[i + a] = self.mag_sat.cam1.edges

                self.pitches2[i + a] = self.mag_sat.cam2.pitch
                self.rolls2[i + a] = self.mag_sat.cam2.roll
                self.edges2[i + a] = self.mag_sat.cam2.edges


    def plotData(self):
        '''
        plots the magnetometer (magData.png) and gyroscope data (magData.png) found in self.data
        '''
        plotData_xyz(self.data)
        # also plot b field of earth
        plot_xyz(self.B_earth, "Earth's Magnetic Field (ECI frame from PySOL)", fileName="B_earth.png", ylabel="Magnetic Field (microteslas)")
        if RUNNING_MAYA:
            plot_multiple_lines([self.pitches1, self.rolls1, self.pitches2, self.rolls2],["Pitch1", "Roll1", "Pitch2", "Roll2"], "Image Processing",fileName="Pitch_Roll.png",ylabel="Angle (Degrees)")
            plot_multiple_lines([self.edges1[:, 0], self.edges1[:, 1], self.edges1[:, 2], self.edges1[:, 3]],["Top", "Right", "Bottom", "Left"], "Edge Intensities 1",fileName="Edges1.png",ylabel="Relative %")
            plot_multiple_lines([self.edges2[:, 0], self.edges2[:, 1], self.edges2[:, 2], self.edges2[:, 3]],["Top", "Right", "Bottom", "Left"], "Edge Intensities 2",fileName="Edges2.png",ylabel="Relative %")


    def plotStates(self):
        '''
        plots the filtered states (filteredQuaternion.png, filteredVelocity.png) found in self.filtered_states
        also plots the states found in self.states
        also also plots the euler angle of our ideal state (with respect to our starting state)
        '''
        plotState_xyz(self.states)
        plot_multiple_lines([self.mode], ["Mode"], "Satellite Mode", fileName="Mode.png", ylabel="Mode")
        plot_multiple_lines([self.nadirError, self.trueNadirError], ["Found Nadir Error", "True Nadir Error"], "Magnitude of Error Quaternion", fileName="Error.png", ylabel="Magnitude of Error")
        plot_multiple_lines([[np.linalg.norm(v) for v in self.states[:, -3:]]], ["Angular Velocity Magnitude"], "Magnitude of Angular Velocity", fileName="Velocity_Magnitude.png", ylabel="Magnitude of Angular Velocity")
        # unpack the filtered quaternion and convert it to euler angles
        # use the error quaternion between our starting state and current state to base angle off of starting point
        plotAngles(np.array([quaternion_to_euler(delta_q(a[:4], QUAT_INITIAL)) for a in self.states]), "Euler angles", fileName="Euler.png")
        # plotAngles(np.array([quaternion_to_euler(*a[:4]) for a in self.filtered_states]), "Euler angles", fileName="Euler.png")

        plotState_xyz(self.filtered_states, True)


    def plotWheelInfo(self):
        '''
        Plot 3 graphs relating to reaction wheel simulation:
        Angular velocity of the wheels (ReactionSpeeds.png), PWM (pulse width modulation) signals (PWM.png), and current to the motors (Current.png)
        The simulated current is determined by the PWM signal output by our PID controller
        '''
        # angular velocity of our 4 wheels at every time step
        plot_xyz(self.rw_speeds, "Reaction Wheel Speeds", fileName="ReactionSpeeds.png")

        # PWM signal output by our controller
        plot_xyz(self.pwms, "PWMs", fileName="PWM.png")

        # simulated current to our 4 wheels
        plot_multiple_lines([self.rw_currents], ["Motor Current"], "Motor Current", fileName="ReactionCurrent.png")


    def plotMagInfo(self):
        '''
        Plots the currents and torque created by our magnetorquers
        '''
        plot_xyz(self.mag_voltages, "Mag Voltages", fileName="MagVoltages.png", ylabel="Voltage (Volts)")
        plot_xyz(self.mag_currents, "Mag Currents", fileName="MagCurrents.png", ylabel="Current (Amps)")
        plot_xyz(self.torques, "Mag Torques", fileName="MagTorques.png", ylabel="Torque (N*m)")
        plot_xyz(self.power_output, "Power Usage", fileName="Power_Output.png", ylabel="Power (Watts)")
        # plot_multiple_lines([self.totalPower],["Total Power"], "Total Power Output",fileName="Total_Power_Output.png",ylabel="Power (Watts)")


    def plot_and_viz_results(self):
        '''
        Plots out filter states, data, and reaction wheel speeds, and creates pdf output + 3D visualization
        Allows us to visualize results of our filter/controls sim
        Based upon RESULT variable in params.py
        '''

        # clear output directory from last simulation
        clearDir(OUTPUT_DIR)

        self.plotData()
        self.plotStates()
        self.plotMagInfo()
        self.plotWheelInfo()

        # 0 = only create pdf output, 1 = show 3D animation visualization, 2 = both, 3 = none
        visualize = RESULT

        if visualize == 1:
            self.visualizeResults(self.states)

        elif visualize == 0:
            self.saveFile(OUTPUT_FILE)

        elif visualize == 2:
            self.saveFile(OUTPUT_FILE)
            self.visualizeResults(self.states)

        # only show plot at end so they all show up
        if not RUNNING_MAYA:
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


    def saveFile(self, fileName):
        '''
        takes all saved pngs and compiles a pdf with the given fileName
        uses the formating function found within saving.py
        stores in OUTPUT_DIR variable declared in params.py and opens completed file
        only prints tests results of printsTests is True
        '''

        # savePNGs(OUTPUT_DIR)

        savePDF(fileName, OUTPUT_DIR, self)

        if not RUNNING_MAYA:
            openFile(fileName)


    def visualizeResults(self, states, i = 0):
        '''
        Given an array of states, visualize the cubesat moving in 3D
        '''
        # TODO: rewrite functions that visualize different data sets: ideal, filtered, data
        #   with plotting, cubesat, etc

        game_visualize(np.array(states), i)

