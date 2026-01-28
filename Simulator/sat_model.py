'''
sat_model.py
Authors: Andrew Gaylord, Michael Paulucci, Sarah Keopfer, Lauren, Kris, Daniel

Contains the satellite model for the magnetorquer-only ADCS
Magnetorquer_Sat class contains the specifications and current state for our NearSpace satellite
'''

import math
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import Simulator.magnetorquer as mag
import Simulator.camera as cam
from Controllers.Pointing.image_processing import processImage
# import params module from parent directory
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from params import *

class Magnetorquer_Sat():
    '''
    Class that represents the specifications and current state for our NearSpace satellite
    '''
    def __init__(self, I_body:np.ndarray, magnetorquers: list[mag.Magnetorquer], w_sat: np.ndarray, B_body: np.ndarray, prevB: np.ndarray, DT: float, gyro_working: bool, kp: float, kd: float):
        '''
        Initialize all parameters needed to model our cubesat and its magnetorquers
        number of coils and area are arrays because our mags have different specifications

        @params:
            I_body (np.ndarray, (3x3)): moment of inertia tensor of satellite
            magnetorquers (list[mag.Magnetorquer]): list of magnetorquer objects
            w_sat (np.ndarray, (1x3)): angular velocity vector (rad/s) (radians not degrees!)
            B_body (np.ndarray, (1x3)): Magnetic field of body (Teslas)
                Note: w_sat and B_body are noisy, as they are measured from our gyro and magnetometer
            prevB (np.ndarray, (1x3)): previous magnetic field of body (Teslas)
            DT (float): timestep
            gyro_working (bool): whether or not the gyroscope is working
            kp (float): proportional gain
            kd (float): derivative gain
        '''

        self.I_body = I_body                        # Initially store moment of inertia tensor
        self.I_body_inv = np.linalg.inv(I_body)

        # current sensor values of magnetic field and angular velocity
        self.w_sat = w_sat
        self.B_body = B_body

        # array of magnetorquer objects
        self.mags = magnetorquers

        # # array of max torques for each magnetorquer (used to check boundaries)
        # self.max_torque = np.array([
        #     AIR_MAX_TORQUE if mag.epsilon == 1 else FERRO_MAX_TORQUE
        #     for mag in self.mags
        # ])

        # array of resistances and inductances for each magnetorquer
        self.resistances = np.array([mag.resistance for mag in self.mags])
        self.inductances = np.array([mag.inductance for mag in self.mags])

        # dt and prev B field for calculations with no gyroscope
        self.dt = DT
        self.prevB = []
        # self.prevB = prevB
        self.gyro_working = gyro_working

        # gains for our pd bang-bang controller
        self.kp = kp
        self.kd = kd

        # what state our satellite is in ("detumble", "search", "point")
        self.state = STARTING_PROTOCOL

        # create two camera objects (facing either direction)
        self.cam1 = cam.Camera()
        self.cam2 = cam.Camera()

        #stores the past four alpha values (not including current reading)
        self.cam1_alphas = np.array([-1,-1,-1,-1], dtype=float)
        self.cam2_alphas = np.array([-1,-1,-1,-1], dtype=float)

        # tracks progression of cameras as edge moves [used for when no edge is detected]
        # TRUE = camera is moving toward all space (alpha is decreasing)
        # FALSE = camera is moving toward all earth (alpha is increasing)
        self.cam1_alpha_increasing = False;
        self.cam2_alpha_increasing = False;


        # all_mags = [0 for _ in range(3)] # create an array holding 3 Magnetorquer objects

        # for index, magnetorquer in enumerate(all_mags):
        #     # update the properties of each magnetorquer
        #     magnetorquer = mag.Magnetorquer(n = n[index], area = area[index], k = k[index], B_body = B_body[index], w_sat = w_sat[index], epsilon = 1)
        #     if (index != 2): # third magnetorquer is the non-ferromagnetic one, calculate special epsilon for the others
        #         ratio = mag_length/core_radius # length-to-radius ratio of the cylindrical magnetorquer
        #
        #         #demag_factor = (4*math.log(ratio - 1)) / ((ratio*ratio) - (4*math.log(ratio)))
        #         demag_factor = calculate_demagnetizing_factor(ratio) # pull equation demagnetizing facotr from fullcalcs.py
        #
        #         ferro_epsilon = 1 + ((rel_perm - 1)/(1 + demag_factor*(rel_perm-1)))
        #         magnetorquer.epsilon = ferro_epsilon


    def momentToVoltage(self, moment: np.ndarray) -> np.ndarray:
        '''
        Converts a desired magnetic moment to the voltage needed to generate that moment
        NOTE: all in body frame, which is why area can be treated as [0, 0 , A]

        @params:
            moment (np.ndarray, (1x3)): desired magnetic moment (Amps * m^2)
        @returns:
            voltage (np.ndarray, (1x3)): voltage needed to generate that moment (Volts)
        '''

        # calculate the voltage needed to generate the desired magnetic moment
        current = np.zeros((3))
        # find current for 2 ferro and 1 air core magnetorquers using dipole / nA*epsilon
        for i in range(3):
            current[i] = moment[i] / (self.mags[i].n * self.mags[i].area * self.mags[i].epsilon)

        # convert current to voltage using Ohm's Law
        voltage = np.array(current) * RESISTANCE_MAG
        # voltage = np.array(current) * self.resistances

        return np.array(voltage)


    def updateAlphaReadings(self) -> bool:
        '''
        Adds the most recent alpha reading and drops oldest reading from array
        Should be called after recieving inputs

        @returns:
            bool: whether or not alpha is increasing
        '''
        # update alpha readings for both cameras
        if (self.cam1.alpha > 0.05 and self.cam1.alpha < .95): #don't update if no edge detected
            self.cam1_alphas = np.roll(self.cam1_alphas, -1)
            self.cam1_alphas[len(self.cam1_alphas)-1] = self.cam1.alpha

        if (self.cam2.alpha > 0.05 and self.cam2.alpha < .95): #don't update if no edge detected
            self.cam2_alphas = np.roll(self.cam2_alphas, -1)
            self.cam2_alphas[len(self.cam2_alphas)-1] = self.cam2.alpha


        # filter -1s and get valid alpha values
        cam1_valid_indices = np.where(self.cam1_alphas != -1)[0]
        cam1_valid_alphas = self.cam1_alphas[cam1_valid_indices]
        cam2_valid_indices = np.where(self.cam2_alphas != -1)[0]
        cam2_valid_alphas = self.cam2_alphas[cam2_valid_indices]

        # Only fit if there are at least 2 valid values
        if len(cam1_valid_alphas) > 1:
            slope1 = np.polyfit(cam1_valid_indices, cam1_valid_alphas, 1)[0]
            self.cam1_alpha_increasing = slope1 > 0
        else:
            self.cam1_alpha_increasing = False

        if len(cam2_valid_alphas) > 1:
            slope2 = np.polyfit(cam2_valid_indices, cam2_valid_alphas, 1)[0]
            self.cam2_alpha_increasing = slope2 > 0
        else:
            self.cam2_alpha_increasing = False

        # print("Slope1:", slope1 if len(cam1_valid_alphas) > 1 else 'N/A')
        # print("Slope2:", slope2 if len(cam2_valid_alphas) > 1 else 'N/A')

        return self.cam1_alpha_increasing, self.cam2_alpha_increasing

    def update_images(self, image1=None, image2=None, degree=0):
        '''
        Updates the images for both cameras
        '''
        # Process images
        image1_results = processImage(image1, degree=degree)
        self.cam1.roll = image1_results[0]
        self.cam1.pitch = image1_results[1]
        self.cam1.alpha = image1_results[2]
        self.cam1.edges = image1_results[3]
        self.cam1.midpoint = image1_results[4]
        self.cam1.ends = image1_results[5]

        image2_results = processImage(image2, degree=degree)
        self.cam2.roll = image2_results[0]
        self.cam2.pitch = image2_results[1]
        self.cam2.alpha = image2_results[2]
        self.cam2.edges = image2_results[3]
        self.cam2.midpoint = image2_results[4]
        self.cam2.ends = image2_results[5]

        cam1_alpha_increasing, cam2_alpha_increasing = self.updateAlphaReadings()

        # new case: if pitch/roll/yaw etc are -1 but alpha is 1 or 0,
        if self.cam1.roll == -1 and self.cam1.pitch == -1 and (self.cam1.alpha == 1.0 or self.cam1.alpha == 0.0):
            self.cam1.alpha = 1.0 if cam1_alpha_increasing else 0.0

        if self.cam2.roll == -1 and self.cam2.pitch == -1 and (self.cam2.alpha == 1.0 or self.cam2.alpha == 0.0):
            self.cam2.alpha = 1.0 if cam2_alpha_increasing else 0.0

        return image1_results, image2_results
