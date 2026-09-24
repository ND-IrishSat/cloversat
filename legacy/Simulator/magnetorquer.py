'''
magnetorquer.py
Author: Daniel, Brian, Sophie

Class that models a single magnetorquer, which is a device that interacts with the Earth's magnetic field to control the orientation of a satellite.

'''

import os
import sys
import math
import numpy as np

# import params module from parent directory
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from params import *
from Utils.transformations import normalize, quaternion_to_euler

class Magnetorquer():
    '''Class to hold a single magnetorquer'''

    def __init__(self, n: int, area: float, k: float, epsilon: float):
        '''
        Initialize all parameters needed to model a magnetorquer
        @params:
            n (int): number of turns of coil for each magnetorquer (num of layers * num of turns per layer?)
            area (float): area of magnetorquer for each magnetorquer (m^2)
            k (float): detumbling constant gain for each magnetorquer
            epsilon (float): added multiplication term for ferromagnetic magnetorquers, 1 for air torquers
        '''
        self.n = n
        self.area = area
        self.k = k
        # accounts for the magnetic permeability and the geometry of the core. 1 for air-core magnetorquers
        self.epsilon = epsilon
        # store our resistance and inductance values for proper torquer
        if (self.epsilon == 1) :
            self.resistance = AIR_RESISTANCE_MAG
            self.inductance = AIR_INDUCTANCE_MAG
            self.max_torque = AIR_MAX_TORQUE
        else:
            self.resistance = FERRO_RESISTANCE_MAG
            self.inductance = FERRO_INDUCTANCE_MAG
            self.max_torque = FERRO_MAX_TORQUE

    def __repr__(self): # representation, returns string with details on the magnetorquer when "mag_object_name" is called
        if (self.epsilon == 1) :
            return "Air Magnetorquer: \n    Number of turns = {}\n    Area = {} m^2\n    Resistance = {}\n    Inductance = {}\n    k = {}".format(self.n, self.area, self.resistance, self.inductance, self.k)
        else: # mention special epsilon if this is a ferromagnetic magnetorquer
            return "Ferro Mumetal Magnetorquer: \n    Number of turns = {}\n    Area = {} m^2\n    Resistance = {}\n    Inductance = {}\n    k = {}\n    Magnetizing factor = {}".format(self.n, self.area, self.resistance, self.inductance, self.k, self.epsilon)