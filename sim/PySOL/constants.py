"""
    constants.py

"""

import numpy as np

def R_E(units = 'km'):
    R_E = 6371 # km
    if units == 'm':
        R_E = R_E*1_000
    elif units == 'cm':
        R_E = R_E*100_000

    return R_E


# Defining gravitational parameter for relevant solar system objects
# Taken from Table A.2 in Curtis

def mu_Earth(units = 'km^3/s^2'):

    mu_Earth =  398600 

    return mu_Earth

def mu_Sun(units = 'km^3/s^2'):

    mu_Sun = 132712000000

    return mu_Sun

def mu_moon(units = 'km^3/s^2'):

    mu_Moon = 4903

    return mu_Moon

