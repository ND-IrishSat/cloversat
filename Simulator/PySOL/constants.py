"""
    constants.py

    Functions Summaries:
    R_E(units='km')                 - returns Earth's radius in either km, m, or cm units

    mu_Earth(units = 'km^3/s^2')    - returns the standard gravitational parameter (μ) for the Earth
    mu_Sun(units = 'km^3/s^2')      - returns the standard gravitational parameter (μ) for the sun
    mu_moon(units = 'km^3/s^2')     - returns the standard gravitational parameter (μ) for the moon
"""

import numpy as np
import constants 

def R_E(units: str = 'km') -> int:
    R_E = 6371 # Earth's Radius in kilometers

    if units == 'm':
        R_E *= 1_000
    elif units == 'cm':
        R_E *= 100_000

    return R_E


# Defining gravitational parameter for relevant solar system objects
# Taken from Table A.2 in Curtis

def mu_Earth(units: str = 'km^3/s^2') -> int:

    mu_Earth = 398600 

    return mu_Earth

def mu_Sun(units: str = 'km^3/s^2') -> int:
    
    mu_Sun = 132712000000

    return mu_Sun

def mu_moon(units: str = 'km^3/s^2') -> int:

    mu_Moon = 4903

    return mu_Moon

