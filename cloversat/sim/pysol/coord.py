''' coord.py

    Module for handling coordinate transformations between different reference frames
    ECI: Earth-Central Interial
    ECEF: Earth-Centered Earth-Fixed
    Transforming position and velocity from an inertial frame (ECI) to a rotating frame (ECEF)

    Juwan Jeremy Jacobe
    University of Notre Dame
'''

import numpy as np

# Angular velocity of Earth with respect to ECI
omega_Earth = np.array([0, 0, 7.3e-5]) # rad/s

# Implement ECI (or GCRF) --> ECEF (and or ITRF) for velocity and acceleration
def v_inert2v_rot(r_inert: np.ndarray, v_inert: np.ndarray, w_ri: np.ndarray = omega_Earth, v_org: np.ndarray = np.array([0, 0, 0])) -> tuple:
    '''  
    Transform velocity in an inertial frame (ECI) to a rotating frame (ECEF).

    Args:
        r_inert: position in inertial frame (ECI)
        v_inert: velocity in inertial frame (ECI)
        w_ri: angular velocity of moving frame (w_Earth)
        v_org: linear velocity of origin of moving frame, default of 0
        
    Returns:
        r_ECEF
        v_ECEF
    '''
    # Not yet implemented
    pass

def v_rot2vinert()
    '''
    Transform velocity in a rotating frame (ECEF) back to an inertial frame (ECI).
    '''
    # Not yet implemented
    pass

