"""
    models.py

"""


import numpy as np

class Orbital_Models:

    def __init__(self, model_name):

        self.md_nm = model_name

        # Defining gravitational parameter for relevant solar system objects
        # Taken from Table A.2 in Curtis
        # All units (km^3/s^2)
        self.mu_earth = 398600
        self.mu_moon = 4903
        self.mu_sun = 132712000000

        self.state_func_dict = {
            'Two-Body' : self.TBP_state_func
        }

        pass

    def get_state_func(self):

        state_func = self.state_func_dict[self.md_nm]

        return state_func

    def TBP_state_func(self, t, S):
        """
            state_func - Runge Kutta integration function

            Forms equation of motion into first order ODE
            F(X) = Xdot

            !!! Currently every thing is ECI !!!
 
            Arguments:
                t (float) : time (seconds)
                s (6x1 array) : state vector (ECI)
                                [x, y, z, dx, dy, dz]^T in km and km/s
        
        """

        # Unpack the state vector
        x, y, z, xdot, ydot, zdot = S

        mu_E = self.mu_earth

        r_ = S[0:3]
        r = np.linalg.norm(S[0:3])
        
        rdot = S[3:6]
        rddot = -(mu_E/r**3)*r_
        
        Sdot = np.concatenate((rdot, rddot))

        return Sdot



        
        

