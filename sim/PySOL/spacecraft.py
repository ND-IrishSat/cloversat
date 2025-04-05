"""
    spacecraft.py - 

"""


import numpy as np

import PySOL.orb_tools as ot 


class Spacecraft():

    def __init__(self, OE_array, t0, verbose = False, color = 'firebrick', name = None):
        """
            Spacecraft object

            Each Spacecraft is initialized with orbital elements

                f - true anamoly [deg]
                a - semi-major axis [km]
                e - eccentricity [0, 1)
                
                i - inclination [deg]
                Om - RA of ascending node [deg]
                w - argument of perigee [deg]        
        """

        self.OE_ = OE_array.get_OE()
        self.f0, self.a0, self.e0, self.i0, self.Om0, self.w0 = self.OE_

        self.state0 = OE_array.get_RV()
        self.time = t0

        self.state_mat = ot.State_Matrix(np.array([self.state0]), np.array([self.time]))
        
        self.color = color
        self.name = name

        if verbose:
            l = 50
            print()
            print('-'*l)
            print('-'*l)
            print('Spacecraft Initialized')
            print('Input OE array')
            print('f: {:5.4f} [deg] | a: {:5.4f} [km] | e: {:5.4f} '.format(self.f0, self.a0, self.e0))
            print('i: {:5.4f} [deg] | Om: {:5.4f} [deg] | w: {:5.4f} [deg]'.format(self.i0, self.Om0, self.w0))
            print('-'*l)
            print('-'*l)


    def calc_B(self, mag_model):

        LALNs = self.state_mat.to_LALN().copy()
        Hs = self.state_mat.H

        times = self.state_mat.times

        time_dys = np.zeros_like(times)
        for i, time in enumerate(times):
            time_dy = ot.dt_to_dec(time)
            time_dys[i] = time_dy

        mag_model.calc_gcc_components(LALNs[:, 0], LALNs[:, 1], Hs, time_dys, degrees = True)

        self.B_ = mag_model.get_Bfield()

    def set_states(self, states, times):

        self.state_mat.append_states(new_STATES= states, new_TIMES= times)


    def set_times(self, times):

        self.state_mat.append_times(new_TIMES = times)

    def get_states(self,):

        return self.states

    def get_state(self,):

        return self.state

    def get_times(self,):

        return self.times

    def get_time(self,):

        return self.time
        
   
    def get_RADEC(self, verbose = False):

        RADEC = OE_array.get_RADEC()

        if verbose:
            print('RA : {:5.4}\N{DEGREE SIGN} | Dec : {:5.4}\N{DEGREE SIGN}'.format(RADEC[0], RADEC[1]))

        return RADEC

    def get_RV(self, verbose = False):

        RV = OE_array.get_RV()
        rx, ry, rz, vx, vy, vz = RV

        if verbose:
            print('r = [{:5.4f}, {:5.4f}, {:5.4f}] km'.format(rx, ry, rz))
            print('v = [{:5.4f}, {:5.4f}, {:5.4f}] km/s'.format(vx, vy, vz))

        return RV

    


if __name__ == '__main__':

    OE_array = ot.OE_array(f = 0, a = 7_000, e = 0.1, i = 37, Om = 0, w = 90)

    sc = Spacecraft(OE_array, verbose = True)

    sc.get_RV(verbose = True)
    sc.get_RADEC(verbose= True)