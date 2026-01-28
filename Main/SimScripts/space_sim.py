'''
space_sim.py
Authors: Andrew Gaylord, Michael Paulucci, Sarah Kopfer, Lauren, Kris, Rene Alzina, Rawan, Abe, Daniel

Main process that calls functions from simulator.py and generates pdf report of results

See params.py for how to set hardware, ADCS type, starting conditions, etc.
'''
import numpy as np

# add to path variable so that subdirectory modules can be imported
import sys, os
sys.path.extend([f'./{name}' for name in os.listdir(".") if os.path.isdir(name)])
# Add root directory of repo to path
sys.path.extend([os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))])

from Simulator.simulator import Simulator
from Simulator.sat_model import Magnetorquer_Sat
from Simulator.magnetorquer import Magnetorquer
from Controllers.PID_controller import PIDController
from params import *

# import PySOL in specific order
# import Simulator.PySOL.wmm as wmm
from Simulator.PySOL.sol_sim import *
# import Simulator.PySOL.spacecraft as sp
# import Simulator.PySOL.orb_tools as ot

def main():
    if ADCS_TYPE == "AD" or RUNNING_MAYA:
        print("ERROR: set params correctly")
        return 1

    # earth magnetic field data (microTeslas)
    B_earth = CONSTANT_B_FIELD_MAG

    if not CONSTANT_B_FIELD:

        if not GENERATE_NEW:
            # use pre-generated data (ensuring OE matches what's in B_FIELD_CSV_FILE)
            # B_earth, gps = get_orbit_data(B_FIELD_CSV_FILE, oe, duration, DT, GPS=True)
            # Remove OE check
            B_earth, gps = get_orbit_data(B_FIELD_CSV_FILE, None, HOURS, DT, GPS=True)
        else:
            # generate new data with orbital elements (see generate_orbit_data in sol_sim.py for more info)
            B_earth, gps = generate_orbit_data(ORBITAL_ELEMENTS, HOURS, DT, B_FIELD_CSV_FILE, store_data=False, GPS=True)

        # if we're simulating for less than pre-generated data was run for, cut it short
        if len(B_earth) > int(TF / DT):
            # TODO: check that timestep lines up, fetch OE
            B_earth = B_earth[:int(TF / DT)]
        elif len(B_earth) < int(TF / DT):
            print("ERROR: not enough data points in B_earth. {} needed, {} created".format(int(TF/DT), len(B_earth)))
            return

    # create 3 Magnetorquer objects to store in Magnetorquer_Sat object
    mag1 = Magnetorquer(n = FERRO_NUM_TURNS, area = FERRO_AREA, k = K, epsilon = FERRO_EPSILON)
    mag2 = Magnetorquer(n = FERRO_NUM_TURNS, area = FERRO_AREA, k = K, epsilon = FERRO_EPSILON)
    mag3 = Magnetorquer(n = AIR_NUM_TURNS, area = AIR_AREA, k = K, epsilon = 1)
    mag_array = np.array([mag1, mag2, mag3])

    # initialize object to hold satellite properties
    # b body overwritten during sim initialization if CONSTANT_B_FIELD is false
    mag_sat = Magnetorquer_Sat(CUBESAT_BODY_INERTIA, mag_array, VELOCITY_INITIAL, CONSTANT_B_FIELD_MAG, np.array([0, 0, 0]), DT, GYRO_WORKING, KP, KD)

    controller = PIDController(KP, KI, KD, DT)

    sim = Simulator(mag_sat, B_earth, controller=controller)
    i = 1

    while i < sim.n:
        # generate fake sensor data in body frame based on last state
        sim.generateData_step(sim.states[i-1], i)

        if not CONSTANT_B_FIELD:
            # find how far we are from nadir (for controller input)
            sim.findTrueNadir(sim.states[i-1], gps[i], i)

        # Filter data to get attitude estimate
        if RUN_UKF:
            sim.determine_attitude(i)

        # check what protocol we should be in and update state
        sim.mag_sat.state = sim.check_state(i)

        # decide voltage for self.voltages[i] (depending on state)
        sim.controls(i)

        # propagate current state (orientation + velocity) based on voltages[i]
        sim.propagate_step(i)

        # calculate total power usage for this time step (Watts)
        # removed from saving.py bc pretty useless lol
        sim.totalPower[i] = sim.power_output[i][0] + sim.power_output[i][1] + sim.power_output[i][2]

        i += 1

    sim.plot_and_viz_results()


if __name__ == '__main__':
    main()
