'''
simple_testbed_simulator.py
Authors: Rawan, Abe, Daniel

Script to simulate a simple 1D magnetorquer test
We want to predict how long it would take us rotate on 1 axis, then compare to real results
'''
import numpy as np

from Simulator.simulator import Simulator
import Simulator.visualizer as visualizer

from Simulator.sat_model import Magnetorquer_Sat
from Simulator.magnetorquer import Magnetorquer
from params import *

# import PySOL in specific order
import Simulator.PySOL.wmm as wmm
from Simulator.PySOL.sol_sim import *
import Simulator.PySOL.spacecraft as sp
import Simulator.PySOL.orb_tools as ot

# add to path variable so that subdirectory modules can be imported
import sys, os
sys.path.extend([f'./{name}' for name in os.listdir(".") if os.path.isdir(name)])


def main():

    if not RUNNING_1D:
        print("ERROR: set params correctly (RUNNING_1D = True)")
        return 1

    w_vector = VELOCITY_INITIAL
    gyro_working = GYRO_WORKING
    B_body = CONSTANT_B_FIELD_MAG
    prevB = np.array([0, 0, 0])
    I_body = ONED_INERTIA
    k = K

    # create 3 Magnetorquer objects to store in Magnetorquer_Sat object
    if TEST_AIRCORE:
        mag1 = Magnetorquer(n = AIR_NUM_TURNS, area = AIR_AREA, k = k, epsilon = 1)
    else:
        # regular config
        mag1 = Magnetorquer(n = FERRO_NUM_TURNS, area = FERRO_AREA, k = k, epsilon = FERRO_EPSILON)
    mag2 = Magnetorquer(n = FERRO_NUM_TURNS, area = FERRO_AREA, k = k, epsilon = FERRO_EPSILON)
    mag3 = Magnetorquer(n = AIR_NUM_TURNS, area = AIR_AREA, k = k, epsilon = 1)
    mag_array = np.array([mag1, mag2, mag3])

    # set constant magnetic field of stenson remick (overridden if CONSTANT_B_FIELD is false)
    B_earth = CONSTANT_B_FIELD_MAG

    if not CONSTANT_B_FIELD:
        # orbital elements (see generate_orbit_data in sol_sim.py for more info)
        oe = ORBITAL_ELEMENTS
        # how long to simulate orbital dynamics for (hours)
        duration = HOURS

        if not GENERATE_NEW:
            # use pre-generated data
            B_earth, gps = get_orbit_data(B_FIELD_CSV_FILE, oe, duration, DT, GPS=True)
        else:
            # generate new data
            B_earth, gps = generate_orbit_data(oe, duration, DT, B_FIELD_CSV_FILE, store_data=False, GPS=True)
        # if we're simulating for less than pre-generated data was run for, cut it short
        if len(B_earth) > int(TF / DT):
            # TODO: check that timestep lines up, fetch OE
            B_earth = B_earth[:int(TF / DT)]
        elif len(B_earth) < int(TF / DT):
            print("ERROR: not enough data points in B_earth. {} needed, {} created".format(int(TF/DT), len(B_earth)))
            return
    
    # initialize object to hold satellite properties
    mag_sat = Magnetorquer_Sat(I_body, mag_array, w_vector, B_body, prevB, DT, gyro_working, KP, KD)

    # run simulation from simulator.py and generate pdf report of results
    sim = Simulator(mag_sat, B_earth)

    i = 1

    while i < sim.n:
        # generate ideal state based on last so that we can better estimate sensor data
        # ideal = sim.find_ideal(i)

        # generate fake sensor data in body frame based on last state
        sim.generateData_step(sim.states[i-1], i)

        # check what protocol we should be in and update state
        sim.mag_sat.state = sim.check_state(i)

        # decide voltage for self.voltages[i] (depending on state)
        # if RUNNING_1D, uses custom controller
        sim.controls(i)

        # propagate current state (orientation + velocity) based on voltages[i]
        sim.propagate_step(i)

        # calculate total power usage for this time step (Watts)
        sim.totalPower[i] = sim.power_output[i][0] + sim.power_output[i][1] + sim.power_output[i][2]

        i += 1
    
    sim.plot_and_viz_results()

if __name__ == "__main__":
    main()
