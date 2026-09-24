import numpy as np
# import PySOL in specific order
import Simulator.PySOL.wmm as wmm
from Simulator.PySOL.sol_sim import *
import Simulator.PySOL.spacecraft as sp
import Simulator.PySOL.orb_tools as ot
from params import *


from Simulator.simulator import Simulator
from Simulator.sat_model import Magnetorquer_Sat
from Simulator.magnetorquer import Magnetorquer

def main():
    kp = 0.01
    kd = 0.01
    
    aggregated_results = []
    B_body = CONSTANT_B_FIELD_MAG
    

    # create 3 Magnetorquer objects to store in Magnetorquer_Sat object
    mag1 = Magnetorquer(n = FERRO_NUM_TURNS, area = FERRO_AREA, k = K, epsilon = FERRO_EPSILON)
    mag2 = Magnetorquer(n = FERRO_NUM_TURNS, area = FERRO_AREA, k = K, epsilon = FERRO_EPSILON)
    mag3 = Magnetorquer(n = AIR_NUM_TURNS, area = AIR_AREA, k = K, epsilon = 1)
    mag_array = np.array([mag1, mag2, mag3])
    
    print("kp: ", kp, "kd: ", kd, "i: ", 0)
    B_earth, gps = get_orbit_data(f'leo_oe_{0}.csv', 3, DT, GPS=True)
    # b body overwritten during sim initialization if CONSTANT_B_FIELD is false
    prevB = np.array([0, 0, 0])

    # initialize object to hold satellite properties
    mag_sat = Magnetorquer_Sat(CUBESAT_BODY_INERTIA, mag_array, VELOCITY_INITIAL, B_body, prevB, DT, GYRO_WORKING, kp, kd)

    # run simulation from simulator.py and generate pdf report of results
    sim = Simulator(mag_sat, B_earth)
    
    i = 1

    while i < sim.n:
        # generate ideal state based on last so that we can better estimate sensor data
        # ideal = sim.find_ideal(i)

        # generate fake sensor data in body frame based on last state
        sim.generateData_step(sim.states[i-1], i)

        # find how far we are from nadir (for controller input)
        sim.findNadir(sim.states[i-1], gps[i], i)

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