'''
Used to find optimal KP and KD gains for magnetorquer PD controller for nadir pointing

Built in a time crunch, so lots of slapdash inefficiencies that could be improved
'''

import numpy as np
import os
import sys

# Add the root directory to the Python path (2 levels up from Pointing/gains_testing/)
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(script_dir))
pysol_dir = os.path.join(root_dir, 'Simulator', 'PySOL')
sys.path.insert(0, root_dir)

# import PySOL in specific order
import Simulator.PySOL.wmm as wmm
from Simulator.PySOL.sol_sim import *
import Simulator.PySOL.spacecraft as sp
import Simulator.PySOL.orb_tools as ot
from params import *

from Simulator.simulator import Simulator
from Simulator.sat_model import Magnetorquer_Sat
from Simulator.magnetorquer import Magnetorquer

def random_leo_elements(i):
    a = EARTH_RADIUS + np.random.uniform(300, 900)  # km
    e = np.random.uniform(0, 0.02)
    i = np.random.uniform(0, 120)
    RAAN = np.random.uniform(0, 360)
    w = np.random.uniform(0, 360)
    nu = np.random.uniform(0, 360)
    return np.array([RAAN, a, e, i, w, nu])

def generate_initial_conditions(num_conditions=10, seed=42):
    '''
    Generate a variety of initial conditions for testing.

    @params:
        num_conditions: Number of different initial conditions to generate
        seed: Random seed for reproducibility

    @returns:
        List of tuples: [(quaternion, velocity), ...]
    '''
    np.random.seed(seed)
    conditions = []

    # Add some predefined interesting starting conditions
    conditions.append((np.array([1.0, 0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0])))  # Nadir, no rotation
    conditions.append((np.array([0.707, 0.707, 0.0, 0.0]), np.array([0.0, 0.0, 0.0])))  # 90 deg rotation

    # Generate random quaternions (uniformly distributed on unit sphere)
    for _ in range(num_conditions - len(conditions)):
        # Random quaternion
        q = np.random.normal(0, 1, 4)
        q = q / np.linalg.norm(q)

        w = np.array([0.0, 0.0, 0.0])

        conditions.append((q, w))

    return conditions

def quat_error_to_angle(q, scalar_first=True):
    """
    Convert a unit quaternion error to angular error.
    q: quaternion error (difference between current and desired orientation)
    scalar_first: True if q = [w, x, y, z], False if q = [x, y, z, w]
    Returns: (angle_radians, angle_degrees)
    """

    q = np.asarray(q, dtype=float)

    # reorder if scalar is last
    if not scalar_first:
        q = q[[3,0,1,2]]

    w = q[0]

    # numerical safety
    w = np.clip(w, -1.0, 1.0)

    angle_rad = 2.0 * np.arccos(w)
    angle_deg = np.degrees(angle_rad)

    return angle_deg


def main():
    sim_lengths = [
        1.5,    # 1.5 hours per simulation
    ]

    # How many starting LEO orbits to use for each gains combination
    # We have 20 stored in pysol_dir/outputs/leo_oe_1.csv ... leo_oe_20.csv
    num_initial_conditions = 10

    # Generate gain sets: logarithmic sweep from max_gain to min_gain
    # For each step, we decrease by one order of magnitude
    max_gain = 0.1
    min_gain = 0.00001
    step_exponent = 1  # Step by 1 order of magnitude (10^1, 10^0, 10^-1, ...)

    # Generate logarithmic sequence: 10^1, 10^0, 10^-1, ..., 10^-5
    # This gives us: 10, 1, 0.1, 0.01, 0.001, 0.0001, 0.00001
    gain_values = []
    exponent = int(np.log10(max_gain))  # Start at 1 (for 10^1)
    while True:
        gain_val = 10.0 ** exponent
        if gain_val < min_gain:
            break
        gain_values.append(gain_val)
        exponent -= step_exponent

    # Ensure we include min_gain exactly (10^-5 = 0.00001)
    if len(gain_values) == 0 or abs(gain_values[-1] - min_gain) > 1e-10:
        gain_values.append(min_gain)

    print(f"\nGain values to test: {gain_values}")
    print(f"Total gain values: {len(gain_values)}")

    # Generate all combinations of KP and KD
    gain_sets = []
    for kp in gain_values:
        for kd in gain_values:
            gain_sets.append((kp, kd))

    print(f"Total gain combinations: {len(gain_sets)}")
    print(f"With {num_initial_conditions} initial conditions and {len(sim_lengths)} simulation lengths:")
    total_sims = len(gain_sets) * num_initial_conditions * len(sim_lengths)
    print(f"Total simulations to run: {total_sims}")

    # Ask for confirmation if running a large number of simulations
    if total_sims > 100:
        response = input(f"\nThis will run {total_sims} simulations. Continue? (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            return
    
    # Wipe all os.path.join(script_dir, "results_new", f'tweaker_results{j}.txt') files first and create empty new ones
    results_dir = os.path.join(script_dir, "results_new")
    os.makedirs(results_dir, exist_ok=True)

    # Wipe all result files first and create empty new ones
    for j in range(num_initial_conditions):
        with open(os.path.join(results_dir, f'tweaker_results{j}.txt'), 'w') as f:
            f.write("")

    # ============ GENERATE INITIAL CONDITIONS ============
    # print("\nGenerating initial conditions...")
    # initial_conditions = generate_initial_conditions(num_initial_conditions)
    # print(f"Generated {len(initial_conditions)} initial conditions")

    # ============ EVALUATE EACH GAIN SET ============
    B_body = CONSTANT_B_FIELD_MAG
    # create 3 Magnetorquer objects to store in Magnetorquer_Sat object
    mag1 = Magnetorquer(n = FERRO_NUM_TURNS, area = FERRO_AREA, k = K, epsilon = FERRO_EPSILON)
    mag2 = Magnetorquer(n = FERRO_NUM_TURNS, area = FERRO_AREA, k = K, epsilon = FERRO_EPSILON)
    mag3 = Magnetorquer(n = AIR_NUM_TURNS, area = AIR_AREA, k = K, epsilon = 1)
    mag_array = np.array([mag1, mag2, mag3])

    end = []
    for kp, kd in gain_sets:
        if kd > kp:
            continue
        sub_end = []
        print("kp: ", kp, "kd: ", kd)
        for j in range(num_initial_conditions):
            # print("kp: ", kp, "kd: ", kd, "i: ", j)

            B_earth, gps = get_orbit_data(os.path.join(pysol_dir, "outputs", f'leo_oe_{j}.csv'), None, sim_lengths[0], DT, GPS=True)
            # if we're simulating for less than pre-generated data was run for, cut it short
            if len(B_earth) > int(TF / DT):
                B_earth = B_earth[:int(TF / DT)]
            elif len(B_earth) < int(TF / DT):
                print("ERROR: not enough data points in B_earth. {} needed, {} created".format(int(TF/DT), len(B_earth)))
                return

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
                sim.findTrueNadir(sim.states[i-1], gps[i], i)

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
            # sim.plot_and_viz_results()

            count = 0
            count2 = 0
            for q in sim.errorQuats:
                count += 1 if quat_error_to_angle(q) < 20 else 0
                count2 += 1 if quat_error_to_angle(q) < 10 else 0
            sub_end += [(kp, kd, count/sim.n, count2/sim.n)]
            # print("count: ", count, "count2: ", count2)
            with open(os.path.join(results_dir, f'tweaker_results{j}.txt'), 'a') as f:
                f.write(str(sub_end))

        avg_count = sum((x[2] for x in sub_end)) / len(sub_end)
        avg_count2 = sum((x[3] for x in sub_end)) / len(sub_end)
        end += [(kp, kd, avg_count, avg_count2)]
    end.sort(key=lambda x: x[2])
    print(str(end))
    with open(os.path.join(results_dir, 'tweaker_results.txt'), 'w') as f:
        f.write(str(end))



def notmain():
    for i in range(1,20):
        oe = random_leo_elements(i)
        print(oe)
        B_earth, gps = generate_orbit_data(oe, 3, DT, os.path.join(script_dir, f'leo_oe_{i}.csv'), store_data=True, GPS=True)

if __name__ == "__main__":
    main()