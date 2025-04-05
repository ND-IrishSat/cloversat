'''
run_UKF.py
Authors: Andrew Gaylord, Claudia Kuczun, Michael Paulucci, Alex Casillas, Anna Arnett
Last modified 10/7/23

Runs IrishSat UKF on generated or real-time data and simulates CubeSat using pygame

TODO:
    
    find freshman who wants to learn UKF
    finish labeling equations in UKF_algorithm

    note: in banquet_demo.py, we don't use gps. instead, pre-generate magnetic fields for known simulated flight path for correct time step
    add to ukf latex documentation: testing cases/results, R and Q research/results, more explanation
    update EOMs to 4 reaction wheels
    not important: interface with gps sensor, find what frame it gives us (ECEF or ECI?) and units?
'''

# add to path variable so that subdirectory modules can be imported
import sys, os
sys.path.extend([f'./{name}' for name in os.listdir(".") if os.path.isdir(name)])

import numpy as np

import ukf.UKF_algorithm as UKF_algorithm
import interface.gps_interface as gps_interface
import ukf.ideal_test_cases as ideal_test_cases
import ukf.hfunc as hfunc
import sim.visualizer as simulator
import sim.PySOL.wmm as wmm
# from interface.happy_sensors import get_imu_data

# from sim.PySOL import spacecraft as sp
# from sim.PySol.spacecraft import *
from sim.PySOL.sol_sim import *
import sim.PySOL.spacecraft as sp
import sim.PySOL.orb_tools as ot
# from sim.PySOL import sol_sim
# from sim.PySOL import orb_tools as ot


def check_zeros(data):
    '''
    check_zeros
        checks validity of data input

    @params
        data: input from sensor real-time (1 x 6)
        
    @returns
        true or false 
    '''
    if int(data[0]) == 0 and int(data[1]) == 0 and int(data[2]) == 0:
        return True


def run_ukf_textfile(start, cov, r, q, filename):
    '''
    run_ukf_textfile
        runs and visualizes UKF algorithm on input data file

    @params
        start: initialized state (1 x n)
        cov: initialized covariance matrix (n x n)
        r: noise vector for predictions (1 x n)
        q: noise vector for sensors (1 x m)
        filename: text file of cleaned sensor data to read from (any length)
    '''

    # set up orbital simulation so can find our position and height for every time step

    # startTime = 2022.321
    t0 = dt.datetime(2022, 3, 21, 0, 0, 0)
    # sim = sol_sim.Simulation(TIME = t0, mag_deg= 12)
    sim = Simulation(TIME = t0, mag_deg= 12)

    # how long we're simulating for
    duration = .02
    OE1 = ot.OE_array(f = 0, a = 6_800, e = 0.00068, i = 51, Om = 30, w = 30)
    sim.create_sc(OE_array= OE1, verbose = True, color = 'green', name = 'Low-Earth Orbit')


    DT = dt.timedelta(hours = duration)
    # resolution = timestep. Must match with rest of ukf
    dt_ukf = .1
    sim.propogate(DT, resolution =  .1)
    orb_laln = sim.scs[0].state_mat.LALN
    orb_h = ot.calc_h(sim.scs[0].state_mat.R_ECEF)

    # print(sim.scs[0].state_mat.R_ECEF.shape)
    # latitude/longitude and height
    print(len(orb_laln))
    print(len(orb_h))

    # Get B field at current lat/long/altitude
    curr_date_time= np.array([2024.1066])
    lat = np.array([41.675])
    long = np.array([-86.252])
    alt = np.array([225.552]) # 740 feet (this is in meters)
    B_true = wmm.bfield_calc(np.array([lat, long, alt, curr_date_time]))

    # find absolute path to text file
    script_path = os.path.abspath(__file__)
    script_dir = os.path.split(script_path)[0]
    abs_file_path = os.path.join(script_dir, filename)
    f = open(abs_file_path, "r")
    
    data = f.readline()
    splitData2 = np.array([float(x) for x in data.split(",")])
    # for test-still: accelerometer, gyro, magnetometer (microteslas)
    # data is in the form of magnetic field (bx, by, zy) and angular velocity (wx, wy, wz)
    splitData = np.concatenate((splitData2[6:], splitData2[3:6]))

    reaction_speeds = np.zeros(4)
    i = 0
    while(i < len(orb_laln) and data):
        # gps_data = gps_interface.get_gps_data()
        # gps_data = gps_interface.ecef_to_latlong(gps_data[0], gps_data[1], gps_data[2]) # add time

        # get gps data and add time stamp
        gps_data = np.array([np.array([orb_laln[i][0]]), np.array([orb_laln[i][1]]), np.array([orb_h[i]]), np.array([2022.257])])

        gps_data = B_true
        # run ukf and visualize output
        start, cov, innov, innovCov = UKF_algorithm.UKF(start, cov, q, r, dt_ukf, gps_data, reaction_speeds, reaction_speeds, splitData)
        # option to just visualize data
        # start = np.concatenate((np.array([0]), splitData))
        simulator.game_visualize(np.array([start[:4]]), i)

        # continue to get data from file until empty
        data = f.readline()
        if(data == ''):
            break
        splitData2 = [float(x) for x in data.split(",")]
        # data is in the form of magnetic field (bx, by, zy) and angular velocity (wx, wy, wz)
        splitData = np.concatenate((splitData2[6:], splitData2[3:6]))

        i+=1

    f.close()


def run_ukf_sensor_iteration(state, cov, r, q, i):
    # data = get_imu_data()
    data = []
    gps_data = gps_interface.get_gps_data()
    gps_data = gps_interface.ecef_to_latlong(gps_data[0], gps_data[1], gps_data[2])
    gps_data.append(2023.8123)

    # do not use data if B-field is all zeros 
    if check_zeros(data): 
        return "Error" 

    b_true = wmm.bfield_calc(gps_data)

    state, cov, innov, innovCov = UKF_algorithm.UKF(state, cov, r, q, b_true, data)
    # Visualize only if needed
    # game_visualize(np.array([state[:4]]), i) 

    return state, cov


def run_ukf_sensor(state, cov, r, q):
    '''
    run_ukf_sensor
        runs and visualizes UKF algorithm using real-time data from magnetometer/pi

    @params
        start: initialized state (1 x n)
        cov: initialized covariance matrix (n x n)
        r: noise vector for predictions (1 x n)
        q: noise vector for sensors (1 x m)
    '''

    # uncomment BNO055 imports to use

    # i = 1
    # gps_data = []
    # calibrate()

    # while(1):
    #     time.sleep(0.5)
    #     data = get_data()
        # gps_data = gps_interface.get_gps_data()
        # gps_data = gps_interface.ecef_to_latlong(gps_data[0], gps_data[1], gps_data[2])
        # gps_data.append(2023.8123)
    #     # do not use data if B-field is all zeros 
    #     if check_zeros(data): continue 

    #     start, cov = UKF_algorithm.UKF(start, cov, r, q, gps_data, data)
    #     game_visualize(np.array([start[:4]]), i)

    #     i += 1


if __name__ == "__main__":

    # Initialize noises and starting state/cov values
    n = 7
    m = n - 1

    start = np.array([1, 0, 0, 0, 0, 0, 0])

    cov = np.identity(n) * 5e-10


    # r: measurement noise (m x m)
    noiseMagnitude = 0.02
    r = np.diag([noiseMagnitude] * m)

    # q: process noise (n x n)
    noiseMagnitude = 0.005
    q = np.diag([noiseMagnitude] * n)


    noise_mag = 0.85
    #r = np.diag([noise_mag] * dim_mes)
    mag = 0.1 
    omg = [1,3,1]
    
    r = [[mag,0,0,0,0,0],
	 [0,mag,0,0,0,0],
	 [0,0,mag,0,0,0],
	 [0,0,0,omg[0],0,0],
	 [0,0,0,0,omg[1],0],
	 [0,0,0,0,0,omg[2]]]
    
    q = [[1e-1,0,0,0,0,0,0],
	 [0,1e-1,0,0,0,0,0],
	 [0,0,1e-1,0,0,0,0],
	 [0,0,0,1e-1,0,0,0],
	 [0,0,0,0,5e4,0,0],
	 [0,0,0,0,0,5e4,0],
	 [0,0,0,0,0,0,5e4]]

    
    # filename = "sensor_data_2.txt"
    # must be relative path to the text file
    filename = "interface/sample_data/test-still.txt"


    # tests ukf with pre-generated and cleaned data file
    run_ukf_textfile(start, cov, r, q, filename)
    # ideal_test_cases.run_moving_test()

    # must uncomment BNO055 imports to use in real-time with sensor
    # run_ukf_sensor(start, cov, r, q)
