'''
bertha.py
Authors:

Combined script for interfacing with our BERTHA test bed.
WIP
First step, connect to VN100 IMU and read datapip

'''

# add to path variable so that subdirectory modules can be imported
import sys, os
sys.path.extend([f'./{name}' for name in os.listdir(".") if os.path.isdir(name)])

import os
import numpy as np

import HardwareInterface.vn100.vn100_interface as vn

import sim.visualizer as simulator

import time

def visualize_data(i, quaternion):
    '''
    use our cubesat simulator to show quaternion (orientation)
    Note: i must start at 0 and then increment

    @params:
        quaternion: orientation from one data read of VN100 IMU (1 x 4)
    '''

    simulator.game_visualize(np.array([quaternion]), i)


'''
read VN100 Software Documentation from documentation folder in SDK->python folder

Control center:
https://www.vectornav.com/downloader?file=https://www.vectornav.com/docs/default-source/software/controlcenter_setup_v3_4_0.exe&key=d2dfe074-c44c-4eb5-940e-c9e0356721c&id=16b36c6b-14d6-4bb8-962a-a081816b205e

'''

if __name__ == "__main__":

    # ==============================================================================
    # connect to VN100 IMU. run setup.py if needed, check and print sensor info, etc

	# declare sensor object
    vn.connect("COM5")

    # count = 100
    # file_name = "test.txt"
    # vn.print_data_to_file(count, file_name)
    # print 'count' counts of data into the file with name 'file_name'

    # vn.disconnect()


    # ==============================================================================
    # once we're connected to IMU, set up a loop to read a stream of data

    # keep track of our iteration count
    i = 0
    count = 50
    file_name = "testing_new_library.txt"
    f = open(file_name, "a+")

    while i < count:

        quat = vn.read_quat()
        visualize_data(i, quat)

        i += 1

        #time.sleep(.1)
        #print("")
		#save to text file in form of magnetometer (magnetic field), angular velocity (gyroscope), and acceleration (accelerometer)
        f.write(vn.get_mag_gyro_quat()) # put mag, gyro, quat data into text file
        if (i < count):
            f.write("\n") # add newline to separate data sets
        # optional: save to text file in form of magnetometer (magnetic field), angular velocity (gyroscope), and acceleration (accelerometer)

    f.close()
    #source = f'./{file_name}'
    #destination = './new_sensor_tests'
    #os.rename(source, destination)

    vn.print_data_to_file(count, "another_test.txt")

    vn.disconnect()
    # ==============================================================================
