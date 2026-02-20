'''
bertha.py
Authors:

Combined script for interfacing with our BERTHA test bed.
WIP
First step, connect to VN100 IMU and read datapip

2/17/24
read from sensors:
	-vectornav: magnetic field, angular acceeraitoon
	-gps: reads from csv file
pass the data to control law (choose)
pass output to the reaction wheels
'''

# add to path variable so that subdirectory modules can be imported
import sys, os
import csv

# Find project root by going up directories until we find HardwareInterface folder
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

import numpy as np

import HardwareInterface.vn100.vn100_interface as vn

import Simulator.visualizer as simulator

import time

data = []
gps_timestep = 0.1

#uptime varaibale while running TODO

def visualize_data(i, quaternion):
    '''
    use our cubesat simulator to show quaternion (orientation)
    Note: i must start at 0 and then increment

    @params:
        quaternion: orientation from one data read of VN100 IMU (1 x 4)
    '''

    simulator.game_visualize(np.array([quaternion]), i)

def get_gps_at_timestamp(elapsed_time):
    '''
    Get GPS data closest to the given elapsed time.
    
    @params:
        elapsed_time (float): Time in seconds since script started
        
    @returns:
        [Bx, By, Bz, X, Y, Z] or None if no data loaded
    '''
    if not data:
        return None
    
    # Find closest timestamp in GPS data
    closest_idx = 0
    min_diff = float('inf')
    
    for idx, gps_row in enumerate(data):
        timestamp = gps_row[0]
        time_diff = abs(timestamp - elapsed_time)
        if time_diff < min_diff:
            min_diff = time_diff
            closest_idx = idx
    
    # Return GPS data (skip timestamp, return [Bx, By, Bz, X, Y, Z])
    return data[closest_idx][1:]

def read_csv():
    '''
    Read GPS and magnetic field data from CSV file.
    
    Data structure:
        Each row in the 'data' list is formatted as:
        [timestamp, Bx, By, Bz, X, Y, Z]
        
        Where:
            timestamp (float): Time in seconds from start of simulation
            Bx (float): Magnetic field X component (microTesla)
            By (float): Magnetic field Y component (microTesla)
            Bz (float): Magnetic field Z component (microTesla)
            X (float): GPS position X coordinate in ECEF frame (km)
            Y (float): GPS position Y coordinate in ECEF frame (km)
            Z (float): GPS position Z coordinate in ECEF frame (km)
    
    Sampling:
        Reads every 10th row from the CSV to align with second marks
    '''
    csv_path = os.path.join(project_root, "Simulator", "PySOL", "outputs", "1d_1orbit_tenth_s_gps.csv")
    with open(csv_path, "r", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)
    
    # Skip header lines (orbital params line and column names line)
    data_rows = rows[2:]
    
    # Sample every 10th row (aligns with second marks)
    for idx, row in enumerate(data_rows[::10]):
        # Calculate timestamp: which data row this is * timestep
        row_index = idx * 10
        timestamp = row_index * gps_timestep
        
        # Convert each string element to float (preserves scientific notation)
        converted_row = [timestamp] + [float(item) for item in row]
        data.append(converted_row)
    print(data)


'''
read VN100 Software Documentation from documentation folder in SDK->python folder

Control center:
https://www.vectornav.com/downloader?file=https://www.vectornav.com/docs/default-source/software/controlcenter_setup_v3_4_0.exe&key=d2dfe074-c44c-4eb5-940e-c9e0356721c&id=16b36c6b-14d6-4bb8-962a-a081816b205e

'''

if __name__ == "__main__":
    start = time.time()
    read_csv()

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
        elapsed = time.time() - start
        gps_data = get_gps_at_timestamp(elapsed)
        
        print(quat)
        visualize_data(i, quat)

        i += 1

        #time.sleep(.1)
        #print("")
		#save to text file in form of magnetometer (magnetic field), angular velocity (gyroscope), and acceleration (accelerometer)
        f.write(vn.get_mag_gyro_quat()) # put mag, gyro, quat data into text file
        if gps_data:
            f.write(gps_data)
        if (i < count):
            f.write("\n") # add newline to separate data sets
        # optional: save to text file in form of magnetometer (magnetic field), angular velocity (gyroscope), and acceleration (accelerometer)
    f.write("\n")
    f.write(str(time.time() - start))
    f.close()
    #source = f'./{file_name}'
    #destination = './new_sensor_tests'
    #os.rename(source, destination)

    vn.print_data_to_file(count, "another_test.txt")

    vn.disconnect()
    # ==============================================================================
