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

#cloversat\Main\SimScripts
#cloversat\ukf
#these are where the multiple ukf py files all seem to be located

#also need to merge Rawan's changes, these will be in the motors branch, but even though she told me there should be a rotate function, I could not find it, reach out to her


# add to path variable so that subdirectory modules can be imported
import sys, os
import csv
import re

# Find project root by going up directories until we find HardwareInterface folder
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

import numpy as np

import HardwareInterface.vn100.vn100_interface as vn

import Simulator.visualizer as simulator

import time

data = []
TARGET_GPS_INTERVAL = 1.0
DEFAULT_CSV_TIMESTEP = 0.1

#uptime varaibale while running TODO

def visualize_data(i, quaternion):
    '''
    use our cubesat simulator to show quaternion (orientation)
    Note: i must start at 0 and then increment

    @params:
        quaternion: orientation from one data read of VN100 IMU (1 x 4)
    '''

    simulator.game_visualize(np.array([quaternion]), i)

def _parse_csv_timestep(metadata_row):
    '''
    Parse timestep from metadata line in CSV header, using regex.
    Defaults to DEFAULT_CSV_TIMESTEP if not found.
    '''
    metadata_line = ",".join(metadata_row)
    timestep_match = re.search(r"timestep\s*=\s*([0-9]*\.?[0-9]+)", metadata_line)
    if timestep_match:
        return float(timestep_match.group(1))
    return DEFAULT_CSV_TIMESTEP

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
    
    # Use elapsed whole-second index so data is consumed at 1-second intervals.
    sample_idx = int(elapsed_time // TARGET_GPS_INTERVAL)
    sample_idx = max(0, min(sample_idx, len(data) - 1))

    # Return GPS data (skip timestamp, return [Bx, By, Bz, X, Y, Z])
    return data[sample_idx][1:]

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
        Computes row stride from CSV timestep metadata to enforce 1-second sampling.
    '''
    csv_path = os.path.join(project_root, "Simulator", "PySOL", "outputs", "1d_1orbit_tenth_s_gps.csv")
    with open(csv_path, "r", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if len(rows) < 3:
        raise ValueError("GPS CSV is missing expected metadata/header rows.")

    csv_timestep = _parse_csv_timestep(rows[0])
    if csv_timestep <= 0:
        raise ValueError(f"Invalid CSV timestep: {csv_timestep}")

    sample_stride = max(1, int(round(TARGET_GPS_INTERVAL / csv_timestep)))
    sampled_interval = sample_stride * csv_timestep
    if abs(sampled_interval - TARGET_GPS_INTERVAL) > 1e-6:
        print(
            f"Warning: nearest possible interval is {sampled_interval:.6f}s, not exactly {TARGET_GPS_INTERVAL:.1f}s"
        )
    
    # Skip header lines (orbital params line and column names line)
    data_rows = rows[2:]

    data.clear()
    for row_index in range(0, len(data_rows), sample_stride):
        row = data_rows[row_index]
        timestamp = row_index * csv_timestep

        # Convert each string element to float (preserves scientific notation)
        converted_row = [timestamp] + [float(item) for item in row]
        data.append(converted_row)

    print(
        f"Loaded {len(data)} GPS rows at {sampled_interval:.3f}s interval from {os.path.basename(csv_path)}"
    )

    if len(data) >= 2:
        measured_interval = data[1][0] - data[0][0]
        check_status = "PASS" if abs(measured_interval - TARGET_GPS_INTERVAL) <= 1e-6 else "FAIL"
        print(
            f"GPS interval check: expected {TARGET_GPS_INTERVAL:.3f}s, measured {measured_interval:.3f}s ({check_status})"
        )


def init_reaction_wheel():
    '''
    Initialize reaction wheel control if dependencies and hardware are available.
    Returns (pi, wheel) or (None, None) when wheel control is unavailable.
    '''
    try:
        import importlib
        pigpio = importlib.import_module("pigpio")
        from HardwareInterface.reaction_wheels import motors as rw_motors
    except Exception as exc:
        print(f"Reaction wheel import unavailable: {exc}")
        return None, None

    pi = pigpio.pi()
    if not pi.connected:
        print("Reaction wheel disabled: pigpio daemon not running.")
        return None, None

    wheel = rw_motors.ReactionWheel(
        pi,
        rw_motors.DAA,
        rw_motors.COMU,
        rw_motors.FREQ,
        rw_motors.PWM,
        rw_motors.BR,
        rw_motors.DIRE,
    )
    wheel.callback = pi.callback(rw_motors.FREQ, pigpio.RISING_EDGE, wheel.get_rpm_callback)
    return pi, wheel


def wheel_command_for_step(step_idx, total_steps):
    '''
    Simple command profile to verify wheel communication in bertha loop.
    '''
    third = max(1, total_steps // 3)
    if step_idx < third:
        return 40
    if step_idx < 2 * third:
        return -40
    return 0


'''
read VN100 Software Documentation from documentation folder in SDK->python folder

Control center:
https://www.vectornav.com/downloader?file=https://www.vectornav.com/docs/default-source/software/controlcenter_setup_v3_4_0.exe&key=d2dfe074-c44c-4eb5-940e-c9e0356721c&id=16b36c6b-14d6-4bb8-962a-a081816b205e

'''

if __name__ == "__main__":
    start = time.time()
    read_csv()

    pi = None
    wheel = None
    try:
        # Connect to VN100 IMU.
        vn.connect("COM5")

        # Optional wheel setup: script still runs if wheel stack is unavailable.
        pi, wheel = init_reaction_wheel()

        count = 50
        file_name = "bertha_imu_wheel_log.csv"
        file_path = os.path.join(project_root, "Main", "HardwareScripts", file_name)

        with open(file_path, "w", newline="") as log_file:
            writer = csv.writer(log_file)
            writer.writerow(
                [
                    "elapsed_s",
                    "q_w",
                    "q_x",
                    "q_y",
                    "q_z",
                    "wheel_cmd",
                    "wheel_rpm",
                    "gps_bx",
                    "gps_by",
                    "gps_bz",
                    "gps_x",
                    "gps_y",
                    "gps_z",
                ]
            )

            i = 0
            while i < count:
                quat = vn.read_quat()
                elapsed = time.time() - start

                # GPS is assumed readable for now; if unavailable, keep row shape stable.
                gps_data = get_gps_at_timestamp(elapsed)
                gps_row = gps_data if gps_data is not None else ["", "", "", "", "", ""]

                wheel_cmd = 0
                wheel_rpm = ""
                if wheel is not None:
                    wheel_cmd = wheel_command_for_step(i, count)
                    wheel.set_speed(wheel_cmd)
                    wheel_rpm = wheel.rpm

                print(f"step={i} quat={quat} wheel_cmd={wheel_cmd}")
                visualize_data(i, quat)

                writer.writerow(
                    [
                        elapsed,
                        quat[0],
                        quat[1],
                        quat[2],
                        quat[3],
                        wheel_cmd,
                        wheel_rpm,
                    ]
                    + gps_row
                )

                i += 1

        print(f"Saved run log to {file_path}")

    finally:
        if wheel is not None:
            wheel.kill()
        if pi is not None:
            pi.stop()
        try:
            vn.disconnect()
        except Exception as exc:
            print(f"VN100 disconnect warning: {exc}")
