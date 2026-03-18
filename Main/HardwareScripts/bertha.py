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
import threading

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
LOG_EVERY_STEPS = 5

#uptime varaibale while running TODO


def log_status(message, level="INFO"):
    '''
    Print timestamped runtime status for quick operator visibility.
    '''
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] [{level}] {message}")

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
    log_status(f"Loading GPS data from {csv_path}")
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
        log_status(
            f"Warning: nearest possible interval is {sampled_interval:.6f}s, not exactly {TARGET_GPS_INTERVAL:.1f}s"
            ,
            level="WARN"
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

    log_status(
        f"Loaded {len(data)} GPS rows at {sampled_interval:.3f}s interval from {os.path.basename(csv_path)}"
    )

    if len(data) >= 2:
        measured_interval = data[1][0] - data[0][0]
        check_status = "PASS" if abs(measured_interval - TARGET_GPS_INTERVAL) <= 1e-6 else "FAIL"
        level = "INFO" if check_status == "PASS" else "WARN"
        log_status(
            f"GPS interval check: expected {TARGET_GPS_INTERVAL:.3f}s, measured {measured_interval:.3f}s ({check_status})",
            level=level,
        )


def init_reaction_wheel():
    '''
    Initialize reaction wheel control if dependencies and hardware are available.
    Returns (pi, wheel) or (None, None) when wheel control is unavailable.
    '''
    log_status("Initializing reaction wheel interface")
    try:
        import importlib
        pigpio = importlib.import_module("pigpio")
        from HardwareInterface.reaction_wheels import motors as rw_motors
    except Exception as exc:
        log_status(f"Reaction wheel import unavailable: {exc}", level="WARN")
        return None, None

    pi = pigpio.pi()
    if not pi.connected:
        log_status("Reaction wheel disabled: pigpio daemon not running.", level="WARN")
        return None, None

    try:
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
        log_status("Reaction wheel connection established")
        return pi, wheel
    except Exception as exc:
        log_status(f"Reaction wheel initialization failed: {exc}", level="WARN")
        pi.stop()
        return None, None


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


def list_available_com_ports():
    '''
    Return detected serial COM ports, or an empty list if enumeration is unavailable.
    '''
    try:
        import importlib
        list_ports = importlib.import_module("serial.tools.list_ports")
    except Exception:
        return []

    return sorted(port.device for port in list_ports.comports())


def resolve_vn100_port(available_ports):
    '''
    Determine VN100 serial port by precedence:
      1) VN100_PORT environment variable
      2) Linux auto-detect (/dev/ttyUSB* or /dev/ttyACM*)
      3) platform default

    Returns:
        (port: str, source: str)
    '''
    env_port = os.getenv("VN100_PORT")
    if env_port:
        return env_port, "env"

    if sys.platform.startswith("linux"):
        for prefix in ("/dev/ttyUSB", "/dev/ttyACM"):
            for port in available_ports:
                if port.startswith(prefix):
                    return port, "linux-auto"
        return "/dev/ttyUSB0", "linux-default"

    if sys.platform.startswith("win"):
        return "COM5", "windows-default"

    return "COM5", "fallback-default"


def connect_vn100_with_timeout(port, timeout_seconds=5.0):
    '''
    Connect to VN100 with a timeout so missing hardware does not block indefinitely.

    Returns:
        (connected: bool, message: str)
    '''
    result = {"connected": False, "error": None}

    def _worker():
        try:
            result["connected"] = vn.connect(port) is True
        except Exception as exc:
            result["error"] = exc

    worker = threading.Thread(target=_worker, daemon=True)
    worker.start()
    worker.join(timeout_seconds)

    if worker.is_alive():
        return False, f"timeout after {timeout_seconds:.1f}s while opening {port}"

    if result["error"] is not None:
        return False, str(result["error"])

    if not result["connected"]:
        return False, "sensor did not verify connectivity"

    return True, "connected"


'''
read VN100 Software Documentation from documentation folder in SDK->python folder

Control center:
https://www.vectornav.com/downloader?file=https://www.vectornav.com/docs/default-source/software/controlcenter_setup_v3_4_0.exe&key=d2dfe074-c44c-4eb5-940e-c9e0356721c&id=16b36c6b-14d6-4bb8-962a-a081816b205e

'''

if __name__ == "__main__":
    log_status("Starting BERTHA hardware script")
    start = time.time()
    read_csv()

    pi = None
    wheel = None
    vn_connected = False
    try:
        # Connect to VN100 IMU.
        available_ports = list_available_com_ports()
        vn_port, vn_port_source = resolve_vn100_port(available_ports)
        log_status(f"Selected VN100 port {vn_port} ({vn_port_source})")
        log_status(f"Attempting VN100 connection on {vn_port}")

        if available_ports:
            log_status(f"Detected COM ports: {', '.join(available_ports)}")
            if vn_port not in available_ports:
                log_status(
                    f"VN100 connection failed: {vn_port} not found in detected COM ports",
                    level="ERROR",
                )
                raise RuntimeError("VN100 not connected")
        else:
            log_status(
                "Unable to enumerate COM ports; continuing with direct connect attempt",
                level="WARN",
            )

        connected, connect_message = connect_vn100_with_timeout(vn_port, timeout_seconds=5.0)
        if not connected:
            log_status(
                f"VN100 connection failed: {connect_message}",
                level="ERROR",
            )
            raise RuntimeError("VN100 not connected")
        vn_connected = True
        log_status(f"VN100 connection established on {vn_port}")

        # Optional wheel setup: script still runs if wheel stack is unavailable.
        pi, wheel = init_reaction_wheel()
        if wheel is None:
            log_status("Running without reaction wheel control", level="WARN")
        else:
            log_status("Reaction wheel control active")

        count = 50
        file_name = "bertha_imu_wheel_log.csv"
        file_path = os.path.join(project_root, "Main", "HardwareScripts", file_name)
        log_status(f"Writing telemetry log to {file_path}")

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
            log_status("CSV header written")

            i = 0
            previous_quat = None
            previous_wheel_cmd = None
            while i < count:
                quat = vn.read_quat()
                elapsed = time.time() - start

                # GPS is assumed readable for now; if unavailable, keep row shape stable.
                gps_data = get_gps_at_timestamp(elapsed)
                gps_row = gps_data if gps_data is not None else ["", "", "", "", "", ""]

                wheel_cmd = 0
                wheel_rpm = None
                if wheel is not None:
                    wheel_cmd = wheel_command_for_step(i, count)
                    wheel.set_speed(wheel_cmd)
                    wheel_rpm = float(wheel.rpm)

                quat_delta = 0.0
                if previous_quat is not None:
                    quat_delta = sum(abs(a - b) for a, b in zip(quat, previous_quat))

                if i % LOG_EVERY_STEPS == 0 or wheel_cmd != previous_wheel_cmd:
                    rpm_text = "n/a" if wheel_rpm is None else f"{wheel_rpm:.2f}"
                    gps_text = "available" if gps_data is not None else "missing"
                    log_status(
                        f"step {i + 1}/{count} elapsed={elapsed:.2f}s wheel_cmd={wheel_cmd} rpm={rpm_text} quat_delta={quat_delta:.6f} gps={gps_text}"
                    )

                visualize_data(i, quat)

                writer.writerow(
                    [
                        elapsed,
                        quat[0],
                        quat[1],
                        quat[2],
                        quat[3],
                        wheel_cmd,
                        "" if wheel_rpm is None else wheel_rpm,
                    ]
                    + gps_row
                )

                previous_quat = quat
                previous_wheel_cmd = wheel_cmd
                i += 1

        log_status(f"Run complete. Saved log to {file_path}")

    except Exception as exc:
        if not vn_connected:
            log_status("VN100 did not connect. Check sensor power, cable, and COM port.", level="ERROR")
        log_status(f"Runtime error: {exc}", level="ERROR")
        sys.exit(1)

    finally:
        log_status("Beginning shutdown")
        if wheel is not None:
            log_status("Stopping reaction wheel")
            wheel.kill()
        if pi is not None:
            log_status("Stopping pigpio handle")
            pi.stop()
        if vn_connected:
            try:
                vn.disconnect()
                log_status("VN100 disconnected")
            except Exception as exc:
                log_status(f"VN100 disconnect warning: {exc}", level="WARN")
        else:
            log_status("VN100 was not connected; skipping disconnect", level="WARN")
