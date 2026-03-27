'''
bertha.py

Combined script for interfacing with the BERTHA test bed.
Reads VN100 quaternion data, samples precomputed GPS rows, and optionally
commands the reaction wheel while logging telemetry.
'''

import csv
import errno
import importlib
import os
import re
import signal
import sys
import time

# Add repo root so project modules can be imported from this script location.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np

import HardwareInterface.vn100.vn100_interface as vn
import Simulator.visualizer as simulator
import Controllers.PID_controller as pid_controller
from Simulator.EOMs import *
from params import *
from ukf.UKF_algorithm import UKF

TARGET_GPS_INTERVAL = 1.0
DEFAULT_CSV_TIMESTEP = 0.1
LOG_EVERY_STEPS = 5
# How many iterations (54000 for entire GPS csv--basically infinite)
SAMPLE_COUNT = 54000


def log_status(message, level="INFO"):
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] [{level}] {message}")


def load_gps_data():
    csv_path = os.path.join(
        PROJECT_ROOT,
        "Simulator",
        "PySOL",
        "outputs",
        "1d_1orbit_tenth_s_gps.csv",
    )
    log_status(f"Loading GPS data from {csv_path}")

    with open(csv_path, "r", newline="") as f:
        rows = list(csv.reader(f))

    if len(rows) < 3:
        raise ValueError("GPS CSV is missing expected metadata/header rows.")

    metadata_line = ",".join(rows[0])
    timestep_match = re.search(r"timestep\s*=\s*([0-9]*\.?[0-9]+)", metadata_line)
    csv_timestep = float(timestep_match.group(1)) if timestep_match else DEFAULT_CSV_TIMESTEP
    if csv_timestep <= 0:
        raise ValueError(f"Invalid CSV timestep: {csv_timestep}")

    sample_stride = max(1, int(round(TARGET_GPS_INTERVAL / csv_timestep)))
    sampled_interval = sample_stride * csv_timestep
    sampled_rows = []

    data_rows = rows[2:]
    for row_index in range(0, len(data_rows), sample_stride):
        row = data_rows[row_index]
        timestamp = row_index * csv_timestep
        sampled_rows.append([timestamp] + [float(item) for item in row])

    log_status(
        f"Loaded {len(sampled_rows)} GPS rows at {sampled_interval:.3f}s interval from {os.path.basename(csv_path)}"
    )

    if len(sampled_rows) >= 2:
        measured_interval = sampled_rows[1][0] - sampled_rows[0][0]
        check_status = "PASS" if abs(measured_interval - TARGET_GPS_INTERVAL) <= 1e-6 else "FAIL"
        level = "INFO" if check_status == "PASS" else "WARN"
        log_status(
            f"GPS interval check: expected {TARGET_GPS_INTERVAL:.3f}s, measured {measured_interval:.3f}s ({check_status})",
            level=level,
        )

    return sampled_rows


def cleanup(wheel=None, wheel_callback=None, pi=None, vn_connected=False):
    """Gracefully shutdown hardware resources."""
    log_status("Beginning shutdown")
    if wheel is not None:
        try:
            log_status("Stopping reaction wheel")
            wheel.kill()
        except Exception as exc:
            log_status(f"Wheel shutdown warning: {exc}", level="WARN")

    if wheel_callback is not None:
        try:
            wheel_callback.cancel()
        except Exception as exc:
            log_status(f"Wheel callback shutdown warning: {exc}", level="WARN")

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


def signal_handler(sig, frame, wheel=None, wheel_callback=None, pi=None, vn_connected=False):
    """Handle Ctrl+C gracefully."""
    print("\nCtrl+C detected. Shutting down BERTHA...")
    cleanup(wheel, wheel_callback, pi, vn_connected)
    sys.exit(0)


def register_signal_handler(wheel, wheel_callback, pi, vn_connected):
    """Register the signal handler for graceful shutdown."""
    signal.signal(signal.SIGINT, lambda sig, frame: signal_handler(sig, frame, wheel, wheel_callback, pi, vn_connected))


def preflight_vn100_port_access(port):
    # Prevent native VN permission aborts by checking Linux serial access first.
    if not sys.platform.startswith("linux"):
        return True, "non-linux platform; skipping serial preflight"

    if not os.path.exists(port):
        return False, f"serial device {port} does not exist"

    if not os.access(port, os.R_OK | os.W_OK):
        return False, f"no read/write access to {port}; add your user to 'dialout' and re-login"

    flags = os.O_RDWR | getattr(os, "O_NONBLOCK", 0)
    flags |= getattr(os, "O_NOCTTY", 0)

    fd = None
    try:
        fd = os.open(port, flags)
    except OSError as exc:
        if exc.errno in (errno.EACCES, errno.EPERM):
            return False, f"permission denied opening {port}; check group permissions (dialout)"
        if exc.errno in (errno.EBUSY, errno.EAGAIN):
            return False, f"serial device {port} is busy (in use by another process)"
        return False, f"unable to open {port}: {exc}"
    finally:
        if fd is not None:
            os.close(fd)

    return True, "serial preflight passed"

def main():
    log_status("Starting BERTHA hardware script")
    start = time.time()
    gps_data = load_gps_data()

    pi = None
    wheel = None
    wheel_callback = None
    vn_connected = False

    try:
        # Detect serial ports when pyserial is available.
        try:
            list_ports = importlib.import_module("serial.tools.list_ports")
            available_ports = sorted(port.device for port in list_ports.comports())
        except Exception:
            available_ports = []

        # Resolve VN100 port (env override first, then platform defaults).
        env_port = os.getenv("VN100_PORT")
        if env_port:
            vn_port = env_port
            vn_port_source = "env"
        elif sys.platform.startswith("linux"):
            vn_port = "/dev/ttyUSB0"
            vn_port_source = "linux-default"
            for prefix in ("/dev/ttyUSB", "/dev/ttyACM"):
                match = next((p for p in available_ports if p.startswith(prefix)), None)
                if match:
                    vn_port = match
                    vn_port_source = "linux-auto"
                    break
        elif sys.platform.startswith("win"):
            vn_port = "COM5"
            vn_port_source = "windows-default"
        else:
            vn_port = "COM5"
            vn_port_source = "fallback-default"

        log_status(f"Selected VN100 port {vn_port} ({vn_port_source})")
        if available_ports:
            log_status(f"Detected COM ports: {', '.join(available_ports)}")
            if vn_port not in available_ports:
                raise RuntimeError(f"VN100 connection failed: {vn_port} not found in detected COM ports")
        else:
            log_status("Unable to enumerate COM ports; continuing with direct connect attempt", level="WARN")

        preflight_ok, preflight_message = preflight_vn100_port_access(vn_port)
        if not preflight_ok:
            raise RuntimeError(f"VN100 preflight failed: {preflight_message}")
        log_status(f"VN100 preflight passed: {preflight_message}")

        log_status(f"Attempting VN100 connection on {vn_port}")
        if vn.connect(vn_port) is not True:
            raise RuntimeError("VN100 sensor did not verify connectivity")
        vn_connected = True
        log_status(f"VN100 connection established on {vn_port}")

        # Optional reaction wheel setup; loop still runs if unavailable.
        try:
            import pigpio
            import serial
            from HardwareInterface.reaction_wheels import motors as rw_motors

            pi = pigpio.pi()
            if not pi.connected:
                log_status("Reaction wheel disabled: pigpio daemon not running.", level="WARN")
                pi.stop()
                pi = None
            else:
                wheel = rw_motors.ReactionWheel(
                    pi,
                    rw_motors.DAA,
                    rw_motors.COMU,
                    rw_motors.FREQ,
                    rw_motors.PWM,
                    rw_motors.BR,
                    rw_motors.DIRE,
                )
                wheel_callback = pi.callback(rw_motors.FREQ, pigpio.RISING_EDGE, wheel.get_rpm_callback)
                log_status("Reaction wheel control active")
        except Exception as exc:
            log_status(f"Running without reaction wheel control: {exc}", level="WARN")
            return

        # Register signal handler for graceful Ctrl+C shutdown
        register_signal_handler(wheel, wheel_callback, pi, vn_connected)

        file_path = os.path.join(PROJECT_ROOT, "Main", "HardwareScripts", "bertha_imu_wheel_log.csv")
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
                    "mag_x",
                    "mag_y",
                    "mag_z",
                    "gyro_x",
                    "gyro_y",
                    "gyro_z",
                    "earth_mag_x",
                    "earth_mag_y",
                    "earth_mag_z",
                    "gps_x",
                    "gps_y",
                    "gps_z",
                ]
            )

            previous_quat = None
            previous_wheel_cmd = None

            first_quat = vn.read_quat()
            our_target = quaternion_multiply(first_quat, TARGET) # 90 degree rotation around x-axis
            state = np.array([1,0,0,0,0,0,0]) #[q0, q1, q2, q3, omega_x, omega_y, omega_z]
            cov = np.identity(7) * 5e-10

            noise_mag = 5
            noise_gyro = 0.1

            # r = np.diag([noise_mag] * dim_mes)
            r = np.array([[noise_mag, 0, 0, 0, 0, 0],
                        [0, noise_mag, 0, 0, 0, 0],
                        [0, 0, noise_mag, 0, 0, 0],
                        [0, 0, 0, noise_gyro, 0, 0],
                        [0, 0, 0, 0, noise_gyro, 0],
                        [0, 0, 0, 0, 0, noise_gyro]])

            # q: process noise (n x n)
            # Should depend on dt
            # try negative noises?
            noise_mag = .05
            # q = np.diag([noise_mag] * dim)
            q = np.array([[dt, 3*dt/4, dt/2, dt/4, 0, 0, 0],
                        [3*dt/4, dt, 3*dt/4, dt/2, 0, 0, 0],
                        [dt/2, 3*dt/4, dt, 3*dt/4, 0, 0, 0],
                        [dt/4, dt/2, 3*dt/4, dt, 0, 0, 0],
                        [0, 0, 0, 0, dt, 2*dt/3, dt/3],
                        [0, 0, 0, 0, 2*dt/3, dt, 2*dt/3],
                        [0, 0, 0, 0, dt/3, 2*dt/3, dt]
            ])
            q = q * noise_mag

            reaction_speeds = [0, 0, 0]
            wheel_rpm = 0

            for i in range(SAMPLE_COUNT):
                quat = np.array(vn.read_quat())
                # Rad/s (seemingly)
                angular_velocity = np.array(vn.read_gyro())
                # Magnetic field in body frame
                b_body = np.array(vn.read_mag())
                elapsed = time.time() - start

                sample_idx = int(elapsed // TARGET_GPS_INTERVAL)
                sample_idx = max(0, min(sample_idx, len(gps_data) - 1))
                gps_row = gps_data[sample_idx][1:] if gps_data else ["", "", "", "", "", ""]

                wheel_cmd = 0
                wheel_rpm = None
                if wheel is not None:
                    command_law = "point" # "point", "spin", "constant", "ukf-point"
                    if command_law == "constant":
                        wheel_cmd = 40 # hardcode for now

                    elif command_law == "spin":
                        kp = 2e-1
                        kd = 0.5e-1
                        print("Speed: ", math.degrees(angular_velocity[2]))
                        pid = pid_controller.PIDController(kp, 0, kd, 0)
                        target_speed = np.array([0.0, 0.0, -math.radians(10.0)]) # degrees/s
                        wheel_cmd = pid.pd_velocity_controller(target_speed=target_speed, current_speed=angular_velocity, kp=kp, kd=kd)
                        wheel_cmd = wheel_cmd[2]

                    elif command_law == "point":
                        dt = 0.05
                        kp = 3e-2
                        kd = 5e-3
                        ki = 1e-4
                        controller = pid_controller.PIDController(kp, ki, kd, dt)
                        wheel_cmd = controller.pid_controller(quat, normalize(TARGET), angular_velocity, [])
                        # wheel_cmd = controller.pid_controller(quat, normalize(updated_target), angular_velocity, [])

                        wheel_cmd = wheel_cmd[2]

                    elif command_law == "ukf-point":
                        dt = 0.05
                        kp = 3e-2
                        kd = 5e-3
                        ki = 1e-4
                        controller = pid_controller.PIDController(kp, ki, kd, dt)
                        old_reaction_speeds = reaction_speeds
                        reaction_speeds = [wheel_rpm, 0 ,0]
                        data = [
                            *b_body.tolist(),
                            *angular_velocity.tolist(),
                        ]
                        state, cov = UKF(state, cov, q, r, gps_row[:3], reaction_speeds, old_reaction_speeds, data)
                        wheel_cmd = controller.pid_controller(np.array(state[:4]), normalize(our_target), np.array(state[4:7]), [])
                        wheel_cmd = wheel_cmd[2]
                    else:
                        wheel_cmd = 0

                    print("PWM command: ", wheel_cmd)
                    wheel.set_speed(wheel_cmd)
                    wheel_rpm = float(wheel.rpm)

                quat_delta = 0.0
                if previous_quat is not None:
                    quat_delta = sum(abs(a - b) for a, b in zip(quat, previous_quat))

                if i % LOG_EVERY_STEPS == 0 or wheel_cmd != previous_wheel_cmd:
                    rpm_text = "n/a" if wheel_rpm is None else f"{wheel_rpm:.2f}"
                    log_status(
                        f"step {i + 1}/{SAMPLE_COUNT} elapsed={elapsed:.2f}s wheel_cmd={wheel_cmd} rpm={rpm_text} quat_delta={quat_delta:.6f}"
                    )

                simulator.game_visualize(np.array([quat]), i)
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
                    + b_body.tolist()
                    + angular_velocity.tolist()
                    + gps_row
                )

                previous_quat = quat
                previous_wheel_cmd = wheel_cmd
                time.sleep(.05)

        log_status(f"Run complete. Saved log to {file_path}")

    except Exception as exc:
        if not vn_connected:
            log_status("VN100 did not connect. Check sensor power, cable, and COM port.", level="ERROR")
        log_status(f"Runtime error: {exc}", level="ERROR")
        sys.exit(1)

    finally:
        cleanup(wheel, wheel_callback, pi, vn_connected)


if __name__ == "__main__":
    main()
