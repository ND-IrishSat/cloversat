'''
bertha.py

Combined script for interfacing with the BERTHA test bed.
Reads VN100 quaternion data, samples precomputed GPS rows, and optionally
commands the reaction wheel while logging telemetry.
'''

import csv
from datetime import datetime
import errno
import importlib
import os
import re
import select
import signal
import sys
import time
import argparse

# Add repo root so project modules can be imported from this script location.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
from ukf.low_pass_filter.lowpassfilter import LowPassFilter


import HardwareInterface.vn100.vn100_interface as vn
import Simulator.visualizer as simulator
import Controllers.PID_controller as pid_controller
from Simulator.EOMs import *
from params import *
# from ukf.UKF_algorithm import UKF

TARGET_GPS_INTERVAL = 1.0
DEFAULT_CSV_TIMESTEP = 0.1
LOG_EVERY_STEPS = 5
# How many iterations (54000 for entire GPS csv--basically infinite)
SAMPLE_COUNT = 54000

ABSOLUTE_KEY_ANGLES = {
    "w": 0.0,      # Return to first_quat
    "d": -90.0,    # 90° clockwise from first_quat
    "a": 90.0,     # 90° CCW from first_quat
    "s": 180.0,    # 180° from first_quat
}

RELATIVE_ROTATION_KEYS = {
    "q": 90.0,     # Rotate 90° CCW from current
    "e": -90.0,    # Rotate 90° CW from current
}


def euler_deg_to_quaternion(roll_deg, pitch_deg, yaw_deg):
    """Convert Euler angles (deg) to quaternion (w, x, y, z)."""
    roll = np.deg2rad(roll_deg)
    pitch = np.deg2rad(pitch_deg)
    yaw = np.deg2rad(yaw_deg)

    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return np.array([w, x, y, z])


def quaternion_to_euler_deg(q):
    """Convert quaternion (w, x, y, z) to Euler angles in degrees (roll, pitch, yaw)."""
    w, x, y, z = q

    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = np.copysign(np.pi / 2, sinp)
    else:
        pitch = np.arcsin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.rad2deg(roll), np.rad2deg(pitch), np.rad2deg(yaw)


def print_rotation_update(label, quat, prev_euler=None):
    """Print rotation in Euler degrees and show which axis changed if prev_euler is provided."""
    roll, pitch, yaw = quaternion_to_euler_deg(quat)

    if prev_euler is not None:
        prev_roll, prev_pitch, prev_yaw = prev_euler
        roll_delta = abs(roll - prev_roll)
        pitch_delta = abs(pitch - prev_pitch)
        yaw_delta = abs(yaw - prev_yaw)

        # Determine which axis changed the most
        max_delta = max(roll_delta, pitch_delta, yaw_delta)
        if abs(max_delta - roll_delta) < 1e-6:
            axis = "ROLL"
        elif abs(max_delta - pitch_delta) < 1e-6:
            axis = "PITCH"
        else:
            axis = "YAW"

        log_status(f"{label}: roll={roll:.1f}deg, pitch={pitch:.1f}deg, yaw={yaw:.1f}deg [{axis}]")
    else:
        log_status(f"{label}: roll={roll:.1f}deg, pitch={pitch:.1f}deg, yaw={yaw:.1f}deg")


def setup_nonblocking_keyboard():
    """Configure stdin for non-blocking single-key reads (POSIX terminals)."""
    if not os.name == "posix" or not sys.stdin.isatty():
        return None

    try:
        import termios
        import tty

        fd = sys.stdin.fileno()
        original_state = termios.tcgetattr(fd)
        tty.setcbreak(fd)
        return (fd, original_state)
    except Exception:
        return None


def restore_keyboard(keyboard_state):
    """Restore terminal state after setup_nonblocking_keyboard."""
    if keyboard_state is None:
        return

    try:
        import termios

        fd, original_state = keyboard_state
        termios.tcsetattr(fd, termios.TCSADRAIN, original_state)
    except Exception:
        pass


def read_keypress():
    """Return (key_type, value, key_name) if a mapped key was pressed, else (None, None, None).
    key_type: 'absolute' (WASD), 'relative' (QE), or None.
    value: angle in degrees or None.
    key_name: readable key name or None.
    """
    if not sys.stdin.isatty():
        return None, None, None

    ready, _, _ = select.select([sys.stdin], [], [], 0)
    if not ready:
        return None, None, None

    key = sys.stdin.read(1)
    if not key:
        return None, None, None

    lowered = key.lower()
    if lowered in ABSOLUTE_KEY_ANGLES:
        return "absolute", ABSOLUTE_KEY_ANGLES[lowered], lowered.upper()

    if lowered in RELATIVE_ROTATION_KEYS:
        return "relative", RELATIVE_ROTATION_KEYS[lowered], lowered.upper()

    return None, None, None


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


def argparse_setup():
    parser =  argparse.ArgumentParser()
    parser.add_argument(
        "--control-law",
        choices = ["point", "constant_speed", "constant_pwm", "none"],
        default = "none",
    )

    parser.add_argument(
        "--visualize",
        choices=["on", "off"],
        default="off",
    )
    parser.add_argument(
        "--filter",
        choices = ["LPF", "Kalman", "None"],
        default = "None",
    )
    parser.add_argument(
        "--output",
        default=f"bertha_results_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv",
        help="Filename for telemetry CSV log (default: bertha_results_{timestamp}.csv)",
    )

    parser.add_argument(
        "--keytest",
        action="store_true",
        help="Enable keyboard control of target quaternion setpoint (WASD + arrow keys).",
    )
    return parser.parse_args()

HARDCODE_CONSTANT_PWM = 40

def main():
    argument = argparse_setup()
    log_status("Starting BERTHA hardware script")
    start = time.time()
    gps_data = load_gps_data()

    pi = None
    wheel = None
    wheel_callback = None
    vn_connected = False
    dt = 0.05
    keyboard_state = None

    try:
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

        # Register signal handler for graceful Ctrl+C shutdown
        register_signal_handler(wheel, wheel_callback, pi, vn_connected)

        if argument.keytest:
            keyboard_state = setup_nonblocking_keyboard()
            if keyboard_state is None:
                log_status("Keyboard setpoint mode unavailable (stdin is not a terminal)", level="WARN")
            else:
                log_status(
                    "Keyboard setpoint mode ON and open to receiving key input:"
                )
                log_status("  ABSOLUTE setpoints (relative to first_quat): W=first_quat, D=90°CW, A=90°CCW, S=180°")
                log_status("  RELATIVE rotations (from current): Q=rotate 90°CCW, E=rotate 90°CW")

        if argument.filter == "Kalman":
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

        if argument.filter == "LPF":
            tau = 0.5
            mag_filter = LowPassFilter(dt, tau)
            gyro_filter = LowPassFilter(dt, tau)

        file_path = os.path.join(PROJECT_ROOT, "Main", "HardwareScripts", "results", argument.output)
        log_status(f"Writing telemetry log to {file_path}")

        with open(file_path, "w", newline="") as log_file:
            writer = csv.writer(log_file)
            csv_headers = [
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
            csv_raw_headers = [
                "mag_x_raw",
                "mag_y_raw",
                "mag_z_raw",
                "gyro_x_raw",
                "gyro_y_raw",
                "gyro_z_raw",
            ]
            if argument.filter == "LPF":
                csv_headers.append(csv_raw_headers)

            writer.writerow(csv_headers)

            previous_quat = None
            previous_wheel_cmd = None

            first_quat = vn.read_quat()
            target_adjusted = quaternion_multiply(first_quat, TARGET)

            print_rotation_update("Initial orientation", first_quat)
            # Extract roll and pitch to keep constant; only yaw will change
            first_roll, first_pitch, first_yaw = quaternion_to_euler_deg(first_quat)
            current_yaw = first_yaw
            current_euler = quaternion_to_euler_deg(target_adjusted)

            reaction_speeds = [0, 0, 0]
            wheel_rpm = 0

            for i in range(SAMPLE_COUNT):
                quat = np.array(vn.read_quat())

                # CHeck for user input to adjust target quaternion
                if argument.keytest:
                    key_type, angle_deg, key_name = read_keypress()
                    if key_type == "absolute":
                        # Absolute setpoint: offset yaw from first_yaw, keep roll and pitch from first_quat
                        current_yaw = first_yaw + angle_deg
                        target_adjusted = euler_deg_to_quaternion(first_roll, first_pitch, current_yaw)
                        print_rotation_update(f"Setpoint updated via {key_name} (absolute yaw offset={angle_deg:.1f}deg)", target_adjusted, current_euler)
                        current_euler = quaternion_to_euler_deg(target_adjusted)
                    elif key_type == "relative":
                        # Relative rotation: rotate current yaw by angle_deg
                        current_yaw += angle_deg
                        target_adjusted = euler_deg_to_quaternion(first_roll, first_pitch, current_yaw)
                        print_rotation_update(f"Setpoint updated via {key_name} (relative yaw rotation={angle_deg:.1f}deg)", target_adjusted, current_euler)
                        current_euler = quaternion_to_euler_deg(target_adjusted)

                # Rad/s (seemingly)
                angular_velocity = np.array(vn.read_gyro())
                # Magnetic field in body frame
                b_body = np.array(vn.read_mag())
                elapsed = time.time() - start

                sample_idx = int(elapsed // TARGET_GPS_INTERVAL)
                sample_idx = max(0, min(sample_idx, len(gps_data) - 1))
                gps_row = gps_data[sample_idx][1:] if gps_data else ["", "", "", "", "", ""]

                angular_velocity_raw = angular_velocity.copy()
                b_body_raw = b_body.copy()

                #applying the filter to the magnetometer data
                if argument.filter == "LPF":
                    b_body = mag_filter.apply(b_body)
                    angular_velocity = gyro_filter.apply(angular_velocity)

                wheel_cmd = 0
                wheel_rpm = None
                if wheel is not None:
                    command_law = argument.control_law # "point", "constant_speed", "constant_pwm"
                    if command_law == "constant_pwm":
                        wheel_cmd = HARDCODE_CONSTANT_PWM # hardcode pwm for now

                    elif command_law == "constant_speed":
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
                        # Because its 1D, only take z axis
                        wheel_cmd = controller.pid_controller(quat, normalize(target_adjusted), angular_velocity, [])[2]

                    # elif command_law == "ukf-point":
                    #     # TODO: add an option for filtering data (none, lowpass, or kalman) no matter which control law we use
                    #     dt = 0.05
                    #     kp = 3e-2
                    #     kd = 5e-3
                    #     ki = 1e-4
                    #     controller = pid_controller.PIDController(kp, ki, kd, dt)
                    #     old_reaction_speeds = reaction_speeds
                    #     reaction_speeds = [wheel_rpm, 0 ,0]
                    #     data = [
                    #         *b_body.tolist(),
                    #         *angular_velocity.tolist(),
                    #     ]
                    #     state, cov = UKF(state, cov, q, r, gps_row[:3], reaction_speeds, old_reaction_speeds, data)
                    #     wheel_cmd = controller.pid_controller(np.array(state[:4]), normalize(target_adjusted), np.array(state[4:7]), [])
                    #     wheel_cmd = wheel_cmd[2]

                    else:
                        wheel_cmd = 0

                    print("PWM command: ", wheel_cmd)
                    wheel.set_speed(wheel_cmd)
                    wheel_rpm = float(wheel.rpm)

                # quat_delta = 0.0
                # if previous_quat is not None:
                    # quat_delta = sum(abs(a - b) for a, b in zip(quat, previous_quat))

                # if i % LOG_EVERY_STEPS == 0 or wheel_cmd != previous_wheel_cmd:
                #     rpm_text = "n/a" if wheel_rpm is None else f"{wheel_rpm:.2f}"
                #     log_status(
                #         f"step {i + 1}/{SAMPLE_COUNT} elapsed={elapsed:.2f}s wheel_cmd={wheel_cmd} rpm={rpm_text} quat_delta={quat_delta:.6f}"
                #     )

                if argument.visualize == "on":
                    simulator.game_visualize(np.array([quat]), i)

                row = [
                        elapsed,
                        quat[0],
                        quat[1],
                        quat[2],
                        quat[3],
                        wheel_cmd,
                        "" if wheel_rpm is None else wheel_rpm,
                    ] + b_body.tolist() + angular_velocity.tolist() + gps_row

                if argument.filter == "LPF":
                    row += b_body_raw.tolist() + angular_velocity_raw.tolist()

                writer.writerow(row)

                previous_quat = quat
                previous_wheel_cmd = wheel_cmd
                time.sleep(dt)

        log_status(f"Run complete. Saved log to {file_path}")

    except Exception as exc:
        if not vn_connected:
            log_status("VN100 did not connect. Check sensor power, cable, and COM port.", level="ERROR")
        log_status(f"Runtime error: {exc}", level="ERROR")
        sys.exit(1)

    finally:
        restore_keyboard(keyboard_state)
        cleanup(wheel, wheel_callback, pi, vn_connected)


if __name__ == "__main__":
    main()
