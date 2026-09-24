"""VectorNav VN-100 IMU driver (testbed IMU).

Purpose:  Implements gyro, magnetometer, and accelerometer interfaces for the
          VN-100 used on the BERTHA testbed.
Inputs:   ``Vn100Params`` (port, baud, sample rate).
Outputs:  ``Vn100Driver`` (``GyroInterface``, ``MagInterface``, ``AccelInterface``).
Status:   [MODIFIED]
Port from: legacy/HardwareInterface/vn100/vn100_interface.py
           (vendored library: ``vn100_library/``, install separately)

Changes from legacy:
    - Class instance instead of a module-level singleton ``VnSensor``.
    - Return ``Sample`` in SI: gyro deg/s -> rad/s, mag µT -> T
      (use ``geometry/units.py``).
    - No ``print``; no file logging (``print_data_to_file`` belongs in tools/).
    - Raw values only; mag calibration moves to ``estimation/mag_conditioning.py``.

Not the flight IMU; flight IMU driver added once the part is chosen (HW).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from cloversat.datatypes.sample_types import Sample, SelfTestResult
from cloversat.datatypes.vector_types import Vec3
from cloversat.devices.interfaces.accel_interface import AccelInterface
from cloversat.devices.interfaces.gyro_interface import GyroInterface
from cloversat.devices.interfaces.mag_interface import MagInterface

if TYPE_CHECKING:
    from cloversat.config.device_params import Vn100Params


class Vn100Driver(GyroInterface, MagInterface, AccelInterface):
    """VN-100 over serial."""

    def __init__(self, params: Vn100Params) -> None:
        raise NotImplementedError

    def init(self) -> None:
        """Connect and verify sensor connectivity."""
        raise NotImplementedError

    def self_test(self) -> SelfTestResult:
        """Verify connectivity and plausible readings."""
        raise NotImplementedError

    def read_gyro(self) -> Sample[Vec3]:
        """Return ``omega_body`` [rad/s]."""
        raise NotImplementedError

    def read_mag(self) -> Sample[Vec3]:
        """Return ``B_body`` [T]."""
        raise NotImplementedError

    def read_accel(self) -> Sample[Vec3]:
        """Return ``a_body`` [m/s^2]."""
        raise NotImplementedError

    def close(self) -> None:
        """Disconnect from the sensor."""
        raise NotImplementedError


# ---- LEGACY START: legacy/HardwareInterface/vn100/vn100_interface.py (reference only, not wired up) ----
# TODO(bug): legacy returns mag in µT and gyro in deg/s; port must return T and rad/s.
'''
vn100_interface.py
Authors: Andrew Gaylord

test script for connecting to and reading data from vn100 imu

Script for changing connectivity to VN100 IMU sensor, and reading and/or getting data from the sensor.
Data such as quaternion, magnetic, angle, acceleration, and temperature.


'''
# [scaffold: disabled] from vnpy import VnSensor
# [scaffold: disabled] import vnpy as Registers  # Create namespace alias for backwards compatibility
# [scaffold: disabled] s = VnSensor()

'''
Alternate connection route:

ez = EzAsyncData.connect('COM5', 115200)

# call the sensor object through the object and read the relevant data
ez.sensor.read_imu_measurements()

# this returns a compositeData object
# lots of helper functions exist to extract the data from this object
data = ez.current_data

print("angular: ", data.angular_rate)

print("quat: ", data.any_attitude.quat)

'''


def get_instance(): 
	'''
	@returns 
		The singleton instance of the VnSensor 
	'''
	return s
def connect(port = 'COM5'):
	'''
	Connects to the VnSensor version using specified COM port

	Prints "CONNECTED" upon succesful connection verification
	'''
	s.connect(port, 115200)  # Connect using vnpy API
	if s.verify_sensor_connectivity():
		print("CONNECTED")
		return True

def disconnect():
	'''
	Disconnects from the VnSensor version using specified COM port

	Prints "DICONNECTED" upon end
	'''
	s.disconnect()
	print("DISCONNECTED")

def read_quat():
	'''
	readquat(): reads current orientation from sensor in quaternion form

	@return 
		A 4 element list containing the quaternion values (w, x, y, z)
		Note: Quaternions do not have units
	'''
	quat = s.read_attitude_quaternion()
	# vnpy returns as Quaternion object, extract w, x, y, z
	return [quat.w, quat.x, quat.y, quat.z]

def read_mag():
	'''
	readmag(): Reads the current magnetic fields from the sensor as three values corresponding to x,y,z dimensions 

	@return 
		A 3 element list containing 3 float values in microtesla (x,y,z)
	'''
	mag = s.read_magnetic_measurements()
	return [mag.x, mag.y, mag.z]


def read_gyro():
	'''
	read_gyro(): reads angular velocity values from sensor

	@return 
		A returns 3 element list containing 3 float values in °/s (x,y,z)
	'''
	gyro = s.read_angular_rate_measurements()
	return [gyro.x, gyro.y, gyro.z]
	

def read_accel():
	'''
	read_accel(): reads acceleration value from sensor

	@return 
		A returns 3 element list containing 3 float values in m/s^2 (x,y,z)
	'''
	accel = s.read_acceleration_measurements()
	return [accel.x, accel.y, accel.z]

def read_all():
	'''
	read_all(): Gets magnetic field and angular velocity
	@return
		Returns a 2 element list containing all data from magnetic field and angular velocity
	'''
	mag = s.read_magnetic_measurements()
	gyro = s.read_angular_rate_measurements()
	return [[mag.x, mag.y, mag.z], [gyro.x, gyro.y, gyro.z]]

def get_mag_gyro_quat():
	'''
	get_mag_gyro_quat: Wraps mag gyro and quaternion data in a String

	@return
		A string comprised of three lists joined and seperated by commas (x,y,z) x (mag, gyro, quat)
	'''

	mag = [ str(a) for a in read_mag()]
	gyro = [ str(b) for b in read_gyro()]
	quat = [ str(c) for c in read_quat()]
	all_three = mag + gyro + quat # create list with all data
	all_three_string = ", ".join(all_three)
	return all_three_string

def print_data_to_file(count, file_name):
	'''
	print_data_to_file(): Updates a text file with data from Magnetometer, Gyroscope, and Accelerometer, seperating each data set on a new line
	
	@params
		count: Number of desried data sets
		file_name: Name of the file
	
	'''
	i = 0 # keep track of our iteration count
	f = open(file_name, "a+")
	while i < count:

		i += 1
		# save to text file in form of magnetometer (magnetic field), angular velocity (gyroscope), and acceleration (accelerometer)
		f.write(get_mag_gyro_quat()) # put mag, gyro, quat data into text file
		if (i < count): f.write("\n") # add newline to separate data sets

	f.close()
	#source = f'./{file_name}'
	#destination = './new_sensor_tests'
	#os.rename(source, destination)
	return

def print_mag_calibration():
	#TODO: Finish implementing this method
	'''
	Note: Unfinished
	
	print_mag_calibration
		A method that prints out the calculated calibration for both B and C
	'''
	mag_cal = s.read_calculated_magnetometer_calibration()
	print("b: ", mag_cal.b) # b and c are objects
	print("c: ", mag_cal.c)

#Returns temp from the sensor as 
def read_temp():
	'''
	read_temp(): reads temperature value from sensor
	@return 
		Returns a float with the temperature in °C
	'''
	return s.readRegister(Registers.IMU.ImuMeas).temperature

#Returns the pressure reading of the sensor as a float 
def read_pressure():
	'''
	read_pressure(): reads pressure value from sensor
	@return
		Returns a float with the pressure in Hectopascals (hPO)
	'''
	return s.readRegister(Registers.IMU.ImuMeas).pressure
# ---- LEGACY END ----
