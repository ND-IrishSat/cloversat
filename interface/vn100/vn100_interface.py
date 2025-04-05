'''
vn100_interface.py
Authors: Andrew Gaylord

test script for connecting to and reading data from vn100 imu

Script for changing connectivity to VN100 IMU sensor, and reading and/or getting data from the sensor.
Data such as quaternion, magnetic, angle, acceleration, and temperature.


'''
from vnpy import *

s = VnSensor()

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

def connect():
	'''
	Connects to the VnSensor version using specified COM port

	Prints "CONNECTED" upon succesful connection verification
	'''
	s.connect('COM5', 115200)
	if(s.verify_sensor_connectivity()):
		print("CONNECTED")

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
	quat = [quat.w, quat.x, quat.y, quat.z]
	return quat

def read_mag():
	'''
	readmag(): Reads the current magnetic fields from the sensor as three values corresponding to x,y,z dimensions 

	@return 
		A 3 element list containing 3 float values in microtesla (x,y,z)
	'''
	mag_reading = s.read_magnetic_measurements()
	return [mag_reading.x, mag_reading.y, mag_reading.z]

def read_gyro():
	'''
	read_gyro(): reads angular velocity values from sensor

	@return 
		A returns 3 element list containing 3 float values in °/s (x,y,z)
	'''
	gyro_reading = s.read_angular_rate_measurements()
	return [gyro_reading.x, gyro_reading.y, gyro_reading.z]
	

def read_accel():
	'''
	read_accel(): reads angular acceleration value from sensor

	@return 
		A returns 3 element list containing 3 float values in °/s^2 (x,y,z)
	'''
	accel_reading = s.read_acceleration_measurements()
	return [accel_reading.x, accel_reading.y, accel_reading.z]

def read_all():
	'''
	read_all(): Gets magnetic acceleration and angular velocity
	@return
		Returns a 2 element list containing all data from magnetic field and angular velocity
	'''
	allData = s.read_magnetic_acceleration_and_angular_rates()
	mag = allData.mag
	gyro = allData.gyro
	return [mag, gyro]

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

	#TODO: Please verify this fuction put the file is in correct folder
	folder_path = '../new_sensor_tests/'
	full_path = folder_path + file_name

	i = 0 # keep track of our iteration count
	f = open(full_path, "a+")
	while i < count:
		i += 1
		# save to text file in form of magnetometer (magnetic field), angular velocity (gyroscope), and acceleration (accelerometer)
		f.write(print_mag_gyro_quat()) # put mag, gyro, quat data into text file
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
	return s.read_imu_measurements().temp

#Returns the pressure reading of the sensor as a float 
def read_pressure():
	'''
	read_pressure(): reads pressure value from sensor
	@return
		Returns a float with the pressure in Hectopascals (hPO)
	'''
	return s.read_imu_measurements().pressure