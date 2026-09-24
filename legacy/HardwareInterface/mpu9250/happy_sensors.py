'''
happy_sensors.py
Authors: Claudia Kuczun, Andrew Gaylord, Juwan Jacobe

calibrates and interfaces with mpu9250, retrieving magnetometer and gyroscope data

'''


import time
from mpu9250_jmdev.registers import *
from mpu9250_jmdev.mpu_9250 import MPU9250
import math
import numpy as np

mpu = MPU9250(
        address_ak = AK8963_ADDRESS,
        address_mpu_master = MPU9050_ADDRESS_68,
        address_mpu_slave = None,
        bus = 1,
        gfs = GFS_1000,
        afs = AFS_8G,
        mfs = AK8963_BIT_16,
        mode = AK8963_MODE_C100HZ)

mpu.configure()

'''
calibrate resources: 
https://docs.nanoframework.net/devicesdetails/Mpu9250/README.html
https://docs.nanoframework.net/devicesdetails/Ak8963/README.html
https://github.com/nanoframework/nanoFramework.IoT.Device/blob/develop/devices/Mpu9250/samples/Program.cs
https://www.instructables.com/Quaternion-Compass/
'''
def calibrate():
    # Calibrate and configure the IMU
    try:
        mpu.configure()
        mpu.calibrate()
        mpu.configure()
        print("SENSOR: Calibrated & configured...\n\n\nBeginning data read...\n\n")
        return True
    
    # Something went wrong
    except:
        return False
    
def custom_calibrate(sensor, num, B_true):
    '''
    custom calibrate function to find and remove soft and hard iron distortions
    uses B_true to find sensor biases with num iterations
    returns magScale (offsets) and mbias (scaling)
    '''
    sensor.configure()

    B_true_x = B_true[0]
    B_true_y = B_true[1]
    B_true_z = B_true[2]

    avg = np.zeros(3)

    for i in range(num):
        data = np.array(get_imu_data())
        avg = avg + np.array(data[:3])
        time.sleep(1)

    avg = avg / num
        
    Bx_offset = float(avg[0] - B_true_x)
    By_offset = float(avg[1] - B_true_y)
    Bz_offset = float(avg[2] - B_true_z)

    print(Bx_offset)
    sensor.magScale = [1, 1, 1]
    sensor.mbias = [Bx_offset, By_offset, Bz_offset]
        
    return [Bx_offset, By_offset, Bz_offset]

def get_imu_data():
    data = [*mpu.readMagnetometerMaster(),
            *convertToRad(mpu.readGyroscopeMaster())]
    print("*IMU data: [{:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}]".format(data[0],data[1],data[2],data[3],data[4],data[5]))
    
    return data

def convertToRad(degrees):
    return [x * math.pi / 180 for x in degrees]


if __name__ == "__main__":
    #tryCalib = calibrate()

    #mpu.magScale = [0.05555, 0.8787878, 1.044]
    #mpu.magScale = [1, 1, 1]
    #mpu.mbias = [-8.9, 8.2, 24.1]
    #mpu.mbias = [-5.6, 28, 26.7]
    
    #mpu.calibrateAK8963()
    mpu.configure()

    #mpu.magScale = [0.05555, 0.8787878, 1.044]
    mpu.magScale = [1, 1, 1]
    #mpu.mbias = [-8.9, 8.2, 24.1]
    mpu.mbias = [30.335109508547003, 59.71757955586081, 38.51908195970696]

    magScale = mpu.magScale
    mbias = mpu.mbias

    print("magScale: ", magScale)
    print("mbais: ", mbias)

    for i in range(20):

        print(mpu.readMagnetometerMaster())
        time.sleep(1)

