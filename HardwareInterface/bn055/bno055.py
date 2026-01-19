# SPDX-FileCopyrightText: 2021 ladyada for Adafruit Industries
# SPDX-License-Identifier: MIT

import time
import board
#import busio
import adafruit_bno055
import serial
from Adafruit_BNO055 import BNO055

_TRIGGER_REGISTER = 0x3F
_PAGE_REGISTER = 0x07
CONFIG_MODE = 0x00

def calibrate():
    # code to get data from imu :)
    time.sleep(3)
    i2c = board.I2C()
    sensor = adafruit_bno055.BNO055_I2C(i2c)
    time.sleep(1)

    # set using external crystal on
    last_mode = sensor.mode
    sensor.mode = CONFIG_MODE
    sensor._write_register(_PAGE_REGISTER, 0X00)
    sensor._write_register(_TRIGGER_REGISTER,0x80)
    sensor.mode = last_mode
    time.sleep(.1)

    print(sensor.external_crystal)
    return 

    if sensor.calibrated: print("CALIBRATED!")
    print("calibration status:", sensor.calibration_status)

    while True:
        print(sensor.calibration_status)
        #if sensor.calibration_status[1] < 3:
            #print("need to calibrate gyro! hold the sensor still")
            #time.sleep(.1)
        #elif sensor.calibration_status[3] < 3:
            #print("need to calibrate magnetometer! move the sensor in figure 8 config")
            #time.sleep(.1)
        time.sleep(1)
        if sensor.calibration_status[1] == 3 and sensor.calibration_status[3] ==3:
            break
        #cnt = 0
        #for i in sensor.calibration_status:
            #if i == 3:
                #cnt+=1
        #if cnt == 2: break

    time.sleep(1)
    print(sensor.calibration_status, "\n")
    data = get_calibration()
    os.environ['CALIBRATION'] = str(data)

    # next time, do just the following
    #set_calibration(os.environ['CALIBRATION'])


def get_data():
    i2c = board.I2C()
    sensor = adafruit_bno055.BNO055_I2C(i2c)
    print(sensor.acceleration)
    res = sensor.magnetic + sensor.gyro
    return res



calibrate()
print('calibrated!')
while True:
    output = get_data()
    print(output)

