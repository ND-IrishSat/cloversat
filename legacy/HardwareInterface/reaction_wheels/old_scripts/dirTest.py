'''
dirTest.py
Author: Tim Roberts

fundamental motors direction test that does not use structure of motors.py

'''

from board import SCL, SDA
import busio
from adafruit_pca9685 import PCA9685
import RPi.GPIO as GPIO
import time
import numpy as np
from sklearn.linear_model import LinearRegression
from hall import checkHall
import random

# Constants
cons = 9100
# max duty cycles
k = 65535
default = 0
enable = 11
pinX = 10
dirX = 0
hallX = [13,19,26]

# Initialize i2c
i2c_bus = busio.I2C(SCL, SDA)
pca = PCA9685(i2c_bus)
pca.frequency = 1500

# GPIO setup
GPIO.setmode(GPIO.BCM)
GPIO.setup(enable,GPIO.OUT)
GPIO.output(enable,True)

GPIO.setup(pinX,GPIO.OUT)
GPIO.setup(dirX,GPIO.OUT)
GPIO.output(dirX, True)

GPIO.setup(hallX[0],GPIO.IN,pull_up_down=GPIO.PUD_DOWN)
GPIO.add_event_detect(hallX[0], GPIO.FALLING)

check = input("START?")

if check.lower() == "y":
	speedPos = int(input("ENTER POS SPEED: "))
	speedNeg = int(input("ENTER NEG SPEED: "))
	
	pca.channels[pinX].duty_cycle = abs(speedPos)
	time.sleep(5)
	pca.channels[pinX].duty_cycle = 0
	time.sleep(3)
	GPIO.output(dirX, False)
	pca.channels[pinX].duty_cycle = abs(speedNeg)
	time.sleep(5)
	pca.channels[pinX].duty_cycle = 0
else:
	pca.channels[pinX].duty_cycle = 0
