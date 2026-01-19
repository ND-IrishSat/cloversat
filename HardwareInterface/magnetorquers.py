'''
magnetorquers.py
Authors: Tim Roberts & Sarah Kopfer

mag interface class
checks hall sensors readings and sets reaction wheel speeds??
initialization must be done through init.py??

MUST GET PINS BEFORE USING

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


# Define actual GPIO pin numbers
PWM4=4
PWM5=5
PWM7=7
PWM6=6
PWM0=0
PWM1=1
PWM3=3
PWM2=2
MAX_DUTY=65535


class Mag():
    def __init__(self, pins):
        self.pins = pins  # List of PWM channel numbers for AIN1, AIN2, BIN1, BIN2

    def magOn(self, duty_cycle, pindex, pca):
        print("Magnetorquer On")
        if duty_cycle > 0:
            pca.channels[self.pins[pindex][0]].duty_cycle = min(abs(duty_cycle), MAX_DUTY)
            pca.channels[self.pins[pindex][1]].duty_cycle = 0
        elif duty_cycle < 0:
            pca.channels[self.pins[pindex][0]].duty_cycle = 0
            pca.channels[self.pins[pindex][1]].duty_cycle = min(abs(duty_cycle), MAX_DUTY)
        else:
            for pin in self.pins[pindex]:
                pca.channels[pin].duty_cycle = MAX_DUTY

    def magOff(self, pindex, pca):
        print("Magnetorquer Off")
        for pin in self.pins[pindex]:
            pca.channels[pin].duty_cycle = MAX_DUTY

# Initialize magnetorquers with their respective PWM channels
mag = Mag([
    [PWM4, PWM5, PWM7, PWM6],  # PWM channels for the first magnetorquer
    [PWM0, PWM1, PWM3, PWM2]   # PWM channels for the second magnetorquer
])

"""
# Example usage
try:
    mag.magOn(50, 0)
    time.sleep(10)
    mag.magOn(-50, 0)
    time.sleep(10)
    mag.magOff(0) # one magnetorquer
    time.sleep(10)
    mag.magOn(50, 1)
    time.sleep(10)
    mag.magOn(-50, 1)
    time.sleep(10)
    mag.magOff(1) # other magnetorquer
except KeyboardInterrupt:
    print("Interrupted by user")
finally:
    # Cleanup code if needed
    print("Cleanup complete.")
"""

