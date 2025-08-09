'''
init.py
Author: Claudia Kuczun

Initialize the GPIO setup and the I2C connections necessary to work with reaction wheels.

'''

from motors import *
from hall import checkHall
import RPi.GPIO as GPIO


def initialize_setup():
    # return pca object that can be passed to motor class

    # GPIO
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(enable,GPIO.OUT)
    GPIO.output(enable,True)
    # note: must also run GPIO.cleanup() at end of script

    # I2C
    i2c_bus = busio.I2C(SCL, SDA)
    pca = PCA9685(i2c_bus)
    pca.frequency = 1500

    return pca

def cleanup(motor = None):
    # Bring the wheels to a stop
    if motor != None:
        motor.target = 0
        motor.changeSpeed()

    # Confirm it is done
    print("STATUS: done with script! ending...")
    GPIO.cleanup()