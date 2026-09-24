'''
kill_motors.py

quickly and easily kills motors

'''

import sys, os
from HardwareInterface.motors import *
from HardwareInterface.init import initialize_setup
from time import sleep
sys.path.extend([f'./{name}' for name in os.listdir(".") if os.path.isdir(name)])

x = Motor(pinX,dirX,hallX,default,default)

GPIO.setmode(GPIO.BCM)
GPIO.setup(enable,GPIO.OUT)
GPIO.output(enable,True)

# STOP the motors!
print("stopping motors...")
speed = 0
x.target = speed
x.checkDir()
print("stopped!")
