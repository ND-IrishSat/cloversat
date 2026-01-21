'''
motors.py
Authors: Tim Roberts

motor interface class 
checks hall sensors readings and sets reaction wheel speeds
initialization must be done through init.py

also contains function to convert pwm signals to predicted RPM

'''
from board import SCL, SDA
import busio
from adafruit_pca9685 import PCA9685
import RPi.GPIO as GPIO
import time
import numpy as np
from sklearn.linear_model import LinearRegression
#from hall import checkHall
#from motorsandmags import mag
import random


# # GPIO
# GPIO.setmode(GPIO.BCM)
# GPIO.setup(enable,GPIO.OUT)
# GPIO.output(enable,True)
# # note: must also run GPIO.cleanup() at end of script

# I2C
# i2c_bus = busio.I2C(SCL, SDA)
# pca = PCA9685(i2c_bus)
# pca.frequency = 1500


# Constants
MAX_RPM = 9100
# max duty cycles
MAX_DUTY = 65535
default = 0
enable = 11
pinX = 10
pinY = 9
pinZ = 8
dirX = 0
dirY = 5
dirZ = 6
hallX = [13,19,26]
hallY = [8,7,1]
hallZ = [16,20,21]

class params:
    # Importing motor parameters - Maxon DCX 8 M (9 volts)
    Rwa = 3.54      # Ohms, winding resistance at ambient temperature
    Lw = 0.424e-3  # Henry
    Kt = 8.82e-3   # Torque constant Nm/A
    Kv = Kt    # Voltage constant V*s/rad
    Jm = 5.1*(1e-7)   # Kg m^2
    bm = 3.61e-6   # [N·m·s/rad] Viscous friction
    Rha = 16.5      # K/W
    Rwh = 2.66     # K/W
    Cwa = 2.31/Rwh     # Thermal Capacitance
    Cha = 162/Rha      # Thermal Capacitance
    alpha_Cu = 0.00393 # copper's temperature coefficient [1/K]
    # Moments of Inertia [g cm^2]- from CAD of CubeSat test bed
    Ixx = 46535.388 
    Ixy = 257.834 
    Ixz = 536.12
    Iyx = 257.834 
    Iyy = 47934.771 
    Iyz = -710.058
    Izx = 546.12 
    Izy = -710.058 
    Izz = 23138.181

    # Moment of Inertia Tensor of full 2U cubesat [kg m^2]
    J_B = (1e-7)*np.array([[Ixx, Ixy, Ixz],
                    [Iyx, Iyy, Iyz],
                    [Izx, Izy, Izz]])
    
    J_B_inv = np.linalg.inv(J_B)

    # Moments of Inertia of rxn wheels [g cm^2] - measured
    Iw1 = (1/2)*38*1.8**2 # I_disc = 1/2 * M * R^2
    Iw2 = Iw1 
    Iw3 = Iw1 
    Iw4 = Iw1

    # Moment of inertia tensor of rxn wheels [kg m^2]
    J_w = (1e-7)*np.array([[Iw1, 0, 0, 0],
                    [0, Iw2, 0, 0],
                    [0, 0, Iw3, 0],
                    [0, 0, 0, Iw4]])

    # External torques (later this can be from magnetorquers)
    L_B = np.array([0, 0, 0])

    # Transformation matrix for NASA config given in Fundamentals pg
    # 153-154
    W = np.array([[1, 0, 0, 1/np.sqrt(3)],
            [0, 1, 0, 1/np.sqrt(3)],
            [0, 0, 1, 1/np.sqrt(3)]])

# motor class!
class Motor():
    # IO setup
    def __init__(self, pin, direction, hallList, lastSpeed, target, pca):
        self.pin = pin
        self.dirPin = direction
        self.hallList = hallList
        self.hData = [[],[],[]]
        self.lastSpeed = lastSpeed
        self.target = target
        self.count = 0
        self.M_switch = 0
        self.rate = 0
        self.pca = pca
        GPIO.setup(self.dirPin,GPIO.OUT)
        GPIO.setup(self.hallList[0],GPIO.IN,pull_up_down=GPIO.PUD_DOWN)
        GPIO.setup(self.hallList[1],GPIO.IN,pull_up_down=GPIO.PUD_DOWN)
        GPIO.setup(self.hallList[2],GPIO.IN,pull_up_down=GPIO.PUD_DOWN)
        GPIO.add_event_detect(self.hallList[0], GPIO.FALLING)
        GPIO.add_event_detect(self.hallList[1], GPIO.FALLING)
        GPIO.add_event_detect(self.hallList[2], GPIO.FALLING)
    
    
    # Set the speed
    def setSpeed(self):
        # self.target = newVal
        self.pca.channels[self.pin].duty_cycle = abs(self.target)


    # Set the direction
    def setDir(self, val):
        GPIO.output(self.dirPin, val)


    # Send output signal to actuators (& checks direction)
    def changeSpeed(self):
        switch_timout = 2.75
        temp_target = self.target
        if self.target < 0 and self.lastSpeed > 0:
            print("Switching to negative")
            self.target = 0
            self.setSpeed()
            time.sleep(switch_timout)

            # reverses the dirPin output, causing negative spin
            self.setDir(True)
            self.target = temp_target
            self.setSpeed()

        elif self.target > 0 and self.lastSpeed < 0:
            print("Switching to posotive")
            self.target = 0
            self.setSpeed()
            time.sleep(switch_timout)

            # defines dirPin output = True as positive spin
            self.setDir(False)
            self.target = temp_target
            self.setSpeed()
        else:
            self.setSpeed()
            '''
            if abs(self.target) > .87 * MAX_DUTY:
                self.count+=1
                if self.count >= 10:
                    #self.M_switch = 1
                    mag.magOn(50,0, self.pca)
                    mag.magOn(50,1, self.pca)
            else:
                self.count = 0
                if abs(self.target) < .87 * MAX_DUTY: 
                    #self.M_switch = 0
                    mag.magOff(0, self.pca)
                    mag.magOff(1, self.pca)
            self.target -= 1 * self.rate # NOTE: M_switch doesn't exist?
            self.setSpeed()
    '''
	

    # convert duty cycle to RPM
    def convertToRPM(self): 
        return (self.target / MAX_DUTY) * MAX_RPM

    # convert RPM to duty cycle
    def convertToDuty(RPM):
        if RPM > MAX_RPM:
            RPM = MAX_RPM
        return (RPM / MAX_RPM) * MAX_DUTY
