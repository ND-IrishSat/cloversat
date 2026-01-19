'''
hall.py
Authors: Tim Roberts

interfaces with hall sensors, allowing us to read motor speed of reaction wheels
GPIO initialization must be done beforehand

'''

import RPi.GPIO as GPIO
import numpy as np
from sklearn.linear_model import LinearRegression
import time
# from motors import *
from motors import MAX_DUTY, MAX_RPM

# Constants (imported from motors.py)
# MAX_RPM = 9100
# max duty cycles
# MAX_DUTY = 65535
# pin numbers for hall sensors [x,y,z]
hallList = [13,8,16]


# Check motor speed (in duty cycles)
def checkHall(sensor): 
	count = 0
	hall_speed = 0
	data = []
	while count < 10:
		if GPIO.event_detected(sensor):
			data.append(time.perf_counter())
			count += 1
	t = np.array(data)
	c = np.arange(10).reshape(-1,1)
	model = LinearRegression().fit(c,t)
	frequency = 1/model.coef_
	hall_speed += ((frequency[0] * 15) / MAX_RPM) * MAX_DUTY 
	# Return the speed according to Hall sensor (duty cycles)
	return hall_speed

# example call
# res = checkHall(hallList[0])
