'''
motor_test.py
Author: Andrew Gaylord

tests that we can read sensor data from the hall sensors 
and run our motors at desired rpm
'''

from motors import *
from hall import checkHall
from init import initialize_setup
from time import sleep


# Initialize for execution\
# need to fix scope for pca
pca = initialize_setup()

# Initialize motor classes (for each of 3 reactions wheels) using global variables from motors.py
x = Motor(pinX,dirX,hallX,default,default, pca)
# y = Motor(pinY,dirY,hallY,default,default)
# z = Motor(pinZ,dirZ,hallZ,default,default)

# GPIO setup
#GPIO.setmode(GPIO.BCM)
#GPIO.setup(enable,GPIO.OUT)
#GPIO.output(enable,True)

count = 0
x.target = 0
x.setSpeed()
x.lastSpeed = 0
time.sleep(3)

while (count < 20):

    speed = 20000

    # set speed
    x.target = speed

    # Check directions & alter speeds
    x.changeSpeed()
    x.lastSpeed = checkHall(x.hallList[0])
    print("hall sensor reading: ", x.lastSpeed)
    
    count += 1

print("CHANGING DIR")
count = 0
time.sleep(5)

while (count < 20):
    speed = -20000

    x.target = speed
    
    # Check directions & alter speeds
    x.changeSpeed()
    x.lastSpeed = -(checkHall(x.hallList[0]))
    print("hall reading:" x.lastSpeed)
    count += 1

print("CHANGING DIR2")  
count = 0
time.sleep(3)


while (count < 20):
    speed = 20000

    # Get the pwm signals
    x.target = speed
    
    x.changeSpeed()
    x.lastSpeed = (checkHall(x.hallList[0]))
    print("hall reading:" x.lastSpeed)
    count += 1

# Bring the wheels to a stop
#speed = 0
#x.target = speed
#x.changeSpeed()

# Confrim it is done
#print("done with script! ending...")

cleanup(x)
