'''
banquet_demo.py
Authors: Claudia Kuczun, Andrew Gaylord, Patrick Schwartz, Juwan Jeremy Jacob, Alex Casillas, Micheal Paulucci
Last updated: 4/18/24

Main script for 1D systems integration test
Implements UKF, PD, hall sensors, IMU, visualizer, and actuators to rotate cubeSat on frictionless table on 1 axis 
For use at 2024 end-of-year banquet on air bearing table

Utilizes 1D version of ukf and hfunc with reduced state (only angle and speed)

'''

import sys, os
sys.path.extend([f'./{name}' for name in os.listdir(".") if os.path.isdir(name)])

import numpy as np
from pygame.locals import * 
from OpenGL.GL import *
from OpenGL.GLU import *
from ukf.PySOL.wmm import *
from sim.visualizer import *
from adc.adc_pd_controller_numpy import pd_controller
from ukf.UKF_algorithm_1D import *
import time
import signal
from interface.mpu9250.happy_sensors import *
from ukf.hfunc_1D import *
from interface.motors import *
from interface.hall import checkHall
# from interface.draw_sat import draw_cube
from interface.init import initialize_setup, cleanup

# Color constants
BLUE = '\033[94m'
RESET = '\033[0m'
TEAL = '\033[96m'


def signal_handler(sig, frame, x):
    '''
    signal_handler

    runs down reaction wheel and cleans up when ctrl + c is entered
    '''
    print()
    print("SIGNAL RECEIVED: shutting down...")
    print()

    cleanup(x)
    # Bring the wheels to a stop
    # x.target = 0
    # x.changeSpeed()

    # # Confirm it is done
    # print("STATUS: done with script! ending...")
    # GPIO.cleanup()
    # sys.exit(0)


# Main method
def main(target):
    '''NOTE
        
        uses janky forced 1D state (reduced EOMs and hfunc)
        very buggy but shows some simple adcs movement
    '''

    line = "="*150
    print(line)
    draw_cube()
    print(BLUE + "STARTING ADCS SCRIPT" + RESET)
    print(line, "\n")

    # GPIO setup used for motor communication
    # GPIO.setmode(GPIO.BCM)
    # GPIO.setup(enable,GPIO.OUT)
    # GPIO.output(enable,True)

    # return pca object from init.py for motor communication
    pca = initialize_setup()

    # Initialize motor classes (for each of 3 reactions wheels) using global variables from motors.py
    x = Motor(pinX,dirX,hallX,default,default, pca)
    #y = Motor(pinY,dirY,hallY,default,default)
    #z = Motor(pinZ,dirZ,hallZ,default,default)

    # Initialize signal handler for clean quitting
    print(line)
    print("STATUS: setting up signal handler.")
    print(line, "\n")
    signal.signal(signal.SIGINT, lambda sig, frame: signal_handler(sig, frame, x))

    # Dimension of state and measurement space
    # state space contains angle from start (euler angle) and angular velocity for one axis
    dim = 2
    # measurement space contains X and Y magnetic field, and angular velocity for 1 axis
    dim_mes = 3

    dt = 0.1
    curr_time_pwm = time.time()

    # Calibrate the IMU
    print(line)
    print("BEGINNING INITIAL CALIBRATION")
    print(line, "\n")
    success = calibrate()
    if not success:
        print("ERROR: failed to calibrate!")
    print(line)
    print("STATUS: initial calibration complete.")
    print(line, "\n")

    # initialize starting state and covariance
    # starting angle, starting speed
    state = np.array([0, 0]) 
    cov = np.identity(dim) * 5e-10

    # r: measurement noise (m x m)
    # smaller value = trust it more = source is less noisy
    noise_mag = 5
    # 50 fine, best results with 5
    noise_gyro = 1
    # .5 fine, 1 best

    r = np.array([[noise_mag, 0, 0],
                [0, noise_mag, 0],
                [0, 0, noise_gyro]])

    # q: process noise (n x n)
    # Should depend on dt
    noise_mag = .5
    # small (.05) = rocky. 0.5 best
    q = np.array([[noise_mag*dt, 0],
                [0, noise_mag* (dt**2)]
    ])
    
    # Offsets and scaling from mag_custom_cal.py, custom written calibration
    # mpu.mbias = [9.748114231447563, 25.470711778301066, -17.96126317098539]
    # mpu.magScale = [1.0332524560289558, 0.9686228670148338, 1.000211279317508]
    mpu.mbias = [-13.76629503262567, -29.023757966288436, -9.806086092427686]
    mpu.magScale = [0.6827787821226224, 0.962944744760604, 2.012414185537061]

    # Mag calibration values
    print(line)
    print("MAGNETOMETER - CUSTOM CALIBRATION VALUES:")
    print(mpu.mbias)
    print(mpu.magScale)
    print(line, "\n")
    
    # Our North West Up true magnetic field in stenson remick should be: 19.42900375, 1.74830615, 49.13746833 [micro Teslas]

    '''
    get a sample of magnetometer readings and caculate true B field based on them
    set B_true as the average of count number of measurements
    if in same config for a while, can run a couple times and hardcode
    '''
    print(line)
    print("CALCULATING AVERAGE B_FIELD.")
    print(line, "\n")
    first_read = np.array(get_imu_data())
    B_true_num = first_read[:3]
    count = 40
    for i in range(count):
            reading = np.array(get_imu_data())
            #print(reading[:3])
            #B_true_num = np.vstack((B_true_num, reading[:3]))
            B_true_num = B_true_num + reading[:3]
            time.sleep(0.1)
    #B_true_num = -B_true_num / count
    B_true_num = B_true_num / count
    B_true = np.zeros(2)
    B_true[0] = B_true_num[0]
    B_true[1] = B_true_num[1]

    print(line)
    print("STATUS: finished get average B_field.")
    print("RESULT: B_true from sensor average: ", B_true_num)
    print(line, "\n")

    # Initialize current step and last step reaction wheel speeds
    # For this test they're 1x3: x, y, skew (z) wheels
    old_reaction_speeds = 0 
    reaction_speeds = 0

    # Initialize pwm speeds
    pwm = np.array([0,0,0,0])
    old_pwm = pwm

    # inialize starting speed because hall sensor is read at end of loop
    x_speed = 0

    dt = .1
    
    print(line)
    print("STARTING TEST RUN.")
    print(line, "\n")

    # Infinite loop to run until you kill it
    i = 0
    while (1):
        print(TEAL + line + RESET)
        print(BLUE + f'RUN {i+1}' + RESET)
        
        # Starting time for loop 
        # start_time = time.time()

        # store reaction wheel speeds of last iteration
        old_reaction_speeds = reaction_speeds

        # Get current imu data (mag*3, gyro*3)
        imu_data = get_imu_data()

        # Angular velocity comes from gyro
        angular_vel = imu_data[3:]

        # Data array to pass into ukf
        data = [0] * dim_mes
        data[0] = imu_data[0]
        data[1] = imu_data[1]
        data[2] = angular_vel[2]

        # Run UKF
        state, cov = UKF(state, cov, q, r, dt, B_true, reaction_speeds, old_reaction_speeds, data)
        print("*Current state: ({:.2f}, {:.2f})".format(state[0],state[1]))

        # convert our euler state to quaternion (and one in proper frame of reference of visualizer)
        quaternion = np.array(euler2quat(0, 0, state[0]))
        visualizer = np.array(euler2quat(0,state[0],0))
        
        # Visualize if you want
        game_visualize(np.array([visualizer]), i) # not working on Pi 0, but working on  Pi 4

        # print current quaternion estimate
        print("*Current quaternion: ({:.2f}, {:.2f}, {:.2f}, {:.2f})".format(quaternion[0],quaternion[1],quaternion[2],quaternion[3]))
        
        # Run PD controller

        # Get current angular velocity of cubesat
        omega = np.array([angular_vel[0], angular_vel[1], angular_vel[2]])
        # PD gains parameters (dependant upon max pwm/duty cycles)
        kp = .4*MAX_DUTY
        # .5 works, .2->.4 best so far
        #kp = .05*MAX_DUTY # old, bad
        kd = .2*MAX_DUTY
        # .1->.2 best so far
        #kd = .01*MAX_DUTY # old, bad
        
        # Find time since last pd call
        end_time_pwm = time.time()
        pwm_total_time = end_time_pwm - curr_time_pwm

        # Run PD controller to generate output for reaction wheels
        pwm = pd_controller(quaternion, target, omega, kp, kd, old_pwm, pwm_total_time)

        curr_time_pwm = time.time()
        old_pwm = pwm
        print(f"*PWM after controller: ({pwm[0]}, {pwm[1]}, {pwm[2]}, {pwm[3]})")
        # pwm[0] = abs(pwm[0])

        # use x.lastSpeed to store prev target for sign comparison
        x.lastSpeed = x.target
        
        # Get the pwm signals
        x.target = pwm[2]
        #y.target = pwm[1]
        #z.target = pwm[3]

        # Check directions & alter speeds

        # so that our wheel continues to spin, enact a minimum speed that activates the wheels
        print(f'-> Target speed = {x.target}')
        if x.target < 5000 and x.target > -5000:
            if x.target > 0:
                x.target = 5000
            if x.target < 0:
                x.target = -5000
            print(f'   -> Updated target to {x.target}')
            x.changeSpeed()
        else:
            x.changeSpeed()

        # some sleep is needed so that sensor can read correctly
        time.sleep(.07)

        # Ending time for loop
        #end_time = time.time()

        # Output total runtime for single loop iteration
        #print("Time to run single iteration of ADC loop: {:.2f}".format(end_time-start_time))

        # Get reaction speeds (in duty cycles) from Hall sensors
        #NOTE: wheel must be spinning to check Hall
        x_speed = checkHall(x.hallList[0]) 
        #y_speed = checkHall(y.hallList[1])
        #z_speed = checkHall(z.hallList[2])
        #y_speed=0
        #z_speed=0

        reaction_speeds = x_speed
 
        # Increment iteration count
        i += 1

        print()


if __name__ == "__main__":
    target = [1, 0, 0, 0]
    main(target)
