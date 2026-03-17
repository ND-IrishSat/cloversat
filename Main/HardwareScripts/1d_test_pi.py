'''
1d_test_pi.py
Authors: Claudia Kuczun, Andrew Gaylord, Patrick Schwartz, Juwan Jeremy Jacob, Alex Casillas
Last updated: 2/25/24

Main script for 1D systems integration test
Collab test between GOAT lab and protoSat
Implements UKF, PD, hall sensors, IMU, visualizer, and actuators to rotate cubeSat on frictionless table on 1 axis
For use at 2024 end-of-year banquet

'''

#cloversat\Main\SimScripts
#cloversat\ukf
#these are where the multiple ukf py files all seem to be located

#also need to merge Rawan's changes, these will be in the motors branch, but even though she told me there should be a rotate function, I could not find it, reach out to her

import sys, os
sys.path.extend([f'./{name}' for name in os.listdir(".") if os.path.isdir(name)])

import numpy as np
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
from ukf.PySOL.wmm import *
from sim.visualizer import *
from adc.adc_pd_controller_numpy import pd_controller
from ukf.UKF_algorithm import *
import time
# from interface.happy_sensors import get_imu_data, calibrate
from interface.happy_sensors import *
from ukf.hfunc import *
from sim.PySOL import wmm
from HardwareInterface.motors import *
from HardwareInterface.hall import checkHall
from HardwareInterface.init import initialize_setup

# MAX_DUTY (65535, used for pwm) and MAX_RPM (9100) are declared in motors.py

# pi info (PUT IN TUTORIAL FILE SOMEWHERE):
# scp -r test irishsat@10.7.85.47:/home/irishsat/test .
# password: irishsat
# download vnc viewer for exact screen mirroring

def main(target=[1, 0, 0, 0]):
    '''NOTE
        - things to take out for UKF only test: motor initialization, reaction wheel update, pd section
        - What is starting state guess? read from sensors?
        - Set proper target quaternion?

        - implement new IMU (redo this script)!!! maybe forget following workarounds for the moment:
            - fix initialization
            - fix file of constants (maybe not with new sensor)
            - seperate script for initialization of imu + b true
        - rewrite/comment motors.py
        - flags for testing only ukf, etc
        - effiency testing somehow
            - update run_UKF?
        - update visualizer with gui
        - update organization of interface folder

        - magnetorquer...? magdriver repo
    '''

    #you can change where you put this but this is reading in the magnetometer calibration values
    fname="interface/calvals.txt"
    fin=open(fname,"r")
    mcoeffs=[0]*3
    mscales=[0]*3
    for i in range(3):
        mcoeffs[i],mscales[i]=[float(x) for x in fin.readline().split()]
    fin.close()

    # Initialize setup for motors (I2C and GPIO): not functional currently
    # i2c needs to be in motors.py???
    # currently non-functional
    # initialize_setup()
    # print("initialized setup\n")

    # GPIO setup
    # NOTE: uncomment only gpio to run motors
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(enable,GPIO.OUT)
    GPIO.output(enable,True)

    # Initialize motor classes (for each of 3 reactions wheels) using global variables from motors.py
    #x = Motor(pinX,dirX,hallX,default,default)
    #y = Motor(pinY,dirY,hallY,default,default)
    #z = Motor(pinZ,dirZ,hallZ,default,default)

    # i2c initialization
    # NOTE: DO NOT INITALIZE HERE. DONE IN MOTORS.PY
    #i2c_bus = busio.I2C(SCL, SDA)
    #pca = PCA9685(i2c_bus)
    #pca.frequency = 1500
    # print("initialized motors\n")

    # Dimension of state and measurement space
    dim = 7
    dim_mes = dim - 1

    dt = 0.1
    curr_time_pwm = time.time()

    # Calibrate the IMU
    success = calibrate()
    if not success:
        print("Something went wrong when calibrating & configuring!!!\n")

    # Read starting state from sensors?
    # start_data = get_imu_data()

    # initialize starting state and covariance
    state = np.array([1, 0, 0, 0, 0, 0, 0]) #[q0, q1, q2, q3, omega_x, omega_y, omega_z]
    cov = np.identity(dim) * 5e-10

    '''
    smaller noise = more trust = less uncertainty = source is less noisy
    higher noise = lower kalman gain = smooth out noise, lower responsivness
    Strive to strike a balance between precision and robustness.
    Too much uncertainty (high covariances) can lead to poor estimation accuracy,
    while too little uncertainty can make the filter overly sensitive to noise and outliers.
    '''
    # r: measurement noise (m x m)
    # smaller value = trust it more = source is less noisy
    noise_mag = 5
    noise_gyro = 0.1

    # r = np.diag([noise_mag] * dim_mes)
    r = np.array([[noise_mag, 0, 0, 0, 0, 0],
                 [0, noise_mag, 0, 0, 0, 0],
                 [0, 0, noise_mag, 0, 0, 0],
                 [0, 0, 0, noise_gyro, 0, 0],
                 [0, 0, 0, 0, noise_gyro, 0],
                 [0, 0, 0, 0, 0, noise_gyro]])

    # q: process noise (n x n)
    # Should depend on dt
    # try negative noises?
    noise_mag = .05
    # q = np.diag([noise_mag] * dim)
    q = np.array([[dt, 3*dt/4, dt/2, dt/4, 0, 0, 0],
                [3*dt/4, dt, 3*dt/4, dt/2, 0, 0, 0],
                [dt/2, 3*dt/4, dt, 3*dt/4, 0, 0, 0],
                [dt/4, dt/2, 3*dt/4, dt, 0, 0, 0],
                [0, 0, 0, 0, dt, 2*dt/3, dt/3],
                [0, 0, 0, 0, 2*dt/3, dt, 2*dt/3],
                [0, 0, 0, 0, dt/3, 2*dt/3, dt]
    ])
    q = q * noise_mag

    # Current lat/long/altitude, which doesn't change for this test
    curr_date_time= np.array([2024.1266])
    lat = np.array([41.675])
    longitude = np.array([-86.252])
    alt = np.array([.225552]) # 740 feet (km)

    # calculate true B field at stenson-remick
    B_true = wmm.bfield_calc(np.array([lat, longitude, alt, curr_date_time]))

    # Set negative of last element to match magnetometer
    # Convert to North East Down to North East Up, matching X, Y, Z reference frame of magnetometer
    B_true[1] *= -1
    B_true[2] *= -1

    # Our North West Up true magnetic field should be: 19.42900375, 1.74830615, 49.13746833 [micro Teslas]
    # can use mag_calibration.py to find mbias
    #offsets = custom_calibrate(mpu, 10, B_true)
    #print(mpu.mbias)
    #print(mpu.magScale)

    # Offsets and scaling from mag_custom_cal.py, custom written calibration
    mpu.mbias = [30.335109508547004, 59.71757955586081, 38.51908195970696]
    mpu.magScale = [1, 1, 1]

    # print(mpu.mbias)
    # print(mpu.magScale)
    #print('B_true: ', B_true)

    # Set B_true as the average of count number of
    # measurements of the magnetometer upon initialization
    # if in same config for a while, can run a couple times and hardcode
    # Also, run as own script separately and store B_true in file
    first_read = np.array(get_imu_data())
    B_true_num = first_read[:3]
    count = 60
    for i in range(count):
            reading = np.array(get_imu_data())
            print(reading[:3])
            #B_true_num = np.vstack((B_true_num, reading[:3]))
            B_true_num = B_true_num + reading[:3]
            time.sleep(0.2)
    #B_true_num = -B_true_num / count
    #B_true_num = np.median(B_true_num, axis = 0)
    B_true_num = B_true_num / count

    print("B_true from sensor average: ", B_true_num)
    B_true = B_true_num

    # Initialize current step and last step reaction wheel speeds
    # For this test they're 1x3: x, y, skew (z) wheels
    old_reaction_speeds = np.array([0,0,0])
    reaction_speeds = np.array([0,0,0])

    # Initialize pwm speeds
    pwm = np.array([0,0,0,0])
    old_pwm = pwm

    # inialize starting speed because hall sensor is read at end of loop
    x_speed = 0

    # calculated b field if fudging mag data
    b_calc = [0, 0, 0]

    # Infinite loop to run until you kill it
    i = 0
    while (i < 10000):
        # Starting time for loop
        # start_time = time.time()

        # Store reaction wheel speeds of last iteration
        old_reaction_speeds = reaction_speeds

        # Get current imu data (mag*3, gyro*3)
        imu_data = get_imu_data()

        #b_calc = list(np.matmul(quaternion_rotation_matrix(state[:4]), B_true))
        #print("\nfake b data: ", b_calc, "\n")

        # Angular velocity comes from gyro
        angular_vel = imu_data[3:]

        # Data array to pass into ukf
        data = [0] * dim_mes
        data[0] = imu_data[0]
        #data[0] = b_calc[0][0]
        data[1] = imu_data[1]
        #data[1] = b_calc[1][0]
        data[2] = imu_data[2]
        #data[2] = b_calc[2][0]
        data[3] = angular_vel[0]
        data[4] = angular_vel[1]
        data[5] = angular_vel[2]

        # Run UKF
        state, cov = UKF(state, cov, q, r, B_true, reaction_speeds, old_reaction_speeds, data)

        # Visualize if you want
        game_visualize(np.array([state[:4]]), i) # not working on Pi 0, but working on  Pi 4
        # print current quaternion estimate
        print("Current state: ", state[:4])


        # Run PD controller
        curr_quaternion = state[:4]

        # get current angular velocity of cubesat
        omega = np.array([angular_vel[0], angular_vel[1], angular_vel[2]])
        # PD gains parameters (dependant upon max pwm/duty cycles)
        kp = .2*MAX_DUTY
        kp = .05*MAX_DUTY
        kd = .1*MAX_DUTY
        kd = .01*MAX_DUTY

        # find time since last pd call
        end_time_pwm = time.time()
        pwm_total_time = end_time_pwm - curr_time_pwm

        # Run PD controller to generate output for reaction wheels
        pwm = pd_controller(curr_quaternion, target, omega, kp, kd, old_pwm, pwm_total_time)

        curr_time_pwm = time.time()
        old_pwm = pwm
        print("pwm after controller: ", pwm)

        # pwm[0] = abs(pwm[0])
        '''
        # Get the pwm signals
        x.target = pwm[0]
        # y.target = pwm[1]
        # z.target = pwm[3]

        # Check directions & alter speeds
        # TODO: rewrite motors.py for clarity

        # so that our wheel continues to spin, enact a minimum speed that activates the wheels
        print(f'target = {x.target}')
        if x.target < 5000:
            x.target = 5000
            print(f'updated target to {x.target}')
            x.changeSpeed()
        else:
            x.changeSpeed()
        time.sleep(1)
        # Ending time for loop
        #end_time = time.time()
        x.current = checkHall(x.hallList[0])

        # Output total runtime for single loop iteration
        #print("Time to run single iteration of ADC loop: {:.2f}".format(end_time-start_time))
        # Clear screen
        #os.system('cls' if os.name == 'nt' else 'clear')

        # Get reaction speeds (in duty cycles) from Hall sensors
        #NOTE: wheel must be spinning to check Hall
        x_speed = checkHall(x.hallList[0])
        # y_speed = checkHall(y.hallList[1])
        # z_speed = checkHall(z.hallList[2])
        # reaction_speeds = [*x_speed, *y_speed, *z_speed]
        reaction_speeds[0] = x_speed
        '''
        # Increment iteration count
        i += 1

    # Bring the wheels to a stop
    # x.target = 0
    # x.checkDir()

    # Confirm it is done
    print("done with script! ending...")

    GPIO.cleanup()


# if __name__ == "__main__":
    # target = [1, 0, 0, 0]
    main()
