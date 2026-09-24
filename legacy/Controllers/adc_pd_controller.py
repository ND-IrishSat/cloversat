# clover_controller.py
'''Feb 6, 2024
   Author: Patrick Schwartz
   PD - controller for 2U cubesat as described in Fundamentals of Spacecraft Attitude Determination and Control book pgs 289 - 293

'''
import numpy as np
import math
import matplotlib.pyplot as plt


MAX_PWM = 65535 # pwm val that gives max speed according to Tim


def quaternionMultiply(a, b):
    '''
    quaternionMultiply
        custom function to perform quaternion multiply on two passed-in matrices

    @params
        a, b: quaternion matrices (4 x 1)
    @returns
        multiplied quaternion matrix
    '''

    return np.array([a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
            a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
            a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
            a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0]])


def delta_q(state,target):
    '''
    delta_q
        returns error quaternion by taking quaternion product (x) of current 
        quaternion and inverse of goal quaternion
        Attitude error kinematics is on pg 76 of Fundamentals book

    @params
        state, target: quaternion matrices (1 x 4) [q0, q1:3]
    @returns
        error quaternion
    '''
    # normalizing factor
    scalar = (target[0]**2 + target[1]**2 + target[2]**2 + target[3]**2)
    if scalar == 0:
        scalar = 1
    # Takes the inverse of the quaternion given on page 39 of Fundamentals book q-1 = q* / ||q||^2 where q*=[q4; -q1:3]
    quat_inv_target = np.array([target[0],
                       -target[1],
                       -target[2],
                       -target[3]])/scalar
    
    return quaternionMultiply(state,quat_inv_target)

def pd_controller(state,target, omega, kp, kd, old_pwm, dt):
    '''
    pd_controller
        proportional derivative controller as described in Fundamentals book pgs 289 -293

    @params
        state: current quaternion of cubesat (4 x 1) [q0 ; q1:3]
        target: goal quaternion of cubesat (4 x 1) [q0 ; q1:3]
        omega: current angular velocity of cubesat
        kp: proportional gain
        kd: derivative gain
    
    @returns
        4 pwm inputs for each of the 4 motors

    '''
    # initialize pwm vector
    pwm = [0,0,0]
    # find error quaternion
    delta_q_out = delta_q(state, target) # outputs 4x1 with the first element being w
    #print("Error Quaternion: ", delta_q_out)
    # loop through list to get 3 pwm vals
    L = -kp * np.sign(delta_q_out[0]) * delta_q_out[1:4] - kd * omega # this is torque (which is proportional to angular acceleration)

    # for i in range(len(pwm)):
    #     pwm[i] = -kp * np.sign(delta_q_out[0]) * delta_q_out[i+1] - kd * omega[i]
    # print('pwm: ', L)

    # transformation matrix for NASA configuration
    alpha = 1/np.sqrt(3)
    beta = 1/np.sqrt(3)
    gamma = 1/np.sqrt(3)

    # If fourth reaction wheel is not mounted exactly how we want we can adjust alpha, beta, gamma
    W =  [[1, 0, 0, alpha],
             [0, 1, 0, beta],
             [0, 0, 1, gamma]]
    
    # normalizing scalar
    n = (1 + alpha**2 + beta**2 + gamma**2)

    # Calculates pseudoinverse needed for transformation (pg 157 of Fundamentals book)
    W_inv = np.array([[(1 + beta**2 + gamma**2), -alpha*beta, -alpha*gamma],
             [-alpha*beta, (1 + alpha**2 + gamma**2), -beta*gamma],
             [alpha*gamma, -beta*gamma, (1 + beta**2 + beta**2)],
             [alpha, beta, gamma]])/n
    
    # convert output for 3 rx wheels to 4
    L = np.matmul(W_inv,L)

    pwm = np.add(L*dt,old_pwm) # this does finite difference to get velocity from acceleration


    # Convert to integers
    pwm = np.array([int(pwm[0]),int(pwm[1]),int(pwm[2]),int(pwm[3])])
    # Convert back to 1x4 list
    #pwm = [int(pwm[0][0]), int(pwm[1][0]), int(pwm[2][0]), int(pwm[3][0])]
    # print("pwm: ", pwm)
    # Ensure pwm is always within limits of RPMs
    np.putmask(pwm, pwm > 0.5*MAX_PWM, 0.5*MAX_PWM)
    np.putmask(pwm, pwm < 0.5*-MAX_PWM, 0.5*-MAX_PWM)

    return pwm

if __name__ == "__main__":
    # Test to make sure it works
    test_state = np.array([0, 1, 0, 0]) # current quaternion
    test_target = np.array([0, 0, 0, 0]) # goal quaternion
    test_omega = np.array([3,3,3]) # rad/s
    dt = 1e-2
    old_pwm = np.array([500,500,500,500])

    # Gains
    kp = .05*MAX_PWM # I think these would be good to start with
    kd = .01*MAX_PWM

    test_pwm = pd_controller(test_state,test_target, test_omega, kp, kd,old_pwm,dt)
    print(test_pwm)
