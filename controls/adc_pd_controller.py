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

    return [a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
            a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
            a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
            a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0]] 

def matrix_multipy(a,b):
    '''
    matrix_multiply
        Function that multiplies a and b

    @params
    a, b: matrices (b must be n x 1 column vector)
    @returns
        multiplied matrix
    '''
    result = [[0],[0],[0],[0]] # this should probably be generalized but for now we know its 4x1
    for i in range(len(a)):
        for k in range(len(b)):
            result[i][0] += a[i][k] * b[k]

    return result

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
    # Takes the inverse of the quaternion given on page 39 of Fundamentals book q-1 = q* / ||q||^2 where q*=[q4; -q1:3]
    quat_inv_target = [target[0]/scalar,
                       -target[1]/scalar,
                       -target[2]/scalar,
                       -target[3]/scalar]
    return quaternionMultiply(state,quat_inv_target)

# Returns 1 if x is positive, -1 if x is negative, 0 if x is 0
def sign(x):
    return (x > 0) - (x < 0)

def pd_controller(state,target, omega, kp, kd):
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
    for i in range(len(pwm)):
        pwm[i] = -kp * sign(delta_q_out[0]) * delta_q_out[i+1] - kd * omega[i]
    #print('pwm: ', pwm)

    # transformation matrix for NASA configuration
    alpha = 1/math.sqrt(3)
    beta = 1/math.sqrt(3)
    gamma = 1/math.sqrt(3)

    # If fourth reaction wheel is not mounted exactly how we want we can adjust alpha, beta, gamma
    W =  [[1, 0, 0, alpha],
             [0, 1, 0, beta],
             [0, 0, 1, gamma]]
    
    # normalizing scalar
    n = (1 + alpha**2 + beta**2 + gamma**2)
    
    # Calculates pseudoinverse needed for transformation (pg 157 of Fundamentals book)
    W_inv = [[(1 + beta**2 + gamma**2)/n, -alpha*beta/n, -alpha*gamma/n],
             [-alpha*beta/n, (1 + alpha**2 + gamma**2)/n, -beta*gamma/n],
             [-alpha*gamma/n, -beta*gamma/n, (1 + beta**2 + beta**2)/n],
             [alpha/n, beta/n, gamma/n]]
    
    
    # convert output for 3 rx wheels to 4
    pwm = matrix_multipy(W_inv,pwm)
    # Convert back to 1x4 list
    pwm = [int(pwm[0][0]), int(pwm[1][0]), int(pwm[2][0]), int(pwm[3][0])]

    # Ensure pwm is always within limits of RPMs
    for i in range(4):
        if pwm[i] > 0.5*MAX_PWM:
            pwm[i] = 0.5*MAX_PWM
        elif pwm[i] < 0.5*-MAX_PWM:
            pwm[i] = 0.5*-MAX_PWM

    return pwm

# Test to make sure it works
test_state = [-1, 1, 0, 0] # current quaternion
test_target = [1, 0, 0, 0] # goal quaternion
test_omega = [30,30,30] # rad/s

# Gains
kp = .05*MAX_PWM # I think these would be good to start with
kd = .01*MAX_PWM

test_pwm = pd_controller(test_state,test_target, test_omega, kp, kd)
print(test_pwm)
