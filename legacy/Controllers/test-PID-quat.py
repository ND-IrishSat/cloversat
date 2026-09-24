#test-PID-quat.py
#this for the quaternion so we using 3 pIDS for 3 values (a) does not matter in teh a,b,c,d of a quaternion
import numpy as np
import matplotlib.pyplot as plt
import math
#q is the quaternion

def pid(state, target, prev, integral, dt, lwe, iwe, dwe): #dont need to pass derivative since it is not based off previous, integral is
    #state/targ is a 4 dimensional quaternion (numpy arrays)
    err = error(state, target)
    
    for i in range(1,len(err)):
        integral[i] += err[i]*dt
        der = (err[i]-prev[i])/dt
        curr[i] = err[i]*lwe+integral[i]*iwe+der*dwe
        prev[i] = err[i]
        
    
    return (np.average(curr), prev, integral)
    
    

def quaternionMultiply(a, b):
    '''
    quaternionMultiply
        custom function to perform quaternion multiply on two passed-in matrices

    @params
        a, b: quaternion matrices (1 x 4)
    @returns
        multiplied quaternion matrix
    '''
    return [[a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3]],
            [a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2]],
            [a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1]],
            [a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0]]]


def error(state,target): #just a linear error function. Make sure that your error has direction (i.e. can be negative or positive. I think linear error might be simplest for what we are doing but other error funcs may work better who knows, certainly not me)
    return np.subtract(target, state)


if __name__ == '__main__':
    
    #set these values elsewhere to start the program, and they will be adjusted as the program is ran (pid is called once an iteration
    integral = 0
    prev=0
    curr=0
    dt=.2 #this will fluctuate with the actual loop time of the program
    lwe=1 #linear weight
    iwe=.01 #integral weight
    dwe=-.01 #derivative weight
    prev = [0,0,0,0]
    curr = [0,0,0]
    target = [0,0,0,0]
    
    movements = []
    for i in range(1000):
        (curr, prev, integral) = pid(state, target, prev, integral, dt, lwe, iwe, dwe)
        movements.append(curr)
        state = [1,1,1,1] #will explain this in person
