'''
testing_functions.py
Authors: Andrew Gaylord and Alex Casillas
Last modified: 10/31/2023

Speed tests for arbitrary functions
'''

import time
import numpy as np
import scipy
import timeit
import run_UKF
import UKF_algorithm
from hfunc import *
import random
from wmm import WMM




def speedTest (func, inputs, n):
    '''
    speedTest
        takes arbitrary function and finds average speed (in milliseconds) over n iterations

    @params
        func: function we want to test
        inputs: list of inputs for function
        n: number of iterations to run speed test on
    @returns
        average: average time to complete n trials (in milliseconds)
    '''
    start = time.time()

    for i in range(n):
        func(*inputs)
    
    total = time.time() - start

    average = (total / n) * 1000

    print("Average time of {} over {} iterations: {} ms".format(func.__name__, n, average))
    return average


if __name__ == '__main__':

    n = 10
    m = 9
    scaling = 3-n
    means = np.random.rand(n)
    cov = np.random.rand(10,10)
    r=np.zeros(n)
    q=np.zeros(m)
    for i in range(m):
        r[i]=random.random()
        q[i]=random.random() * .1
    u_k = [35.69314751, -114.13385513, 424.7491012, np.array([2023.811])]

    data = np.random.rand(6)

    # speedTest(UKF_algorithm.sigma, [means, cov, n, scaling], 10000)
    # about half a millisecond

    speedTest(UKF_algorithm.UKF, [means, cov, r, q, u_k, data], 1000)
    # about 7.9 ms
    # 90 ms with new hfunc?!?!

    # Bfield = bfield_calc(u_k)
    # speedTest(hfunc, [means, Bfield], 1000)
    # .01 ms

    # speedTest(bfield_calc, [u_k], 1000)
    # 8.6 or 7




