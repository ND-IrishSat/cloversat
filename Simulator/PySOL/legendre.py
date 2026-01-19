''' legendre.py

    Module for generating the Schmidt semi-normalized associated Legendre functions, for calculation of the
    geomagnetic field in wmm.py

    Juwan Jeremy Jacobe
    University of Notre Dame
    IrishSat

    Last modified: 25 Oct 2022
'''
import numpy as np
import math
from scipy.special import lpmn

def lpmn_alt_not(n, x):
    ''' Function to switch between P_l^m notation and the P_l,m
    notation where

    P_l,m = (-1)^m * P_l^m 

    Args:
        n (int): max order want to calculate
        x (float): argument of Legendre polynomial

    Return:
        lpmv_alt_not (m+1, n+1), for all orders 0..m and degrees 0..n
    '''

    # Get (m+1, n+1) array (in this case m = n)
    lpmv_alt_not = lpmn(n, n, x)[0] # indexing 0th element to grab only polynomial, not derivative
    
    # Transpose so that it is (n+1, m+1) array
    lpmv_alt_not = np.transpose(lpmv_alt_not)

    # Multiply with (-1)**m factor to correct for alternate notation
    for n_ite in np.arange(0, n + 1, 1):
        for m_ite in np.arange(0, n_ite + 1, 1):
            lpmv_alt_not[n_ite, m_ite] = (-1)**m_ite * lpmv_alt_not[n_ite, m_ite]

    return lpmv_alt_not

def ssn_lpmv(n_max, x):
    ''' Schmidt semi normalized Legendre function. Written to emulate's pyshtools PlmSchmidt() function to replace

    See US/UK World Magnetic Model 2020-2025 | Technical Report

    Args:
        n_max: 
        x (float): argument of SSN Legendre function

    Return
        ssn_lpmv (np.array of size (n_max + 1)*(n_max + 2)/ 2), for all orders 0..n and degrees 0..m, where index n * (n+1) / 2 + m would give the
        Schmidt semi normalized Legendre of 
    '''

    lpmv_alt_not_vals = lpmn_alt_not(n_max, x)

    ssn_lpmv_vals = np.zeros(int((n_max + 1)*(n_max+2) / 2 ))

    # Normalizing factor sqrt( 2 * (n - m)!/ (n+m)! )
    for n_ite in np.arange(0, n_max + 1, 1):
        for m_ite in np.arange(0, n_ite + 1, 1):
            if m_ite == 0:
                ssn_lpmv_vals[int(n_ite * (n_ite+1) / 2 + m_ite)] = lpmv_alt_not_vals[n_ite, m_ite]
            if m_ite > 0:
                ssn_lpmv_vals[int(n_ite * (n_ite+1) / 2 + m_ite)] = np.sqrt(2 *math.factorial(n_ite - m_ite)/math.factorial(n_ite + m_ite) ) * lpmv_alt_not_vals[n_ite, m_ite]
    
    return ssn_lpmv_vals

if __name__ == "__main__":
    import pyshtools

    print('Pyshtools output')
    print(pyshtools.legendre.PlmSchmidt(12, 0.5))
    print('----------------------\n')

    print('Legendre')
    print(ssn_lpmv(12, 0.5))
    print('-----------------------\n')
