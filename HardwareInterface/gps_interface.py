'''
gps_interface.py
Authors: Claudia Kuczun, Andrew Gaylord
Last modified: 11/5/2023

Interfaces with and calibrates GPS used on CubeSat
'''

import numpy as np
import pyproj

def get_gps_data():
    '''
    get_gps_data
        TODO: Interfaces with gps sensor and returns result in ECEF?? or ECI??

    @returns
        np array of ecef coordinates (x, y, z)
    '''

    #TODO
    x, y, z = generate_gps()

    return np.array([x, y, z])

def generate_gps():
    x = 0
    y = 0
    z = 0
    #TODO

    return x, y, z


def ecef_to_latlong(x, y, z):
    """
    This function takes ECEF coordinates and converts them to WGS84 latitude, longitude, and altitude
    found online but we can easily find a different one
    such as https://gis.stackexchange.com/questions/230160/converting-wgs84-to-ecef-in-python
    
    Parameters:
        x (float): The x-coordinate in meters
        y (float): The y-coordinate in meters
        z (float): The z-coordinate in meters
    
    Returns:
        np array containing the latitude, longitude (in degrees), and altitude
    """

    # Define the ECEF and WGS84 coordinate systems
    ecef = pyproj.Proj(proj='geocent', ellps='WGS84', datum='WGS84')
    wgs84 = pyproj.Proj(proj='latlong', ellps='WGS84', datum='WGS84')
    
    # Convert the ECEF coordinates to WGS84 latitude and longitude
    lon, lat, alt = pyproj.transform(ecef, wgs84, x, y, z, radians=False)
    
    # Return the latitude and longitude
    return np.array([lat, lon, alt])
