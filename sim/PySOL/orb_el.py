

from matplotlib import bezier
import scipy as sci
import numpy as np





mu_earth = 398600

def calc_RV2OE(S, verbose = False):
    """
        calc_OE - function which takes state vector, S
            and calculates the six orbital elements

            f - true anamoly [deg]
            a - semi-major axis [km]
            e - eccentricity [0, 1)
            i - inclination [deg]
            Om - RA of ascending node [deg]
            w - argument of perigee [deg]  

            Args:
                S (1x6 array): state vector [km]/[km/s]
                    x, y, z, vx, vy, vz in ECI frame  
                
                Opt Args:
                verbose (bool): if True, prints solved OE

            Returns:
                OE_array (1x6 array) - solved orbital elements
                    note: all angle quants in deg
                    [f, a, e, i, Om, w]
    
    """

    r_ = S[0:3]
    rx, ry, rz = r_
    v_ = S[3:]
    vx, vy, vz = v_
    r = np.linalg.norm(r_)
    v = np.linalg.norm(v_)
    vr = np.dot(v_, r_/r)

    # Angular momentum 
    # Technically OE but not used here
    h_ = np.cross(r_, v_)
    h = np.linalg.norm(h_)

    # Node line calc | Note quad ambiguity
    K_ = np.array([0, 0, 1])
    N_ = np.cross(K_, h_)
    N = np.linalg.norm(N_)
    Nx, Ny, Nz = N_

    ### Begin OE Calculations

    # Eccentrcity calc
    e_ = 1/mu_earth*np.cross(v_, h_) - r_/r
    e = np.linalg.norm(e_)
    ex, ey, ez = e_

    print('e', e_)

    # Semi-major axis calc
    a = h**2/(mu_earth*(1-e**2))

    # Inclination calc
    i = np.arccos(h_[2]/h)

    # RA of Ascending Node 
    if Ny >= 0:
        Om = np.arccos(Nx/N)
    else:
        Om = 2*np.pi - np.arccos(Nx/N)

    if ez >= 0:
        w = np.arccos(np.dot(N_/N, e_/e))
    else:
        w = 2*np.pi - np.arccos(np.dot(N_/N, e_/e))

    if vr >= 0:
        f = np.arccos(np.dot(e_/e, r_/r))
    else:
        f = 2*np.pi - np.arccos(np.dot(e_/e, r_/r))


    i_deg = np.rad2deg(i)
    Om_deg = np.rad2deg(Om)
    w_deg = np.rad2deg(w)
    f_deg = np.rad2deg(f)

    OE_array = np.array([f_deg, a, e, i_deg, Om_deg, w_deg])

    if verbose:
        l = 50
        print()
        print('-'*l)
        print('-'*l)
        print('RV2OE Input')
        print('r = [{:5.4f}, {:5.4f}, {:5.4f}] km'.format(rx, ry, rz))
        print('v = [{:5.4f}, {:5.4f}, {:5.4f}] km/s'.format(vx, vy, vz))
        print('-'*l)
        print('-'*l)
        print('Solved Orbital Elements')
        print('-'*l)
        print('f: {:5.4f}  [deg] | a: {:5.4f} [km] | e: {:5.4f} '.format(f_deg, a, e))
        print('i: {:5.4f} [deg] | Om: {:5.4f} [km] | w: {:5.4f} '.format(i_deg, Om_deg, w_deg))

    return OE_array

def calc_OE2RV(OE_array, verbose = False):
    """
        calc_OE2V - function takes in orbital element array
            and calculates ECI position and velocity vectors

            Method taken from: https://youtu.be/ZiLxfVevkI8

        Args:
            OE_array (1x6 array): 
                note: all angle quants in deg
                    [f, a, e, i, Om, w]

        Returns:
            S (1x6 array): state vector in ECI [km]/[km/s]

    """

    f, a, e, i, Om, w = OE_array

    f_rad = np.deg2rad(f)
    i_rad = np.deg2rad(i)
    Om_rad = np.deg2rad(Om)
    w_rad = np.deg2rad(w)

    # First calc r, v in perifocal frame
    # note _PQW denotes perifocal frame
    # note _IJK denotes ECI frame 
    p = a*(1-e**2)
    r = p/(1+e*np.cos(f_rad))

    r_PQW = r*np.array([np.cos(f_rad), np.sin(f_rad), 0])
    v_PQW = np.sqrt(mu_earth/p)*np.array([-np.sin(f_rad), e + np.cos(f_rad), 0])


    R_Om = Rotate_Z(Om_rad)
    R_w = Rotate_Z(w_rad)
    R_i = Rotate_X(i_rad)

    R_PQQ_2_ECI = R_Om @ R_i @ R_w

    r_ECI = R_PQQ_2_ECI @ r_PQW
    v_ECI = R_PQQ_2_ECI @ v_PQW

    S = np.concatenate((r_ECI, v_ECI))

    if verbose:
        l = 50
        print()
        print('-'*l)
        print('-'*l)
        print('OE2RV Input')
        print('f: {:5.4f} [deg] | a: {:5.4f} [km] | e: {:5.4f} '.format(f, a, e))
        print('i: {:5.4f} [deg] | Om: {:5.4f} [km] | w: {:5.4f} '.format(i, Om, w))
        print('-'*l)
        print('-'*l)
        print('Solved Position and Velocity')
        print('r = [{:5.4f}, {:5.4f}, {:5.4f}] km'.format(r_ECI[0], r_ECI[1], r_ECI[2]))
        print('v = [{:5.4f}, {:5.4f}, {:5.4f}] km/s'.format(v_ECI[0], v_ECI[1], v_ECI[2]))
        print('-'*l)
        print('-'*l)

    return S


def calc_R2RADEC(r_):
    """
        calc_RADEC - calculates the Right Ascension and Declination 
            of r_ position vector in ECI coordinates

            Args:
                r_ (1x3 array) : position vector in IJK ECI frame

            Return:
                RA_deg (float) : right ascension in degrees
                Dec (float) : declination in degrees
    
    """

    X, Y, Z = r_
    r = np.linalg.norm(r_)

    Xprojr = X/r # scalar projection of component along r_
    Yprojr = Y/r
    Zprojr = Z/r 

    Dec = np.arcsin(Zprojr)

    if Yprojr > 0:
        RA = np.cos(Xprojr/np.cos(Dec))
    else:
        RA = 2*np.pi - np.cos(Xprojr/np.cos(Dec))

    Dec_deg = np.rad2deg(Dec)
    RA_deg = np.rad2deg(RA)

    return RA_deg, Dec_deg 

def Rotate_Z(psi):
    """
        Rotate_Z - rotation matrix about the Z axis
            Args:
                psi (float): rotation in radians

            Returns:
                R_Z (3x3 array): rotation matrix
    """
    cos = np.cos 
    sin = np.sin 

    R_Z = np.array([
            [cos(psi), -sin(psi), 0],
            [sin(psi), cos(psi) , 0],
            [0       , 0        , 1] 
    ])

    return R_Z

def Rotate_X(psi):
    """
        Rotate_X - rotation matrix about the X axis
            Args:
                psi (float): rotation in radians

            Returns:
                R_X (3x3 array): rotation matrix
    """
    cos = np.cos 
    sin = np.sin 

    R_X = np.array([
            [1      , 0        , 0         ],
            [0      , cos(psi) , -sin(psi) ],
            [0      , sin(psi) , cos(psi)  ] 
    ])

    return R_X


if __name__ == '__main__':
    

    r_ = np.array([-6045, -3490, 2500])
    v_ = np.array([-3.457, 6.618, 2.533])
    S_ = np.concatenate((r_, v_))

    OE_ = calc_RV2OE(S_, verbose=True)
    S_ =calc_OE2RV(OE_, verbose = True)

    
