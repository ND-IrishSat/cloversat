'''
nadir_point.py
Authors: Peyton Reynolds, Lauren Catalano, David Scully, Michael Kuczun, Andrew Gaylord, Sean Wilson

Controller for nadir pointing given an error quaterion
Outputs the voltage necissary to orient us towards nadir
Attitude Control acutation (step 2 of our nadir pointing process)

'''

import math
import numpy as np
import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
# if RUNNING_MAYA:
    # from NearSpace.Pointing.image_processing import *
    # from Horizon_Sensor_Sim.Simulator.all_EOMs import normalize, delta_q

from Simulator.all_EOMs import *
from Simulator.sat_model import Magnetorquer_Sat
from Simulator.camera import Camera
from params import *


unsupported_case_count = 0

def determine_attitude(mag_sat: Magnetorquer_Sat):
    '''
    Based on image processing output from our two cameras, find our current orientation with respect to nadir pointing

    @params:
        mag_sat: represents current state of our satellite (including image processing results)

    @returns:
        determined_vector: 3D vector in body frame of satellite representing nadir
        determined_angles: list of [roll, pitch, yaw] in body frame of satellite representing our current orientation with respect to nadir (degrees) - for backwards compatibility
        determined_quaternion: quaternion [w, x, y, z] in body frame of satellite representing our current orientation with respect to nadir
        vector_angle_error: angle error between body -Z and nadir vector (degrees)
    '''
    # print("")
    # print("Cam 1 results: roll {}, pitch {}, alpha {}, edges {}, midpoint {}, ends {}".format(
    #     mag_sat.cam1.roll, mag_sat.cam1.pitch, mag_sat.cam1.alpha, mag_sat.cam1.edges, mag_sat.cam1.midpoint, mag_sat.cam1.ends))
    # print("Cam 2 results: roll {}, pitch {}, alpha {}, edges {}, midpoint {}, ends {}".format(
    #     mag_sat.cam2.roll, mag_sat.cam2.pitch, mag_sat.cam2.alpha, mag_sat.cam2.edges, mag_sat.cam2.midpoint, mag_sat.cam2.ends))

    sideways_thresh = 0.70
    bottom_thresh = 0.9

    # Case 1: earth in bottom of both cams
    # if mag_sat.cam1.edges[0] < mag_sat.cam1.edges[2] and mag_sat.cam2.edges[0] < mag_sat.cam2.edges[2]:
    if mag_sat.cam1.edges[2] >= bottom_thresh and mag_sat.cam2.edges[2] >= bottom_thresh:
        return AD_earth_bottom(mag_sat)

    # Case 2: one cam sees all earth, other sees some earth
    elif (mag_sat.cam1.alpha == 1) or (mag_sat.cam2.alpha == 1):
        if (mag_sat.cam1.alpha == 1):
            roll = 180
            pitch = 200
            yaw = 0
            eulers = [roll, pitch, yaw]
            nadir_vector = euler_to_vector(eulers)
            q = euler_to_quaternion(roll, pitch, yaw)
            return nadir_vector, eulers, q, -1
        else:
            roll = 0
            pitch = 45
            yaw = 0
            eulers = [roll, pitch, yaw]
            nadir_vector = euler_to_vector(eulers)
            q = euler_to_quaternion(roll, pitch, yaw)
            return nadir_vector, eulers, q, -1

    # Case 3: 1 cam sees space, the other sees some earth
    elif (mag_sat.cam1.alpha == 0 and mag_sat.cam2.alpha > 0) or (mag_sat.cam1.alpha > 0 and mag_sat.cam2.alpha == 0):
        good_cam = mag_sat.cam1 if mag_sat.cam1.alpha > 0 else mag_sat.cam2
        good_cam_num = 1 if mag_sat.cam1.alpha > 0 else 2
        return AD_one_cam(good_cam, good_cam_num)

    # # Case 4: earth at top of both cams
    # elif (mag_sat.cam1.edges[0] == 1 and mag_sat.cam2.edges[0] == 1):
    #     return AD_earth_top(mag_sat)

    # Case 5: earth at edge of both cams
    elif (mag_sat.cam1.edges[1] >= sideways_thresh and mag_sat.cam2.edges[3] >= sideways_thresh) or (mag_sat.cam2.edges[1] >= sideways_thresh and mag_sat.cam1.edges[3] >= sideways_thresh):
        left_of_cam1 = mag_sat.cam1.edges[3] >= sideways_thresh
        return AD_earth_sideways(mag_sat, left_of_cam1)

    # Case 6: earth in opposite corners
    elif (   (mag_sat.cam1.edges[0] > mag_sat.cam2.edges[0] and mag_sat.cam1.edges[3] > mag_sat.cam2.edges[3] and mag_sat.cam2.edges[1] > mag_sat.cam1.edges[1] and mag_sat.cam2.edges[2] > mag_sat.cam1.edges[2])
          or (mag_sat.cam2.edges[0] > mag_sat.cam1.edges[0] and mag_sat.cam2.edges[3] > mag_sat.cam1.edges[3] and mag_sat.cam1.edges[1] > mag_sat.cam2.edges[1] and mag_sat.cam1.edges[2] > mag_sat.cam2.edges[2])
          or (mag_sat.cam1.edges[0] > mag_sat.cam2.edges[0] and mag_sat.cam1.edges[1] > mag_sat.cam2.edges[1] and mag_sat.cam2.edges[2] > mag_sat.cam1.edges[2] and mag_sat.cam2.edges[3] > mag_sat.cam1.edges[3])
          or (mag_sat.cam2.edges[0] > mag_sat.cam1.edges[0] and mag_sat.cam2.edges[1] > mag_sat.cam1.edges[1] and mag_sat.cam1.edges[2] > mag_sat.cam2.edges[2] and mag_sat.cam1.edges[3] > mag_sat.cam2.edges[3])):
        # Determine whether the earth (combined) lies to the left of cam1 or to the right.
        # edges indices: [top, right, bottom, left]
        # Basic rule: if cam1 shows more left intensity than right, the earth is left-of-cam1.
        left_of_cam1 = mag_sat.cam1.edges[3] > mag_sat.cam1.edges[1]

        # If ambiguous (left vs right nearly equal), use the stronger cue across cams
        # or fall back to cam roll sign (negative roll -> tilted left in our convention)
        horiz_diff = abs(mag_sat.cam1.edges[3] - mag_sat.cam1.edges[1])
        if horiz_diff < 0.12:
            # prefer the cam that shows a stronger left intensity compared to the other cam
            if mag_sat.cam1.edges[3] != mag_sat.cam2.edges[3]:
                left_of_cam1 = mag_sat.cam1.edges[3] > mag_sat.cam2.edges[3]
            else:
                # final fallback: use roll sign (negative -> earth appears left)
                left_of_cam1 = (mag_sat.cam1.roll < 0)

        return AD_earth_opposite_corners_vector(mag_sat, left_of_cam1)

    else:
        return [-1, -1, -1], [-1, -1, -1], [0.707, 0.707, 0.707, 0.0], -1

    # # Case ?? (too vague): both cams see some earth and some space
    # elif mag_sat.cam1.alpha != 0 and mag_sat.cam2.alpha != 0:
    #     return AD_cross_product(mag_sat)

    # else:
    #     global unsupported_case_count
    #     unsupported_case_count += 1
    #     print(f"Error: Case not supported, cannot determine attitude. (Count: {unsupported_case_count})")
    #     print("************************************************************************************************************************************************************************************************************************************************************************************************************************")

    #     # Print debug info
    #     print(f"DEBUG - Cam1: alpha={mag_sat.cam1.alpha:.3f}, edges={mag_sat.cam1.edges}")
    #     print(f"DEBUG - Cam2: alpha={mag_sat.cam2.alpha:.3f}, edges={mag_sat.cam2.edges}")

    #     raise ValueError("Case not supported, cannot determine attitude.")
    #     return None, None, None, None


def AD_cross_product(mag_sat: Magnetorquer_Sat):
    '''
    Use cross product of 3D vectors to find nadir
    Good for case 1, fails for all (?) others
    '''
    # 1) Assume you have P_s1 (image one endpoint), P_s2 (image one endpoint), P_m_pt1 (image one midpoint), and P_m_pt2 (image two midpoint) from computing with their cartesian coordinates (x,y) in pixels for the two images
    # 2.) Calculate theta = horizontal component of P_m - 12
    theta1 = mag_sat.cam1.midpoint[0] - 12

    # 3.) Calculate A = 32 - vertical component of P_m + 1.4545
    A1 = 32 - mag_sat.cam1.midpoint[1] + 1.4545

    # 4.) Convert theta and A to degrees (theta/horizontal is 2.916 deg/pixel and A/vertical is 3.4375 deg/pixel) and call them theta_o and A_o
    # Note that _o denotes the degree version of the variable as opposed to the pixel version
    theta1_o = theta1 * 2.916
    A1_o = A1 * 3.4375

    # 5.) Calculate phi_o as sqrt(theta_o^2+A_o^2)
    # phi1_o = math.sqrt(theta1_o**2 + A1_o**2)
    phi1_o = A1_o

    # 6.) Repeats steps 1-5 for the other image
    theta2 = mag_sat.cam2.midpoint[0] - 12
    A2 = 32 - mag_sat.cam2.midpoint[1] + 1.4545
    theta2_o = theta2 * 2.916 + 180  # add 180 degs because cam2 is looking opposite direction
    A2_o = A2 * 3.4375
    # phi2_o = math.sqrt(theta2_o**2 + A2_o**2)
    phi2_o = A2_o

    # 7.) Convert both points to 3D cartesian vectors, calculating the x, y, and z components as [sin(phi_o)*cos(theta_o), sin(phi_o)*sin(theta_o), sin(theta_o)*cos(phi_o)] and save them as P_m_3d_pt1 and P_m_3d_pt2
    # NOTE: define image 1 as looking towards -x
    P_m_3d_pt1 = np.array([math.sin(math.radians(phi1_o)) * math.cos(math.radians(theta1_o)),
                           math.sin(math.radians(phi1_o)) * math.sin(math.radians(theta1_o)),
                           math.cos(math.radians(phi1_o))])
    print("3d vector for midpoint of horizon line in image 1: ", P_m_3d_pt1)

    P_m_3d_pt2 = np.array([math.sin(math.radians(phi2_o)) * math.cos(math.radians(theta2_o)),
                           math.sin(math.radians(phi2_o)) * math.sin(math.radians(theta2_o)),
                           math.cos(math.radians(phi2_o))])
    print("3d vector for midpoint of horizon line in image 2: ", P_m_3d_pt2)

    # 8.) Find the 3D vector between the two to find the vector between them, and save it as V1
    V1 = P_m_3d_pt2 - P_m_3d_pt1
    print("3d vector between midpoints of 2 horizon lines: ", V1)

    # 9.) Find a line between the two horizon endpoints P_s1 and P_s2, and save it as V2
    # The x component is 0 because it doesn't matter how close/far these lines are along the axis the cams are aligned with
    V2 = np.array([
        0,
        mag_sat.cam1.ends[1][0] - mag_sat.cam1.ends[0][0],
        mag_sat.cam1.ends[1][1] - mag_sat.cam1.ends[0][1]
    ])
    print("horizon line for image 1: ", V2)

    # 10.) Cross Product the endpoint slope vector with the midpoint vector (V1 x V2)
    # v1 x v2 gives us vector opposite of earth's normal vector (maybe)
    # [0, 0, -1] is pointing down z axis (bottom of cubesat)
    # [0, 0, 0] is center of cubesat)
    # [0, 1, 0] is pointing perpendicular to cams
    cross_product = np.cross(V1, V2)

    # 11.) Find the nadir vector (normalized cross product)
    nadir_vector = normalize(cross_product)
    print("Nadir vector: ", nadir_vector)

    # Ensure down direciton is actually down (because cross product can be pointing opposite)
    # if np.dot(BODY_DOWN, nadir_vector) < 0:
        # nadir_vector = -nadir_vector
    # print("Nadir vector: ", nadir_vector)

    # Find the error between body down vector and nadir vector
    vector_angle_error = determine_angle_error(nadir_vector, BODY_DOWN)
    print("Angle error between body -Z and nadir vector (deg): {:.2f}".format(vector_angle_error))

    # 12.) Compute roll, pitch (yaw cannot determine from horizon data alone)
    determined_angles = vector_to_euler(nadir_vector)
    print("Estimated Euler angles (deg): Roll={:.2f}, Pitch={:.2f}, Yaw={:.2f}".format(*determined_angles))

    # 13.) Compute quaternion from body -Z to nadir vector
    q = vector_to_quaternion(BODY_DOWN, nadir_vector)

    print("Quaternion (w, x, y, z):", q)

    # if the horizon is at the top of both images, flip the nadir vector
    if (mag_sat.cam1.edges[0] == 1 and mag_sat.cam2.edges[0] == 1):
        nadir_vector = -nadir_vector

    return nadir_vector, determined_angles, q, vector_angle_error


def AD_earth_bottom(mag_sat: Magnetorquer_Sat):
    '''
    Make an educated nadir guess when both cams see earth at bottom of frame
    '''
    roll = np.average([ mag_sat.cam1.roll, -mag_sat.cam2.roll ])
    # +15 has been found by testing
    pitch = np.average([ -mag_sat.cam1.pitch, mag_sat.cam2.pitch ]) + 15
    yaw = 0
    eulers = [roll, pitch, yaw]

    nadir_vector = euler_to_vector(eulers)

    return nadir_vector, eulers, euler_to_quaternion(*eulers), determine_angle_error(nadir_vector, BODY_DOWN)


def AD_one_cam(cam: Camera, cam_num: int):
    '''
    Make an educated nadir guess when only 1 cam has earth
    '''
    # cam 1 = -roll
    if cam_num == 1:
        roll = -cam.roll
    else:
        roll = cam.roll
    # Check cam number: if cam 1, subtract instead of add (want pitch -75 instead of 75)
    # TODO: add more if alpha is small (account for edge distortion). Model 110 in eq based on alpha?
    if cam_num == 1:
        pitch = -cam.pitch - 110
    else:
        pitch = cam.pitch + 110
    yaw = 0
    eulers = [roll, pitch, yaw]

    nadir_vector = euler_to_vector(eulers)

    return nadir_vector, eulers, euler_to_quaternion(*eulers), determine_angle_error(nadir_vector, BODY_DOWN)


def AD_earth_sideways(mag_sat: Magnetorquer_Sat, left_of_cam1: bool):
    '''
    Make an educated nadir guess when both cams see earth at the edge of the image
    '''
    # TODO: take abs(roll), as the sign is likely to switch easily as line becomes vertical
    # Big idea: just power left or right towards earth
    # Move towards cam with larger alpha?
    # Sign of roll in each cam tells us tilt
    if left_of_cam1:
        # print("Earth is to left of cam 1")
        # roll = -90 + 100 * abs(mag_sat.cam1.alpha - mag_sat.cam2.alpha) * 1.5
        roll = -90
        # pitch =  180 + 10 * (mag_sat.cam2.pitch - mag_sat.cam1.pitch)
        pitch = 90
    else:
        # print("Earth is to right of cam 1")
        # roll = 90 - 100 * abs(mag_sat.cam1.alpha - mag_sat.cam2.alpha) * 1.5
        roll = 90
        # pitch = 180 - 10 * (mag_sat.cam1.pitch - mag_sat.cam2.pitch)
        pitch = 90
    yaw = 0
    eulers = [roll, pitch, yaw]

    nadir_vector = euler_to_vector(eulers)
    # print("Vector to nadir (body frame):", nadir_vector)

    error = determine_angle_error(nadir_vector, BODY_DOWN)
    # print("Error: {:.2f} degrees".format(error))

    return nadir_vector, eulers, euler_to_quaternion(*eulers), error


def AD_earth_opposite_corners_vector(mag_sat: Magnetorquer_Sat, left_of_cam1: bool):
    '''
    Simpler opposite-corners solver.
    - Map each camera's measured roll/pitch into body-frame eulers (Maya mapping).
    - Compute circular mean for roll and linear mean for pitch.
    - Enforce roll sign using left_of_cam1 (True -> negative roll, False -> positive roll).
    - Set yaw = 0 and return (nadir_vector, eulers, quaternion, angle_error).
    '''
    cam1_r = mag_sat.cam1.roll
    cam1_p = mag_sat.cam1.pitch
    cam2_r = mag_sat.cam2.roll
    cam2_p = mag_sat.cam2.pitch

    mount_offset = 90 - CAM_MOUNT_ANGLE
    cam1_body_eulers = [cam1_r - mount_offset, cam1_p, 0]
    cam2_body_eulers = [-cam2_r + 180 - mount_offset, -cam2_p, 180]

    def circular_mean_deg(angles):
        rads = np.radians(angles)
        x = np.mean(np.cos(rads))
        y = np.mean(np.sin(rads))
        mean_rad = math.atan2(y, x)
        return math.degrees(mean_rad)

    # mean roll (in degrees, may be outside -180..180)
    mean_roll = circular_mean_deg([cam1_body_eulers[0], cam2_body_eulers[0]])
    # normalize to [-180, 180]
    mean_roll = ((mean_roll + 180) % 360) - 180

    # enforce sign from left_of_cam1: left -> negative roll, right -> positive roll
    mean_roll = -abs(mean_roll) if left_of_cam1 else abs(mean_roll)

    mean_pitch = float(np.mean([cam1_body_eulers[1], cam2_body_eulers[1]]))
    mean_yaw = 0.0

    determined_angles = [mean_roll, mean_pitch, mean_yaw]
    nadir_vector = euler_to_vector(determined_angles)
    q = euler_to_quaternion(*determined_angles)

    return nadir_vector, determined_angles, q, determine_angle_error(nadir_vector, BODY_DOWN)


def determine_angle_error(nadir_vector, body_down):
    '''
    Based on image processing output from our two cameras, find the angle error between body -Z and nadir vector

    @params:
        nadir_vector: the current nadir vector
        body_down: the body -Z vector
    '''
    return math.degrees(math.acos(np.clip(np.dot(body_down, nadir_vector), -1.0, 1.0)))


def nadir_point(error_quat, mag_sat):
    '''
    Proportional-derivative controller to orient us towards a target quaternion
    Seeks to minimize the mangetude of the error quaternion

    @params:
        error_quat (1x4 np.ndarray): error quaternion between current and target nadir
            represents the difference in orientation; [1, 0, 0, 0] meaning that they're aligned
        mag_sat (Magnetorquer_satellite object): encapsulates current state of our cubesat
    @returns:
        voltage_in (1x3 np.ndarray): vector representing voltage we are sending along all three magnetorquers (Volts)
    '''

    # Define the torque using the error quaternion and angular velocity (equation 7.7 from Fund of Spacecraft Att Det)
    #   Derivative term  responds to how fast the error quaternion is changing over time (which is related to how fast we're spinning)
    #   this allows us to anticipate and dampen rapid changes, opposing quick changes and preventing overshooting
    torque = - mag_sat.kp * error_quat[1:4] - mag_sat.kd * mag_sat.w_sat

    # Define the magnetic moment by taking a cross product of the magnetic field with the previously defined torque
    # TODO: only works if they're all orthogonal
    # by doing this, our actual torque will not be aligned with desired torque if they're not orthogonal
    # https://math.stackexchange.com/questions/32600/whats-the-opposite-of-a-cross-product
    m = np.cross( mag_sat.B_body, torque )

    # According to Attitude Control by Magnetic Torquers paper (page 1048)
    m = m / math.pow(np.linalg.norm(mag_sat.B_body), 2)

    # convert magnetic moment to voltage
    voltage = mag_sat.momentToVoltage(m)

    # OR, just scale our max voltage according to the magnitude of the magnetic moment
    # voltage = MAX_VOLTAGE * m_unit

    # OR, compute the scaling factor to make the largest component equal to MAX_VOLTAGE
    # largest_component = np.max(np.abs(m))
    # scaling_factor = MAX_VOLTAGE / largest_component

    # voltage = m * scaling_factor

    # Ensure that we don't exceed power constraint from Nearspace bus (or we get power cycled)
    # P = V^2 / R (watts)
    P_total = np.sum((voltage * voltage) / mag_sat.resistances)
    # Giving some leeway to not exceed in any situation
    if P_total > (MAX_POWER_NADIR * 0.92):
        scale_factor = math.sqrt((MAX_POWER_NADIR * 0.92) / P_total)
        # BRING EVERYTHING DOWN TO FIT IN THE POWER
        voltage *= scale_factor

    return voltage


# OLD VERSION OF NADIR POINTING
'''
def nadir_point(mag_sat):

    Given the current state of the satellite (with image processing already stored in cam1 and cam2 objects),
    try to center our cams and achieve nadir (earth) pointing
    Note: quaternions are based on the center of the image (which is [1, 0, 0, 0])

    @params:
        mag_sat (Magnetorquer_satellite object): encapsulates current state of our cubesat


    # NOTE: cams can be facing different directions because the FOV's overlap

    # do we need to account for yaw sometime? Is that why one axis is always spinning?
    # if roll/pitch flips (cam rotates too much), pd will suddenly try to go other way towards 0 (undoing momentum in had previously?)
    #   want to ensure we're never relying on cam that is rotating too much?
    #   STAY AWAY FROM CAM THAT IS NEAR TRANSITION

    # need to convert voltage to constant frame depending on which cam we're trusting
    # TODO: define which way +x spin with regards to voltage
    #       seems that +x voltage is trying to move up with respect to cam2
    #       seems that -x velocity is downwards with respect to cam2

    # +x and -z voltages are trying to move cam1 up and cam2 down (yayyy)

    # things to test:
    #   Start at different position (to test different constant magnetic fields)
    #   ** Start at nadir and set STANDSTILL = False **
    #   Keep searching if one cam sees space (utilize upside down one by bang-banging?)
    #   Different averages between voltages (easing, more rules for trust)
    #       Trust a cam less as it approaches edge
    #   Try to define 1 quaternion instead of 2
    #   Change bangbang.py to not take cross product--scale max torque instead?

    # create quaternion from first EHS
    roll1 = math.radians(mag_sat.cam1.roll)
    pitch1 = math.radians(mag_sat.cam1.pitch)
    q1 = normalize(euler_to_quat(roll1, pitch1, 0.0))

    # create quaternion from second EHS
    roll2 = math.radians(mag_sat.cam2.roll)
    pitch2 = math.radians(mag_sat.cam2.pitch)
    q2 = normalize(euler_to_quat(roll2, pitch2, 0.0))

    # define target orientation as ~24 degrees pitched up (everything else = 0)
    # if current quat is set to [1, 0, 0, 0], this incites a constant angular y velocity
    # ALPHA$ method (alpha% = 70.2%)
    target_orientation = IDEAL_QUAT

    # try to even the two cams if near nadir pointing
    # target_orientation = normalize(euler_to_quat((roll1+roll2)/2, (pitch1+pitch2)/2, 0.0))
    # "slerp" is a method of getting midpoint of quaternions but couldn't get working
    # target_quaternion = np.quaternion.slerp_evaluate(q1, q2, 0.5)

    # get voltages required to move us towards target quaternion from both cams
    voltage1 = BangBang(q1, target_orientation, mag_sat)

    voltage2 = BangBang(q2, target_orientation, mag_sat)
    # define cam1 as truly aligned: therefore, to turn roll counterclockwise (for this cam) we need a negative current
    # for cam1, a clockwise roll is negative current
    voltage2 *= -1

    # weight 1 = cam1 trusted, weight 0 = cam2 trusted
    weight = 0.5
    # if we see more than this amount of earth, don't trust that cam
    ALPHA_CUTOFF = .9

    # EASE into going back to average?

    # edges are top, right, bottom, left intensities (0-1)
    if (mag_sat.cam1.edges[0] > mag_sat.cam1.edges[2] and mag_sat.cam2.edges[0] < mag_sat.cam2.edges[2]) or (mag_sat.cam1.alpha >= ALPHA_CUTOFF):
        # if first cam is upside down (bottom less than top) and second is not, or first cam is seeing a large amount of earth
        # then trust cam2
        weight = 0
    elif (mag_sat.cam2.edges[0] > mag_sat.cam2.edges[2] and mag_sat.cam1.edges[0] < mag_sat.cam1.edges[2]) or (mag_sat.cam2.alpha >= ALPHA_CUTOFF):
        # if second cam is upside down (bottom less than top) and first is not, or second cam is seeing a large amount of earth
        # then trust cam1
        weight = 1
    # elif (mag_sat.cam2.alpha < mag_sat.cam1.alpha):
        # take the cam that's seeing less of earth (further from danger zones..?)
        # or could take whichever is closer to or further from alpha$...?
        # weight = 0

    # take a weighted sum of the voltages from our 2 cams
    voltage = weight * voltage1 + (1 - weight) * voltage2

    return voltage, weight
'''

'''
BangBang.py
Authors: Peyton Reynolds

Bang Bang algorithm works to minimize error quaternion by writing current to magnetorquers to actuate the satellite


def BangBang (current, target, mag_sat):

    Bang-Bang/proportional-derivative controller to orient us towards a target quaternion

    @params:
        current (np.ndarray): current quaternion of satellite (1x4)
        target (np.ndarray): target quaternion of satellite (1x4)
        mag_sat (Magnetorquer_Sat): object from sat_model.py that represents our satellite and its magnetorquers, including gains and mag specs
    @returns:
        voltage_in (np.ndarray): vector representing voltage we are sending along all three magnetorquers (Volts)


    # Find the error quaternion between current and target quaternion
    # represents the difference in orientation; [1, 0, 0, 0] meaning that they're aligned
    error_quat = delta_q(current, target)

    # Define the torque using the error quaternion and angular velocity (equation 7.7 from Fund of Spacecraft Att Det)
    #   Derivative term  responds to how fast the error quaternion is changing over time (which is related to how fast we're spinning)
    #   this allows us to anticipate and dampen rapid changes, opposing quick changes and preventing overshooting
    torque = - mag_sat.kp * error_quat[1:4] - mag_sat.kd * mag_sat.w_sat

    # find part of torque that is perpendicular to B
    # B_norm_sq = np.dot(B, B)
    # if B_norm_sq == 0:
    #     raise ValueError("Magnetic field vector cannot be zero.")

    # # Project desired torque onto the plane perpendicular to B
    # T_d_perp = T_d - (np.dot(T_d, B) / B_norm_sq) * B

    # # Compute the required magnetic moment
    # m = np.cross(B, T_d_perp) / B_norm_sq

    # Define the magnetic moment by taking a cross product of the magnetic field with the previously defined torque
    # TODO: only works if they're all orthogonal??
    # by doing this, our actual torque will not be aligned with desired torque if they're not orthogonal
    # https://math.stackexchange.com/questions/32600/whats-the-opposite-of-a-cross-product
    m = np.cross( mag_sat.B_body, torque )

    # normalize the magnetic moment so that we're just getting a direction without magnitude
    # m_unit = normalize(m)

    # convert magnetic moment to voltage
    voltage = mag_sat.momentToVoltage(m)

    # OR, just scale our max voltage according to the magnitude of the magnetic moment
    # voltage = MAX_VOLTAGE * m_unit

    # OR, compute the scaling factor to make the largest component equal to MAX_VOLTAGE
    # largest_component = np.max(np.abs(m))
    # scaling_factor = MAX_VOLTAGE / largest_component

    # voltage = m * scaling_factor


    return voltage

'''


# def axisangle_to_q(v, theta):
#     v = normalize(v)
#     x, y, z = v
#     theta /= 2
#     w = math.cos(theta)
#     x = x * math.sin(theta)
#     y = y * math.sin(theta)
#     z = z * math.sin(theta)
#     return w, x, y, z

# def q_conjugate(q):
#     w, x, y, z = q
#     return (w, -x, -y, -z)

# def qv_mult(q1, v1):
#     q2 = np.concatenate(([0.0], v1))
#     return q_mult(q_mult(q1, q2), q_conjugate(q1))[1:]

# def q_mult(q1, q2):
#     w1, x1, y1, z1 = q1
#     w2, x2, y2, z2 = q2
#     w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2

if __name__ == '__main__':
    nadir_point()

def get_unsupported_case_count():
    global unsupported_case_count
    return unsupported_case_count

def reset_unsupported_case_count():
    global unsupported_case_count
    unsupported_case_count = 0
    print("Unsupported case count reset to 0")
