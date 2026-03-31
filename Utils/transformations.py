'''
transformations.py

Various angle, coordinate, matrix, and rotational transformations.

'''

import numpy as np
import math
from scipy.spatial.transform import Rotation as R
import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from params import *

def normalize(v):
    # normalizes the vector v (usuallly a quaternion)
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

def quaternion_rotation_matrix(Q):
    '''
    Covert a quaternion into a full three-dimensional rotation matrix.

    @params
        Q: A 4 element array representing the quaternion (q0,q1,q2,q3)

    @returns
        rot_matrix: A 3x3 element matrix representing the full 3D rotation matrix.
            This rotation matrix converts a point in the local reference
            frame to a point in the global reference frame.
    '''
    # Extract the values from Q
    q0 = Q[0]
    q1 = Q[1]
    q2 = Q[2]
    q3 = Q[3]

    # First row of the rotation matrix
    r00 = 2 * (q0 * q0 + q1 * q1) - 1
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)

    # Second row of the rotation matrix
    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = 2 * (q0 * q0 + q2 * q2) - 1
    r12 = 2 * (q2 * q3 - q0 * q1)

    # Third row of the rotation matrix
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = 2 * (q0 * q0 + q3 * q3) - 1

    # 3x3 rotation matrix
    rot_matrix = np.array([[r00, r01, r02],
                           [r10, r11, r12],
                           [r20, r21, r22]])

    return rot_matrix

def quaternion_difference(q1, q2):
    """
    Calculate the angular difference between two quaternions in degrees

    @params:
        q1, q2: quaternions in [w, x, y, z] format
    @returns:
        angular difference in degrees
    """
    # Convert to scipy format and calculate difference
    r1 = R.from_quat([q1[1], q1[2], q1[3], q1[0]])
    r2 = R.from_quat([q2[1], q2[2], q2[3], q2[0]])

    # Calculate relative rotation
    rel_rot = r1.inv() * r2

    # Get magnitude of rotation
    return np.degrees(rel_rot.magnitude())

def quaternion_multiply(a, b):
    '''
    quaternion_multiply
        custom function to perform quaternion multiply on two passed-in matrices

    @params
        a, b: quaternion matrices (4 x 1) [q0 ; q1:3]
    @returns
        multiplied quaternion matrix [q0 ; q1:3]
    '''

    return np.array([a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
            a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
            a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
            a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0]])

def euler_to_quaternion(roll, pitch, yaw):
    '''
    Convert radians Euler angles (roll, pitch, yaw) to a quaternion in (w, x, y, z) order.

    @params:
        roll: The roll (rotation around x-axis) angle in radians.
        pitch: The pitch (rotation around y-axis) angle in radians.
        yaw: The yaw (rotation around z-axis) angle in radians.

    @returns
        qw, qx, qy, qz: orientation in quaternion [w,x,y,z] format
    '''

    # r = R.from_euler('xyz', [roll, pitch, yaw], degrees=False)
    # quat_scipy = r.as_quat()  # returns [x, y, z, w]
    # return np.array([quat_scipy[3], quat_scipy[0], quat_scipy[1], quat_scipy[2]])  # convert to [w, x, y, z]

    roll, pitch, yaw = roll/2, pitch/2, yaw/2 # all trigs need half angles
    qw = np.cos(roll) * np.cos(pitch) * np.cos(yaw) + np.sin(roll) * np.sin(pitch) * np.sin(yaw)
    qx = np.sin(roll) * np.cos(pitch) * np.cos(yaw) - np.cos(roll) * np.sin(pitch) * np.sin(yaw)
    qy = np.cos(roll) * np.sin(pitch) * np.cos(yaw) + np.sin(roll) * np.cos(pitch) * np.sin(yaw)
    qz = np.cos(roll) * np.cos(pitch) * np.sin(yaw) - np.sin(roll) * np.sin(pitch) * np.cos(yaw)

    # Supposedly comes out normalized
    return np.array([qw, qx, qy, qz])

def angle_to_quaternion(psi):
    ''' Takes in angle psi (Euler angle about Z axis) and returns quaternion
    '''

    # Build rotation matrix (Z rotation matrix)
    Rmat = np.array([[np.cos(psi), -np.sin(psi), 0],
                     [np.sin(psi), np.cos(psi), 0],
                     [0, 0, 1]])

    # Grab components of rotation matrices
    m00 = Rmat[0,0]
    m01 = Rmat[0,1]
    m02 = Rmat[0,2]
    m10 = Rmat[1,0]
    m11 = Rmat[1,1]
    m12 = Rmat[1,2]
    m20 = Rmat[2,0]
    m21 = Rmat[2,1]
    m22 = Rmat[2,2]

    if m00 < -m11:
        t = 1 - m00 - m11 + m22
        quat = np.array([m20 + m02, m12 + m21, t, m01 - m10])
    else:
        t = 1 + m00 + m11 + m22
        quat = np.array([m12 - m21, m20 - m02, m01 - m10, t])

    quat = (0.5/np.sqrt(t)) * quat
    quat = quat / np.linalg.norm(quat)
    #quat[0] = 1

    return quat

def quaternion_to_euler(quat):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)

    Uses ZXY convention to avoid angle wrapping issues for pitch > 90°.
    This ensures that the returned angles match the original Euler angles
    used to create the quaternion, avoiding the mathematical ambiguity
    where (0°,120°,0°) and (180°,60°,180°) represent the same rotation.

    @returns:
        Tuple with:
        roll is rotation around x in degrees (counterclockwise)
        pitch is rotation around y in degrees (counterclockwise)
        yaw is rotation around z in degrees (counterclockwise)
    """
    r = R.from_quat([quat[1], quat[2], quat[3], quat[0]])  # scipy uses [x, y, z, w]

    # Use ZXY convention to preserve original angles better for large pitch values
    try:
        euler_zxy = r.as_euler('zxy', degrees=True)  # returns [yaw, roll, pitch]
        # Reorder to [roll, pitch, yaw] to match expected output format
        euler_deg = np.array([euler_zxy[1], euler_zxy[2], euler_zxy[0]])
    except:
        # Fallback to XYZ if ZXY fails (shouldn't happen but safety first)
        euler_deg = r.as_euler('xyz', degrees=True)  # roll, pitch, yaw in degrees

    return euler_deg
    # w, x, y, z = quat

    # t0 = +2.0 * (w * x + y * z)
    # t1 = +1.0 - 2.0 * (x * x + y * y)
    # roll_x = math.atan2(t0, t1)

    # t2 = +2.0 * (w * y - z * x)
    # t2 = +1.0 if t2 > +1.0 else t2
    # t2 = -1.0 if t2 < -1.0 else t2
    # pitch_y = math.asin(t2)

    # t3 = +2.0 * (w * z + x * y)
    # t4 = +1.0 - 2.0 * (y * y + z * z)
    # yaw_z = math.atan2(t3, t4)

    # return roll_x, pitch_y, yaw_z # in radians

def quaternion_to_angle(q):
    """
    Convert a unit quaternion error to angular error.
    q: quaternion error (difference between current and desired orientation)
    scalar_first: True if q = [w, x, y, z], False if q = [x, y, z, w]
    Returns: angle_degrees
    """

    q = np.asarray(q, dtype=float)
    w = q[0]

    # numerical safety
    w = np.clip(w, -1.0, 1.0)

    angle_rad = 2.0 * np.arccos(abs(w))
    angle_deg = np.degrees(angle_rad)

    return angle_deg

def off_nadir_angle(nadir_body):
    """
    Compute angle between satellite's down axis and true nadir.
    nadir_body: Earth-pointing vector in body frame
    BODY_DOWN: typically [0, 0, -1] for -Z down
    """
    # Normalize vectors
    nadir_normalized = nadir_body / np.linalg.norm(nadir_body)
    body_down_normalized = np.array(BODY_DOWN) / np.linalg.norm(BODY_DOWN)

    # Dot product gives cos(angle)
    cos_angle = np.dot(nadir_normalized, body_down_normalized)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Handle numerical errors

    angle_rad = np.arccos(cos_angle)
    return np.degrees(angle_rad)

def delta_q(q_actual, q_target):
    '''
    delta_q
        Returns error quaternion by taking quaternion product (x)
            between actual quaternion and conjugate of target quaternion.
        Tells us what rotation is needed to reach target

    @params
        q_actual, q_target: normalized (unit) quaternion matrices (1 x 4) [q0, q1:3]
    @returns
        error quaternion: always normalized. equals [1, 0, 0, 0] when q_actual and q_target are equal
    '''

    # because we're using unit quaternions, inverse = conjugate
    # q_actual_inverse = np.array([q_actual[0], -q_actual[1], -q_actual[2], -q_actual[3]])
    q_target_inverse = np.array([q_target[0], -q_target[1], -q_target[2], -q_target[3]])

    q_error = quaternion_multiply(q_actual, q_target_inverse)
    # q_error = quaternion_multiply(q_target, q_actual_inverse)

    # since a quaternion can represent 2 relative orientations, we also want to ensure that the error quaternion is the shortest path
    # from: Quaternion Attitude Control System of Highly Maneuverable Aircraft
    if q_error[0] < 0:
        # if desired rotation is > pi away, then the actual closest rotation is the inverse
        q_error = -q_error

    # error_range = 0.1
    # if np.linalg.norm(q_error[1:4]) < error_range:
        # TODO: if we're close enough to the target, don't waste energy on micro movements?
        #print("close enough")
        # return np.array([1, 0, 0, 0])
    # else:
        #print("error: ", q_error)
        # return q_error

    return q_error

def vector_to_quaternion(v1, v2):
    '''
    Return quaternion that represents rotation from v1 to v2

    @returns:
        quaternion in [w, x, y, z] format
    '''
    dot = np.dot(v1, v2)
    axis = np.cross(v1, v2)

    if np.linalg.norm(axis) < 1e-8:
        # Already aligned or opposite
        if dot > 0.9999:     # aligned in same direction
            q = np.array([1, 0, 0, 0])
        else:                # opposite direction
            q = np.array([0, 1, 0, 0])  # 180 degrees about X
    else:
        w = 1.0 + dot
        q = np.concatenate(([w], axis))
        q /= np.linalg.norm(q)

    return q

def vector_to_euler(v1):
    '''
    Convert a 3D vector to roll, pitch, yaw (degrees)
    Based on down and direction (front) vectors
    '''

    x, y, z = v1[0], v1[1], v1[2]
    # assuming z is "down" axis, y is "side" axis, x is "forward" axis
    roll = math.degrees(math.atan2(y, z))
    pitch = math.degrees(math.atan2(-x, math.sqrt(y**2 + z**2)))
    yaw = 0.0

    return roll, pitch, yaw

def euler_to_vector(eulers_deg):
    """Convert Euler angles (roll, pitch, yaw in degrees) to a nadir vector."""

    roll_rad = math.radians(eulers_deg[0])
    pitch_rad = math.radians(eulers_deg[1])
    yaw_rad = math.radians(eulers_deg[2])
    q = euler_to_quaternion(roll_rad, pitch_rad, yaw_rad)

    # Apply the rotation to BODY_DOWN to see where it points after the rotation
    rotated_down = quaternion_to_vector(q)
    return rotated_down

    # NOTE: must apply rotation in same order as maya (which is xyz)
    eulers_deg = [-angle for angle in eulers_deg]
    r = R.from_euler('xyz', eulers_deg, degrees=True)
    # Apply rotation to our down direction
    vec = r.apply(BODY_DOWN)
    return normalize(vec)

def quaternion_to_vector(quat):
    """Convert quaternion [w, x, y, z] to a nadir vector."""
    r = R.from_quat([quat[1], quat[2], quat[3], quat[0]])  # scipy uses [x, y, z, w]
    # Apply rotation to the nadir vector [0, 0, -1] (down)
    vec = r.apply(BODY_DOWN)
    return vec