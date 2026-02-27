r'''
maya_simulator.py
Author: Andrew Gaylord

Maya Python script that simulates earth horizon sensor images taken from a satellite in Low Earth Orbit (LEO)

Usage:

1. Open Maya

1. Clone IrishSat's Python Simulated Orbital Library (PySOL) and this repo to the scripts folder in Maya.
    Mine looks like this: C:\Users\agaylord\Documents\maya\scripts\cloversat and C:\Users\agaylord\Documents\maya\scripts\PySOL
        git clone https://github.com/ND-IrishSat/PySOL.git
        git clone https://github.com/ND-IrishSat/cloversat.git
    The generate_orbit_data function (PySOL) and everything else can be imported from there

2. The IrishSat_Simulator folder (in the Maya_project folder) is your project folder
    C:\Users\agaylord\Documents\maya\scripts\cloversat\Maya_project\IrishSat_Simulator

4. Open Maya and open the IrishSat project (set project)
    File -> Project Window

5. Open the earth_scene.mb file from the scenes folder

6. Open the script window (look for the semicolon in the bottom right corner) and open the maya_simulator.py file

7. Open VScode and open the \Documents\maya\scripts\cloversat folder
    Turn on autosave
    Install pdf extension (will pop up when you try to open pdf)

8. Import all required libraries to python maya by navigating to C:\Program Files\Autodesk\Maya2025\bin in the command prompt/vscode terminal
    Run .\mayapy -m pip install -r ~/path_to_nearspace/requirements_maya.txt
    Or manually run .\mayapy -m pip install library_name in that directory for each library error that pops up when you try to run the script
        For the cv2 error, restart maya after running .\mayapy -m pip install opencv-python
        For openGL error, run .\mayapy -m pip install PyOpenGL

9. Run the script by clicking the play button in the script window

Edit params.py and nadir_point.py in VSCode to change simulation parameters/control laws
    Make sure you hit play several times (before hitting confirm) to reload the libraries and make sure they are updated

Rendered images are created in project folder -> images
    For example, C:\Users\agaylord\Documents\maya\scripts\cloversat\Maya_project\IrishSat_Simulator\images
To change clip speed in DaVinci resolve while stitching together images:
    "DaVinci resolve" tab => preferences => user => editing => standard still duration

'''

# NOTE: add this to params.py if getting import errors
# if RUNNING_MAYA:
    # sys.path.insert(1, os.path.join(sys.path[0], 'Simulator'))
    # sys.path.insert(1, os.path.join(sys.path[0], 'ukf'))

import maya.cmds as mc
import maya.api.OpenMaya as om

# need to import all libraries (to maya) using
# C:\Program Files\Autodesk\Maya2025\bin>.\mayapy -m pip install numpy
import numpy as np
import math
import time
from importlib import reload, import_module
import math
import cv2
import json

import inspect
from os.path import dirname, realpath
import sys, os

def resetSessionForScript(projectName):
    '''
    Delete all modules loaded under projectName. Replaces importlib.reload
    '''
    toDelete = []
    # Iterate over all the modules that are currently loaded
    for key, module in sys.modules.items():

        # some modules don't like being inspected
        try:
            # Use the "inspect" library to get the moduleFilePath that the current module was loaded from
            moduleFilePath = inspect.getfile(module).lower()

            # mark all modules that are related to our project
            if projectName in key or projectName.lower() in moduleFilePath:
                toDelete.append(key)

        except Exception as e:
            pass

    # delete all custom modules
    for module in toDelete:
        del (sys.modules[module])

# DELETE ALL EVIL MODULES THAT WANT TO OVERSTAY THEIR WELCOME
resetSessionForScript("cloversat")

# import all needed modules (they should all update automatically after being deleted)
p = import_module("cloversat.params")
# reload(p)
thing = import_module("cloversat.Controllers.B_dot")
# reload(thing)
thing = import_module("cloversat.Simulator.graphing")
# reload(thing)
thing = import_module("cloversat.Simulator.saving")
# reload(thing)
thing = import_module("cloversat.Simulator.camera")
# reload(thing)
thing = import_module("cloversat.Controllers.Pointing.image_processing")
# reload(thing)
thing = import_module("cloversat.Controllers.Pointing.nadir_point")
# reload(thing)

mag = import_module("cloversat.Simulator.magnetorquer")
# reload(mag)
sat = import_module("cloversat.Simulator.sat_model")
# reload(sat)
s = import_module("cloversat.Simulator.simulator")
# reload(s)

# import PySOL in specific orderopenGL: PyOpenGL
import PySOL.wmm as wmm
import PySOL.sol_sim
import PySOL.spacecraft as sp
import PySOL.orb_tools as ot

reload(PySOL.sol_sim)


# ============== PARAMETERS =====================================

# array to store all camera objects created
cam_objects = []

# hide the correct groups based on option selected
if p.USING_IR:
    mc.hide(p.SUN_EARTH_GROUP)
    mc.showHidden(p.IR_EARTH_GROUP)
else:
    mc.hide(p.IR_EARTH_GROUP)
    mc.showHidden(p.SUN_EARTH_GROUP)

# ============== FUNCTIONS =====================================

def delete_old():
    '''
    Deletes all leftover "orbit" cubes or "ehs" cams
    '''
    orbit_objects = mc.ls("orbit*")
    if orbit_objects:
        mc.delete(orbit_objects)
    cam_objects = mc.ls("ehs*")
    if cam_objects:
        mc.delete(cam_objects)


# -------------- IO helpers --------------
def _ensure_dir_for_file(path):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def write_metadata_entry(output_dir, entry):
    meta_path = os.path.join(output_dir, p.METADATA_OUTPUT_FILE)
    _ensure_dir_for_file(meta_path)
    with open(meta_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


# -------------- Minimal orientation helpers --------------
def _camera_forward_world(cam_name: str) -> om.MVector:
    """Return the world-space forward (-Z local) unit vector for a camera."""
    mat_list = mc.xform(cam_name, q=True, ws=True, m=True)
    m = om.MMatrix(mat_list)
    xform = om.MTransformationMatrix(m)
    e = xform.rotation(om.MEulerRotation.kXYZ)
    q = e.asQuaternion()
    f_local = om.MVector(0.0, 0.0, -1.0)
    f = f_local.rotateBy(q)
    try:
        f.normalize()
    except Exception:
        f = om.MVector(0.0, -1.0, 0.0)
    return f

def _camera_position_world(cam_name: str) -> om.MVector:
    """Return the world-space position of a camera as a vector."""
    t = mc.xform(cam_name, q=True, ws=True, t=True)
    return om.MVector(float(t[0]), float(t[1]), float(t[2]))

def _angle_between_deg(a: om.MVector, b: om.MVector) -> float:
    """Angle in degrees between two non-zero vectors (robustly clamped)."""
    try:
        a = om.MVector(a.x, a.y, a.z); a.normalize()
    except Exception:
        a = om.MVector(1.0, 0.0, 0.0)
    try:
        b = om.MVector(b.x, b.y, b.z); b.normalize()
    except Exception:
        b = om.MVector(1.0, 0.0, 0.0)
    dot = max(-1.0, min(1.0, a.x*b.x + a.y*b.y + a.z*b.z))
    return math.degrees(math.acos(dot))


def create_gui(default_oe):
    '''
    Creates the graphical interface that allows user to define the orbit
    '''
    window_name = "orbital_elements_window"

    if mc.window(window_name, exists=True):
        mc.deleteUI(window_name)

    mc.window(window_name, title="Orbital Elements Input", widthHeight=(500, 300))
    mc.columnLayout(adjustableColumn=True)

    mc.text(label="Enter Orbital Elements:")
    mc.separator(height=30)

    # True anomaly (degrees)
    true_anomaly_field = mc.floatFieldGrp(label="True Anomaly (degrees):", value1=default_oe[0], columnAlign=(1, "center"), columnWidth=(1, 250))

    # Semi-major axis (km)
    semi_major_axis_field = mc.floatFieldGrp(label="Semi-major Axis (km):", value1=default_oe[1], columnAlign=(1, "center"), columnWidth=(1, 250))

    # Eccentricity
    eccentricity_field = mc.floatFieldGrp(label="Eccentricity:", value1=default_oe[2], columnAlign=(1, "center"), columnWidth=(1, 250), precision=6)

    # Inclination (degrees)
    inclination_field = mc.floatFieldGrp(label="Inclination (degrees):", value1=default_oe[3], columnAlign=(1, "center"), columnWidth=(1, 250))

    # Right Ascension of Ascending Node (degrees)
    ra_ascending_node_field = mc.floatFieldGrp(label="Right Ascension of Ascending Node (degrees):", value1=default_oe[4], columnAlign=(1, "center"), columnWidth=(1, 250))

    # Argument of Perigee (degrees)
    arg_perigee_field = mc.floatFieldGrp(label="Argument of Perigee (degrees):", value1=default_oe[5], columnAlign=(1, "center"), columnWidth=(1, 250))

    mc.separator(height=30)

    # Submit button to apply the orbital elements
    def on_confirm_clicked(*args):
        oe = [
            mc.floatFieldGrp(true_anomaly_field, q=True, value1=True),
            mc.floatFieldGrp(semi_major_axis_field, q=True, value1=True),
            mc.floatFieldGrp(eccentricity_field, q=True, value1=True),
            mc.floatFieldGrp(inclination_field, q=True, value1=True),
            mc.floatFieldGrp(ra_ascending_node_field, q=True, value1=True),
            mc.floatFieldGrp(arg_perigee_field, q=True, value1=True)
        ]
        print("Orbital Elements:", oe)
        # Close the window after applying the orbital elements
        mc.deleteUI(window_name)
        # Call the main function to draw the render the cameras
        main(oe)

    # Confirm button
    mc.button(label="Confirm", command=on_confirm_clicked)

    mc.showWindow(window_name)


def create_dataset(gps, ram, output_dir, earth_object=p.EARTH_OBJECT):
    '''
    Create a labelled dataset by rotating satellite in place

    @params:
        gps (1x3 array): array of gps coordinates to create dataset at
            NOTE: ignored for now lol
        ram (1x3 array): array of ram coordinates to use as up hint
            NOTE: also ignored
        output_dir (string): full path of location to store images in
        earth_object: name of the Earth object in the scene
    '''
    sample_count = 0
    # Store images and truth values in separate folders
    image_output_dir = os.path.join(output_dir, p.DATASET_OUTPUT_DIR)
    metadata_output_dir = os.path.join(output_dir, p.METADATA_OUTPUT_DIR)

    # Initialize the camera position along the -Y axis at the proper radial distance
    # Scale matches elsewhere (km scaled by 0.001 in this scene)
    gps = [0, -p.ORBITAL_ELEMENTS[1] * 0.001, 0]
    ram = [-0.00087425, -0.00130909, -0.0075605]

    for roll in p.SAMPLE_ROLL_DEG:
        for pitch in p.SAMPLE_PITCH_DEG:
            for yaw in p.SAMPLE_YAW_DEG:
                unique_tag = f"s{sample_count}"
                print(f"[render] yaw={yaw} pitch={pitch} roll={roll}")
                # Use the original, proven orientation path: point at Earth with RAM as up,
                # then apply added eulers. This respects CAM_MOUNT_ANGLE and existing math.
                image1_name, image2_name = create_two_cams(
                    gps=gps,
                    curr_quat=None,
                    output_dir=image_output_dir,
                    direction=ram,
                    earth_object=earth_object,
                    tilt=None,
                    eulers=[roll, pitch, yaw],
                    unique_image_tag=unique_tag,
                )
                sample_count += 1
                # Compute yaw-invariant angle to nadir using the acute-angle bisector (craft bottom)
                cam2 = cam_objects[-1]
                cam1 = cam_objects[-2]
                f1 = _camera_forward_world(cam1)
                f2 = _camera_forward_world(cam2)
                # Acute-angle bisector direction (sum of unit forwards)
                bis = om.MVector(f1.x + f2.x, f1.y + f2.y, f1.z + f2.z)
                try:
                    bis.normalize()
                except Exception:
                    bis = om.MVector(0.0, -1.0, 0.0)
                # Nadir from the camera position (use cam1 position; both share same root)
                pos = _camera_position_world(cam1)
                nadir = om.MVector(-pos.x, -pos.y, -pos.z)
                try:
                    nadir.normalize()
                except Exception:
                    nadir = om.MVector(0.0, -1.0, 0.0)
                yaw_invarient_degrees = _angle_between_deg(bis, nadir)
                entry = {
                    "tag": int(sample_count),
                    "yaw": int(round(yaw, p.ANGLE_DECIMAL_PLACES)),
                    "pitch": int(round(pitch, p.ANGLE_DECIMAL_PLACES)),
                    "roll": int(round(roll, p.ANGLE_DECIMAL_PLACES)),
                    "image1": image1_name,
                    "image2": image2_name,
                    "yaw_invarient_degrees": float(round(yaw_invarient_degrees, 6)),
                }
                write_metadata_entry(metadata_output_dir, entry)

                # If we're flat, only do 1 sample of yaw (all others are identical)
                if roll == 0 or roll == 180:
                    break


def quat_rotate(obj, quat, second=None):
    """
    Apply a quaternion rotation to an object in Maya.

    @params:
        obj: cube/cam to be rotated
        quat: quaternion in (w, x, y, z) form
            https://danceswithcode.net/engineeringnotes/quaternions/quaternions.html
            error quat: https://stackoverflow.com/questions/23860476/how-to-get-opposite-angle-quaternion
        second (optional): second ehs cam. They will be tilted by CAM_MOUNT_ANGLE
    """
    # Convert quaternion to MQuaternion object (x, y, z, w)
    q = om.MQuaternion(quat[1], quat[2], quat[3], quat[0])
    q = q.normal()

    # Convert the quaternion to Euler angles
    euler_rotation = q.asEulerRotation()

    # Convert Euler angles to degrees (Maya uses degrees for rotations)
    euler_rotation_degrees = [angle * (180.0 / 3.14159265359) for angle in euler_rotation]
    if second:
        # define the euler rotation that has the cameras face towards each other
        # TODO: are they oriented on correct face (x, y, etc)? Does it matter for our controls?
        cam2_euler = [-euler_rotation_degrees[0] + 180, -euler_rotation_degrees[1], euler_rotation_degrees[2] + 180]

        # apply camera tilt if we're workin with two cam objects
        euler_rotation_degrees[0] -= (90 - p.CAM_MOUNT_ANGLE)
        cam2_euler[0] -= (90 - p.CAM_MOUNT_ANGLE)
        mc.xform(second, rotation=cam2_euler, worldSpace=True)

    # Apply the rotation to the object
    mc.xform(obj, rotation=euler_rotation_degrees, worldSpace=True)


def orient_towards(source, target, ram, second=None, tilt=None, added_eulers=p.DEFAULT_ADDED_EULERS):
    '''
    Orient source object towards target using quaternion rotation
        Bases direction of up upon direction of travel (ram)
        If CAM_MOUNT_ANGLE is not set, adjusts orientation so the object points perfectly towards the horizon
        Otherwise, orient at CAM_MOUNT_ANGLE from nadir (plus added angles)

    @params:
        source: cam object to rotate
        target: object to point towards (usually Earth)
        ram: 1x3 vector of velocity direction (used to define 'up' direction)
        second (optional): second ehs cam object. They will point away from first object
        tilt (1x9 array, optional): added to quaternion rotation matrix to add small random tilt
        added_eulers (1x3 array, optional): rotate both cams by this amount after orienting them towards nadir (degrees)
            NOTE: this depends on where (x,y,z) the cameras are located
            Ex: [90, 0, 0] rotates both cams 90 degrees on x
            roll is subtracted from both Z's, pitch is added/subtracted from both X's, and yaw is added/subtracted from both Y's
    '''

    if tilt is None:
        tilt = np.zeros((9))

    # Get position of two objects (source and target)
    source_pos = mc.xform(source, q=True, ws=True, t=True)
    target_pos = mc.xform(target, q=True, ws=True, t=True)

    # Calculate direction vector from source to target
    vector = om.MVector(target_pos[0] - source_pos[0],
                        target_pos[1] - source_pos[1],
                        target_pos[2] - source_pos[2]).normalize()

    # Set the 'up' direction based on the 'ram' velocity vector
    up_dir = om.MVector(ram[0], ram[1], ram[2]).normalize()
    # print("Normalized direction to face (ram): ", up_dir)

    # Cross product to calculate the right vector (perpendicular to forward and up)
    right = vector ^ up_dir
    right.normalize()

    # Recompute the 'up' vector to make sure it's perpendicular to both the 'vector' and 'right'
    up_dir = right ^ vector
    up_dir.normalize()

    # Create the quaternion rotation matrix for first cam (adding random tilt)
    quat_matrix = om.MMatrix([right.x + tilt[0], right.y + tilt[1], right.z + tilt[2], 0,
                              up_dir.x + tilt[3], up_dir.y + tilt[4], up_dir.z + tilt[5], 0,
                              -vector.x + tilt[6], -vector.y + tilt[7], -vector.z + tilt[8], 0,
                              0, 0, 0, 1])

    # Convert the rotation matrix to Euler angles
    transform = om.MTransformationMatrix(quat_matrix)
    eulers = transform.rotation(om.MEulerRotation.kXYZ)

    # Calculate the vector from the center of the Earth to the source position (assumed Earth's radius = 6.378 km)
    dx = source_pos[0]  # Earth's center assumed at (0,0,0)
    dy = source_pos[1]
    dz = source_pos[2]
    d = math.sqrt(dx**2 + dy**2 + dz**2)

    if not p.CAM_MOUNT_ANGLE:
        # Calculate the angle from the Earth's center to the object's position
        theta_center_to_X = math.acos((p.EARTH_RADIUS * .001) / d)
        # Angle of the line connecting the center to the tangent plane
        tangent_angle = math.pi / 2 - theta_center_to_X

        # Convert the angle to the tangent line to degrees
        # this is the ideal angle our cams would be pointed to directly point at horizon
        angle_degrees = math.degrees(tangent_angle)
    else:
        # record the angle that our cams are mounted at
        angle_degrees = p.CAM_MOUNT_ANGLE

    # adjust the x axis so that the object faces the horizon while still facing ram
    new_angle_x = om.MAngle(eulers.x).asDegrees() + angle_degrees
    # print("new angle_x: ", new_angle_x)

    # Apply the transformation to the source object
    mc.xform(source, rotation=(new_angle_x + added_eulers[1],
                                om.MAngle(eulers.y).asDegrees() + added_eulers[2],
                                om.MAngle(eulers.z).asDegrees() - added_eulers[0]), worldSpace=True)

    if second:
        # set up direction facing opposite direction as ram
        up_dir = om.MVector(-ram[0], -ram[1], -ram[2]).normalize()
        right = vector ^ up_dir
        right.normalize()
        up_dir = right ^ vector
        up_dir.normalize()
        # quat matrix for second cam (subtracting the tilt so it goes opposite way)
        quat_matrix_second = om.MMatrix([right.x - tilt[0], right.y - tilt[1], right.z - tilt[2], 0,
                                up_dir.x - tilt[3], up_dir.y - tilt[4], up_dir.z - tilt[5], 0,
                                -vector.x + tilt[6], -vector.y + tilt[7], -vector.z + tilt[8], 0,
                                0, 0, 0, 1])

        # Convert the rotation matrix to Euler angles
        transform = om.MTransformationMatrix(quat_matrix_second)
        eulers = transform.rotation(om.MEulerRotation.kXYZ)
        # add horizon angle for second cam
        new_angle_x = om.MAngle(eulers.x).asDegrees() + angle_degrees

        mc.xform(second, rotation=(new_angle_x - added_eulers[1],
                                    om.MAngle(eulers.y).asDegrees() - added_eulers[2],
                                    om.MAngle(eulers.z).asDegrees() - added_eulers[0]), worldSpace=True)


def set_cam_fov(cam, horizontal_fov, vertical_fov):
    '''
    Set cam object to have the specified field of view and sensor size
    '''
    aspect_ratio = p.IMAGE_WIDTH / p.IMAGE_HEIGHT

    # set film back -> aperature to sensor size of camera
    mc.setAttr(f'{cam}.horizontalFilmAperture', p.SENSOR_WIDTH / 25.4)
    mc.setAttr(f'{cam}.verticalFilmAperture', p.SENSOR_HEIGHT / 25.4)

    # Convert FOV to focal length
    fov_radians = math.radians(p.CAM_FOV_VERTICAL)
    focal_length = p.SENSOR_HEIGHT / (2 * math.tan(fov_radians / 2))
    # hardcode focal length that gives 110 angle of view (which I think is for entire square then cut short by resolution)
    focal_length = 9.033

    # Set the focal length for the camera
    mc.setAttr(f'{cam}.focalLength', focal_length)


def create_two_cams(gps, curr_quat, output_dir, direction=None, earth_object=None, tilt=None, eulers=p.DEFAULT_ADDED_EULERS, unique_image_tag=None):
    '''
    Create two earth horizon sensors (EHS) at current orientation
    Render their image and return from proper directory
    @params:
        gps (1x3 array): current gps coordinates of satellite
        curr_quat (1x4 array): current orientation of satellite
        output_dir (string): full path of location to store images in
        Optionally for perfect nadir pointing:
            direction (1x3 vector, optional): if provided, always points cams towards perfect nadir
            earth_object (string, optional): name of earth object to point cams towards
            tilt (1x9 array, optional): if provided, adds this tilt to the ideal nadir pointing
            eulers (1x3 array, optional): rotate both cams by this amount after orienting them towards nadir (degrees)
            unique_image_tag (string, optional): tag to add to both image names to make it unique. If this is not passed, pitch and roll are not added to file name
    @returns:
        if not p.CREATE_DATASET:
            image1 (24x32 array): rendered image as loaded by openCV
            image2 (24x32 array): rendered image as loaded by openCV
        else:
            filepath1 (string): file name of first rendered image (stored in output_dir + p.DATASET_OUTPUT_DIR)
            filepath2 (string): file name of second rendered image (stored in output_dir + p.DATASET_OUTPUT_DIR)
    '''

    # create camera and move to current GPS
    mc.camera(name = "ehs")
    mc.move(gps[0], gps[1], gps[2])

    # get created object (name not setting correctly for some reason)
    first_cam = mc.ls(sl=True)[0]
    # add to our list to render later
    cam_objects.append(first_cam)
    # set the FOV of the camera
    set_cam_fov(first_cam, p.CAM_FOV_HORIZONTAL, p.CAM_FOV_VERTICAL)

    # create second cam
    mc.camera(name = "ehs")
    mc.move(gps[0], gps[1], gps[2])
    second_cam = mc.ls(sl=True)[0]
    # add to our list to render later
    cam_objects.append(second_cam)
    set_cam_fov(second_cam, p.CAM_FOV_HORIZONTAL, p.CAM_FOV_VERTICAL)

    # orient our cameras towards current orientation
    if p.ADCS_TYPE != "none" and p.SIMULATING and curr_quat is not None:
        quat_rotate(first_cam, curr_quat, second_cam)
    else:
        # if not simulating, always point satellite towards the earth
        if p.ADCS_TYPE != "none" and tilt is None:
            tilt = np.random.normal(0, .5, 9)
        orient_towards(first_cam, earth_object, direction, second_cam, tilt, eulers)

    if p.RENDER_IMAGES:

        if unique_image_tag is not None:
            # Store pitch/roll angles in image name + unique tag (sample number)
            # common_image_name = f"roll{int(eulers[0])}_pitch{int(eulers[1])}_yaw{int(eulers[2])}_{unique_image_tag}"
            common_image_name = f"{unique_image_tag}_roll{int(eulers[0])}_pitch{int(eulers[1])}_yaw{int(eulers[2])}"

            # cam1_prefix = os.path.join(output_dir, f"cam1_{common_image_name}")
            # cam2_prefix = os.path.join(output_dir, f"cam2_{common_image_name}")
            cam1_prefix = os.path.join(output_dir, f"{common_image_name}_cam1")
            cam2_prefix = os.path.join(output_dir, f"{common_image_name}_cam2")

        else:
            cam1_prefix = os.path.join(output_dir, f"{first_cam}_IR_first")
            cam2_prefix = os.path.join(output_dir, f"{first_cam}_IR_second")

        mc.setAttr("defaultRenderGlobals.imageFilePrefix", cam1_prefix, type="string")
        # render first earth horizon sensor (EHS)
        mc.arnoldRender(camera=first_cam, render=True)

        mc.setAttr("defaultRenderGlobals.imageFilePrefix", cam2_prefix, type="string")
        # render second earth horizon sensor (EHS)
        mc.arnoldRender(camera=second_cam, render=True)

        if not p.CREATE_DATASET:
            # fetch our recently rendered images with openCV
            # Construct the absolute path to the image
            image_path = os.path.join(output_dir, f"{first_cam}_IR_first_1.png")
            # read our image
            image1 = cv2.imread(image_path)
            image_path = os.path.join(output_dir, f"{first_cam}_IR_second_1.png")
            image2 = cv2.imread(image_path)

            return image1, image2

        else:
            # Return the file paths of the rendered images
            return f"cam1_{common_image_name}_1.png", f"cam2_{common_image_name}_1.png"

    else:
        return -1, -1


def place_cam_pair_at_direction_roll(gps, direction_vec, up_hint, roll_deg, output_dir, unique_image_tag=None):
    """
    Place a camera pair so their bisector aligns with the provided direction
    vector and apply an intuitive roll about that bisector. This reuses
    orient_towards/create_two_cams by creating a temporary target transform
    along the desired direction and passing roll via added_eulers.

    Params:
      - gps: world position (km) for the camera roots
      - direction_vec: desired bisector direction (3-vector, world)
      - up_hint: optional world up-hint for basis (passed to orient_towards as 'ram')
      - roll_deg: roll about the bisector (degrees)
      - output_dir: image output directory prefix used by Arnold
      - unique_image_tag: optional unique ID to include in filenames

    Returns: (image1_name, image2_name) file names when creating dataset
    """
    # Normalize direction
    f = om.MVector(direction_vec[0], direction_vec[1], direction_vec[2])
    if f.length() < 1e-9:
        f = om.MVector(0.0, 0.0, -1.0)
    else:
        f.normalize()

    # Create a temporary target object along the given direction
    # so orient_towards can compute a proper basis.
    scale = max(1000.0, float(p.EARTH_RADIUS))  # km scale far enough away
    # Place target opposite the desired direction so -vector in the
    # orientation matrix aligns object forward with +direction.
    target_pos = [gps[0] - f.x * scale, gps[1] - f.y * scale, gps[2] - f.z * scale]
    tmp = mc.spaceLocator(name="ehs_dir_target#")[0]
    mc.move(target_pos[0], target_pos[1], target_pos[2], tmp, absolute=True, worldSpace=True)

    # Ensure up_hint is a valid vector
    if up_hint is None or (abs(up_hint[0]) + abs(up_hint[1]) + abs(up_hint[2])) < 1e-9:
        up_hint = [0.0, 1.0, 0.0]

    # Temporarily disable automatic horizon/mount tilt so bisector == direction
    old_mount = p.CAM_MOUNT_ANGLE
    try:
        p.CAM_MOUNT_ANGLE = 0
        # Map roll to forward-axis roll and split pair by original mount angle via added_eulers pitch
        added = [roll_deg, float(old_mount), 0.0]
        # Reuse existing function to create and orient both cameras
        return create_two_cams(
            gps=gps,
            curr_quat=None,
            output_dir=output_dir,
            direction=up_hint,
            earth_object=tmp,
            tilt=None,
            eulers=added,
            unique_image_tag=unique_image_tag,
        )
    finally:
        # Clean up the temporary target
        if mc.objExists(tmp):
            mc.delete(tmp)
        # Restore mount angle
        p.CAM_MOUNT_ANGLE = old_mount


def main(oe):
    '''
    Given orbital elements and the parameters from file header,
    generate orbital data and draw cameras to simulate EHS readings
    '''

    if not p.RUNNING_MAYA:
        print("ERROR: set params correctly (RUNNING_MAYA must = True)")
        print("Check 3D options section of params.py closely as well.")
        return 1

    if p.CREATE_DATASET and (p.SIMULATING or not p.TWO_CAMS or not p.RENDER_IMAGES or p.CUBES_PATH_NO_CAMS):
        print("ERROR: set params correctly to create dataset")
        print("SIMULATING = False, TWO_CAMS = True, RENDER_IMAGES = True, CUBES_PATH_NO_CAMS = False")
        return 1

    # get gps data in ecef frame from python orbital simulated library
    # also get ram velocity vector for each step (km/s)
    # TODO: add ram to get_orbit_data
    if not p.GENERATE_NEW:
        # If you don't pass oe, it uses saved orbital elements from csv without checking match
        B_earth, gps = PySOL.sol_sim.get_orbit_data(p.B_FIELD_CSV_FILE, GPS=True)
    else:
        B_earth, gps, ram = PySOL.sol_sim.generate_orbit_data(oe, p.HOURS, p.DT, p.B_FIELD_CSV_FILE, store_data=False, GPS=True, RAM=True)
        ram = ram * .001
    if len(B_earth) > int(p.TF / p.DT):
        B_earth = B_earth[:int(p.TF / p.DT)]
    elif len(B_earth) < int(p.TF / p.DT):
        print("ERROR: not enough data points in B_earth. {} needed, {} created".format(int(p.TF/p.DT), len(B_earth)))
        return
    # convert to km
    gps = gps * .001
    if p.STANDSTILL:
        firstPoint = gps[0]
        gps = np.array([firstPoint for _ in gps])

    # initialize current state
    current_state = np.zeros((p.STATE_SPACE_DIMENSION))
    # set arnold renderer and different settings
    # find output directory (current project folder -> images)
    project_path = mc.workspace(query=True, rootDirectory=True)
    output_dir = os.path.join(project_path, "images")
    mc.setAttr("defaultRenderGlobals.currentRenderer", "arnold", type="string")
    mc.setAttr("defaultArnoldDriver.ai_translator", "png", type="string")
    mc.setAttr("defaultResolution.width", max(p.IMAGE_WIDTH, p.IMAGE_HEIGHT))
    mc.setAttr("defaultResolution.height", max(p.IMAGE_WIDTH, p.IMAGE_HEIGHT))

    if p.IMAGE_HEIGHT != p.IMAGE_WIDTH:
        # crop our image to specified resolution
        mc.setAttr("defaultRenderGlobals.useRenderRegion", 1)
        # change sides to make 24 pixels horizontally (hopefully 70 FOV)
        mc.setAttr("defaultRenderGlobals.leftRegion", (p.IMAGE_HEIGHT-p.IMAGE_WIDTH) / 2)
        mc.setAttr("defaultRenderGlobals.rightRegion", p.IMAGE_HEIGHT - (p.IMAGE_HEIGHT-p.IMAGE_WIDTH) / 2 - 1)
        # don't change top/bottom to keep 110 FOV
        mc.setAttr("defaultRenderGlobals.bottomRegion", 0)
        mc.setAttr("defaultRenderGlobals.topRegion", p.IMAGE_HEIGHT-1)
    else:
        # undo cropping for when we want non-ir or square pics
        mc.setAttr("defaultRenderGlobals.useRenderRegion", 0)

    if p.CREATE_DATASET:
        output_dir = os.path.join(project_path, "images")
        create_dataset(gps[0], ram[0], output_dir, p.EARTH_OBJECT)
        return

    if p.SIMULATING:
        # create 3 Magnetorquer objects to store in Magnetorquer_Sat object
        mag1 = mag.Magnetorquer(n = p.FERRO_NUM_TURNS, area = p.FERRO_AREA, k = p.K, epsilon = p.FERRO_EPSILON)
        mag2 = mag.Magnetorquer(n = p.FERRO_NUM_TURNS, area = p.FERRO_AREA, k = p.K, epsilon = p.FERRO_EPSILON)
        # mag3 = mag.Magnetorquer(n = FERRO_NUM_TURNS, area = FERRO_AREA, k = K, epsilon = FERRO_EPSILON)
        mag3 = mag.Magnetorquer(n = p.AIR_NUM_TURNS, area = p.AIR_AREA, k = p.K, epsilon = 1)
        mag_array = np.array([mag1, mag2, mag3])

        # initialize object to hold satellite properties
        mag_sat = sat.Magnetorquer_Sat(p.CUBESAT_BODY_INERTIA, mag_array, p.VELOCITY_INITIAL, p.CONSTANT_B_FIELD_MAG, np.array([0.0, 0.0, 0.0]), p.DT, p.GYRO_WORKING, p.KP, p.KD)

        # run simulation from simulator.py and generate pdf report of results
        sim = s.Simulator(mag_sat, B_earth)

    for i, element in enumerate(gps):

        # don't exceed simulation time, even if we have more gps data
        if (p.SIMULATING and i >= sim.n):
            break

        # sim starts on i = 1
        if (p.SIMULATING and i == 0):
            continue

        # protocal that replaces run_b_dot_sim for ehs simulator
        if p.SIMULATING:
            # generate ideal state based on last so that we can better estimate sensor data
            # ideal_state = sim.find_ideal(i)

            # generate fake sensor data in body frame based on ideal guess
            sim.generateData_step(sim.states[i - 1], i)

            if not p.CUBES_PATH_NO_CAMS and i % p.PIC_INTERVAL == 0:

                # direction that our cam should be oriented
                if p.ADCS_TYPE == "none":
                    direction = ram[i]
                    earth_object = p.EARTH_OBJECT
                else:
                    direction = None
                    earth_object = None

                # generate ehs, render image, and fetch from dir
                image1, image2 = create_two_cams(element, sim.states[i - 1][:4], output_dir, direction, earth_object)

                if p.RENDER_IMAGES and p.USING_IR:
                    # process our images and store results in mag_sat
                    sim.process_images(image1, image2, i)

            # find how far we are from nadir (for controller)
            sim.findTrueNadir(sim.states[i - 1], gps[i], i)

            # check what protocol we should be in and update state
            sim.mag_sat.state = sim.check_state(i)

            # decide voltage for self.voltages[i] (depending on state)
            sim.controls(i)

            # propagate based on voltages[i]
            current_state = sim.propagate_step(i)
            # print("current state: ", current_state)

            # calculate total power usage for this time step (Watts)
            sim.totalPower[i] = sim.power_output[i][0] + sim.power_output[i][1] + sim.power_output[i][2]

        if p.CUBES_PATH_NO_CAMS and i % (p.DT * 3000) == 0: # 200 for dt = .5
            # generate cubes every so often that show orbit
            mc.polyCube(name = "orbit" + str(i))
            mc.move(element[0], element[1], element[2])
            mc.scale(.3,.3,.3)
            if p.SIMULATING:
                quat_rotate("orbit" + str(i), current_state[:4])

    if p.RENDER_IMAGES and not p.SIMULATING:
        # render all cameras that we created
        print("render every ", p.PIC_INTERVAL, " frames")

        # create output directory
        project_path = mc.workspace(query=True, rootDirectory=True)
        output_dir = os.path.join(project_path, "images")

        # set arnold renderer and different settings
        mc.setAttr("defaultRenderGlobals.currentRenderer", "arnold", type="string")
        mc.setAttr("defaultArnoldDriver.ai_translator", "png", type="string")
        mc.setAttr("defaultResolution.width", max(p.IMAGE_WIDTH, p.IMAGE_HEIGHT))
        mc.setAttr("defaultResolution.height", max(p.IMAGE_WIDTH, p.IMAGE_HEIGHT))

        for i, cam in enumerate(cam_objects):
            # set file name and render for every cam we stored
            render_prefix = os.path.join(output_dir, f"{cam}")
            if p.USING_IR:
                render_prefix = os.path.join(output_dir, f"{cam}_IR")

            if p.IMAGE_HEIGHT != p.IMAGE_WIDTH:
                # crop our image to specified resolution
                mc.setAttr("defaultRenderGlobals.useRenderRegion", 1)
                # change sides to make 24 pixels horizontally (hopefully 70 FOV)
                mc.setAttr("defaultRenderGlobals.leftRegion", (p.IMAGE_HEIGHT-p.IMAGE_WIDTH) / 2)
                mc.setAttr("defaultRenderGlobals.rightRegion", p.IMAGE_HEIGHT - (p.IMAGE_HEIGHT-p.IMAGE_WIDTH) / 2 - 1)
                # don't change top/bottom to keep 110 FOV
                mc.setAttr("defaultRenderGlobals.bottomRegion", 0)
                mc.setAttr("defaultRenderGlobals.topRegion", p.IMAGE_HEIGHT-1)

            mc.setAttr("defaultRenderGlobals.imageFilePrefix", render_prefix, type="string")
            # render every earth horizon sensor (EHS)
            mc.arnoldRender(camera=cam, render=True)

        print("All scenes rendered correctly to project folder->images!")

    if p.SIMULATING:
        sim.plot_and_viz_results()

# delete all cube and cam objects from previous iterations
delete_old()

# create the gui (which calls main when "confirm" button is clicked)
create_gui(p.ORBITAL_ELEMENTS)

