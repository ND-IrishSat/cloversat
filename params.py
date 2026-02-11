'''
params.py

Reference file that contains all important variables for our system
Here, all factors that we would want to change are easily accessible
'''

import numpy as np
import math
import sys, os, inspect

frame = inspect.currentframe()
here = os.path.abspath(os.path.dirname(inspect.getfile(frame)))

if here not in sys.path:
    sys.path.insert(0, here)

from Simulator.gui import GUI

DEGREES = False

# ============  INITIAL VALUES  ======================================

QUAT_INITIAL = np.array([1.0, 0.0, 0.0, 0.0])
# degrees/s. Detumbling starts at ~15 deg/s
VELOCITY_INITIAL = np.array([0.0, 0.0, 0.0])
# convert to rad/s
if not DEGREES:
    VELOCITY_INITIAL *= math.pi / 180
RW_CURRENTS_INITIAL = np.array([0.0, 0.0, 0.0, 0.0])
RW_VOLTAGES_INITIAL = np.array([0.0, 0.0, 0.0, 0.0])
MAG_CURRENT_INITIAL = np.array([0.0, 0.0, 0.0])
MAG_VOLTAGE_INITIAL = np.array([0.0, 0.0, 0.0])
RW_INITIAL = np.array([0.0, 0.0, 0.0, 0.0])
STARTING_PROTOCOL = "target_point" # "detumble", "point", "target_point", "idle"
PROTOCOL_MAP = {"demagnetize": -2, "detumble": -1, "idle": 0, "point": 1, "target_point": 2}

# ============  ORBITAL DYNAMICS  ==================================================

EARTH_RADIUS = 6378
# see generate_orbit_data in sol_sim.py for details on OE args
# sun-synchronous LEO orbit
ORBITAL_ELEMENTS = np.array([180, EARTH_RADIUS + 450, 0.0005, 98.06, 105, 0.001])
# Starts us at nadir (with quat = 0.707, 0.0, 0.0, -0.707)
# ORBITAL_ELEMENTS = np.array([0, EARTH_RADIUS + 450, 0.0005, 90.0, 60, 0.001])

# standard gravitational parameter for earth (m^3/s^2)
GRAVITY_EARTH = 3.986004418e14

# time for one orbit from function (hours)
TIME_PER_ORBIT = 1.550138888888889 # 1.525 from trial/error
# orbital period (time to complete one orbit) according to Kepler's 3rd law (seconds)
ORBITAL_PERIOD = 2 * np.pi * math.sqrt((ORBITAL_ELEMENTS[1] * 1000)**3 / GRAVITY_EARTH)

# Our North West Up true magnetic field [micro Teslas] and GPS data in Stinson Remick
CONSTANT_B_FIELD_MAG = np.array([19.42900375, 1.74830615, 49.13746833])
curr_date_time= np.array([2024.1066])
lat = np.array([41.675])
long = np.array([-86.252])
alt = np.array([225.552]) # 740 feet (this is in meters)
CURRENT_GPS_DATA = np.array([lat, long, alt, curr_date_time])

EARTH_MAGNETIC_FIELD_LEO = 30e-6  # Average magnetic flux density in LEO [T]

# ============  SIM OPTIONS  ==============================================================

# total time to run sim (unrounded hours)
# HOURS = ORBITAL_PERIOD / 3600
HOURS = 0.06
print("simulation time: ", round(HOURS, 6), "hours")
# total time to run sim (seconds)
TF = int(HOURS * 3600)
# time step (how long between each iteration)
DT = .1
SOLVER_METHOD = "rk4" # "rk4", "euler"
# threshold for when we consider our satellite detumbled (degrees/s)
DETUMBLE_THRESHOLD = 0.5
# convert to rad/s
if not DEGREES:
    DETUMBLE_THRESHOLD *= math.pi / 180

# bitmask that represents whether we have wheels on x, y, z, or skew axes
RW_AXES = np.array([1, 1, 1, 0])
MAG_AXES = np.array([0, 0, 0])
# bitmask for which axes we can rotate about
FREEDOM_OF_MOVEMENT_AXES = np.array([1, 1, 1])

STATE_SPACE_DIMENSION = 7
MEASUREMENT_SPACE_DIMENSION = 6

# whether to generate new pySOL data or not
GENERATE_NEW = True
# csv to get pre-generated pysol b field from
B_FIELD_CSV_FILE = "leo_oe_10.csv"
# B_FIELD_CSV_FILE = "1_and_half_orbit.csv" # .05 timestep
# B_FIELD_CSV_FILE = "1_and_half_orbit_quarter.csv" # .025 timestep
# B_FIELD_CSV_FILE = "1_orbit.csv" # .1 timestep

# if false, use PySOL to calculate orbital magnetic field
CONSTANT_B_FIELD = True
SENSOR_NOISE = True
STANDSTILL = False # keep satellite in same position around the earth

# if true, turn off our magnetorquers to take magnetometer readings
ACCURATE_MAG_READINGS = False
# how often we want to take a new magnetometer reading (seconds)
MAG_READING_INTERVAL = .45
# how long to turn off our torquers for before taking reading (seconds)
TORQUER_OFF_TIME = .25
assert(MAG_READING_INTERVAL <= np.pi / (2 * (MAG_READING_INTERVAL - TORQUER_OFF_TIME) * 0.26))

# How often to run our controls loop (seconds)
NADIR_INTERVAL = 2.0

# Whether to run our unscented kalman filter or not
RUN_UKF = False
# for simple 1D testbed validation (with a suboption for detumble)
RUNNING_1D = False
# whether you're running in Debart with 3D rendering (check 3D settings at bottom if so)
RUNNING_MAYA = False
# 0 = only create pdf output, 1 = show 3D animation visualization, 2 = both, 3 = none
RESULT = 0
OUTPUT_DIR = "plotOutput"
OUTPUT_FILE = "output.pdf"

# time array for our graphs (hours)
TIME_GRAPHING_ARRAY = np.arange(0, TF, DT)
TIME_GRAPHING_ARRAY = TIME_GRAPHING_ARRAY / 3600

# set gyroscope to working (true) or not working (false)
GYRO_WORKING = True
MAG_READINGS_STORED = 5

# if SENSOR_DATA_FILE points to a valid file, we are using data stored in that file
# if SENSOR_DATA_FILE is None, we are using live data from the sensors
SENSOR_DATA_FILE = "data.txt"


# =======  UKF  =================================================

# should be true if not doing controls
RUN_STATISTICAL_TESTS = False

COVARIANCE_INITIAL_MAG = 5e-7

# Filter process noise Q represents uncertainty in our state transition model
# Higher values mean we trust our model less and measurements more
PROCESS_NOISE_MAG = 0.00001
PROCESS_NOISE_K = 10

# Filter measurement noise R
# Higher values mean we trust our measurements less
MEASUREMENT_MAGNETOMETER_NOISE = 0.001
MEASUREMENT_GYROSCOPE_NOISE = 0.01

# scaling parameters used to calculate scaling factor and sigma point weights
# alpha and k scale points around the mean. To capture the kurtosis of a gaussian distribution, a=1 and k=3-n should be used
#   If a decrease in the spread of the SPs is desired, use κ = 0 and alpha < 1
#   If an increase in the spread of the SPs is desired, use κ > 0 and alpha = 1
ALPHA = 0.001
K_SCALING = 0
# beta minimizes higher order errors in covariance estimation
BETA = 2

# =======  PHYSICS  ================================================

# moment of inertia tensor of 2U CubeSat (w/o reaction wheel inertias) (kg m^2)
CUBESAT_BODY_INERTIA = (1e-7) * np.array([[46535.388, 257.834, 536.12],
                                          [257.834, 47934.771, -710.058],
                                          [536.12, -710.058, 23138.181]])
CUBESAT_BODY_INERTIA_INVERSE = np.linalg.inv(CUBESAT_BODY_INERTIA)

# Moments of Inertia of reaction wheels [g cm^2] - measured
Iw1 = (1/2)*38*1.8**2 # I_disc = 1/2 * M * R^2
Iw2 = Iw1
Iw3 = Iw1
Iw4 = Iw1
# Moment of inertia tensor of rxn wheels [kg m^2]
# this gets multiplied by SPIN_AXIS_INERTIA during EOMs calculation
RW_CONFIG_INERTIA = np.array([[Iw1, 0, 0, 0],
                              [0, Iw2, 0, 0],
                              [0, 0, Iw3, 0],
                              [0, 0, 0, Iw4]])

# Principal moment of inertia for reaction wheels about spin axis and about axis transverse to spin axis, respectively
RW_SPIN_AXIS_INERTIA = 5.1e-7 # 1e-7 also mentioned
# TODO: what does this represent??
RW_TRANVERSE_AXIS_INERTIA = 0.0

# Calculate contributions of reaction wheel to moment of inertia tensor due to principal moment transverse to the spin axis
# wtf does this mean?????
for i in np.arange(RW_CONFIG_INERTIA.shape[1]):
    CUBESAT_BODY_INERTIA = CUBESAT_BODY_INERTIA + RW_TRANVERSE_AXIS_INERTIA*(np.identity(3) - np.matmul(RW_CONFIG_INERTIA[:, i], np.transpose(RW_CONFIG_INERTIA[:, i])))

# Transformation matrix for NASA config given in Fundamentals pg 153-154
TRANSFORMATION = np.array([[1, 0, 0, 1/np.sqrt(3)],
                           [0, 1, 0, 1/np.sqrt(3)],
                           [0, 0, 1, 1/np.sqrt(3)]])

# =======  CONTROLS  =======================================

# target orientation for if we're simulating controls
TARGET = np.array([1.0, 0.0, 1.0, 0.0])

# Quaternion error tolerances define how close we need to be to our target orientation
QUAT_ERROR_TOLERANCE = 0.01  # Maximum acceptable quaternion error magnitude
                            # 0.01 ≈ 1.15 degrees of rotation error
                            # sqrt(1 - cos(theta/2)) for small angles
ANGULAR_RATE_TOLERANCE = 0.001  # rad/s, maximum acceptable angular rate when "settled"

# PID controller
# .001 was flashed for both kp and kd on firmware
KP = 1e-3                   # Proportional gain
# close to kp allows for narrowing in on target, but not too close
# smaller = oscillating more frequently, larger = overshooting more
KI = 1e-4                   # Integral gain
# if this is too high, it overrotates
KD = 1e-3                   # Derivative gain

if GYRO_WORKING:
    K = 1.25e-4 # detumble gain on firmware
    # proposed value for k for b-cross algorithm
    # k = 2 * 1/(rad(a^3/GM))(1+sinζ)λmin  "This is bcross" - also andrew
    # K = 2 * ((ORBITAL_ELEMENTS[1] * 1000 ) ** 3 / (GRAVITY_EARTH))**(-0.5) * (1 + math.sin(INCLINATION_RAD)) * min(CUBESAT_eigenvalues)
else:
    # proportional gain for B_dot without gyro
    K = 5e-5 # .275e4 (non-time-varying) 7.5e-6 (varying)
    # Avanzini and Giulietti (https://arc.aiaa.org/doi/10.2514/1.53074):
    #     k =2n(1+sinζ)Imin "this is bdot"-andrew
    #     GM - grav constant, a - orbit semi-major axis, lambda - J minimum eigenvalue, n is mean motion of satellite,
    #     ζ is the inclination of the orbit with respect to the geomagnetic equator, and Imin is the value of minimum moment of inertia.
    # INCLINATION_RAD = math.radians(ORBITAL_ELEMENTS[3]) # inclination of orbit (radians)
    # CUBESAT_eigenvalues, CUBESAT_eigenvectors = np.linalg.eig(CUBESAT_BODY_INERTIA)
    # I_min = np.min(np.diagonal(CUBESAT_BODY_INERTIA))
    # MEAN_MOTION = 2 * np.pi / ORBITAL_PERIOD # average angular velocity over one orbit
    # K = 2 * MEAN_MOTION * (1 + math.sin(INCLINATION_RAD)) * I_min
    # ORBIT_RATE = math.sqrt(GRAVITY_EARTH / (ORBITAL_ELEMENTS[1] * 1000)**3) # orbital angular rate (rad/s): how fast it orbits earth
    # K = 2 * ORBIT_RATE * (1 + math.sin(INCLINATION_RAD)) * I_min

# =======  SENSORS  ==================================================

# noise sd = noise density * sqrt(sampling rate)
# vn100 imu sampling rate from user manual = 200 Hz

# mag noise density from vn100 website = 140 uGauss /sqrt(Hz)
# TODO: confirm these noise densitites
SENSOR_MAGNETOMETER_SD = (140 * 10e-6) * np.sqrt(200)
# SENSOR_MAGNETOMETER_SD *= 50

# gyro noise density from vn100 website = 0.0035 degree/s /sqrt(Hz)
SENSOR_GYROSCOPE_SD = 0.0035 * np.sqrt(200)
# SENSOR_GYROSCOPE_SD *= 50
if not DEGREES:
    SENSOR_GYROSCOPE_SD *= (np.pi / 180)

# =======  REACTION WHEELS  ================================================

# Pulse Width Modulation (PWM) signal that generates the max speed in our motors
MAX_PWM = 65535
# (TODO) max torque that our wheels can handle (Nm)
MAX_RW_TORQUE = 0.02

# motor model parameters (Maxon DCX 8 M (9 volts)) used for controls sim
# TODO: find accurate numbers for these
MAX_CURRENT_RW = 1
MIN_CURRENT_RW = -1
THERMAL_RESISTANCE = 0.01  # °C per A^2 (or Kelvin per A^2). how much the current flowing through the system causes heat generation
COOLING_CONSTANT = 0.1     # 1/s (rate of cooling). how quickly the temperature difference between the system and its surroundings dissipates
WHEEL_COUPLING_FACTOR = 0.5  # coupling between ambient and reaction wheel temperature

# external torque on the wheels
RW_EXTERNAL_TORQUE = np.array([0.0, 0.0, 0.0, 0.0])

RWA = 3.54      # Ohms, winding resistance at ambient temperature
LW = 0.424e-3  # Henry
KT = 8.82e-3   # Torque constant Nm/A
KV = KT    # Voltage constant V*s/rad
JM = 5.1*(1e-7)   # Kg m^2
BM = 3.61e-6   # [N·m·s/rad] Viscous friction
ALPHA_CU = 0.00393 # copper's temperature coefficient [1/K]
Rha = 16.5      # K/W
Rwh = 2.66     # K/W
Cwa = 2.31/Rwh     # Thermal Capacitance
Cha = 162/Rha      # Thermal Capacitance

# =======  MAGNETORQUERS  ============================================

# Max power for nadir pointing mode (W)
MAX_POWER_NADIR = 0.5

# Max torque for both Torquers
MAX_VOLTAGE_MAG = 5  # Maximum voltage [V]
# what voltage to shoot as a burst to demagnitize core
DEMAGNITIZING_VOLTAGE = 0
# we have ~1 amp total between all torques
MAX_CURRENT_MAG = 0.4  # Maximum current [A]
# .4 = max for single torquer (5 volts / 12 ohms)
# we can pull max 1 amp from NSL for all torquers
RESISTANCE_MAG = 12 # Resistance [Ohm]
INDUCTANCE_MAG = 146 # Inductance [H]

#===============  AIRCORE TORQUER  =====================================================

# Magnetorquer geometry (rectangular air core)
AIR_NUM_TURNS = 341  # green expiremental green air toruqer (from robby)
AIR_AREA = 0.007901 # Area of magnetorquer [m^2]

# want 60 mA current at max voltage
# resistance controls the max current
AIR_RESISTANCE_MAG = 41.9 # (Ohms)
# THIS CONTROLS RATE OF CHANGE OF CURRENT (lower = lower time constant/charging speed)
AIR_INDUCTANCE_MAG = 8 # Inductance [H]

AIR_MAX_TORQUE = 4.997917534360683e-05 # N·m
AIR_MAX_MAGNETIC_MOMENT = AIR_NUM_TURNS * AIR_AREA * MAX_CURRENT_MAG # A * m^2

# ================  MUMETAL TORQUER  ======================================================

RELATIVE_PERM_MM = 80000  # Relative permeability of MuMetal (between 80,000 and 100,000)
FERRO_LENGTH = 7 # length of the rod [cm]
FERRO_ROD_RADIUS = 0.32 # Core rod radius [cm]
# FERRO_NUM_TURNS = 1845 # number of turns of coil
FERRO_NUM_TURNS = 2110 # expiremntal ferro torquer (from robby)
FERRO_AREA = np.pi * (FERRO_ROD_RADIUS / 100)**2 # Area of magnetorquer [m^2]

# want 225 mA current at max voltage
FERRO_RESISTANCE_MAG = 20.5 # (Ohms)
# THIS CONTROLS RATE OF CHANGE OF CURRENT (lower = lower time constant/charging speed)
FERRO_INDUCTANCE_MAG = 2 # Inductance [H]

# taken from Sarah's optimizing code (with 80000 permeability)
FERRO_MAX_TORQUE = 3.185e-5 # n*m

# equations for relative permeability and demagnetizing factor from fullcalcs.py
FERRO_RADIUS = FERRO_LENGTH/FERRO_ROD_RADIUS # length-to-radius ratio of the cylindrical magnetorquer
FERRO_DEMAG_FACTOR = (4 * np.log(FERRO_RADIUS - 1)) / (FERRO_RADIUS * FERRO_RADIUS - 4 * np.log(FERRO_RADIUS))
# theorized epsilon should be between 100 and 300
# we can tweak our torque output by changing this value
FERRO_EPSILON = ( 1 + (RELATIVE_PERM_MM - 1) ) / (1 + FERRO_DEMAG_FACTOR * (RELATIVE_PERM_MM-1))
# alternate method by "Attitude Control by Magnetic Torquer"
# ferro_epsilon = 1 / (1/rel_perm + (((2*core_radius)**2) / (core_length**2))*(np.log(2*core_length/core_radius) - 1))
# print("Epsilon (ferro core magnetizing factor): ", FERRO_EPSILON)

FERRO_MAX_MAGNETIC_MOMENT = FERRO_NUM_TURNS * FERRO_AREA * MAX_CURRENT_MAG * FERRO_EPSILON # A * m^2

# =======  SIMPLE 1D TEST  =================================================

if (RUNNING_1D):

    # bitmask that represents which axis our torquer is mounted upon
    MAG_AXES = np.array([1,1,1])
    # bitmask for orientation of table: only movement along z axis should be allowed
    FREEDOM_OF_MOVEMENT_AXES = np.array([0,0,1])
    # if true, replace proper torquer with aircore
    TEST_AIRCORE = False

    # TODO: find inertia of bowling ball testbed
    #inertia of bowling ball testbed in (the np array is in kg m^2)
    TESTBED_INERTIA = (0.00029263965) * np.array([[-0.41, 0.00, 0.91, 
                                                   0.91, 0.00, 0.41, 
                                                  0.00, 1.00, 0.00]])
    
    
    # CUBESAT_BODY_INERTIA = ???
    # CUBESAT_BODY_INERTIA_INVERSE = np.linalg.inv(CUBESAT_BODY_INERTIA)

    VELOCITY_INITIAL = np.array([0.0, 0.0, 0.0])*FREEDOM_OF_MOVEMENT_AXES
    # what magnetic moment to create along each axis (should only be 1 axis)
    if TEST_AIRCORE:
        #using maxing to return an array
        DESIRED_MAGNETIC_MOMENTS = AIR_MAX_MAGNETIC_MOMENT*MAG_AXES
    else:
        DESIRED_MAGNETIC_MOMENTS = FERRO_MAX_MAGNETIC_MOMENT*MAG_AXES
    DESIRED_ANGLE = np.array([-90, 0, 0]) # desired angle for x axis

    # ==========  1D DETUMBLE/GUI  =====================================
    DETUMBLE_1D = True
    GUI_ON = True

    if DETUMBLE_1D and not GUI_ON:
        VELOCITY_INITIAL = np.array([0.0, 0.0, 20.0])
        HOURS = 25 / 60 # simulation time in hours
    DETUMBLE_THRESHOLD_1D = 0.2
    # 3e-3 for fast detubmle (< 100 seconds)
    K = 1.25e-4

    if GUI_ON:
        gui = GUI(None)
         # gui.title('Test')
        gui.mainloop() #this will run until it closes
        gui.velocity = float(gui.velocity)
        if (gui.axes == "x"):
            MAG_AXES = np.array([1,0,0])
       # VELOCITY_INITIAL = np.array([gui.velocity, 0.0, 0.0])*FREEDOM_OF_MOVEMENT_AXES

        elif (gui.axes == "y"):
            MAG_AXES = np.array([0,1,0])
        #VELOCITY_INITIAL = np.array([0.0, gui.velocity, 0.0])

        else:
            MAG_AXES = np.array([0,0,1])
        VELOCITY_INITIAL = np.multiply(np.array([0.0, 0.0, gui.velocity]),FREEDOM_OF_MOVEMENT_AXES)
        gui.time = float(gui.time)
        HOURS = gui.time / 3600 # simulation time in hours
        print(gui.time,gui.velocity,gui.axes)

    TF = int(HOURS * 3600)
    TIME_GRAPHING_ARRAY = np.arange(0, TF, DT)
    TIME_GRAPHING_ARRAY = TIME_GRAPHING_ARRAY / 3600
    # CUBESAT_BODY_INERTIA = TODO: find real inertia of cardboard testbed
    # CUBESAT_BODY_INERTIA_INVERSE = np.linalg.inv(CUBESAT_BODY_INERTIA)

    # other default params we want to always set for 1D
    if GUI_ON:
        # show 3D animation if doing gui (probably showcasing)
        RESULT = 2
    ACCURATE_MAG_READINGS = False

    # convert to rad/s
    if not DEGREES:
        VELOCITY_INITIAL *= math.pi / 180
        DETUMBLE_THRESHOLD_1D *= math.pi / 180

# ================  3D OPTIONS  ======================================================

# whether to run physics simulator or not
SIMULATING = True
# option to only highlight orbit path with new cams + images
CUBES_PATH_NO_CAMS = False
RENDER_IMAGES = True
TWO_CAMS = True
# whether to use our AD vs perfect knowledge vs always point towards nadir (no noise) (no controls)
ADCS_TYPE = "perfect" # "AD", "perfect", "none"
# Keep sat still and get images from rotating in place
CREATE_DATASET = False
DEFAULT_ADDED_EULERS = [0,0,0] # default extra euler angles to add to each cam (roll, pitch, yaw)
SAMPLE_YAW_DEG   = [0]  # yaw sweep (about global up)
SAMPLE_PITCH_DEG = list(np.arange(0, 360, 30))  # pitch (about global right)
SAMPLE_ROLL_DEG  = list(np.arange(0, 360, 30))  # roll (about global forward)
DATASET_OUTPUT_DIR = "dataset"
METADATA_OUTPUT_DIR = "metadata"
METADATA_OUTPUT_FILE = "metadata.jsonl"
ANGLE_DECIMAL_PLACES = 5 # how much to round for stored data entry
# whether to render as color cam or ir cam (hides correct group)
USING_IR = True
# how many pairs of EHS images to create (roughly, depends on timestep)
PIC_COUNT = 10
# how often to space cams along orbit
PIC_INTERVAL = int(HOURS * 3600 / DT / PIC_COUNT)
# how often we want to take pic (70 = every 70 timesteps)
# Firmware takes new pic every 7 seconds
PIC_INTERVAL = 70

if USING_IR:
    IMAGE_WIDTH = 24
    IMAGE_HEIGHT = 32
    # sensor width (in mm) and desired FOV
    CAM_FOV_VERTICAL = 110.0
    CAM_FOV_HORIZONTAL = 70
else:
    # settings for higher quality color cam
    # TODO: fix cropping/distortion issue with square pics
    IMAGE_WIDTH = 512
    IMAGE_HEIGHT = 512
    CAM_FOV_VERTICAL = 90.0
    CAM_FOV_HORIZONTAL = 90.0
SENSOR_WIDTH = 25.8
SENSOR_HEIGHT = 17.8

# In Maya, Y-axis is UP, so -Y points toward Earth center (nadir)
# Cameras should point along -Y axis when nadir-pointing
BODY_DOWN = [0, 0, -1]
# angle at which our cams are mounted (degrees) with respect to axis we want to nadir point
CAM_MOUNT_ANGLE = 60
EARTH_OBJECT = "earth"
SUN_EARTH_GROUP = "earth_sun"
IR_EARTH_GROUP = "earth_IR"

if CUBES_PATH_NO_CAMS:
    RENDER_IMAGES = False

# ======= ENVIRONMENTAL DISTURBANCES ======================

# Typical disturbance torques for a 2U CubeSat in LEO
GRAVITY_GRADIENT_MAX = 1e-7  # Nm, maximum gravity gradient torque
                            # Varies with orbit altitude and satellite orientation
                            # Typically ~10^-7 Nm for 2U in 400km orbit

SOLAR_PRESSURE_MAX = 1e-8   # Nm, maximum solar radiation pressure torque
                           # Depends on surface area, reflectivity, sun angle
                           # Typically ~10^-8 Nm for 2U

AERO_DRAG_MAX = 2e-7       # Nm, maximum aerodynamic drag torque
                          # Varies with altitude (atmospheric density)
                          # Typically ~10^-7 Nm at 400km

MAGNETIC_RESIDUAL = 1e-6   # Am^2, residual magnetic dipole of spacecraft
                          # Creates torque when interacting with Earth's field
                          # Typical value for CubeSat with basic magnetic cleanliness

# Combined disturbance for simulation
TOTAL_DISTURBANCE_MAX = (GRAVITY_GRADIENT_MAX +
                        SOLAR_PRESSURE_MAX +
                        AERO_DRAG_MAX +
                        MAGNETIC_RESIDUAL * np.linalg.norm([19e-6, 1.7e-6, 49e-6]))  # Total maximum disturbance torque

# # Add environmental disturbances
# disturbance_direction = np.random.rand(3) - 0.5  # Random direction
# disturbance_direction = disturbance_direction / np.linalg.norm(disturbance_direction)
# disturbance_magnitude = np.random.uniform(0, TOTAL_DISTURBANCE_MAX)
# disturbance_torque = disturbance_magnitude * disturbance_direction

# # Add disturbance to dynamics
# angular_acceleration += np.dot(self.J_B_inv, disturbance_torque)
