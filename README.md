<h2 align="left">CloverSat Computing Projects for IrishSat 2024-2025</h2>

Collaborative base for developing Attitude Determination and Control System (ADCS), which include a Proportional-Integral-Derivative (PID) Controller and Unscented Kalman Filter (UKF) for state estimation.

This computing squad enganges in an iterative development model in order to fulfil the club's technical needs. Members apply their own unique technical background while engaging with research, professors, and industry contacts. 

## Organization

-- ukf: contains scripts directly relating to state estimation, including the main unscented kalman filter algorithm, cubesat visualizer, and PySOL repository. 

-- adc: holds our pd controller to determine our reaction wheel inputs

-- interface: contains all scripts that interface with physical components of the cubesat, such as imu, hall sensors, and reaction wheel motors

## Notes

Those sub directories contain testing scripts that are confined to that section of the ADCS process. 

High-level scripts that implement more than one of these sub-sections are contained in main and simply import from the correct sub-directory. 

Members: instead of working on main, make sure to create a feature-branch and push to dev (our most updated but somewhat expiremental code)
