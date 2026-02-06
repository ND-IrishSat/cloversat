# CloverSat Computing Projects

Collaborative codebase for developing CloverSat's Attitude Determination and Control System (ADCS), which include a Proportional-Integral-Derivative (PID) Controller and Unscented Kalman Filter (UKF) for state estimation.

Also contains extensive simulation capabilities, including sensor modeling, orbital dynamics, 3D visualization, and performance report generation.

Lots of code was created for the Nearspace mission: https://github.com/ND-IrishSat/NearSpace

This computing squad enganges in an iterative development model in order to fulfil the club's technical needs. Members apply their own unique technical background while engaging with research, professors, and industry contacts.

Members: instead of working on main, make sure to create a feature-branch and push to dev (our most updated but somewhat expiremental code)

## Organization

* ukf: contains scripts directly relating to state estimation, including the main unscented kalman filter (UKF) algorithm.

* Controllers: various controllers (like PID) for Attitude Control.

* HardwareInterface: contains all scripts that interface with physical components of the cubesat, such as imu, hall sensors, and reaction wheel motors.

* Simulator: helper files that allow us to simulate our orbit (PySOL library), prpogate the sim and store the data (`simulator.py`), generate a PDF report with graphs + info, and visualize cubesat in 3D. See `Simulator/README.md` for more info.

* Utils: functions used everywhere (like speed tests, conversions, etc)

* Maya_project: integrates our simulator with Maya (3D modelling software found on Debart computers)

`Params.py`: holds all global constants/parameters/design specifications/starting states in one place. See `Simulator/README.md` for more info.

## Setup

Install Python **3.12** (with tcl/tk and IDLE checked).

Install git.

Close this git repo in folder of your choosing. `cd` to the root of this directory **in a `Git Bash` console/terminal**, then run `./venv_help.sh create`. After running `. ./venv_help.sh activate` (note the period!) or `source ./venv_help.sh activate` to activate it, scripts can be run by doing `python B_dot_simulator.py` (for example) in that same terminal.

If getting the error permission denied after trying to run `./venv_help.sh create`, try running `chmod +x venv_help.sh`, then rerun `./venv_help.sh create`
