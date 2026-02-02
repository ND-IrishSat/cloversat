# Simulator Overview

Technical overview of Irishsat's orbital simulator.

## PySOL

Before worrying about the specifics of our satellite sytem, we must know what orbit we're simulating. The Python Simulated Orbital Library (PySOL) is a custom orbital generator created by Drew, Juwan, and Eli (all alumni). See the PyPi page for more info (TODO).

A Low Earth Orbit (LEO) can be uniquely identified by 6 numbers, its Orbital Elements (OE). These 6 numbers represent the altitude, starting point, angle, etc of a rotation.

Given a set of Orbital Elements, PySOL tells us the magnetic field and GPS data for every timestep. It also tells us the orbit speed (and RAM direction) of the satellite. PySOL can either generate new data every time for a specific set of OE's, or it can fetch pregenerated data from a file (which is faster).

## Simulator Object vs Satellite Object

The `Simulator` class is the heart of the project. An instance of this class Knows Everything (the absolute truth of the state of the satellite for all time steps). It's omniscient. It stores all the info about our simulation, like all the currents for our reaction wheels, the reaction wheels' speeds, the power output for every timestep, Earth's magnetic field, TONS of information.

This is not be confused with the `Magnetorquer_Sat` object, which contains WHAT THE SIMULATED SATELLITE KNOWS AT ANY GIVEN TIME. It aligns with what actual flight software would have access too. It tracks the sensor values the satellite is receiving from the IMU/cams, its inertia, etc. It stores all current sensor data (**which is noisy**).

## Steps

See `Main/` for examples of simulator usage, like `space_sim.py`. It first sets up/creates/initializes objects (like the 3 magnetorquers and the `Simulator`), then enters the 6-7 steps of the simulation loop (LOL).

1. `generateDat_step`: generate fake sensor data by adding noise to our perfect state, which involves transforming the magnetic field from the earth frame to the body frame.

2. Finding true nadir was used for Nearspace, not really needed anymore.

3. `determine_attitude`: use state estimation methods to smooth out sensor data. We use an Unscented Kalman Filter (UKF); basically, we cannot trust our sensors because they are so noisy, so the Kalman filter (grad school-level topic!!) takes in the sensor data, then uses physics predictions to give you a better estimate of where you are. (Finding orientation this way requires a GPS so that's why we couldn't use it with the NearSpace project). Extended, normal, or multiplicative Kalman Filters also exist and may be added at some point.


4. `check_state`: decide which “protocol” we should be in. This is like checking the transitions of our Finite State Machine! The satellite is only in one state at a time, but it can jump between modes when different conditions are met. Example: if we're in the detumbling state, we'll switch to sun-pointing when we see our speed is below the threshold.

5. `controls`: once you know what state you are in, you enact the corresponding control law (like a PID, custom nadir pointing, etc). **Outputs a PWM/voltage** to our actuators to move us towards some desired target. Right now there's a bunch of ugly `if` statements that could be improved to make it more plug-and-play.

6. `propagate_step`: handles the heavy physics lifting of the simulator. Based on the current state and output from our control law, finds our state (quaternion and angular velocity) at the **next** time step. This involves calculating how our system's current changes, what torques are applied, and what power is used.
- The main thing this function does is call our EQUATIONS OF MOTION (EOM's), which are the backbone of the simulator that which dictate how our simulated system moves and changes. The EOM's use physics equations to calculate the first derivative of our velocity and quaternion.
- After finding the first derivative, the state must combine that info about how the state is changing with the current value. In other words, actually propagate the state forward in time. There are two ways to do this: Euler's method (which is computationally faster but accumulates accuracy errors over time) and RK4 (which is more expensive but more accurate)

## Params.py

Before running the sim, all parameters in `params.py` should be checked carefully. You can change how many hours you want to run the simulation for (which affects how many orbits are generated), starting speed/orientation/mode, and much more. It contains all physics constants we've obtained over the years. A hour-long simulation may be like 3 minutes to run (kind of slow) so be aware of that. `RW_AXES` and `MAG_AXES` are bitmasks that represent what hardware should be simulated (which reaction wheels and magnetorquers are mounted on which axes).

## Utils

We can find helper functions here! Whenever you want to convert something from one unit to another, check out Util/transformations.py to see if we have a function for that already!
