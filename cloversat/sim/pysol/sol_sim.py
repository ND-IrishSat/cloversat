"""
sol_sim.py
Author: Jeremy Juwan, added to by Andrew Gaylord

Contains main simulation class and supporting functions for PySOL repository.

Newest interface functions:
    generate_orbit_data (generates orbit data and stores in CSV file)
    get_orbit_data (retrieves orbit data from CSV file)

To use, import PySOL in specific order:
    import Simulator.PySOL.wmm as wmm
    from Simulator.PySOL.sol_sim import *
    import Simulator.PySOL.spacecraft as sp
    import Simulator.PySOL.orb_tools as ot

"""

# location for storing magnetic field CSV's
OUTPUT_DIR = 'outputs'

from itertools import count
from math import remainder
from time import strftime
# from tkinter.tix import DisplayStyle
import spacecraft as sp
import orb_tools as ot
import models
import astropy.time as astro_time

import os
import h5py
import datetime
import numpy as np
import scipy.integrate as sci
import matplotlib.pyplot as plt
import constants

import geopandas as gpd
import geodatasets

from wmm import WMM


# fix for geopandas depricating their dataset, probably slower
# url = "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
# countries = gpd.read_file(url)

# fix #2 using geodatasets library
countries = gpd.read_file(geodatasets.get_path('naturalearth.land'))
# this method is depricated in geopandas 1.0
# countries = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))

# import matplotlib.pyplot as plt
# # Read world Countries
# world = gpd.read_file(
# gpd.datasets.get_path("naturalearth_lowres")
# )
# # # Read world cities
# # cities = gpd.read_file(
# #  gpd.datasets.get_path("naturalearth_cities")
# # )
# # Read Graticules
# graticules = gpd.read_file(
#  "ne_110m_graticules_10/ne_110m_graticules_10.shp"
# )

class Simulation():

    def __init__(self, model_name = 'Two-Body', mag_deg = 1, TIME = None):
        """
            __init__ - initialization of PySol Simulation

                Opt. Args:
                    model_name (str) :
                        Dynamical model name
                        Current dynamical models:
                            ['Two-Body']
                    mag_deg (int): [1-12]
                        degree of the WMM model Legendre polynomial

                    TIME (dt.datetime object) :
                        UTC time to initialize simulation
                        if None, sim time set to current UTC

        """

        # Define Constants used in simulation
        self.R_E = constants.R_E()

        # Generate dynamical model object
        self.model_nm = model_name

        ####!!!!!!! FLAG
        model = models.Orbital_Models(model_name)
        # find the directory of current file and append to get absolute path
        wmm_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'WMMcoef.csv')

        # wmm_model = WMM(mag_deg, 'WMMcoef.csv')
        wmm_model = WMM(mag_deg, wmm_path)

        self.mag_model = wmm_model

        # State integration function
        self.state_func = model.get_state_func()

        # Set simulation clock
        if TIME == None:
            self.t0  = datetime.datetime.utcnow()
            self.time = self.t0
        else:
            self.t0 = TIME
            self.time = TIME

        # Intitialize spacecraft and sim time lists
        self.scs = []
        self.times = []

        self.calculated_B = False # Will flip to True once B is calculated

        pass

    def create_sc(self, OE_array, verbose = False, color = 'firebrick', name = None):
        """
            create_sc - function takes OE_array and generates sc object in memeory

                Args:
                    OE_array (OE_array obj): OE_array object to initialize sc

                Returns:
                    sc (sc obj): spacecraft object
        """

        # Time of init is taken from current sim time
        TIME = self.time

        sc = sp.Spacecraft(OE_array, TIME, verbose = verbose, color = color, name = name)
        self.scs.append(sc)

        return sc

    def propogate(self, DT, resolution = 10, tol = [1e-7, 1e-4], integrator = 'RK45',
        event_func = None):
        """
            propogate - propogates sc and simulation forward in time using dynamical model

            Args:
                DT (datetime.timedelta) : time of integration
                resolution (float) : output time spacing in seconds
                tol (1x2 list) : tolerance of RK integrator
                integrator (str) : num. integration method to use
                    ['RK23', 'RK45', 'DOP853', 'Radau']
                event_func (function) : function to record events

            Return:
                None
        """


        dt_seconds = DT.seconds
        n_outputs = int(dt_seconds/resolution)

        # Propogate each sc and record output to sc object
        for i, sc in enumerate(self.scs):
            sc_props = self.propogate_func(sc, dt_seconds, n_outputs, tol, integrator, event_func)

            # Add new datetime objects to sc time list
            t_s = sc_props[1]
            times = []
            for s in t_s:
                delta_t = datetime.timedelta(seconds  = s)
                times.append(self.time + delta_t)

            # Add new state vectors to sc time list
            states = sc_props[0]
            self.scs[i].set_states(states[1:], times[1:])### First item is a repeat of init
            self.scs[i].set_times(times[1:])

        # Set the sim time to the last sc recorded time
        self.time = times[-1]
        self.times = times

    def propogate_func(self, sc, dt_sec, n_outputs, tol, integrator, event_func):
        """
            propogate_func:
                function for use in the simulation.propogate_parallel,
                    simulation.propogate methods

                    Args:
                        sc (sc object): sc object for propogation
                        tau_f (float): final non-dim time
                        n_outputs (int): number of integration outputs
                                        (NOT # of integration steps!)
                        tol (tuple): absolute and relative intergation tolerance, set to
                            [1e-12, 1e-10] for high precision
                        event_func: func to record events

                    Returns:
                        s_new (6xN): sc states at each t_eval
                        t_new (N): evaluated times

                    Opt. Returns:
                        y_hits: states at event triggers
                        t_hits: times at event triggers

        """

        # Set up integration
        state = sc.state_mat.S_[-1]
        print("state: ", state)
        t_eval = np.linspace(0, dt_sec, n_outputs)

        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html
        # Numerical integration
        sol = sci.solve_ivp(
            fun = self.state_func,  # Integrtion func
            y0 = state,             # Initial state
            t_span = [0, dt_sec],   # Init/Final int time
            method = integrator,    # Int Algorithm
            t_eval = t_eval,        # Times to output
            max_step = 10,          # Max time step [s] in int
            atol = tol[0],
            rtol = tol[1],
            #args = [eval_STM, p_event, epoch],
            events = event_func
        )

        # propogated states and time arrays
        s_new = sol.y.T
        t_new = sol.t.T

        return s_new, t_new

    def calc_B(self):
        if self.calculated_B:
            raise Exception('B field already calculated!')
        else:
            self.calculated_B = True

        for sc in self.scs:
            sc.calc_B(mag_model = self.mag_model)

    def save_sim(self, file_name = None, save_states = True, save_times = True,
                    save_B = False):

        if file_name == None:
            file_name = datetime.datetime.now().strftime("%Y-%m-%d_PySol")
        file_path = 'save_sim/{}.hdf5'.format(file_name)

        # append current file path to file_path to get absolute path
        # find directory of current file and append to get absolute path
        file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), file_path)

        ### if file already exists, append a number to the end of the file name
        exists = os.path.exists(file_path)
        i = 0
        while exists == True:
            i += 1
            i_str = str(i)
            temp_name = '_'.join([file_name, i_str])
            temp_path = 'save_sim/{}.hdf5'.format(temp_name)
            temp_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), temp_path)
            exists = os.path.exists(temp_path)
        if i >0:
            file_path = temp_path
        #######################################################


        sc = self.scs[0] # only saving the first spacecraft trajectory
        ST_MT = sc.state_mat
        # Open file
        f = h5py.File(file_path, "a")

        f.attrs['t0'] = astro_time.Time(self.t0).jd
        f.attrs['tf'] = astro_time.Time(self.time).jd

        times = ST_MT.times
        dt = times[3] - times[2]
        f.attrs['dt'] = dt.total_seconds()

        f.attrs['N Sc'] = len(self.scs)
        f.attrs['Dyn Model'] = self.model_nm
        # f.attrs['Mag Model'] = self.mag_model

        if save_states:
            states_grp = f.create_group('states')

            ECI_states = states_grp.create_group('ECI')
            #### ECI States ####
            ECI_states.create_dataset(name = 'S_', data = ST_MT.S_)
            ECI_states.create_dataset(name = 'R_', data = ST_MT.R_)
            ECI_states.create_dataset(name = 'V_', data = ST_MT.V_)
            ECI_states.create_dataset(name = 'R', data = ST_MT.R)
            ECI_states.create_dataset(name = 'V', data = ST_MT.V)
            ECI_states.create_dataset(name = 'X', data = ST_MT.X)
            ECI_states.create_dataset(name = 'Y', data = ST_MT.Y)
            ECI_states.create_dataset(name = 'Z', data = ST_MT.Z)
            ECI_states.create_dataset(name = 'VX', data = ST_MT.VX)
            ECI_states.create_dataset(name = 'VY', data = ST_MT.VY)
            ECI_states.create_dataset(name = 'VZ', data = ST_MT.VZ)
            ECI_states.create_dataset(name = 'H', data = ST_MT.H)

            ang_states = states_grp.create_group('angular')
            ang_states.create_dataset(name = 'OE', data = ST_MT.OE_)
            ang_states.create_dataset(name = 'RADEC', data = ST_MT.RADEC)
            ang_states.create_dataset(name = 'LALN', data = ST_MT.LALN)

            ECEF_states = states_grp.create_group('ECEF')
            ECEF_states.create_dataset(name = 'r_', data = ST_MT.R_ECEF)
            ECEF_states.create_dataset(name = 'x', data = ST_MT.R_ECEF[:, 0])
            ECEF_states.create_dataset(name = 'y', data = ST_MT.R_ECEF[:, 1])
            ECEF_states.create_dataset(name = 'z', data = ST_MT.R_ECEF[:, 2])

            print('Saved states..')

        if save_times:
            times_grp = f.create_group('times')
            #times_grp.create_dataset(name = 'UTC', data = ST_MT.times)
            times_grp.create_dataset(name = 'JD', data = ST_MT.times_jd)

            print('Saved times..')

        if save_B:
            B_grp = f.create_group('B')

            B = sc.B_.astype(np.double)
            B_grp.create_dataset(name = 'B', data = np.linalg.norm(B, axis = 0)*1e-3)
            B_grp.create_dataset(name = 'Bx', data = B[0]*1e-3)
            B_grp.create_dataset(name = 'By', data = B[1]*1e-3)
            B_grp.create_dataset(name = 'Bz', data = B[2]*1e-3)


        print('HDF5 file saving is donezo..')
        print('Simulation file saved to ' + file_path)


    def plot_orbit(self, D2 = False, tau_f = None, Earth = True,
        lims = [8000, 8000, 8000], IJK = True):

        # plt.rcParams.update({'font.sans-serif': 'Helvetica'})

        xlim, ylim, zlim = lims

        if D2:
            fig = plt.figure(figsize = [10, 10])
            ax = fig.add_subplot()
            ax.set_aspect('equal')

            if Earth:
                earth = self.__earth_2d()
                ax.add_artist(earth)
        else:

            fig = plt.figure(figsize = [8, 8])
            ax = fig.add_subplot(projection = '3d')
            ax.set_xlim(-xlim, xlim)
            ax.set_ylim(-ylim, ylim)
            ax.set_zlim(-zlim, zlim)
            ax.set_box_aspect([1, ylim/xlim, zlim/xlim])

            ax.set_xlabel('X [km]')
            ax.set_ylabel('Y [km]')
            ax.set_zlabel('Z [km]')

            if Earth:
                earth = self.__earth_3d()
                ax.plot_wireframe(earth[0], earth[1], earth[2],
                    color = 'mediumblue', alpha = 0.1, zorder = 0)

            if IJK:
                I = xlim*np.array([[0, 0, 0], [1, 0, 0]])
                J = ylim*np.array([[0, 0, 0], [0, 1, 0]])
                K = zlim*np.array([[0, 0, 0], [0, 0, 1]])

                plt.plot(I[:, 0], I[:, 1], I[:, 2], color = 'black')
                plt.plot(J[:, 0], J[:, 1], J[:, 2], color = 'black')
                plt.plot(K[:, 0], K[:, 1], K[:, 2], color = 'black')

        tf = self.scs[0].state_mat.times[-1]
        t0 = self.scs[0].state_mat.times[0]

        ax.set_title('PySOL | ECI | Dynamics: {} | '.format(self.model_nm) + t0.strftime('%Y/%m/%d') +
            '\n' + t0.strftime('%H:%M:%S – ') + tf.strftime('%H:%M:%S UTC'))
        for sc in self.scs:
            #X = sc.state_mat.R_ECEF[:, 0]
            #Y = sc.state_mat.R_ECEF[:, 1]
            #Z = sc.state_mat.R_ECEF[:, 2]
            X = sc.state_mat.X
            Y = sc.state_mat.Y
            Z = sc.state_mat.Z
            ax.plot(X, Y, Z, color = sc.color, zorder = 2)
            ax.scatter(X[-1], Y[-1], Z[-1], s = 100,
                fc = 'black', ec = sc.color, marker = '.', zorder = 3, label = sc.name)

        ax.legend()

    def plot_RADEC(self,):

        fig = plt.figure()
        ax = fig.add_subplot()

        for sc in self.scs:
            RADEC = sc.state_mat.to_RADEC()
            ax.scatter(RADEC[:, 0], RADEC[:, 1], s = 5)

        ax.set_xlabel('Right Ascension [deg]')
        ax.set_ylabel('Declination [deg]')

    def plot_LALN(self):

        fig, ax = plt.subplots()

        # world.plot(ax=ax, color="lightgray")
        # graticules.plot(ax=ax, color="lightgray", linewidth=0.5)

        countries.plot(ax = ax, color= 'gray', alpha = 0.3, edgecolor='black')


        for sc in self.scs:
            LALN = sc.state_mat.to_LALN()
            ax.scatter(LALN[:, 1], LALN[:, 0], s = 5, color = sc.color, label = sc.name)

        ax.set_xlabel('Longitude [deg]')
        ax.set_ylabel('Lattitude [deg]')

        ax.set_xlim(-180, 180)
        ax.set_ylim(-90, 90)

        ax.legend()

        tf = self.scs[0].state_mat.times[-1]
        t0 = self.scs[0].state_mat.times[0]

        ax.set_title('PySOL | Ground Tracks | Dynamics: {} | '.format(self.model_nm) + t0.strftime('%Y/%m/%d') +
            '\n' + t0.strftime('%H:%M:%S – ') + tf.strftime('%H:%M:%S UTC'))
        return ax

    def plot_B(self):

        fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows = 4, ncols= 1, sharex= True)



        ylabels = ['$B_{north}$', '$B_{east}$', '$B_{down}$', '$|{B}|$']
        for sc in self.scs:

            LALN = sc.state_mat.to_LALN()
            B = sc.B_.astype(np.double)
            times = sc.state_mat.times

            for i, ax in enumerate([ax1, ax2, ax3, ax4]):
                if i > 2.1:
                    ax.plot(times, np.linalg.norm(B, axis = 0)*1e-3, color = sc.color)
                    ax.set_xlabel('Time [UTC]')
                    ax.set_ylabel(r'{}'.format(ylabels[i]) + r' [$\mu T$]')
                    ax.set_ylim(0, 80)
                else:
                    ax.plot(times, B[i]*1e-3, color = sc.color)
                    ax.set_ylabel(ylabels[i] + r' [$\mu T$]')

                    ax.set_ylim(-70, 30)
                ax.tick_params('x', which = 'major', length = 0.1, direction = 'in', labelsize = 8)



        fig.legend()



    def plot_XYZ(self):

        fig, axs = plt.subplots(nrows = 3, ncols = 1, figsize = [12, 6], sharex = True)

        # plt.rcParams.update({'font.sans-serif': 'Helvetica'})

        for sc in self.scs:
            labels = ['X [km]', 'Y [km]', 'Z [km]']
            states = sc.state_mat.S_
            times = sc.state_mat.times
            for i, ax in enumerate(axs):
                if i < 1:
                    axs[i].plot(times, states[:, i], color = sc.color, label = sc.name)
                else:
                    axs[i].plot(times, states[:, i], color = sc.color)

                if i > 1:
                    axs[i].set_xlabel('Time [UTC]')

                ax.minorticks_on()
                ax.tick_params('both', which = 'major', length = 10, direction = 'in')
                ax.tick_params('both', which = 'minor', length = 5, direction = 'in')
                ax.set_ylabel(labels[i])

        axs[0].legend(loc = 0)


    def __earth_2d(self):

        R_e = 6371 #km
        earth = plt.Circle((0, 0), R_e, color = 'mediumblue')

        return earth

    def __earth_3d(self):
        """
            __earth_3d:
            produces 3D wireplot state vector of Earth in the synodic frame

                Args:
                    r (1x6 array): position of Earth in ndim synodic frame

                Returns
                    earth (1x6 array): Earth wireframe state vector
        """

        R_e = 6371 #km

        u, v = np.mgrid[0:2*np.pi:100j, 0:np.pi:100j]
        x = (np.cos(u)*np.sin(v))*R_e
        y = (np.sin(u)*np.sin(v))*R_e
        z = (np.cos(v))*R_e

        earth = np.array([x, y, z])

        return earth


def generate_orbit_data(OE_array, total_time, timestep, file_name="b_field_data.csv", store_data=False, GPS=False, RAM=False):
    '''
    Generate the magnetic field data for a given orbit
    If store_data is True, the magnetic field data is saved to a CSV file in the OUTPUT_DIR folder
        First line of CSV file contains orbit info

    @params:
        OE_array ( 1x6 array ): orbital elements of the spacecraft. 6 numbers to describe the shape and orientation of the orbit
            Position of spacecraft along 1D motion path:
                f - true anomaly [deg]: angle between point closest to earth and current location
            Shape of ellipse in 2D:
                a - semi-major axis [km]: the distance from the center of an ellipse to the longer end of the ellipse=radius/altitude lol
                e - eccentricity [0, 1): how stretched the ellipse is, circle = 0
            Orient 2D ellipse in 3D:
                i - inclination [deg]: angle between equator and orbit plane
                    NOTE: bugs out halfway around orbit if 90 (b field x and y swap)
                Om - right ascension (RA) of ascending node [deg]: angle between ascending node and non-rotating coordinate system of earth (geocentric equatorial coordinate system)
                    NOTE: cannot be 0, also bugs out if 90
                w - argument of perigee/periapsis [deg]: angle between ascending node and point of closest approach to earth (periapsis)

        total_time (float): total time of the simulation (hours)
        timestep (float): time step of the simulation (seconds)
        file_name (str): name of CSV file
        store_data (bool): whether to create CSV and store data within
        GPS (bool): whether to find and store GPS data (ECEF) or not
        RAM (bool): whether to find and store RAM velocity vector (direction of travel) or not

    @returns:
        B_fields ( (3 x n) np.array): magnetic field data for all n time steps (microTesla)
        if GPS:
            return additional element:
                GPS ( (3 x n) np.array): location for each time step in efec frame (km)
        if RAM:
            return additional element:
                RAM ( (3 x n) np.array): RAM velocity vector for each step (km/s)
    '''

    # initialize simulation object (with time = current date and time)
    sim = Simulation(mag_deg = 12)

    # create spacecraft object with specified orbital elements
    OE = ot.OE_array(*OE_array)
    sim.create_sc(OE_array= OE, verbose = True, color = 'green', name = file_name)

    # how long we're simulating for
    DT = datetime.timedelta(hours = total_time)
    # propogate and solve the simulation
    # resolution = timestep (seconds)
    sim.propogate(DT, resolution =  timestep)

    # calculate b field based on GPS data from model
    sim.calc_B()

    # extract the magnetic field from the sim
    B_field = np.array(sim.scs[0].B_)
    # Reshape B_field to be [n_steps, 3] where each row contains [Bx, By, Bz]
    B_field = B_field.T
    # convert to uT (microTesla)
    B_earth = B_field * 1e-3

    if GPS:
        # scs[0].state_mat.X gives x in ECI frame
        # get x, y, z in ECEF frame (rotates with earth)
        gps_x = sim.scs[0].state_mat.R_ECEF[:, 0]
        gps_y = sim.scs[0].state_mat.R_ECEF[:, 1]
        gps_z = sim.scs[0].state_mat.R_ECEF[:, 2]

        gps_data = np.array(list(zip(gps_x, gps_y, gps_z)))
    if RAM:
        # extract orbital elements
        # orbital_elements = sim.scs[0].state_mat.OE_
        # convert to position/velocity (eci) using orb tools function
        # pos_velocity = np.array([ot.calc_OE2RV(element) for element in orbital_elements])

        # get position and velocity in eci frame
        eci_pos = sim.scs[0].state_mat.R_
        eci_vel = sim.scs[0].state_mat.V_
        # get datetime objects for every step
        times = sim.scs[0].state_mat.times
        ecef = []

        for i in range(len(eci_pos)):
            # find pos and velocity in ecef frame
            pos, vel = ot.calc_ECEF_R(eci_pos[i], eci_vel[i], times[i])
            ecef.append(np.concatenate((pos, vel)))

        # extract ecef velocity from array
        ram_velocity = np.array([a[3:] for a in ecef])

    if store_data:
        # Get the directory of the current script
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # Create OUTPUT_DIR as a subdirectory of the script location
        output_dir = os.path.join(script_dir, OUTPUT_DIR)
        os.makedirs(output_dir, exist_ok=True)

        # Full path to output file
        output_path = os.path.join(output_dir, file_name)

        # Save data to CSV
        with open(output_path, 'w') as f:
            # Write header with simulation parameters
            f.write(f"# f={OE_array[0]},a={OE_array[1]},e={OE_array[2]},i={OE_array[3]},Om={OE_array[4]},w={OE_array[5]},total_time={total_time},timestep={timestep}\n")
            if GPS:
                # combine b field and gps data
                combined_data = np.hstack((B_earth, gps_data))
                np.savetxt(f, combined_data, delimiter=',', header='Bx,By,Bz,X,Y,Z', comments='')
            else:
                # Write B-field data
                np.savetxt(f, B_earth, delimiter=',', header='Bx,By,Bz', comments='')
        print(f"Data saved to {output_path}")

    if RAM and not GPS:
        return B_earth, ram_velocity
    if GPS:
        if RAM:
            return B_earth, gps_data, ram_velocity
        return B_earth, gps_data
    else:
        return B_earth


def get_orbit_data(file_name, oe=None, total_time=None, timestep=None, GPS=False):
    '''
    Get the magnetic field data from the specified CSV file in OUTPUT_DIR folder
    and check orbital parameters against those in the CSV file.

    @params:
        file_name (str): name of the CSV file to get the data from
        oe ( 1x6 array ): orbital elements of the spacecraft. 6 numbers to describe the shape and orientation of the orbit
                f - true anomaly [deg]: angle between point closest to earth and current location
                a - semi-major axis [km]: the distance from the center of an ellipse to the longer end of the ellipse=radius/altitude lol
                e - eccentricity [0, 1): how stretched the ellipse is, circle = 0
                i - inclination [deg]: angle between equator and orbit plane
                Om - right ascension (RA) of ascending node [deg]: angle between ascending node and non-rotating coordinate system of earth (geocentric equatorial coordinate system)
                w - argument of perigee/periapsis [deg]: angle between ascending node and point of closest approach to earth (periapsis)
        total_time (float): total time for the orbit
        timestep (float): timestep for the orbit
        GPS (bool): whether GPS data is in csv or not

    @returns:
        B_fields ( (3 x n) np.array): magnetic field data for all n time steps (microTesla)
    '''

    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Full path to output file
    output_path = os.path.join(script_dir, OUTPUT_DIR, file_name)

    # Read the first few lines to get orbital elements, total time, and timestep
    with open(output_path, 'r') as file:
        header_lines = [file.readline().strip() for _ in range(3)]

    # Parse the first line to get the orbital parameters
    first_line = header_lines[0].split(',')
    csv_orbital_elements = [
        float(first_line[0].split('=')[1]),
        float(first_line[1].split('=')[1]),
        float(first_line[2].split('=')[1]),
        float(first_line[3].split('=')[1]),
        float(first_line[4].split('=')[1]),
        float(first_line[5].split('=')[1])]

    # Check if orbital elements match
    if oe is not None:
        for i, element in enumerate(oe):
            if np.isclose(element, csv_orbital_elements[i]) == False:
                raise ValueError(f"CSV has orbital elements: {csv_orbital_elements}.")

    csv_total_time = float(first_line[6].split('=')[1])
    csv_timestep = float(first_line[7].split('=')[1])

    # Check if total time and timestep match
    if total_time is not None and total_time > csv_total_time:
        raise ValueError(f"{total_time} is too long, CSV only has {csv_total_time} hours.")

    if timestep is not None and not timestep == csv_timestep:
        raise ValueError(f"Timestep mismatch: Expected {timestep}, got {csv_timestep}.")

    # read the signified data
    if GPS:
        data = np.genfromtxt(output_path, delimiter=',')

        # extract magnetic field and gps data from csv
        B_fields = data[1:, :3]
        gps = data[1:, -3:]

        return B_fields, gps
    else:
        B_fields = np.genfromtxt(output_path, delimiter=',')

        # remove header
        B_fields = B_fields[1:, :]

        return B_fields


if __name__ == '__main__':
    oe = [121, 6_800, 0.0000922, 51, -10, 80]
    total_time = 1 / 60
    timestep = 1.0
    file_name = "test.csv"
    store_data = False
    generate_GPS = True
    generate_RAM = True

    B_field, gps, ram = generate_orbit_data(oe, total_time, timestep, file_name, store_data, generate_GPS, generate_RAM)

    # print(gps)
    print("ram: ", ram)
    # print(get_orbit_data(file_name))
    '''

    # 3/21, 2022
    t0 = datetime.datetime(2022, 3, 21, 0, 0, 0)
    sim = Simulation(TIME = t0, mag_deg= 12)

    # True anomaly
    # Semi-major axis
    # Eccentricity
    # Inclination
    # Longitude of the ascending node
    # Argument of periapsis
    OE1 = ot.OE_array(f = 121, a = 6_800, e = 0.0000922, i = 51, Om = -10, w = 80)
    sim.create_sc(OE_array= OE1, verbose = True, color = 'green', name = 'CLOVER-Sat')

    #OE2 = ot.OE_array(f = 7.5, a = 42_000, e = 0.001, i = 51.64, Om = 300, w = 74)
    #sim.create_sc(OE_array= OE2, verbose = True, color = 'blue', name = 'Geostationary Orbit')

    #OE3 = ot.OE_array(f = 20, a = 8_000, e = 0.0003, i = 86, Om = 10, w = 20)
    #sim.create_sc(OE_array= OE3, verbose = True, color = 'red', name = 'Polar Orbit')

    # OE2 = ot.OE_array(f = 108, a = 8_000, e = 0.07, i = -20, Om = 0, w = 0)
    # sim.create_sc(OE_array= OE2, verbose = True, color = 'purple', name = 'Graces Orbit')

    # OE3 = ot.OE_array(f = 62.5, a = 42000, e = 0.005, i = 0.1, Om = 15.5, w = 35.2)
    # sim.create_sc(OE_array= OE3, verbose = True, color = 'red', name = 'Juwan')

    # OE4 = ot.OE_array(f = 0, a = 20_000, e = 0.36, i = 46, Om = 4, w = 8)
    # sim.create_sc(OE_array= OE4, verbose = True, name = 'Bridget\'s Orbit', color = 'Pink')

    # OE5 = ot.OE_array(f = 0, a = 10_000, e = 0.1, i = 69, Om = 42, w = 63)
    # sim.create_sc(OE_array= OE5, verbose = True, color = 'darkorange', name = 'Owen')

    # total sim time, hours
    hours = 4
    DT = datetime.timedelta(hours = hours)

    # time step, seconds
    timestep = 20.0
    sim.propogate(DT, resolution = timestep)

    # calculate lat/long and height for each time step
    orb_laln = sim.scs[0].state_mat.LALN
    orb_h = ot.calc_h(sim.scs[0].state_mat.R_ECEF)

    print("shape: ", sim.scs[0].state_mat.R_ECEF.shape)
    print("lat/long: ", orb_laln[:5])
    print("hight: ", orb_h[:5])

    sim.plot_orbit(lims = [8_000, 8_000, 8_000])

    sim.calc_B()

    # save_sim saves magnetic field, ECI, ECEF, angular velocity, and time data in hdf5 format
    sim.save_sim(file_name = 'test', save_B=True)

    sim.plot_LALN()
    sim.plot_B()

    # sim.plot_RADEC()

    # sim.plot_XYZ()

    # plt.show()

    # GET CHANGE IN B-FIELD DATA FOR GOAT
    B_field = sim.scs[0].B_

    # print("B field: ", B_field)

    time_array = sim.scs[0].state_mat.times
    time_array = time_array - datetime.datetime.min

    start = time_array[0].total_seconds()

    for i in np.arange(time_array.shape[0]):
        time_array[i] = time_array[i].total_seconds() - start

    print("timesteps: ", time_array[:5])

    new_shape = (B_field.shape[0], B_field.shape[1] - 1)
    B_field_diff = np.zeros(new_shape)
    B_field_dir = np.zeros(new_shape)

    for i in np.arange(new_shape[1]):
        B_field_diff[:, i] = B_field[:, i+1] - B_field[:, i] # difference in magnetic field, in nT
        B_field_dir[:, i] = B_field_diff[:, i] / timestep # time derivative, calculated roughly

    fig1 = plt.figure()
    fig2 = plt.figure()
    axes1 = fig1.subplots(3)
    axes2 = fig2.subplots(3)

    comp = ['x', 'y', 'z']

    axes1[0].set_title('Difference in B-field at from previous time step')
    axes2[0].set_title('dB/dt')

    for i, direc in enumerate(comp):
        axes1[i].plot(time_array[:-1], B_field_diff[i, :] * 1e-3)
        axes1[i].set_ylabel('dB' + direc + ' (muT)')
        axes2[i].plot(time_array[:-1], B_field_dir[i, :] * 1e-3)
        axes2[i].set_ylabel('dB' + direc + '/dt + (muT)')

    # np.savetxt('B_out.csv', (time_array, B_field[0, :] * 1e-3, B_field[1, :] * 1e-3, B_field[2, :] * 1e-3), delimiter = ',')
    plt.show()

    '''

