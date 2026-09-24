"""
run_sim.py  -- 

runs thing


"""

import csv
#import numba
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
import h5py
import geopandas as gpd
import geodatasets
import astropy.time as astro_time
import os

from wmm import WMM

# fix for geopandas depricating their dataset, probably slower
# https://stackoverflow.com/questions/76548222/how-to-get-maps-to-geopandas-after-datasets-are-removed
# url = "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
# countries = gpd.read_file(url)

# fix #2 using geodatasets library
countries = gpd.read_file(geodatasets.get_path('naturalearth.land'))
# this method is depricated in geopandas 1.0
# countries = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))


class OAT:

    def __init__(self, fn, path = 'save_sim/', output_path = 'outputs/'):

        print('Initializing OAT simulation..')

        # set the font globally
        plt.rcParams.update({'font.family':'sans-serif'})

        # configure the path correctly
        script_path = os.path.abspath(__file__)
        script_dir = os.path.split(script_path)[0]
        abs_file_path = os.path.join(script_dir, path)

        self.sim_path = abs_file_path
        
        abs_file_path = os.path.join(script_dir, output_path)
        self.out_path = abs_file_path

        f = h5py.File(self.sim_path + fn, 'r')
        print('Loading ' + fn + '..')

        self.dt = f.attrs['dt']

        self.X = f['states']['ECI']['X']
        self.Y = f['states']['ECI']['Y']
        self.Z = f['states']['ECI']['Z']
        self.LALN = f['states']['angular']['LALN']
        self.H = f['states']['ECI']['H']
        self.OE_ = f['states']['angular']['OE']

        self.B = f['B']['B']
        self.Bx = f['B']['Bx']
        self.By = f['B']['By']
        self.Bz = f['B']['Bz']

        self.times_jd = astro_time.Time(f['times']['JD'], format = 'jd').jd

        self.times_utc = astro_time.Time(f['times']['JD'], format = 'jd').datetime

        print('data successfully loaded..')


    def orb_plot(self, i, speed, ax):

        ax.clear()
        lims = [8000, 8000, 8000]
        xlim, ylim, zlim = lims
        ax.set_xlim(-xlim, xlim)
        ax.set_ylim(-ylim, ylim)
        ax.set_zlim(-zlim, zlim)
        ax.set_box_aspect([1, ylim/xlim, zlim/xlim])

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        # ax.set_xlabel('X [km]')
        # ax.set_ylabel('Y [km]')
        # ax.set_zlabel('Z [km]')
        I = xlim*np.array([[0, 0, 0], [1, 0, 0]])
        J = ylim*np.array([[0, 0, 0], [0, 1, 0]])
        K = zlim*np.array([[0, 0, 0], [0, 0, 1]])

        ax.plot(I[:, 0], I[:, 1], I[:, 2], color = 'black')
        ax.plot(J[:, 0], J[:, 1], J[:, 2], color = 'black')
        ax.plot(K[:, 0], K[:, 1], K[:, 2], color = 'black')

        last_frame = i*speed
        #ax.plot(self.X, self.Y, self.Z, color = 'goldenrod', lw = 1, )
        ax.plot(self.X[0:last_frame], self.Y[0:last_frame], self.Z[0:last_frame], 
            color = 'navy', lw = 5)
        ax.scatter(self.X[last_frame], self.Y[last_frame], self.Z[last_frame], s = 100, 
                fc = 'goldenrod', ec = 'navy', marker = '.', zorder = 3)
        earth = self.__earth_3d()
        ax.plot_wireframe(earth[0], earth[1], earth[2], 
            color = 'steelblue', alpha = 0.7, zorder = 0)


    def groundtrack_plot(self, i, speed, ax):

        ax.clear()
        countries.plot(ax = ax, color= 'gray', alpha = 0.3, edgecolor='black')

        last_frame = i*speed
        ax.scatter(self.LALN[:, 1], self.LALN[:, 0], s = 0.1, 
            color = 'goldenrod', alpha = 0.1)
        ax.scatter(self.LALN[0:last_frame, 1], self.LALN[0:last_frame, 0], s = 5, 
            color = 'navy' )

        ax.set_xlabel('Longitude [deg]')
        ax.set_ylabel('Lattitude [deg]')

        ax.set_xlim(-180, 180)
        ax.set_ylim(-90, 90)

        ax.set_yticks(np.arange(-90, 91, 15))
        ax.set_xticks(np.arange(-180, 180, 30))
        ax.grid()

    def text_plot(self, i, speed, ax):

        tsize = 8

        last_frame = i*speed
        ax.clear()
        ax.set_yticks([])
        ax.set_xticks([])
        ax.text(0.02, 0.98, 8*' ' + 'Orbital Data' + 8*' ', fontsize = 14, horizontalalignment='left',
            verticalalignment='top', transform=ax.transAxes, 
            bbox = dict(facecolor='navy', alpha=0.5))

        ax.text(0.02, 0.90, 'f  : {:3.2f}°'.format(self.OE_[last_frame, 0]), 
            fontsize = tsize, horizontalalignment='left',
            verticalalignment='top', transform=ax.transAxes)
        ax.text(0.02, 0.85, 'a : {:5.2f} [km]'.format(self.OE_[last_frame, 1]), 
            fontsize = tsize, horizontalalignment='left',
            verticalalignment='top', transform=ax.transAxes)
        ax.text(0.02, 0.80, 'e : {:3.5f}'.format(self.OE_[last_frame, 2]), 
            fontsize = tsize, horizontalalignment='left',
            verticalalignment='top', transform=ax.transAxes)
        ax.text(0.02, 0.75, 'i  : {:3.2f}°'.format(self.OE_[last_frame, 3]), 
            fontsize = tsize, horizontalalignment='left',
            verticalalignment='top', transform=ax.transAxes)
        ax.text(0.02, 0.70, r'$\Omega$' +  ' : {:3.2f}°'.format(self.OE_[last_frame, 4]), 
            fontsize = tsize, horizontalalignment='left',
            verticalalignment='top', transform=ax.transAxes)
        ax.text(0.02, 0.65, r'$\omega$'+' : {:3.2f}°'.format(self.OE_[last_frame, 2]), 
            fontsize = tsize, horizontalalignment='left',
            verticalalignment='top', transform=ax.transAxes)

        ax.text(0.27, 0.90, 'h   : {:3.2f} [km]'.format(self.H[last_frame]), 
            fontsize = tsize, horizontalalignment='left',
            verticalalignment='top', transform=ax.transAxes)
        ax.text(0.27, 0.85, 'Lat : {:3.2f}°'.format(self.LALN[last_frame,0]), 
            fontsize = tsize, horizontalalignment='left',
            verticalalignment='top', transform=ax.transAxes)
        ax.text(0.27, 0.80, 'Lat : {:3.2f}°'.format(self.LALN[last_frame, 1]), 
            fontsize = tsize, horizontalalignment='left',
            verticalalignment='top', transform=ax.transAxes)

    def plot_B(self, i, speed, ax):

        # times = sc.state_mat.times

        last_frame = i*speed
        ax.clear()
        ax.set_xlabel('Time [UTC]')
        ax.set_ylabel(r'[$\mu T$]')
        ax.set_ylim(-80, 80)

        ax.plot(self.times_utc[0: last_frame], self.B[0:last_frame], label = r'|B|')
        ax.plot(self.times_utc[0: last_frame], self.Bx[0: last_frame], label = r'$B_x$')
        ax.plot(self.times_utc[0: last_frame], self.By[0: last_frame], label = r'$B_y$')
        ax.plot(self.times_utc[0: last_frame], self.Bz[0: last_frame], label = r'$B_z$')

        ax.grid()

        ax.legend(loc = 0)

    def plot_func(self, i, speed, output, o_fn):

        if output:
            with open(self.out_path + o_fn, 'w') as o_f:

                text = np.arange(0, i, 1)
                writer = csv.writer(o_f)

                writer.writerow(text)

        self.fig.suptitle('IrishSat OAT Laboratory', size = 30)

        self.orb_plot(i, speed, ax = self.ax1)
        self.groundtrack_plot(i, speed, ax = self.ax2)
        self.text_plot(i, speed, ax = self.ax3)
        self.plot_B(i, speed, ax = self.ax4)

    def __earth_3d(self):

        R_e = 6371 #km

        u, v = np.mgrid[0:2*np.pi:100j, 0:np.pi:100j]
        x = (np.cos(u)*np.sin(v))*R_e
        y = (np.sin(u)*np.sin(v))*R_e
        z = (np.cos(v))*R_e

        earth = np.array([x, y, z])

        return earth 


    def run_OAT(self, speed = 10, fps = 2, output = False, o_fn = 'oat_output'):

        self.fig = plt.figure(figsize= [15, 8])

        self.fig.patch.set_facecolor('slategray')
        self.fig.patch.set_alpha(0.6)
        self.fig.tight_layout()

        self.ax1 = self.fig.add_subplot(221, projection = '3d')
        self.ax2 = self.fig.add_subplot(222)
        self.ax3 = self.fig.add_subplot(223)
        self.ax4 = self.fig.add_subplot(224)
        
        ani = animation.FuncAnimation(self.fig, self.plot_func, interval= int(1e3/fps),  
            cache_frame_data=False,
            fargs= [speed, output, o_fn])

        plt.show()



if __name__ == '__main__':

    oat = OAT('test.hdf5')

    oat.run_OAT(fps = 1, speed = 20, output= True)