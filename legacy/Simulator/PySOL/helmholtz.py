''' Module containing analytical models for the B field of the frames of the Helmholtz cage + Helmholtz cage
    proper with all 6 frames implemented.
    
    by Juwan Jeremy Jacobe
    University of Notre Dame
    IrishSat OAT Lab
'''

import numpy as np
import matplotlib.pyplot as plt

#####################################################################
#     Frame X, Y, Z Classes                                         #
#####################################################################
# These FrameX, FrameY, and FrameZ classes are used to put together the HelmholtzCage class. The equations
# for the B-Field are derived from the Biot-Savart Law
#
# With the way these are implemented right now, the code is redundant in making distinctions between
# FrameX, FrameY and FrameZ as different objects. Ideally, we should have this written as one Frame class
# (which even more ideally, is made up of a Wire class instead of having the constituent be a Frame class),
# especially as the equations are mostly the same. 
#
# However, the effort to make this code more elegant is probably not worth the effort unless the
# person in question reworking the code is very dedicated and also has a lot of downtime to funnel
# into a pursuit with no price but pride :>
#
# NOTE: The B-field from the frames are independently confirmed to be correct for simple cases such as B-field
# at the center of the frame.

# Frame displaced -x from the origin, generates +Bx components
class FrameX():
    def __init__(self, length, num, x_disp):
        
        # x displacement the frame from the origin
        self.x_disp = x_disp
 
        # y displacement of wires 1 and 3 from origin
        self.y_disp1 = length/2
        self.y_disp3 = -length/2
        
        # z displacement of wires 2 and 4 from origin
        self.z_disp2 = length/2
        self.z_disp4 = -length/2
        
        # Length of square + number of wires wrapped around
        self.L = length
        self.N = num
        
        self.mu_0 = 1.256637062e-6
    
        # Constant in front of Biot-Savart Law: mu_0*N/(4*pi)
        self.const = self.mu_0*num/(4*np.pi)
        
    def get_B_factors(self, x, y, z):
        
        # integrated over z'
        B1_integral_term1 = ( self.L - 2*z )/( np.sqrt( 4*( (x-self.x_disp)**2 + (y-self.y_disp1)**2 ) + (self.L - 2*z)**2 ) ) 
        B1_integral_term2 = ( self.L + 2*z )/( np.sqrt( 4*( (x-self.x_disp)**2 + (y-self.y_disp1)**2 ) + (self.L + 2*z)**2 ) )
        B1_integral = (B1_integral_term1 + B1_integral_term2)/( (x - self.x_disp)**2 + (y - self.y_disp1)**2 )

        self.B1_factor = self.const * B1_integral 
        
        # integrated over y'
        B2_integral_term1 = ( self.L - 2*y )/( np.sqrt(4 * ( (x-self.x_disp)**2 + (z-self.z_disp2)**2 ) + (self.L - 2*y)**2 ) )
        B2_integral_term2 = ( self.L + 2*y )/( np.sqrt(4 * ( (x-self.x_disp)**2 + (z-self.z_disp2)**2 ) + (self.L + 2*y)**2 ) ) 
        B2_integral = -(B2_integral_term1 + B2_integral_term2)/( (x - self.x_disp)**2 + (z - self.z_disp2)**2 )
        
        self.B2_factor = self.const * B2_integral
        
        # integrated over z'
        B3_integral_term1 = ( -self.L + 2*z )/( np.sqrt( 4*( (x-self.x_disp)**2 + (y-self.y_disp3)**2 ) + (self.L - 2*z)**2 ) ) 
        B3_integral_term2 = ( -self.L - 2*z )/( np.sqrt( 4*( (x-self.x_disp)**2 + (y-self.y_disp3)**2 ) + (self.L + 2*z)**2 ) )
        B3_integral = (B3_integral_term1 + B3_integral_term2)/( (x - self.x_disp)**2 + (y - self.y_disp3)**2 )

        self.B3_factor = self.const * B3_integral

        # integrated over y'
        B4_integral_term1 = ( -self.L + 2*y )/( np.sqrt( 4*( (x-self.x_disp)**2 + (z-self.z_disp4)**2 ) + (self.L - 2*y)**2 ) )
        B4_integral_term2 = ( -self.L - 2*y )/( np.sqrt( 4*( (x-self.x_disp)**2 + (z-self.z_disp4)**2 ) + (self.L + 2*y)**2 ) )
        B4_integral = -(B4_integral_term1 + B4_integral_term2) / ( (x - self.x_disp)**2 + (z - self.z_disp4)**2 )

        self.B4_factor = self.const * B4_integral

    def get_Bx(self, I, r):
    
        x = r[0]
        y = r[1]
        z = r[2]
        
        self.get_B_factors(x, y, z)
        
        B_x = ( I*(-self.B1_factor*(y - self.y_disp1) + self.B2_factor*( z - self.z_disp2 ) + 
            -self.B3_factor*(y - self.y_disp3) + self.B4_factor*( z - self.z_disp4 )) )
        
        return B_x
        
    def get_Bfield(self, I, r):
        x = r[0]
        y = r[1]
        z = r[2]
        
        self.get_B_factors(x, y, z)
        
        B_x = ( I*(-self.B1_factor*(y - self.y_disp1) + self.B2_factor*( z - self.z_disp2 ) + 
            -self.B3_factor*(y - self.y_disp3) + self.B4_factor*( z - self.z_disp4 )) )
            
        B_y = ( I*(self.B1_factor*(x - self.x_disp) + (self.B3_factor* (x-self.x_disp) )) )
        
        B_z = ( I* ( -self.B2_factor*(x - self.x_disp) + -(self.B4_factor * (x - self.x_disp) )) )
        
        return np.array([B_x, B_y, B_z])


class FrameY():
    def __init__(self, length, num, y_disp):
        
        # x displacement the frame from the origin
        self.y_disp = y_disp
 
        # y displacement of wires 1 and 3 from origin
        self.x_disp1 = -length/2
        self.x_disp3 = length/2
        
        # z displacement of wires 2 and 4 from origin
        self.z_disp2 = length/2
        self.z_disp4 = -length/2
        
        # Length of square + number of wires wrapped around
        self.L = length
        self.N = num

        # Vacuum permeability
        self.mu_0 = 1.256637062e-6
    
        # Constant in front of Biot-Savart Law: mu_0*N/(4*pi)
        self.const = self.mu_0*num/(4*np.pi)
        
    def get_B_factors(self, x, y, z):
        
        # integrated over z'
        B1_integral_term1 = ( self.L - 2*z )/( np.sqrt( 4*( (x-self.x_disp1)**2 + (y-self.y_disp)**2 ) + (self.L - 2*z)**2 ) ) 
        B1_integral_term2 = ( self.L + 2*z )/( np.sqrt( 4*( (x-self.x_disp1)**2 + (y-self.y_disp)**2 ) + (self.L + 2*z)**2 ) )
        B1_integral = (B1_integral_term1 + B1_integral_term2)/( (x - self.x_disp1)**2 + (y - self.y_disp)**2 )
        
        # integrated over x'
        B2_integral_term1 = ( self.L - 2*x )/( np.sqrt(4 * ( (y-self.y_disp)**2 + (z-self.z_disp2)**2 ) + (self.L - 2*x)**2 ) )
        B2_integral_term2 = ( self.L + 2*x )/( np.sqrt(4 * ( (y-self.y_disp)**2 + (z-self.z_disp2)**2 ) + (self.L + 2*x)**2 ) ) 
        B2_integral = (B2_integral_term1 + B2_integral_term2)/( (y - self.y_disp)**2 + (z - self.z_disp2)**2 )
        
        # integrated over z'
        B3_integral_term1 = ( -self.L + 2*z )/( np.sqrt( 4*( (x-self.x_disp3)**2 + (y-self.y_disp)**2 ) + (self.L - 2*z)**2 ) ) 
        B3_integral_term2 = ( -self.L - 2*z )/( np.sqrt( 4*( (x-self.x_disp3)**2 + (y-self.y_disp)**2 ) + (self.L + 2*z)**2 ) )
        B3_integral = (B3_integral_term1 + B3_integral_term2)/( (x - self.x_disp3)**2 + (y - self.y_disp)**2 )
        
        # integrated over x'
        B4_integral_term1 = ( -self.L + 2*x )/( np.sqrt(4 * ( (y-self.y_disp)**2 + (z-self.z_disp4)**2 ) + (self.L - 2*x)**2 ) )
        B4_integral_term2 = ( -self.L - 2*x )/( np.sqrt(4 * ( (y-self.y_disp)**2 + (z-self.z_disp4)**2 ) + (self.L + 2*x)**2 ) ) 
        B4_integral = (B4_integral_term1 + B4_integral_term2)/( (y - self.y_disp)**2 + (z - self.z_disp4)**2 )
        
        self.B1_factor = self.const * B1_integral 
        self.B2_factor = self.const * B2_integral 
        self.B3_factor = self.const * B3_integral 
        self.B4_factor = self.const * B4_integral

    def get_Bfield(self, I, r):
        x = r[0]
        y = r[1]
        z = r[2]
        
        self.get_B_factors(x, y, z)
        
        B_x = ( I * ( -self.B1_factor * (y - self.y_disp) + -self.B3_factor * (y - self.y_disp) ) )
        
        B_y = ( I * ( self.B1_factor * (x - self.x_disp1) + -self.B2_factor * (z - self.z_disp2) + self.B3_factor * (x - self.x_disp3)
            + -self.B4_factor * (z - self.z_disp4) ) )
        B_z = ( I * ( self.B2_factor * (y - self.y_disp) + self.B4_factor * (y - self.y_disp)) )
        
        return np.array([B_x, B_y, B_z])
        
class FrameZ():
    def __init__(self, length, num, z_disp):
        
        # z displacement of frame from the origin
        self.z_disp = z_disp
        
        # y displacement of wires 1 and 3 from origin
        self.y_disp1 = length/2
        self.y_disp3 = -length/2
        
        # x displacement of wires 2 and 4 from origin
        self.x_disp2 = -length/2
        self.x_disp4 = length/2
        
        # Length of square + number of wires wrapped around
        self.L = length
        self.N = num
        
        self.mu_0 = 1.256637062
    
        # Constant in front of Biot-Savart Law: mu_0*N/(4*pi)
        self.const = self.mu_0*num/(4*np.pi)
        
    def get_B_factors(self, x, y, z):
        
        # integrated over x'
        B1_integral_term1 = ( -self.L + 2*x )/( np.sqrt(4 * ( (y-self.y_disp1)**2 + (z-self.z_disp)**2 ) + (self.L - 2*x)**2 ) )
        B1_integral_term2 = ( -self.L - 2*x )/( np.sqrt(4 * ( (y-self.y_disp1)**2 + (z-self.z_disp)**2 ) + (self.L + 2*x)**2 ) ) 
        B1_integral = (B1_integral_term1 + B1_integral_term2)/( (y - self.y_disp1)**2 + (z - self.z_disp)**2 )
        
        self.B1_factor = self.const * B1_integral 
        
        # integrated over y'
        B2_integral_term1 = ( self.L - 2*y )/( np.sqrt(4 * ( (x-self.x_disp2)**2 + (z-self.z_disp)**2 ) + (self.L - 2*y)**2 ) )
        B2_integral_term2 = ( self.L + 2*y )/( np.sqrt(4 * ( (x-self.x_disp2)**2 + (z-self.z_disp)**2 ) + (self.L + 2*y)**2 ) ) 
        B2_integral = -(B2_integral_term1 + B2_integral_term2)/( (x - self.x_disp2)**2 + (z - self.z_disp)**2 )
        
        self.B2_factor = self.const * B2_integral
        
        # integrated over x'
        B3_integral_term1 = ( self.L - 2*x )/( np.sqrt(4 * ( (y-self.y_disp3)**2 + (z-self.z_disp)**2 ) + (self.L - 2*x)**2 ) )
        B3_integral_term2 = ( self.L + 2*x )/( np.sqrt(4 * ( (y-self.y_disp3)**2 + (z-self.z_disp)**2 ) + (self.L + 2*x)**2 ) ) 
        B3_integral = (B3_integral_term1 + B3_integral_term2)/( (y - self.y_disp3)**2 + (z - self.z_disp)**2 )

        self.B3_factor = self.const * B3_integral

        # integrated over y'
        B4_integral_term1 = ( -self.L + 2*y )/( np.sqrt( 4*( (x-self.x_disp4)**2 + (z-self.z_disp)**2 ) + (self.L - 2*y)**2 ) )
        B4_integral_term2 = ( -self.L - 2*y )/( np.sqrt( 4*( (x-self.x_disp4)**2 + (z-self.z_disp)**2 ) + (self.L + 2*y)**2 ) )
        B4_integral = -(B4_integral_term1 + B4_integral_term2) / ( (x - self.x_disp4)**2 + (z - self.z_disp)**2 )

        self.B4_factor = self.const * B4_integral
        
    def get_Bfield(self, I, r):
        x = r[0]
        y = r[1]
        z = r[2]
        
        self.get_B_factors(x, y, z)
        
        B_x = ( I * ( -self.B2_factor * (z - self.z_disp) + -self.B4_factor * (z - self.z_disp) ) )
        B_y = ( I * ( self.B1_factor * (z - self.z_disp) + self.B3_factor * (z - self.z_disp) ) )
        B_z = ( I * ( self.B1_factor * (y - self.y_disp1) + -self.B2_factor * (x - self.x_disp2) + self.B3_factor * 
            (y - self.y_disp3) + -self.B4_factor * (x - self.x_disp4) ) )
            
        return np.array([B_x, B_y, B_z])

###############################################################
#     Helmholtz Cage Analytical Model                         #
###############################################################
#  This class allows one to solve for the B-field due to a Helmholtz cage at position r
#  where the origin lies in the very center of this Helmholtz cage :>

class HelmholtzCage():
    def __init__(self, length, num, x_disp, y_disp, z_disp):
        ''' Initialize the cage object. Assumes that each frame is of equal side length + equal num of coils wrapped
        around the frame + the x, y, z frames are displaced equally from the origin.
            
            Args:
                length: the length of the frames making up the cage
                num: the number of coils wrapped around the frames
                x_disp: displacements of frames generating B_x from origin, where the origin is the center of the cage 
                y_disp: displacements of frames generating B_y from origin, where the origin is the center of the cage
                z_disp: displacements of frames generating B_z from origin, where the origin is the center of the cage
        '''
        
        # Frame generating +B_x, displaced -x_disp from origin
        self.frame_xminus = FrameX(length, num, -x_disp)
        
        # Frame generating -B_x, displace +x_disp from origin
        self.frame_xplus = FrameX(length, num, x_disp)
        
        # Frame generating +B_y, displaced -y_disp from origin
        self.frame_yminus = FrameY(length, num, -y_disp)
        
        # Frame generating -B_y, displaced +y_disp from origin
        self.frame_yplus = FrameY(length, num, y_disp)
        
        # Frame generating +B_z, displaced -z_disp from origin
        self.frame_zminus = FrameZ(length, num, -z_disp)
        
        # Frame generating -B_z, displaced +z_disp from origin
        self.frame_zplus = FrameZ(length, num, z_disp)
        
        # Create tuple of all frames together, for coding elegance purposes :>
        self.cage = (self.frame_xminus, self.frame_xplus, self.frame_yminus, self.frame_yplus, self.frame_zminus, self.frame_zplus)
        
    def calc_Bfield(self, r, I_xminus=0.1, I_xplus= 0.1, I_yminus= 0.1, I_yplus= 0.1, I_zminus=0.1, I_zplus= 0.1):
        ''' Calculate B field due to Helmholtz cage at position r
        
            Args:
                r (np.array): array of x, y, z positions at which to calculate B field for
                I_xminus (number): current running through Frame -X   (A)
                I_xplus (number): current running through Frame +X    (A) 
                I_yminus (number): current running through Frame -Y   (A)
                I_yplus (number): current running through Frame +Y    (A)
                I_zminus (number): current running through Frame -Z   (A)
                I_zplus (number): current running through Frame +Z    (A)
            
            Out:
                B_field (np.array): array of magnetic field components, Bx, By, Bz
        '''
        
        B_field = np.zeros(3)
        
        currents = (I_xminus, I_xplus, I_yminus, I_yplus, I_zminus, I_zplus)
        
        for i, frame in enumerate(self.cage):
            B_field = B_field + frame.get_Bfield(currents[i], r)
            
        return B_field
        
    def calc_frame_vertices(self, frame):
        ''' Find the four vertices of the frame:
        
            Args:
                frame: a FrameX, FrameY, or FrameZ object
        '''
        if isinstance(frame, FrameX):
            vertix1 = [frame.x_disp, frame.y_disp1, frame.z_disp2]
            vertix2 = [frame.x_disp, frame.y_disp3, frame.z_disp2]
            vertix3 = [frame.x_disp, frame.y_disp3, frame.z_disp4]
            vertix4 = [frame.x_disp, frame.y_disp1, frame.z_disp4]
        elif isinstance(frame, FrameY):
            vertix1 = [frame.x_disp1, frame.y_disp, frame.z_disp2]
            vertix2 = [frame.x_disp3, frame.y_disp, frame.z_disp2]
            vertix3 = [frame.x_disp3, frame.y_disp, frame.z_disp4]
            vertix4 = [frame.x_disp1, frame.y_disp, frame.z_disp4]
        elif isinstance(frame, FrameZ):
            vertix1 = [frame.x_disp2, frame.y_disp1, frame.z_disp]
            vertix2 = [frame.x_disp2, frame.y_disp3, frame.z_disp]
            vertix3 = [frame.x_disp4, frame.y_disp3, frame.z_disp]
            vertix4 = [frame.x_disp4, frame.y_disp1, frame.z_disp]
            
        return np.array([vertix1, vertix2, vertix3, vertix4, vertix1])
        
    def draw_cage(self, ax):
        ''' Draws cage on a figure
            
            Arg: 
                ax (Figure): figure object to plot cage onto
        '''
        
        for frame in self.cage:
            verts = self.calc_frame_vertices(frame)
            ax.plot(verts[:, 0], verts[:, 1], verts[:, 2], marker = 'o', color = 'blue')
            
    def plot_Bfield(self, xlim, ylim, zlim, grid_num = 4, I_xminus=0.1, I_xplus= 0.1, I_yminus=0, I_yplus= 0, I_zminus= 0, I_zplus= 0):
        ''' Plot a 3D vector field of the B field due to the Helmholtz cage
        
            Args:
                xlim (number): + and - x limits to make grid for  (m)
                ylim (number): + and - y limits to make grid for  (m)    
                zlim (number): + and - z limits to make grid for  (m)
                grid_num (int): number of points per axis in mesh grid
                I_xminus (number): current running through Frame -X   (A)
                I_xplus (number): current running through Frame +X    (A) 
                I_yminus (number): current running through Frame -Y   (A)
                I_yplus (number): current running through Frame +Y    (A)
                I_zminus (number): current running through Frame -Z   (A)
                I_zplus (number): current running through Frame +Z    (A)
        '''
        
        # Create 3D mesh grid for x, y, z over limits
        x = np.linspace(-xlim, xlim, grid_num)
        y = np.linspace(-ylim, ylim, grid_num)
        z = np.linspace(-zlim, zlim, grid_num)
        x_mesh, y_mesh, z_mesh = np.meshgrid(x, y, z)
        
        # Initialize B field components as a 3D mesh
        B_cage_x = np.zeros((grid_num, grid_num, grid_num))
        B_cage_y = np.zeros((grid_num, grid_num, grid_num))
        B_cage_z = np.zeros((grid_num, grid_num, grid_num))
        
        # Find Bx, By, Bz for every x, y, z combination
        # Is what I'm doing naughty here? Yes. Is there another way to do it? Probably. Can it be easily implemented? Who knows?
        for i in range(grid_num):
            for j in range(grid_num):
                for k in range(grid_num):
                    B_field = cage.calc_Bfield(np.array([x[i], y[j], z[k]]), I_xminus=I_xminus, I_xplus= I_xplus, I_yminus= I_yminus, I_yplus= I_yplus, I_zminus= I_zminus, I_zplus= I_zplus)
                    B_cage_x[i, j, k] = B_field[0]
                    B_cage_y[i, j, k] = B_field[1]
                    B_cage_z[i, j, k] = B_field[2]
        
        # Draw vector field
        ax = plt.figure().add_subplot(projection = '3d')
        ax.quiver(x_mesh, y_mesh, z_mesh, B_cage_x, B_cage_y, B_cage_z, length = 0.1, normalize = True, color = 'black')
        
        # Draw cage
        self.draw_cage(ax)
        
        # Labels
        ax.set_title('B-Field of the Helmholtz Cage')
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_zlabel('z (m)')
        plt.show()
        
if __name__ == "__main__":
    # Test case
    L = 1
    num = 80

    # Set displacements of frame from origin (center of cage)
    x_disp = 0.25
    y_disp = 0.25
    z_disp = 0.25
    I = 1
    r = np.array([0.0, 0.0, 0.0])
    lim = 0.4
    
    cage = HelmholtzCage(L, num, x_disp, y_disp, z_disp)
    cage.plot_Bfield(lim, lim, lim)
    
