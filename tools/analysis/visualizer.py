'''
visualizer.py
Authors: Juwan Jacobe, Cody, Dao Wei, Martin, Jacob
Last updated: 3/3/24

Attitude propogator and cubesat visualizer from protOat repo

'''

import numpy as np
import pygame
from pygame.locals import * 
from OpenGL.GL import *
from OpenGL.GLU import *
import matplotlib.animation as animation

from pyquaternion import Quaternion


##################################################################################################################
## AttitudePropagator - the important part, propagates states numerically using constant angular velocity model ##
##################################################################################################################
class AttitudePropagator:
    def __init__(self, q_init: np.ndarray, w_init: np.ndarray):
        ''' For generating a set of states for a constant angular velocity model, given an initial angular velocity vector

        Args:
            q_init (np.ndarray, (1x4)): initial quaternion
            init_w (np.ndarray, (1x3)): initial angular velocity

        '''
        self.q_init = q_init
        self.w = w_init

    def attitude_propagator(self, quaternion: np.ndarray, w_vect: np.ndarray, dt: float, normalize: bool):
        ''' Function to propagate quaternion/attitude given an angular velocity vector w. Equations of motion
        derived from https://ahrs.readthedocs.io/en/latest/filters/angular.html

        Form of propagtion equation is 
                    q_t+1 = K*q_t
        where K is
                ( cos(w*t/2)*I_4 + 1/w * sin(w*t/2) * Big_Omega(w) )

        See reference for more info

        Args:  
            quaternion (np.ndarray): quaternion before propagation step
            w_vect (np.ndarray): angular velocity array
            dt (float): timestep
            normalize (bool): boolean, where if True, normalize the new quaternion
        '''
        
        # Store norm of w_vect as separate variable, as w_magnitude. Also separate out components of w
        w_magnitude = np.linalg.norm(w_vect)
        w_x = w_vect[0]
        w_y = w_vect[1]
        w_z = w_vect[2]

        # Calculate factors for calculating matrix K
        cos_fact = np.cos(w_magnitude * dt/2)
        sin_fact = np.sin(w_magnitude * dt/2)
        big_omega =  np.array([[0, -w_x, -w_y, -w_z],
                               [w_x,  0,  w_z, -w_y],
                               [w_y, -w_z, 0,   w_x],
                               [w_z,  w_y, -w_x,  0]])

        # Putting together to calculate matrix K
        K = cos_fact * np.identity(4) + (1/w_magnitude) * sin_fact * big_omega

        # Propagate q_t+1
        new_quaternion = K @ quaternion

        # Normalize -- technically not needed but recommended to get rid off round off errors
        new_quaternion = new_quaternion / np.linalg.norm(new_quaternion)

        return new_quaternion

    def propagate_states(self, t_0: float, t_f: float, N: int):
        ''' Propagate attitude from its initial conditions from t = t_0 to t = t_f, using
        N number of steps

        Args:
            t_0 (float): starting time
            t_f (float): ending time
            N (int): number of steps between t_0 and t_f
        '''

        # Calculate dt from time interval and N number of steps
        dt = (t_f - t_0)/N 

        # Allocate state matrix and save initial quaternion to first row of state matrix
        quaternion_states = np.zeros((N+1, 4))
        quaternion_states[0] = self.q_init

        # Propagate N number of times
        for i in np.arange(1, N+1, 1):
            quaternion_states[i] = self.attitude_propagator(quaternion_states[i-1], self.w, dt, normalize = True)

        return quaternion_states

##############################################################################################################
## Display and 3D rendering stuff, not very important tbh, unless you're into it, jk jk haha... unless?   ####
##############################################################################################################
def rotate_points_by_quaternion(points, quaternion):
    ''' Rotates set of 3-vectors by the transformation described by the input quaternion

        Args:
            points (np.ndarray, (Nx3)): array of 3-vectors
            quaternion (np.ndarray, (1x4)): quaternion, using q0, q1, q2, q3 convention where q0 is the scalar component
    
    '''
    v_prime = np.zeros(points.shape)
    
    quaternion = Quaternion(quaternion)
    
    for i in np.arange(len(points)):
        v = points[i]
        v_prime[i] = quaternion.rotate(v)
    
    return v_prime

def Draw(vertices, edges):
    ''' Draw edges using defined vertices
    '''
    
    glBegin(GL_LINES)

    for edge in edges:
        for vertex in edge:
            glVertex3fv(vertices[vertex])

    glEnd()

def game_visualize(states, a):
    ''' Visualization of time evolution of state using PyGame + OpenGL
    '''     

    # Initial vertices describe cube's position
    vertices_cube = np.array([[1, -1, -1], [1, 1, -1],[-1, 1, -1],[-1, -1, -1],\
        [1, -1, 1],[1, 1, 1],[-1, -1, 1],[-1, 1, 1]])
    
    # Vertices defining label z on one of the cube's face
    vertices_z_label = np.array([[-0.90, 0.90, 1], [-0.80, 0.90, 1], [-0.90, 0.80, 1], [-0.80, 0.80, 1]])
    edges_z_label = (
        (0, 1),
        (1, 2),
        (2, 3)
    )

    # Vertices defining label y on one of the cube's face
    vertices_y_label = np.array([[-0.85, 1, 0.85], [-0.70, 1, 0.85], [-0.775, 1, 0.75], [-0.775, 1, 0.70]])
    edges_y_label = (
        (0, 2),
        (1, 2),
        (2, 3)
    )

    # Vertices defining label x on one of the cube's face
    vertices_x_label = np.array([[1, -0.90, 0.90], [1, -0.80, 0.80], [1, -0.90, 0.80], [1, -0.80, 0.90]])
    edges_x_label = (
        (0, 1),
        (2, 3)
    )

    # List of tuples describing set of commands of how to draw edges. Ex: The first tuple, (0, 1), describes a command to draw a line from vertix 0 to vertix 1
    edges_cube = [
        (0,1),
        (0,3),
        (0,4),
        (2,1),
        (2,3),
        (2,7),
        (6,3),
        (6,4),
        (6,7),
        (5,1),
        (5,4),
        (5,7),
        ]

    # Define vertices of reaction wheels using lin space to define circle
    num = 100
    theta = np.linspace(0, 2*np.pi, num)
    vertices_RW_z = np.zeros((num, 3))
    vertices_RW_y = np.zeros((num, 3))
    vertices_RW_x = np.zeros((num, 3))

    sines = 0.5*np.cos(theta)
    cosines = 0.5*np.sin(theta)    

    vertices_RW_z[:, 0] = sines
    vertices_RW_z[:, 1] = cosines
    vertices_RW_z[:, 2] = 1.1*np.ones(num)

    vertices_RW_y[:, 0] = cosines
    vertices_RW_y[:, 1] = 1.1*np.ones(num)
    vertices_RW_y[:, 2] = sines

    vertices_RW_x[:, 0] = 1.1*np.ones(num)
    vertices_RW_x[:, 1] = sines
    vertices_RW_x[:, 2] = cosines

    # Define two rotation matrices to create fourth reaction wheel from the x-reaction wheel
    ang1 = np.pi/4
    ang2 = -np.pi/4

    c1 = np.cos(ang1)
    s1 = np.sin(ang1)

    Rz_mat = np.array([[c1, -s1, 0],
                       [s1, c1, 0],
                       [0, 0, 1]])
    
    c2 = np.cos(ang2)
    s2 = np.sin(ang2)
    
    Ry_mat = np.array([[c2, 0, s2],
                       [0, 1, 0],
                       [-s2, 0, c2]])
    
    vertices_RW_4 = np.zeros((num, 3))

    for i in np.arange(num):
        vertices_RW_4[i, :] = np.matmul(Rz_mat, np.matmul(Ry_mat, vertices_RW_x[i, :] + np.array([0.40, 0, 0])))

    # Define how to draw reaction wheel's edges
    edges_RW = []

    for i in np.arange(num-1):
        edges_RW = edges_RW + [(i, i+1)]

    edges_RW = edges_RW + [(num-1, 0)]

    # Vertices describing the lines showing the satellite's x, y, z axes / body frame
    vertices_x_axis = np.array([[0, 0, 0], [1.5, 0, 0]])
    vertices_y_axis = np.array([[0, 0, 0], [0, 1.5, 0]])
    vertices_z_axis = np.array([[0, 0, 0], [0, 0, 1.5]])

    edges_axis = (
        (0, 1),
        (0, 0)
    )

    # Initialize window, perspective, and "view" of camera
    if a == 0:
        pygame.init()
        display = (900, 700)
        pygame.display.set_mode(display, DOUBLEBUF|OPENGL)
        gluPerspective(50, (display[0]/display[1]), 0.1, 50.0)
        glTranslatef(0.0, 0.0, -6)

    clock = pygame.time.Clock()
    Q_array = states[:, :4]  # array of quaternions (for each entry, take first four items -> array of [a,b,c,d])
    i = 0

    num_states = states.shape[0]

    # Draw the states iteratively in PyGame
    while True:
        clock.tick_busy_loop(25)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)

        glEnable(GL_DEPTH_TEST)

        vertices_cube_new = rotate_points_by_quaternion(vertices_cube, Q_array[i])

        vertices_z_label_new = rotate_points_by_quaternion(vertices_z_label, Q_array[i])
        vertices_y_label_new = rotate_points_by_quaternion(vertices_y_label, Q_array[i])
        vertices_x_label_new = rotate_points_by_quaternion(vertices_x_label, Q_array[i])

        vertices_x_axis_new = rotate_points_by_quaternion(vertices_x_axis, Q_array[i])
        vertices_y_axis_new = rotate_points_by_quaternion(vertices_y_axis, Q_array[i])
        vertices_z_axis_new = rotate_points_by_quaternion(vertices_z_axis, Q_array[i])

        vertices_RW_z_new = rotate_points_by_quaternion(vertices_RW_z, Q_array[i])
        vertices_RW_y_new = rotate_points_by_quaternion(vertices_RW_y, Q_array[i])
        vertices_RW_x_new = rotate_points_by_quaternion(vertices_RW_x, Q_array[i])
        vertices_RW_4_new = rotate_points_by_quaternion(vertices_RW_4, Q_array[i])

        Draw(vertices_cube_new, edges_cube)

        Draw(vertices_z_label_new, edges_z_label)
        Draw(vertices_y_label_new, edges_y_label)
        Draw(vertices_x_label_new, edges_x_label)

        Draw(vertices_RW_z_new, edges_RW)
        Draw(vertices_RW_y_new, edges_RW)
        Draw(vertices_RW_x_new, edges_RW)
        Draw(vertices_RW_4_new, edges_RW)

        Draw(vertices_x_axis_new, edges_axis)
        Draw(vertices_y_axis_new, edges_axis)
        Draw(vertices_z_axis_new, edges_axis)

        pygame.display.flip()
        i += 1

        if i == num_states:
            break

def draw_cube():
    print("   +------+")  # Top face
    print("  /      /|")
    print(" /      / |")
    print("+------+  |")
    print("|      |  +")  # Right face
    print("|      | /")
    print("+------+")