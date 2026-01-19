''''
graphing.py
Author: Andrew Gaylord

contains the graphing functionality for kalman filter visualization
can plot data, state (quaternion and angular velocity), and 3D vectors
when 2D data is plotted, it is also saved as a png file in the plotOutput directory

'''


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
from saving import *


def plot_multiple_lines(data, labels, title, x=0, y=0, text="", fileName="default.png"):
    ''' 
    plots multiple lines on the same graph
    stores plot as a png file in the plotOutput (global var in saving.py) directory
    note: does not call plt.show()

    @params:
        data: A list of lists of data points.
        labels: A list of labels for each line.
        title: title for graph
        x, y: pixel location on screen
        text: sentence to go on bottom of graph
        fileName: name of png file to save graph as
    '''
    # Create a figure and axes
    fig, ax = plt.subplots()

    # Plot each line.
    for i, line in enumerate(data):
        ax.plot(line, label=labels[i])

    # Add a legend
    ax.legend()

    plt.title(title)

    if text != "":
        fig.text(.01, .01, "    " + text)
        # fig.subplots_adjust(top=0.5)

    # moves figure to x and y coordinates
    move_figure(fig, x, y)

    # save the figure as a png using saving.py
    saveFig(fig, fileName)

    # Show the plot
    # plt.show()


def move_figure(f, x=0, y=0):
    ''' 
    move figure's upper left corner to pixel (x, y)
    
    @params:
        f: figure object returned by calling pt.subplot
        x, y: coordinates to move to
    '''
    backend = matplotlib.get_backend()
    if backend == 'TkAgg':
        f.canvas.manager.window.wm_geometry("+%d+%d" % (x, y))
    elif backend == 'WXAgg':
        f.canvas.manager.window.SetPosition((x, y))
    # else:
        # This works for QT and GTK
        # You can also use window.setGeometry
        # f.canvas.manager.window.move(x, y)


def plot_xyz(data, title, x=0, y=0, fileName="default.png"):
    ''' 
    given an arbitrary numpy 2D list (where every element contains x, y, z or a quaternion a, b, c, d), plot them on a 2D graph

    @params:
        data: 2D array of xyz coordinates or quaternions
        title: graph title
        x, y: coordinates for graph on screen
        fileName: name of png file to save graph as
    '''

    newData = data.transpose()

    if len(data[0]) == 4:
        # plot quaternion
        plot_multiple_lines(newData, ["a", "b", "c", "d"], title, x, y, fileName=fileName)
    else:
        # plot xyz
        plot_multiple_lines(newData, ["x", "y", "z"], title, x, y, fileName=fileName)


def plotAngles(data, title, fileName="default.png"):
    '''
    plot a list of Euler Angles with special graph formatting

    @params:
        data: 2D array of Euler Angles
        title: graph title
        fileName: name of png file to save graph as
    '''
    data = data.transpose()
    labels = ["roll (x)", "pitch (y)", "yaw (z)"]

    # Create a figure and axes
    fig, ax = plt.subplots()

    # Plot each line.
    for i, line in enumerate(data):
        ax.plot(line, label=labels[i])

    # Add a legend
    ax.legend()

    # Set the y-axis ticks and labels in terms of pi
    # The ticks should be multiples of pi (e.g., -pi, -pi/2, 0, pi/2, pi)
    pi_ticks = [-np.pi, -np.pi/2, 0, np.pi/2, np.pi]
    pi_labels = [r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$']

    # Set the y-axis limits if needed
    plt.ylim([pi_ticks[0]-.2, pi_ticks[len(pi_ticks)-1]+.2])

    # Apply the custom ticks
    plt.yticks(pi_ticks, pi_labels)

    plt.title(title)

    # save the figure as a png using saving.py
    saveFig(fig, fileName)


def plotState_xyz(data, ideal=False):
    '''
    plots IrishSat's 7 dimensional state (quaternion and angular velocity)
    creates 2 graphs that show quaternion and angular velocity for all time steps

    @params:
        data: 2D array of states to be graphed
        ideal: boolean value that signifies where to place graphs
            true = top left, false = bottom middle
    '''

    # separate quaternion and angular velocity from data array
    quaternions = np.array([data[0][:4]])
    velocities = np.array([data[0][4:]])

    for i in range(1, len(data)):
        quaternions = np.append(quaternions, np.array([data[i][:4]]), axis=0)
        velocities = np.append(velocities, np.array([data[i][4:]]), axis=0)

    if ideal:
        # plot two graphs in top left
        plot_xyz(velocities, "Ideal Angular Velocity", 50, 0, fileName="idealVelocity.png")
        plot_xyz(quaternions, "Ideal Quaternion", 0, 0, fileName="idealQuaternion.png")
    else:
        plot_xyz(velocities, "Filtered Angular Velocity", 575, 370, fileName="filteredVelocity.png")
        plot_xyz(quaternions, "Filtered Quaternion", 525, 370, fileName="filteredQuaternion.png")


def plotData_xyz(data):
    '''
    plots 6 dimensional data
    creates 2 graphs that show magnetic field and angular velocity for all time steps

    @params:
        data: 2D array of data readings (magnetometer and gyroscoped)
    '''

    # separate magnetometer and gyroscope data
    magData = np.array([data[0][:3]])
    gyroData = np.array([data[0][3:]])

    for i in range(1, len(data)):
        magData = np.append(magData, np.array([data[i][:3]]), axis=0)
        gyroData = np.append(gyroData, np.array([data[i][3:]]), axis=0)

    plot_xyz(gyroData, "Gyroscope Data", 1100, 0, fileName="gyroData.png")
    plot_xyz(magData, "Magnetometer Data", 1050, 0, fileName="magData.png")

    
def plot3DVectors(vectors, plotSegment=111):
    '''
    plots 3D vectors
    note: does not call plt.show()
    
    @params:
        vectors: list of 3D vectors to plot
        plotSegment (optional): 3 digit number 'nmi'
            split plot into n by m plots and picks ith one
            ex: 121 splits into 1 row, 2 columns and picks the first (the left half)
    '''  

    # set bounds of graph
    bounds = [-1.5, 1.5]

    result = np.array([np.concatenate(([0, 0, 0], vectors[0]))])
    # print(result)

    # formats vectors correctly so that they can be graphed
    for i in range (1, len(vectors)):
        v = vectors[i]

        result = np.append(result, np.array([np.concatenate(([0, 0, 0], v))]), axis=0)

    # original = np.concatenate(([0, 0, 0], vectors[0]))
    # rotated = np.concatenate(([0, 0, 0], vectors[1]))
    # rotated2 = np.concatenate(([0, 0, 0], vectors[2]))
    # soa = np.array([original, rotated])
    # X, Y, Z, U, V, W = zip(*soa)
    # print(X, Y, Z, U, V, W)

    fig = plt.figure()
    ax = fig.add_subplot(plotSegment, projection='3d')
    # ax.quiver(X, Y, Z, U, V, W, cmap=cm.coolwarm)
    # print(np.array([original, rotated, rotated2]))
    # ax.quiver(*[x for x in zip(*np.array([original, rotated, rotated2]))])

    # add a variable number of vectors to plot
    ax.quiver(*[x for x in zip(*result)])

    ax.set_xlim([bounds[0], bounds[1]])
    ax.set_ylim([bounds[0], bounds[1]])
    ax.set_zlim([bounds[0], bounds[1]])

    # plt.show()


def plotData3D(data, numVectors, plotSegment=111):
    '''
    plot some number of evenly-spaced elements of data on the same 3D graph

    @params:
        data: 2D list of sensor data (magnetomer, gyro)
        numVectors: number of elements of data to print
        plotSegment (optional): 3 digit number 'nmi'
            split plot into n by m plots and picks ith one
            ex: 121 splits into 1 row, 2 columns and picks the first (the left half)
    '''

    section = int(len(data) / (2*numVectors))

    # split magnetomer portion of data
    result = np.array([data[0][:3]])

    # take numVector elements of data to be printed
    for i in range(1, numVectors):
        result = np.append(result, np.array([data[i*section][:3]]), axis=0)
    
    # print(result)

    # plot3DVectors(np.array([ukf.B_true, data[50][:3], data[100][:3], data[150][:3]]), 121)
    plot3DVectors(result, plotSegment)
