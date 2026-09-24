'''
camera.py
Author: Andrew Gaylord

Represents our Earth Horizon Sensor (EHS) cams and their current image processing output

'''

class Camera():

    def __init__(self):

        # store pitch and roll of found horizon (None = all earth or all space)
        self.pitch = -1
        self.roll = -1
        # store the percentage of frame filled with earth
        self.alpha = -1
        # array of the average intensity of our 4 edges (top, right, bottom, left, respectively) (all between [0-1])
        self.edges = [-1, -1, -1, -1]

        # x, y coordinates of the midpoint of the horizon line
        self.midpoint = (-1, -1)

        # ((x1, y1), (x2, y2)) represent line endpoints
        self.ends = ((-1, -1), (-1, -1))