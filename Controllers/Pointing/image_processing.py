'''
image_processing.py
Authors: Andrew, Brian, Kris, Rawan, Daniel, Chau, Andres, Abe, Sophie

Image processing script for finding horizon edges, regression line, pitch and roll of satellite
Input: sample satellite Earth Horizon Sensor (EHS) images

NOTE: removed because it's prioprietary
See https://github.com/ND-IrishSat/NearSpace to replace

'''

import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys, os
import math
import time
import json

# if RUNNING_MAYA:
    # from Nearspace.params import *

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from params import *


def processImage(image=None, degree="auto", img_name = None):
    return None, None, None, None, None, None


def firmwareImageProcessing(image):
    return None, None, None, None, None, None
