import csv
import os
import sys
from time import sleep
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

import Simulator.visualizer as simulator

def load_csv_to_arrays(csv_path):
    arrays = []
    with open(csv_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            arrays.append(row[1:5])
        arrays.remove(arrays[0])  # Remove header row
    return arrays

def main():
    arrays = load_csv_to_arrays(os.path.join(PROJECT_ROOT, 'Main', 'HardwareScripts', 'bertha_imu_wheel_log.csv'))
    for i in range(len(arrays)):
        # print(arrays[i])
        simulator.game_visualize(np.array([arrays[i]]), i)
        sleep(0.1)

main()

