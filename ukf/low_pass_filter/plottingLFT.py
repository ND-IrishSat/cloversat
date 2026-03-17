import numpy as np
import matplotlib.pyplot as plt
from lowpassfilter import LowPassFilter

# settings
dt = 0.01
tau = 0.05

filt = LowPassFilter(dt, tau)

raw_data = []
filtered_data = []

input_file = "../../HardwareInterface/sample_data/sensor_data_2.txt"


# read fake sensor file
with open(input_file, "r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        values = [float(x) for x in line.split(",")]

        # example: use gyro x-axis
        x = values[3]  # gx

        y = filt.apply(x)

        raw_data.append(x)
        filtered_data.append(y)

# plot
plt.figure()
plt.plot(raw_data, label="Raw Signal")
plt.plot(filtered_data, label="Filtered Signal")

plt.title("Low-Pass Filter Visualization (Gyro X)")
plt.xlabel("Time Step")
plt.ylabel("Value")
plt.legend()

plt.show()