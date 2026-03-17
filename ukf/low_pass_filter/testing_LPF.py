import numpy as np
from lowpassfilter import LowPassFilter

# sampling time (assume 100 Hz)
dt = 0.01

# choose tau values
tau_mag = 0.10
tau_gyro = 0.03

mag_filter = LowPassFilter(dt, tau_mag)
gyro_filter = LowPassFilter(dt, tau_gyro)

input_file = "../../HardwareInterface/sample_data/sensor_data_2.txt"
output_file = "filtered_output.txt"

with open(input_file, "r") as f_in, open(output_file, "w") as f_out:
    for line in f_in:
        line = line.strip()
        if not line:
            continue

        values = [float(x) for x in line.split(",")]

        mag = np.array(values[0:3])
        gyro = np.array(values[3:6])

        mag_f = mag_filter.apply(mag)
        gyro_f = gyro_filter.apply(gyro)

        out = ",".join(map(str, [
            *mag_f,
            *gyro_f
        ]))

        print(out)            # 👈 THIS is your output in terminal
        f_out.write(out + "\n")  # 👈 saved to file