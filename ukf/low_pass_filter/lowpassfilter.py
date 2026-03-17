'''
applying a loss pass filter on magnetometer and gyroscope signals
3/5/2026
'''

import numpy as np

'''
equation for Low Pass Filter: y[i] = alpha * x[i] + (1 - alpha) * y[i-1]

x[n] = current sensor reading (magnetometer or gyro)
y[n] = filtered output
y[n−1] = previous filtered value
α = smoothing factor (between 0 and 1)
'''
class LowPassFilter:
    def __init__(self, dt, tau):
        self.dt = float(dt)
        self.tau = float(tau)
        self.alpha = self.dt / (self.tau + self.dt)
        self.prev_output = None

    def apply(self, input_signal):
        input_signal = np.asarray(input_signal, dtype=float)

        if self.prev_output is None:
            output = input_signal.copy()
        else:
            output = self.alpha * input_signal + (1 - self.alpha) * self.prev_output

        self.prev_output = output
        return output
    
