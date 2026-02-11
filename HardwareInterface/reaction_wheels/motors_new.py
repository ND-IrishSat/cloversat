#Reaction wheels

import time
import pigpio

'''
3 input pins:
    dir: 1/0 for cloclwise/counterclockwise
    br: breaking while high
    PWM: motor strength
Output:
    Freq: hall sensor output--high every time wheel rotates
'''

DAA = 17
COMU = 24
FREQ = 27
PWM = 12 # PWM input signal
BR = 23 # BR is for brake control
DIRE = 22 # direction control

NUMBER_POLE_PAIRS = 4

# motor class!
class ReactionWheel:
     #initializing the class

    def __init__(self, pi, daa, comu, freq, pwm, br, dire):
        self.pi = pi
        self.daa = daa
        self.comu = comu
        self.freq = freq
        self.pwm = pwm
        self.br = br
        self.dire = dire
        self.position = 0
        self.last_tick = None
        self.callback = None
        self.rpm = 0.0 #revolutions per minute
        self._setup_()

    def _setup_(self):
        #input pins to the pi
        self.pi.set_mode(self.freq, pigpio.INPUT)

        #output pins from the PI to the wheels pi. set_mode(self.pwm, pigpio.OUTPUT)
        self.pi.set_mode(self.pwm, pigpio.OUTPUT)
        self.pi.set_mode(self.daa, pigpio.OUTPUT)
        self.pi.set_mode(self.comu, pigpio.OUTPUT)
        self.pi.set_mode(self.br, pigpio.OUTPUT)
        self.pi.set_mode(self.dire, pigpio.OUTPUT)

        #initial state
        self.pi.write(self.br, 0)
        #when 0 its clockwise
        self.pi.write(self.dire, 0)
        self.pi.write(self.daa, 0)
        self.pi.write(self.comu, 0)

        # Initialize hardware PWM OFF
        self.pi.hardware_PWM(self.pwm, 20000, 0)
        time.sleep(0.1)

    def _set_speed_(self, duty_0_255: int): #duty call goes from 0 to 255

        if (duty_0_255 < 0):
            self.pi.write(self.dire, 1)
        else:
            self.pi.write(self.dire, 0)

        self.pi.write(self.br, 0)
        duty = max(0, min(255, int(abs(duty_0_255))))
        self.pi.hardware_PWM(self.pwm, 20000, duty * 1_000_000 // 255)

    def slow_down(self, total_time=1.0, final_stop_condition = False):
        current_PWM = self.pi.get_PWM_dutycycle(self.pwm)
        if current_PWM <= 0:
            if  final_stop_condition:
                self.pi.write(self.br, 1)
            return

        time_increment = 60
        delay = total_time / time_increment
        step = max(1, int(current_PWM / time_increment))  # depends on current PWM

        self.pi.write(self.br, 0)  # release brake during ramp
        duty = current_PWM
        while duty > 0:
            duty = max(0, duty - step)
            self.pi.set_PWM_dutycycle(self.pwm, duty)
            time.sleep(delay)

        if final_stop_condition:
            self.pi.write(self.br, 1)

    def getPWMFrequency(self):
        return self.pi.get_PWM_frequency(self.pwm)

    def kill(self):
        self.callback.cancel()
        self.pi.hardware_PWM(self.pwm, 20000, 0)
        self.pi.write(self.br, 1)

    def get_rpm_callback(self, gpio, level, tick):
        """
        Called on each rising edge of FREQ.
        tick is in microseconds (pigpio's internal time base).
        """
        if level != 1:
            return

        if self.last_tick is None:
            self.last_tick = tick
            return

        # pigpio.tickDiff handles wrap-around safely
        interval_us = pigpio.tickDiff(self.last_tick, tick)
        self.last_tick = tick

        if interval_us > 0:
            frequency_hz = 1_000_000.0 / interval_us
            self.rpm = (frequency_hz * 60.0) / NUMBER_POLE_PAIRS


if __name__ == '__main__':
    #stop and close functions
    pi = pigpio.pi()
    if not pi.connected:
        raise RuntimeError("pigpio daemon not running. Start with: sudo systemctl start pigpiod")

    wheel = ReactionWheel(pi, DAA, COMU, FREQ, PWM, BR, DIRE)

    # Keep checking freq in the background
    wheel.callback = pi.callback(FREQ, pigpio.RISING_EDGE, wheel.get_rpm_callback)

    wheel._set_speed_(200)
    for i in range(15):
        print(wheel.getPWMFrequency())
        print(f"RPM: {wheel.rpm:.2f}")
        time.sleep(2)

    wheel.kill()
    pi.stop()