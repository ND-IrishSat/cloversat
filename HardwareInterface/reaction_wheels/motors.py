'''

motors.py
Authors: Rawan, Nic, Michael P, Andrew

Class for interfacing with reaction wheel motors and Irishat control board

3 input pins:
    dir: 1/0 for cloclwise/counterclockwise
    br: breaking while high
    PWM: motor strength
Output:
    Freq: hall sensor output--high every time wheel rotates
'''

import time
import pigpio
import signal
import sys

DAA = 17
COMU = 24
FREQ = 27
PWM = 12 # PWM input signal
BR = 23 # BR is for brake control
DIRE = 22 # direction control

# Direction control: low (0) is clockwise, high (1) is counterclockwise
CW = 0
CCW = 1

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

        self.pi.set_pull_up_down(self.freq, pigpio.PUD_UP)
        self.pi.set_glitch_filter(self.freq, 300)


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


    def set_speed(self, speed_0_255: int):
        '''
        Set motors to specified speed
        If negative speed, set direction to counterclockwise, otherwise clockwise
        Then set the PWM pin duty cycle to absolute value of speed, converted to duty cycle (0-1_000_000)
        '''

        dir_switch_sleep = 2

        #check if sign of input is different than what we are now
        # If so, sleep for a bit to allow the motor to slow down before switching direction, then switch direction
        curr_dir = self.pi.read(DIRE)
        if (speed_0_255 < 0 and curr_dir == CW):
            # Go from CC to CCW
            time.sleep(dir_switch_sleep)
            self.pi.write(self.dire, CCW)
        elif (speed_0_255 > 0 and curr_dir == CCW):
            # Go from CCW to CC
            time.sleep(dir_switch_sleep)
            self.pi.write(self.dire, CW)

        self.pi.write(self.br, 0)
        pwm_to_set = max(0, min(255, int(abs(speed_0_255))))
        self.pi.hardware_PWM(self.pwm, 20000, int(pwm_to_set/255*1_000_000)) # this converts pwm (0-255) to duty cycle (0-1_000_000)


    def slow_down(self, total_time=1.0, final_stop_condition = False):
        current_duty = self.pi.get_PWM_dutycycle(self.pwm)
        if current_duty <= 0:
            if  final_stop_condition:
                self.pi.write(self.br, 1)
            return

        time_increment = 60
        delay = total_time / time_increment
        step = max(1, int(current_duty / time_increment))  # depends on current duty cycle

        self.pi.write(self.br, 0)  # release brake during ramp
        duty = current_duty
        while duty > 0:
            duty = max(0, duty - step)
            self.pi.hardware_PWM(self.pwm, 20000, duty)
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
            rpm = (frequency_hz * 60.0) / NUMBER_POLE_PAIRS

            direction = self.pi.read(DIRE)
            sign = -1 if direction == 1 else 1
            self.rpm = sign * rpm


def signal_handler(sig, frame, wheel=None):
    print("\nCtrl+C detected. Killing reaction wheel...")
    wheel.kill()
    sys.exit(0)

def register_signal_handler(wheel):
    # Allow us to access ReactionWheel instance in the signal handler by using a lambda
    signal.signal(signal.SIGINT, lambda sig, frame: signal_handler(sig, frame, wheel))


if __name__ == '__main__':
    #stop and close functions
    pi = pigpio.pi()
    if not pi.connected:
        raise RuntimeError("pigpio daemon not running. Start with: sudo systemctl start pigpiod")

    wheel = ReactionWheel(pi, DAA, COMU, FREQ, PWM, BR, DIRE)

    # Register handler for Ctrl+C
    register_signal_handler(wheel)

    # Keep checking freq in the background
    wheel.callback = pi.callback(FREQ, pigpio.RISING_EDGE, wheel.get_rpm_callback)

    # 200 = ~700 rpm
  #  wheel.set_speed(300)
    for i in range(-300, 200, 5):
       # print(wheel.getPWMFrequency())
        wheel.set_speed(i)
        print(f"RPM: {wheel.rpm:.2f}")
        time.sleep(20/60)

    wheel.kill()
    pi.stop()

