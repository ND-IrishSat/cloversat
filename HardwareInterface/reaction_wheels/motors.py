'''
motors.py
Authors: Rawan, Nic R, Michael P, Andrew

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
from USB_serial import USBSerialManager

# Set this flag to True to use the Arduino for PWM control
USING_ARDUINO = False

DAA = 17
COMU = 24
FREQ = 27
PWM = 12 # PWM input signal (used for pigpio PWM)
BR = 23 # BR is for brake control (breaks while high)
DIRE = 22 # direction control

# Direction control: low (0) is clockwise, high (1) is counterclockwise
CW = 0
CCW = 1

MAX_PWM = 255

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

        self.serial_manager = None
        if USING_ARDUINO:
            self.serial_manager = USBSerialManager()

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
        # comu should be hanging (disconnected)
        self.pi.write(self.comu, 1)

        if USING_ARDUINO:
            # devices = self.serial_manager.list_usb_devices()
            #if not devices:
            #    raise RuntimeError("No USB devices found for Arduino connection.")
            # Connect to the first device, assuming it's the Arduino
            # device_path = devices[0]['device']
            device_path = "/dev/ttyACM0"
            if not self.serial_manager.connect_device(device_path, 115200, 'motor_controller'):
                raise RuntimeError(f"Failed to connect to Arduino at {device_path}")
            print("Arduino connected for PWM control.")
            time.sleep(2) # Wait for Arduino to initialize
        else:
            # Initialize hardware PWM OFF
            self.pi.hardware_PWM(self.pwm, 20000, 0)

        time.sleep(0.3)


    def set_speed(self, pwm_input: int):
        '''
        Set motors to specified pwm speed (-255 to 255)
        If negative speed, set direction to counterclockwise, otherwise clockwise
        Then set the PWM pin duty cycle to absolute value of speed, converted to duty cycle (0-1_000_000)
        '''

        # Max time to wait for wheel to slow down (seconds)
        dir_switch_sleep = 2

        #check if sign of input is different than what we are now
        # If so, sleep for a bit to allow the motor to slow down before switching direction, then switch direction
        curr_dir = self.pi.read(DIRE)
        if (pwm_input < 0 and curr_dir == CW):
            # Go from CC to CCW
            '''choose direction
                stop/switch safely (brake)
                set new duty cycle with pwm_to_set to actually run at requested speed.'''
            self.slow_down(target_rpm=0, max_time=dir_switch_sleep)
            self.pi.write(self.dire, CCW)

        elif (pwm_input > 0 and curr_dir == CCW):
            # Go from CCW to CC
            self.slow_down(target_rpm=0, max_time=dir_switch_sleep)
            self.pi.write(self.dire, CW)

        self.pi.write(self.br, 0)
        pwm_to_set = max(0, min(MAX_PWM, int(abs(pwm_input))))

        if USING_ARDUINO:
            self.serial_manager.send_pwm_byte('motor_controller', pwm_to_set)
            #time.sleep(.5)
            #response = self.serial_manager
        else:
            self.pi.hardware_PWM(self.pwm, 20000, int(pwm_to_set/MAX_PWM*1_000_000)) # this converts pwm (0-255) to duty cycle (0-1_000_000)


    def slow_down(self, target_rpm=0.0, max_time=4.0):
        target_rpm = abs(target_rpm)
        step_duty = 20
        sleep_dt = 0.05
        start_time = time.time()

        self.pi.write(self.br, 1)  # set brake so we slow down

        while True:
            current_rpm = abs(self.rpm)
            if current_rpm <= target_rpm:
                break
            if time.time() - start_time > max_time:
                break

            if target_rpm != 0:
                # slowly decrease wheel speed by reducing duty cycle in steps until we reach target RPM or max time
                current_duty = self.pi.get_PWM_dutycycle(self.pwm)
                new_duty = max(0, current_duty - step_duty)
                self.pi.hardware_PWM(self.pwm, 20000, new_duty)
            time.sleep(sleep_dt)

        self.pi.write(self.br, 0) # release brake after slowing down


    def getPWMFrequency(self):
        return self.pi.get_PWM_frequency(self.pwm)


    def kill(self):
        if self.callback:
            self.callback.cancel()

        if USING_ARDUINO and self.serial_manager:
            print("Killing motor (Arduino)...")
            self.serial_manager.send_pwm_byte('motor_controller', 0)
            self.serial_manager.close_all()
        else:
            self.pi.hardware_PWM(self.pwm, 20000, 0)

        # Set break to stop quickly
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


def main():
    #stop and close functions
    pi = pigpio.pi()
    if not pi.connected:
        raise RuntimeError("pigpio daemon not running. Start with: sudo systemctl start pigpiod")

    wheel = ReactionWheel(pi, DAA, COMU, FREQ, PWM, BR, DIRE)

    # Register handler for Ctrl+C
    register_signal_handler(wheel)

    # Keep checking freq in the background
    wheel.callback = pi.callback(FREQ, pigpio.RISING_EDGE, wheel.get_rpm_callback)

    wheel.set_speed(30)
    print(f"RPM: {wheel.rpm:.2f}")
    time.sleep(6)
    wheel.set_speed(140)
    print(f"RPM: {wheel.rpm:.2f}")
    time.sleep(6)
    wheel.set_speed(MAX_PWM)
    print(f"RPM: {wheel.rpm:.2f}")
    time.sleep(6)
    wheel.kill()
    pi.stop()
    return

    for i in range(0, MAX_PWM, 5):
       # print(wheel.getPWMFrequency())
        print(f"Setting PWM {i}")
        wheel.set_speed(i)
        print(f"RPM: {wheel.rpm:.2f}")
        time.sleep(10/60)
    wheel.set_speed(200)
    time.sleep(10)

    wheel.kill()
    pi.stop()

if __name__ == "__main__":
    main()
