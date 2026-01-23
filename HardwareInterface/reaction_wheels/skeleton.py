#when running add: 
#sudo apt-get install pigpio python3-pigpio
#sudo systemctl enable pigpiod
#sudo systemctl start pigpiod


#!/usr/bin/env python3
import time
import pigpio

# BCM pin numbers (your mapping)
DAA  = 17
COMU = 24
FREQ = 27
PWM  = 25
BR   = 23
DIR  = 22

NUMBER_POLE_PAIRS = 4

# Shared RPM state
last_tick = None
rpm = 0.0

def fg_callback(gpio, level, tick):
    """
    Called on each rising edge of FREQ.
    tick is in microseconds (pigpio's internal time base).
    """
    global last_tick, rpm

    if level != 1:
        return

    if last_tick is None:
        last_tick = tick
        return

    # pigpio.tickDiff handles wrap-around safely
    interval_us = pigpio.tickDiff(last_tick, tick)
    last_tick = tick

    if interval_us > 0:
        frequency_hz = 1_000_000.0 / interval_us
        rpm = (frequency_hz * 60.0) / NUMBER_POLE_PAIRS


def set_pin_three_state(pi, pin, mode_sel):
    """
    mode_sel:
      0 -> drive LOW
      1 -> drive HIGH
      2 -> INPUT (high impedance)
    """
    if mode_sel < 2:
        pi.set_mode(pin, pigpio.OUTPUT)
        pi.write(pin, mode_sel)
    else:
        pi.set_mode(pin, pigpio.INPUT)
        # optional: choose pull up/down if you need it:
        # pi.set_pull_up_down(pin, pigpio.PUD_OFF)


def main():
    global rpm

    pi = pigpio.pi()
    if not pi.connected:
        raise RuntimeError("pigpio daemon not running. Start with: sudo systemctl start pigpiod")

    # Set up pins
    pi.set_mode(DAA, pigpio.OUTPUT)
    pi.set_mode(COMU, pigpio.OUTPUT)
    pi.set_mode(FREQ, pigpio.INPUT)
    pi.set_mode(PWM, pigpio.OUTPUT)
    pi.set_mode(BR, pigpio.OUTPUT)
    pi.set_mode(DIR, pigpio.OUTPUT)

    # If your FG/Hall output is open-collector/open-drain, enable a pull-up:
    # pi.set_pull_up_down(FREQ, pigpio.PUD_UP)

    # Initial states (matching your Arduino setup)
    pi.write(BR, 0)      # brake low
    pi.write(DIR, 0)     # direction low
    pi.write(DAA, 0)
    pi.write(COMU, 0)

    # PWM: Arduino analogWrite(128) ~= 50% duty.
    # pigpio hardware PWM uses duty 0..1_000_000
    pwm_freq_hz = 20000  # choose a reasonable PWM freq; change if your driver expects something else
    pi.hardware_PWM(PWM, pwm_freq_hz, int(128 / 255 * 1_000_000))

    # Delay like Arduino delay(10000)
    time.sleep(10)

    # Attach rising-edge callback for FREQ
    cb = pi.callback(FREQ, pigpio.RISING_EDGE, fg_callback)

    try:
        # Main test loops (mirrors your Arduino logic)
        for duty_8bit in range(63, 256, 64):
            pi.hardware_PWM(PWM, pwm_freq_hz, int(duty_8bit / 255 * 1_000_000))
            time.sleep(4)

            for daa in range(0, 3):
                set_pin_three_state(pi, DAA, daa)

                for comu in range(0, 3):
                    set_pin_three_state(pi, COMU, comu)

                    time.sleep(2)

                    print(f"PWM: {duty_8bit}  DAA: {daa}  COMU: {comu}  RPM: {rpm:.2f}")

        # The extra brake/PWM sequences from your Arduino code
        pi.write(BR, 0)
        pi.hardware_PWM(PWM, pwm_freq_hz, 1_000_000)
        time.sleep(5)

        pi.hardware_PWM(PWM, pwm_freq_hz, 0)
        time.sleep(10)

        pi.write(BR, 1)
        pi.hardware_PWM(PWM, pwm_freq_hz, 1_000_000)
        time.sleep(5)

        pi.hardware_PWM(PWM, pwm_freq_hz, 0)
        time.sleep(5)

    finally:
        cb.cancel()
        pi.hardware_PWM(PWM, 0, 0)
        pi.write(BR, 0)
        pi.stop()


if __name__ == "__main__":
    main()
