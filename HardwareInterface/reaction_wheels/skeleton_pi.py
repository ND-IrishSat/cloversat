#!/usr/bin/env python3
"""
Port of skeleton.ino for Raspberry Pi using pigpio.

- Uses hardware PWM for smoother drive (pins 12/13/18/19 only).
- Computes RPM from hall FG pulses on rising edge.
- Sweeps PWM and DAA/COMU states like the Arduino demo.
"""

import time
import pigpio

# BCM pin numbers — change to match your wiring.
DAA_PIN = 17       # tri-state control 0/1/Hi-Z
COMU_PIN = 24      # tri-state control 0/1/Hi-Z
FREQ_PIN = 27      # hall / FG input
PWM_PIN = 12       # MUST be 12, 13, 18, or 19 for hardware_PWM
BR_PIN = 23        # brake (active high)
DIR_PIN = 22       # direction (0/1)

PWM_FREQ_HZ = 20_000
NUMBER_POLE_PAIRS = 4


def duty_from_8bit(val: int) -> int:
    """Convert 0–255 duty to pigpio hardware PWM range 0–1_000_000."""
    val = max(0, min(255, int(val)))
    return val * 1_000_000 // 255


def set_pin_three_state(pi: pigpio.pi, pin: int, mode_sel: int) -> None:
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
        pi.set_pull_up_down(pin, pigpio.PUD_OFF)


class RpmCounter:
    """Tracks RPM from hall sensor pulses."""

    def __init__(self, pi: pigpio.pi, fg_pin: int, pole_pairs: int):
        self._pi = pi
        self._fg_pin = fg_pin
        self._pole_pairs = pole_pairs
        self._last_tick = None
        self.rpm = 0.0
        self._cb = pi.callback(fg_pin, pigpio.RISING_EDGE, self._cb_func)

    def _cb_func(self, gpio, level, tick):
        if level != 1:
            return
        if self._last_tick is None:
            self._last_tick = tick
            return
        interval_us = pigpio.tickDiff(self._last_tick, tick)
        self._last_tick = tick
        if interval_us > 0:
            freq_hz = 1_000_000.0 / interval_us
            self.rpm = (freq_hz * 60.0) / self._pole_pairs

    def cancel(self):
        self._cb.cancel()


def main():
    pi = pigpio.pi()
    if not pi.connected:
        raise RuntimeError(
            "pigpio daemon not running. Start it with: sudo systemctl start pigpiod"
        )

    # Set up pins
    for pin in (DAA_PIN, COMU_PIN, BR_PIN, DIR_PIN):
        pi.set_mode(pin, pigpio.OUTPUT)
    pi.set_mode(FREQ_PIN, pigpio.INPUT)
    pi.set_pull_up_down(FREQ_PIN, pigpio.PUD_UP)  # hall often open-collector

    # Initial states
    pi.write(BR_PIN, 0)
    pi.write(DIR_PIN, 0)
    pi.write(DAA_PIN, 0)
    pi.write(COMU_PIN, 0)
    pi.hardware_PWM(PWM_PIN, PWM_FREQ_HZ, duty_from_8bit(128))

    # Give hardware time to settle (Arduino delay(10000))
    time.sleep(10)

    rpm_counter = RpmCounter(pi, FREQ_PIN, NUMBER_POLE_PAIRS)

    try:
        # Mirror the Arduino sweep: PWM 63..255 in steps of 64
        for duty_8bit in range(63, 256, 64):
            pi.hardware_PWM(PWM_PIN, PWM_FREQ_HZ, duty_from_8bit(duty_8bit))
            time.sleep(4)

            for daa in (0, 1, 2):
                set_pin_three_state(pi, DAA_PIN, daa)

                for comu in (0, 1, 2):
                    set_pin_three_state(pi, COMU_PIN, comu)
                    time.sleep(2)
                    print(
                        f"PWM: {duty_8bit}  DAA: {daa}  COMU: {comu}  RPM: {rpm_counter.rpm:.2f}"
                    )

        # Final brake / PWM sequences from the Arduino sketch
        pi.write(BR_PIN, 0)
        pi.hardware_PWM(PWM_PIN, PWM_FREQ_HZ, 1_000_000)
        time.sleep(5)

        pi.hardware_PWM(PWM_PIN, PWM_FREQ_HZ, 0)
        time.sleep(10)

        pi.write(BR_PIN, 1)
        pi.hardware_PWM(PWM_PIN, PWM_FREQ_HZ, 1_000_000)
        time.sleep(5)

        pi.hardware_PWM(PWM_PIN, PWM_FREQ_HZ, 0)
        time.sleep(5)

    finally:
        rpm_counter.cancel()
        pi.hardware_PWM(PWM_PIN, 0, 0)
        pi.write(BR_PIN, 0)
        pi.stop()


if __name__ == "__main__":
    main()
