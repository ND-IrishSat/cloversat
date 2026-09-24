# devices/hardware

One `_driver.py` per real part, each implementing one or more interfaces from `../interfaces/`.

| File | Part | Status |
|---|---|---|
| `vn100_driver.py` | VectorNav VN-100 (testbed IMU: gyro, mag, accel) | [MODIFIED] |
| `torquer_driver.py` | Magnetorquer H-bridges | [MODIFIED] |
| `wheel_driver.py` | Reaction wheel motors + control board | [MODIFIED] [BLOCKED: OQ-3] |
| `watchdog_driver.py` | Pi `/dev/watchdog` (bcm2835_wdt) + RTC | [NEW] |
| `storage_driver.py` | File-backed triple-copy persistence | [NEW] |
| `vn100_library/` | Vendored VectorNav `vnpy` library, copied as-is from legacy | [EXISTING] |

The **flight IMU, sun sensor, GPS, health sensor, and deployment** drivers are added once
those parts are chosen (see `docs/open_questions.md`, HW).

Hardware packages (`pigpio`, `adafruit-circuitpython-pca9685`, `smbus2`, `vnpy`, ...) are not in the
top-level `requirements.txt`; add them back as each driver is ported. Drivers must import cleanly
on a dev laptop: import hardware libraries inside `init()`, not at module level.

`vnpy` install: `pip install ./cloversat/devices/hardware/vn100_library` (the legacy copy ships a
Windows-only prebuilt `.pyd`; on the Pi it must be built from `libvncxx`).
