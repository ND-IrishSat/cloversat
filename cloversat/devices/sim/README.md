# devices/sim  [LATER]

Simulated devices. One `_sim.py` per interface in `../interfaces/`, fed by truth from `cloversat/sim/`
(via `sim/sensor_models.py` for noise, bias and faults). Flight code cannot tell a sim device from a real one.

Expected files:

| File | Implements |
|---|---|
| `gyro_sim.py` | `GyroInterface` |
| `mag_sim.py` | `MagInterface` |
| `accel_sim.py` | `AccelInterface` |
| `sun_sensor_sim.py` | `SunSensorInterface` |
| `gps_sim.py` | `GpsInterface` |
| `health_sim.py` | `HealthInterface` |
| `torquer_sim.py` | `TorquerInterface` (returns dipole to `sim/dynamics.py`) |
| `wheel_sim.py` | `WheelInterface` |
| `deployable_sim.py` | `DeployableInterface` |
| `storage_sim.py` | `StorageInterface` (in-memory) |
| `watchdog_sim.py` | `WatchdogInterface` (sim clock) |
