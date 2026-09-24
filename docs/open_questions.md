# Open questions

Files tagged `[BLOCKED: OQ-n]` wait on these decisions. When a question is
settled, record the answer here (date + short rationale) and drop the tag from
the affected files.

| ID | Question | Blocks |
|---|---|---|
| OQ-1 | Attitude filter: MEKF vs extend UKF | `estimation/attitude_filter.py` |
| OQ-2 | Field model: IGRF (degree ~8) vs tilted dipole | `estimation/igrf_model.py` |
| OQ-3 | Wheel command interface: torque/current vs speed setpoint | `devices/interfaces/wheel_interface.py`, `devices/hardware/wheel_driver.py`, `control/wheel_loop.py` |
| OQ-4 | Numeric requirements: detumble threshold/time, pointing accuracy, power margins | `executive/detumble_mode.py`, gains/thresholds in `config/` |
| OQ-5 | Initial attitude acquisition: sun search + TRIAD vs two-epoch mag TRIAD vs filter convergence | `estimation/attitude_init.py`, `executive/orient_transition_mode.py` |
| OQ-6 | Operational sub-mode selection: priority scheduler vs fixed cycle | `executive/operational_mode.py` |
| OQ-7 | SAFE exit criteria and SAFE-mode comms | `executive/safe_mode.py` |
| OQ-8 | No uplink vs radio licensing (transmitter shut-off). Owned by comms. | `executive/telemetry.py`, `estimation/sgp4_propagator.py` |
| OQ-9 | Telemetry record format. Decided with comms. | `executive/telemetry.py`, `datatypes/health_types.py` |
| OQ-10 | Flight language/platform (Python on Pi vs C) | Whole package |
| HW | Flight computer, IMU, sun sensor, GPS receiver, wheel driver, health sensors, deployment mechanism, patch placement, inertia/tip-off rates | `config/spacecraft_params.py`, `config/device_params.py`, `devices/*`, `control/ground_target.py` |

## Decisions

_None yet._
