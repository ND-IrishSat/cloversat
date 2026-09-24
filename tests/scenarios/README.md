# Scenario tests `[LATER]`

End-to-end runs of the real flight code against `cloversat/sim/` through the
`devices/sim/` implementations. Planned scenarios:

- Detumble from tip-off rates to below the DETUMBLE exit threshold (OQ-4).
- Attitude acquisition after detumble (OQ-5).
- Sun pointing through an eclipse entry/exit.
- GPS / ground-station pointing preempting sun pointing (OQ-6).
- Momentum dump triggered by wheel momentum threshold, with hysteresis.
- Deployment sequence: inhibit timer, burn limit, retries, `safe()` on failure.
- Reset mid-mode: boot restores `PersistedState` and re-enters correctly.
- Boot loop: repeated resets trip boot-loop protection.
- Sensor faults (stuck gyro, dead magnetometer) force SAFE via FDIR.
- Watchdog: a stalled subsystem goes LATE → STALLED → FAILED.
