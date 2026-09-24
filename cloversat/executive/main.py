"""Flight software entry point.

Purpose:  Load a profile, build every module with its params object, pick the
          device backend, and launch the periodic tasks (control loop,
          heartbeat, telemetry, slow I/O).
Inputs:   Profile name ("flight", "sim", "bertha") from the command line.
Outputs:  Running flight software; exits only on shutdown.
Status:   [NEW]
Port from: legacy/Main/HardwareScripts/bertha.py (testbed loop),
           legacy/Main/SimScripts/space_sim.py (sim loop)

This is the only module allowed to import ``cloversat.config`` at runtime. It
hands each module its own params object; modules never import config.

One control tick: devices -> estimation (``StateEstimate``) -> state manager
(``ModeCommand``) -> controller (``ActuatorCommand``) -> allocation ->
actuation scheduler -> devices.

Related open questions: OQ-10 (Python on Pi vs C decides the task model).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from cloversat.config.flight_profile import FlightConfig


def load_profile(name: str) -> FlightConfig:
    """Return the ``FlightConfig`` for profile ``name`` ("flight", "sim", "bertha")."""
    raise NotImplementedError


def build_devices(config: FlightConfig) -> dict[str, Any]:
    """Construct device objects for ``config.device_backend``, keyed by role."""
    raise NotImplementedError


def build_modules(config: FlightConfig, devices: dict[str, Any]) -> dict[str, Any]:
    """Construct estimation, control, and executive modules with their params."""
    raise NotImplementedError


def control_tick(modules: dict[str, Any], t: float) -> None:
    """Run one control tick (see module docstring for the data flow)."""
    raise NotImplementedError


def run(config: FlightConfig) -> None:
    """Launch the control loop, heartbeat, telemetry, and slow-I/O tasks."""
    raise NotImplementedError


def main(argv: Optional[list[str]] = None) -> int:
    """Command-line entry point. Returns a process exit code."""
    raise NotImplementedError


if __name__ == "__main__":
    raise SystemExit(main())
