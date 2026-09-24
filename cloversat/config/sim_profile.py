"""Simulation profile: flight params with simulated devices.

Purpose:  Same params as flight, ``device_backend="sim"``, plus sim settings.
Inputs:   ``flight_profile.build_flight_config()``, ``sim/sim_params.py``.
Outputs:  ``build_sim_config()``.
Status:   [LATER]
Port from: legacy/params.py (SIM OPTIONS section)
"""
from __future__ import annotations

from cloversat.config.flight_profile import FlightConfig


def build_sim_config() -> FlightConfig:
    """Build a ``FlightConfig`` that selects ``devices/sim/`` implementations."""
    raise NotImplementedError
