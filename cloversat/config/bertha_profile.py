"""BERTHA testbed profile.

Purpose:  Which devices are real on the testbed, 1-axis limits, safety caps.
Inputs:   ``flight_profile.build_flight_config()``.
Outputs:  ``build_bertha_config()``.
Status:   [LATER]
Port from: legacy/params.py (``RUNNING_1D``, ``FREEDOM_OF_MOVEMENT_AXES``),
           legacy/Main/HardwareScripts/bertha.py
"""
from __future__ import annotations

from cloversat.config.flight_profile import FlightConfig


def build_bertha_config() -> FlightConfig:
    """Build a ``FlightConfig`` for the 1-axis BERTHA testbed."""
    raise NotImplementedError
