"""Flight profile: assembles ``FlightConfig`` and selects hardware drivers.

Purpose:  One place that builds every params object for a flight run and
          says which device backend to use.
Inputs:   Params dataclasses from this package.
Outputs:  ``FlightConfig``, ``build_flight_config()``.
Status:   [NEW] [BLOCKED: HW]
Port from: legacy/params.py

``sim_profile.py`` and ``bertha_profile.py`` build the same ``FlightConfig``
with a different ``device_backend``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from cloversat.config.control_params import ControlParams
from cloversat.config.device_params import DeviceParams
from cloversat.config.estimation_params import EstimationParams
from cloversat.config.executive_params import ExecutiveParams
from cloversat.config.spacecraft_params import SpacecraftParams


@dataclass
class FlightConfig:
    """Everything ``executive/main.py`` needs to build the flight software.

    Attributes:
        device_backend: "hardware", "sim", or "bertha".
    """

    spacecraft: SpacecraftParams = field(default_factory=SpacecraftParams)
    devices: DeviceParams = field(default_factory=DeviceParams)
    estimation: EstimationParams = field(default_factory=EstimationParams)
    control: ControlParams = field(default_factory=ControlParams)
    executive: ExecutiveParams = field(default_factory=ExecutiveParams)
    device_backend: str = "hardware"


def build_flight_config() -> FlightConfig:
    """Build the flight configuration with hardware drivers selected."""
    raise NotImplementedError
