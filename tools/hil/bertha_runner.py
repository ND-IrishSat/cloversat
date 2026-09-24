"""BERTHA 1-axis testbed runner.

Purpose:  Run the flight code on the BERTHA testbed with the real devices the
          bertha profile selects (VN-100, wheel), inject setpoints, and log
          telemetry to CSV for ``tools/analysis``.
Inputs:   ``config/bertha_profile.build_bertha_config()``, operator keypresses
          or ``setpoint_injector``.
Outputs:  CSV log of attitude, rates, wheel commands.
Status:   [LATER]
Port from: legacy/Main/HardwareScripts/bertha.py (reference only)
Related open questions: HW, OQ-3

Known legacy bugs (do not carry over):
    TODO(bug): bertha.py recreates the PID controller every loop, so the
        integral term resets each iteration.
    TODO(bug): bertha.py uses ``csv_headers.append(csv_raw_headers)``; should be
        ``extend`` (it nests a list inside the header row).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from cloversat.config.flight_profile import FlightConfig


def run_bertha(config: FlightConfig, log_path: str, duration_s: Optional[float] = None) -> None:
    """Build the flight software on testbed devices and run until stopped."""
    raise NotImplementedError


def main() -> None:
    """CLI entry point."""
    raise NotImplementedError


if __name__ == "__main__":
    main()
