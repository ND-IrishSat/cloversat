"""Setpoint injector for hardware-in-the-loop runs.

Purpose:  Feed attitude/rate setpoints into a HIL run (keyboard or scripted
          sequence), replacing the pointing target the state manager would pick.
Inputs:   Keypresses or a setpoint script; current attitude.
Outputs:  ``AttitudeTarget`` for the controller.
Status:   [LATER]
Port from: legacy/Main/HardwareScripts/bertha.py (``ABSOLUTE_KEY_ANGLES``,
           ``RELATIVE_ROTATION_KEYS``, keyboard handling)
Related open questions: none
"""
from __future__ import annotations

from typing import Optional

from cloversat.datatypes.state_types import AttitudeTarget
from cloversat.datatypes.vector_types import Quat


class SetpointInjector:
    """Produces testbed setpoints relative to a reference attitude."""

    def __init__(self, q_reference: Quat) -> None:
        ...

    def poll(self, q_current: Quat) -> Optional[AttitudeTarget]:
        """Return a new target if one was requested since the last poll."""
        raise NotImplementedError
