"""Momentum dump control law.

Purpose:  Magnetorquer dipole that unloads stored wheel momentum.
Inputs:   Field ``B_body`` [T], wheel momentum ``h_wheels_body`` [N*m*s],
          gain k, start/stop thresholds [N*m*s].
Outputs:  Dipole ``m_body`` [A*m^2]; whether dumping is active.
Status:   [NEW]
Port from: none

Law:  m = k (B x h) / |B|^2
Hysteresis: start when |h| > dump_h_start, stop when |h| < dump_h_stop.
|B| == 0 -> zero dipole. The wheel controller absorbs the resulting torque
to hold attitude.

Related open questions: OQ-4 (momentum limits), OQ-6 (when dump preempts).
"""
from __future__ import annotations

from cloversat.datatypes.vector_types import Vec3


def momentum_dump_dipole(B_body: Vec3, h_wheels_body: Vec3, gain: float) -> Vec3:
    """Dump dipole ``m = k (B x h) / |B|^2`` [A*m^2]; zero if |B| == 0."""
    raise NotImplementedError


def dump_active(h_wheels_body: Vec3, was_active: bool, h_start: float, h_stop: float) -> bool:
    """Hysteresis on |h|: turn on above ``h_start``, off below ``h_stop`` [N*m*s]."""
    raise NotImplementedError
