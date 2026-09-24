"""Reference frame transformations.

Purpose:  ECI <-> ECEF (via GMST), ECEF <-> geodetic (WGS-84), body <-> inertial.
Inputs:   Vectors in SI with frame suffixes, UTC time as Julian date.
Outputs:  Vectors in the requested frame; geodetic (lat, lon [rad], alt [m]).
Status:   [NEW]
Port from: none. Replaces ``pyproj`` (used in
           legacy/HardwareInterface/gps_interface.py via the deprecated
           ``pyproj.transform``) so flight code has no pyproj dependency.

ECI here means the frame the orbit propagator outputs (TEME from SGP4 is
treated as ECI for this stage; note any difference where it matters).

Related open questions: OQ-10 (flight language may change implementation).
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt

from cloversat.datatypes.vector_types import Quat, Vec3


def gmst(jd_ut1: float) -> float:
    """Greenwich mean sidereal time [rad] at Julian date ``jd_ut1``."""
    raise NotImplementedError


def eci_to_ecef_matrix(jd_ut1: float) -> npt.NDArray[np.float64]:
    """3x3 rotation taking ECI vectors to ECEF (Earth rotation by GMST)."""
    raise NotImplementedError


def eci_to_ecef(r_eci: Vec3, jd_ut1: float) -> Vec3:
    """Rotate a position/direction from ECI to ECEF."""
    raise NotImplementedError


def ecef_to_eci(r_ecef: Vec3, jd_ut1: float) -> Vec3:
    """Rotate a position/direction from ECEF to ECI."""
    raise NotImplementedError


def ecef_to_geodetic(r_ecef: Vec3) -> tuple[float, float, float]:
    """WGS-84 geodetic ``(lat [rad], lon [rad], alt [m])`` from ECEF position."""
    raise NotImplementedError


def geodetic_to_ecef(lat: float, lon: float, alt: float) -> Vec3:
    """ECEF position [m] from WGS-84 geodetic ``lat, lon [rad], alt [m]``."""
    raise NotImplementedError


def inertial_to_body(q: Quat, v_inertial: Vec3) -> Vec3:
    """Express an inertial vector in body frame using ``q`` (body <- inertial)."""
    raise NotImplementedError


def body_to_inertial(q: Quat, v_body: Vec3) -> Vec3:
    """Express a body vector in the inertial frame using ``q`` (body <- inertial)."""
    raise NotImplementedError
