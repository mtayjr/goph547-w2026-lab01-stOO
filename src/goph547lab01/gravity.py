from __future__ import annotations
import numpy as np


def gravity_potential_point(x, xm, m, G=6.674e-11):
    """
    Gravity potential due to a point mass.

    Parameters
    ----------
    x : array_like, shape (3,)
        Observation point [x, y, z]
    xm : array_like, shape (3,)
        Mass location [xm, ym, zm]
    m : float
        Mass of the anomaly
    G : float
        Gravitational constant
    """
    x = np.asarray(x, dtype=float)
    xm = np.asarray(xm, dtype=float)

    r = np.linalg.norm(x - xm)
    if r == 0.0:
        raise ValueError("r cannot be zero")

    return G * m / r


def gravity_effect_point(x, xm, m, G=6.674e-11):
    """
    Vertical gravity effect due to a point mass (positive downward).
    """
    x = np.asarray(x, dtype=float)
    xm = np.asarray(xm, dtype=float)

    r_vec = x - xm
    r = np.linalg.norm(r_vec)
    if r == 0.0:
        raise ValueError("r cannot be zero")

    dz = r_vec[2]  # z - zm
    return G * m * dz / (r**3)
