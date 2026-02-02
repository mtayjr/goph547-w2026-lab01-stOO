import numpy as np
from goph547lab01.gravity import gravity_potential_point, gravity_effect_point


def test_gravity_potential_point():
    x = np.array([0.0, 0.0, 0.0])
    xm = np.array([0.0, 0.0, -10.0])
    m = 1.0e7
    G = 6.674e-11

    r = np.linalg.norm(x - xm)
    U_potential = G * m / r

    assert np.isclose(
        gravity_potential_point(x, xm, m, G=G),
        U_potential
    )


def test_gravity_effect_point_positive_downward():
    x = np.array([0.0, 0.0, 0.0])
    xm = np.array([0.0, 0.0, -10.0])
    m = 1.0e7
    G = 6.674e-11

    r_vec = x - xm
    r = np.linalg.norm(r_vec)
    dz = r_vec[2]

    U_potential = G * m * dz / (r**3)
    out = gravity_effect_point(x, xm, m, G=G)

    assert out > 0
    assert np.isclose(out, U_potential)

