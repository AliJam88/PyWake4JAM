from py_wake import np


def madsen_a0(ct_ilk, a0p=np.array([0.2460, 0.0586, 0.0883])):
    """
    BEM axial induction approximation by
    Madsen, H. A., Larsen, T. J., Pirrung, G. R., Li, A., and Zahle, F.: Implementation of the blade element momentum model on a polar grid and its aeroelastic load impact, Wind Energ. Sci., 5, 1–27, https://doi.org/10.5194/wes-5-1-2020, 2020.
    """
    # Evaluate with Horner's rule.
    # a0_ilk = a0p[2] * ct_ilk**3 + a0p[1] * ct_ilk**2 + a0p[0] * ct_ilk
    a0_ilk = ct_ilk * (a0p[0] + ct_ilk * (a0p[1] + ct_ilk * a0p[2]))
    return a0_ilk

def mom1d_a0(ct_ilk):
    """
    1D momentum with CT forced to below 1.
    """
    a0_ilk = 0.5 * (1. - np.sqrt(1. - np.minimum(0.999, ct_ilk)))
    return a0_ilk
