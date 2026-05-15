"""Compute B_inc, B_ref using GSN (Sasaki-Nakamura) integration.

Placeholder — falls back to pybhpt spectral.
"""

import numpy as np


def compute_amplitudes_gsn(a_vals, omega_vals, s=-2, l=2, m=2,
                             r_max=500.0, n_r=256, verbose=True):
    """Placeholder: delegates to spectral teacher."""
    from amplitude_teacher.spectral_teacher import compute_amplitudes_pybhpt

    if verbose:
        print("[gsn_teacher] GSN not yet implemented, falling back to pybhpt spectral.")
    return compute_amplitudes_pybhpt(a_vals, omega_vals, s=s, l=l, m=m,
                                      r_max=r_max, n_r=n_r, verbose=verbose)
