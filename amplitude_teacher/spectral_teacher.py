"""Compute B_inc, B_ref using pybhpt spectral (Leaver continued fraction) solver.

Convention:
  R(r→∞) ≈ B_inc * r^{-1} * e^{-iωr_*} + B_ref * r^3 * e^{+iωr_*}

We use pybhpt's standard normalisation where the "In" solution has unit
ingoing amplitude (B_inc = 1). B_ref is extracted from R/A_up at large r,
where the B_inc * A_down contribution is negligible (O(r^{-4}) suppressed).
"""

import numpy as np


def _pybhpt_amplitudes(a, omega, s=-2, l=2, m=2, r_max=500.0, n_r=256):
    """Compute B_inc=1, B_ref from pybhpt In solution.

    The "In" solution from pybhpt is normalized to unit ingoing wave
    at infinity: R_in(r→∞) ≈ 1 * A_down(r) + C_ref * A_up(r).

    We extract B_ref = C_ref from the large-r asymptote where
    A_down (∝ r^{-1}) is negligible compared to A_up (∝ r^3).

    Returns (B_inc=1.0, B_ref).
    """
    from pybhpt.radial import RadialTeukolsky

    r_h = 1.0 + np.sqrt(1.0 - a**2)
    r_min = max(r_h + 1e-3, 2.0)
    r = np.linspace(r_min, r_max, n_r)

    rad = RadialTeukolsky(s=s, j=l, m=m, a=float(a), omega=float(omega), r=r)
    rad.solve()
    R = np.asarray(rad.radialsolutions("In"))

    # B_ref: fit R/A_up at largest r (B_inc*A_down term is ~r^{-4} suppressed)
    n_fit = max(20, n_r // 5)
    r_fit = r[-n_fit:]
    R_fit = R[-n_fit:]
    A_up_tail = r_fit ** 3 * np.exp(1j * float(omega) * r_fit)

    B_ref_vals = R_fit / A_up_tail
    B_ref = complex(np.median(B_ref_vals.real) + 1j * np.median(B_ref_vals.imag))

    return 1.0 + 0.0j, B_ref


def compute_amplitudes_pybhpt(a_vals, omega_vals, s=-2, l=2, m=2,
                                r_max=500.0, n_r=256, verbose=True):
    """Compute B_inc, B_ref for each (a, omega) using pybhpt.

    Returns:
        B_inc: (N,) complex128 — 1.0+0.0j (pybhpt standard normalisation)
        B_ref: (N,) complex128 — reflection amplitude
    """
    n = len(a_vals)
    B_inc = np.ones(n, dtype=np.complex128)
    B_ref = np.zeros(n, dtype=np.complex128)

    for i in range(n):
        try:
            inc, ref = _pybhpt_amplitudes(
                float(a_vals[i]), float(omega_vals[i]),
                s=s, l=l, m=m, r_max=r_max, n_r=n_r,
            )
            B_inc[i] = inc
            B_ref[i] = ref
            if verbose and (i == 0 or i == n - 1 or (i + 1) % 10 == 0):
                print(f"  [{i+1}/{n}] a={a_vals[i]:.4f} omega={omega_vals[i]:.6e} "
                      f"|B_ref|={abs(ref):.4e}")
        except Exception as e:
            print(f"  [{i+1}/{n}] a={a_vals[i]:.4f} omega={omega_vals[i]:.6e} FAILED: {e}")
            B_inc[i] = np.nan
            B_ref[i] = np.nan

    return B_inc, B_ref
