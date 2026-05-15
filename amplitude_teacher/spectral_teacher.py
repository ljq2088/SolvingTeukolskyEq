"""Compute B_inc, B_ref using pybhpt spectral (Leaver continued fraction) solver.

Convention (matches physical_ansatz.u_basis):
  R(r→∞) ≈ B_inc * A_down(r) + B_ref * A_up(r)
  A_up(r)   = r^3 * exp(+i ω r_*(r,a))
  A_down(r) = r^{-1} * exp(-i ω r_*(r,a))

pybhpt's "In" solution normalisation: B_inc = 1 (unit ingoing at infinity).
B_ref is extracted from R/A_up in the large-r tail, where the
B_inc * A_down term is O(r^{-4}) suppressed relative to B_ref * A_up.
"""

import numpy as np


def _r_plus(a, M=1.0):
    return M + np.sqrt(M**2 - a**2)


def _r_minus(a, M=1.0):
    return M - np.sqrt(M**2 - a**2)


def _r_star_np(r, a, M=1.0):
    """Tortoise coordinate matching physical_ansatz.prefactor.r_star."""
    rp = _r_plus(a, M)
    rm = _r_minus(a, M)
    denom = rp - rm
    return (r
            + (rp**2 + a**2) / denom * np.log(np.abs(r - rp) / (2 * M))
            - (rm**2 + a**2) / denom * np.log(np.abs(r - rm) / (2 * M)))


def _A_up(r, a, omega):
    return r**3 * np.exp(1j * omega * _r_star_np(r, a))


def _A_down(r, a, omega):
    return r**(-1) * np.exp(-1j * omega * _r_star_np(r, a))


def _pybhpt_amplitudes(a, omega, s=-2, l=2, m=2, r_max=1000.0, n_r=512):
    """Compute (B_inc=1, B_ref) from pybhpt In solution.

    B_inc=1 by pybhpt normalisation. B_ref extracted from R/A_up
    at large r where the B_inc*A_down term is negligible.
    """
    from pybhpt.radial import RadialTeukolsky

    r_h = 1.0 + np.sqrt(1.0 - a**2)
    r_min = max(r_h + 1e-3, 2.0)
    r = np.linspace(r_min, r_max, n_r)

    rad = RadialTeukolsky(s=s, j=l, m=m, a=float(a), omega=float(omega), r=r)
    rad.solve()
    R = np.asarray(rad.radialsolutions("In"))

    n_fit = max(40, n_r // 4)
    r_fit = r[-n_fit:]
    R_fit = R[-n_fit:]
    A_up_tail = _A_up(r_fit, a, omega)

    B_ref_vals = R_fit / A_up_tail
    B_ref = complex(np.median(B_ref_vals.real) + 1j * np.median(B_ref_vals.imag))

    return 1.0 + 0.0j, B_ref


def compute_amplitudes_pybhpt(a_vals, omega_vals, s=-2, l=2, m=2,
                                r_max=1000.0, n_r=512, verbose=True):
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
