#!/usr/bin/env python3
"""Compare pybhpt normalization against Leaver prefactor P*h2 at horizon."""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from physical_ansatz.mapping import r_plus, r_minus
from physical_ansatz.prefactor import Leaver_prefactors, build_prefactor_primitives
from physical_ansatz.transform_y import h_factor
from utils.compute_lambda_usage import compute_lambda


def main():
    a, omega = 0.5, 0.0257
    ell, m, s = 2, 2, -2
    M = 1.0
    lam = compute_lambda(a, omega, ell, m, s=s)

    print(f"a={a}, omega={omega}, lambda={lam:.10f}")

    # Get pybhpt solution on a finer grid near horizon
    from pybhpt_usage.compute_solution import compute_pybhpt_solution
    rp = float(M + np.sqrt(M**2 - a**2))
    rm = float(M - np.sqrt(M**2 - a**2))
    r_grid = np.logspace(np.log10(rp + 1e-6), np.log10(rp + 0.1), 100)
    r_ref, R_ref = compute_pybhpt_solution(a, omega, ell=ell, m=m, r_grid=r_grid, timeout=30.0)
    R_ref = np.asarray(R_ref, dtype=np.complex128)

    # Compute P*h2 on the same grid
    dtype = torch.float64
    cdtype = torch.complex128
    a_t = torch.tensor([a], dtype=dtype)
    omega_t = torch.tensor([omega], dtype=dtype)
    r_t = torch.tensor(r_ref, dtype=dtype)

    rp_t = r_plus(a_t, M)
    _, rm_t, _, _, _ = build_prefactor_primitives(r_t, a_t, M=M, need_rs=False)
    P, P_r, P_rr = Leaver_prefactors(r_t, a_t, omega_t, m=m, M=M, s=s, rp=rp_t, rm=rm_t)
    h2 = h_factor(a_t, omega_t, m=m, M=M, s=s)
    P_np = P.detach().numpy()
    h2_np = complex(h2.detach().numpy()[0])

    S = R_ref / (P_np * h2_np)

    print(f"\n=== Horizon analysis ===")
    print(f"rp = {rp:.6f}, rm = {rm:.6f}")
    print(f"r_min = {r_ref[0]:.8f}, r_max = {r_ref[-1]:.6f}")

    # Near-horizon behavior
    eps_vals = r_ref - rp
    print(f"\n=== Near-horizon S = R/(P*h2) ===")
    for i in range(min(5, len(r_ref))):
        print(f"  r={r_ref[i]:.8f} (r-rp={eps_vals[i]:.2e}): S = {S[i]:.6e}")

    # Fit power-law near horizon: S ~ S0 * (r-rp)^alpha
    # If S → const as r → rp, then alpha ≈ 0
    log_eps = np.log(eps_vals[:20])
    log_S_abs = np.log(np.abs(S[:20]))
    fit = np.polyfit(log_eps, log_S_abs, 1)
    print(f"\n  Near-horizon power-law fit: |S| ~ (r-rp)^{fit[0]:.4f}")
    print(f"  S0_abs = exp({fit[1]:.4f}) = {np.exp(fit[1]):.4f}")

    # Extrapolate S to r=rp
    S_at_horizon = np.exp(fit[1]) * np.exp(1j * np.angle(S[0]))
    print(f"\n  Extrapolated S(r=rp) = {S_at_horizon:.6e}")
    print(f"  |S(r=rp)| = {np.abs(S_at_horizon):.6f}")
    print(f"  Expected: S(r=rp) = 1+0j")

    # Check: what if we rescale h2?
    norm_factor = S_at_horizon
    print(f"\n=== Normalization mismatch ===")
    print(f"  S(r=rp) = {norm_factor:.6e}")
    print(f"  |S(r=rp)| = {np.abs(norm_factor):.6f}")
    print(f"  To fix: divide h2 by {norm_factor:.6e}")
    print(f"  Or: multiply P by {norm_factor:.6e}")
    print(f"  Or equivalently: R_ansatz = P * h2 * S / {norm_factor:.6e}")

    # Check the theoretical prefactor
    # pybhpt: R_in ~ B^trans * Delta^2 * exp(-i*k*r*)
    # Let's compute B^trans
    Omega_H = a / (2 * M * rp)
    k = omega - m * Omega_H
    sigma_p = (2 * omega * rp - m * a) / (rp - rm)
    print(f"\n=== Theoretical parameters ===")
    print(f"  Omega_H = {Omega_H:.6f}")
    print(f"  k = {k:.6f}")
    print(f"  sigma_p = {sigma_p:.6f}")

    # pp = -s - i*sigma_p
    # pm = -1-s + 2i*omega + i*sigma_p (for s=-2: pm = 1 + 2i*omega + i*sigma_p)
    pp = complex(-s - 1j * sigma_p)
    pm = complex(-1 - s + 2j * omega + 1j * sigma_p)
    print(f"  pp = {pp:.6f}")
    print(f"  pm = {pm:.6f}")

    # P*h2 ~ (r-rp)^pp * (rp-rm)^pm * exp(i*omega*rp) * h2
    # h2 = exp(-i*(omega+k)*rp) * 2^(2i*k) * (rp-rm)^(1-2i*(omega+k))
    # P*h2 ~ (r-rp)^pp * (rp-rm)^(pm+1-2i*(omega+k)) * exp(-i*k*rp) * 2^(2i*k)
    import cmath
    leading = (rp - rm)**(pm + 1 - 2j*(omega + k)) * cmath.exp(-1j*k*rp) * (2**(2j*k))
    print(f"\n  P*h2 leading coeff (excluding (r-rp)^pp): {leading:.6e}")

    # pybhpt: R_in ~ B^trans * Delta^2 * exp(-i*k*r*)
    # Delta = (r-rp)(r-rm), Delta^2 ~ (r-rp)^2 * (rp-rm)^2
    # r* ~ (rp^2+a^2)/(rp-rm)*log((r-rp)/(2M))
    # exp(-i*k*r*) ~ (r-rp)^(-i*k*(rp^2+a^2)/(rp-rm)) * (2M)^(i*k*(rp^2+a^2)/(rp-rm))
    rp2_p_a2 = rp**2 + a**2
    pybhpt_exp = -1j * k * rp2_p_a2 / (rp - rm)
    pybhpt_coeff = (rp - rm)**2 * (2*M)**(1j*k*rp2_p_a2/(rp-rm))
    print(f"\n  pybhpt horizon: (r-rp)^(2 {pybhpt_exp:.6f})")
    print(f"  pybhpt leading coeff: {pybhpt_coeff:.6e}")

    # Compare the exponents
    print(f"\n  P*h2 exponent: {pp:.6f}")
    print(f"  pybhpt exponent: 2 {pybhpt_exp:.6f}")
    print(f"  Match: {abs(pp - (2 + pybhpt_exp)) < 1e-10}")

    ratio = leading / pybhpt_coeff
    print(f"\n  P*h2 / pybhpt ratio at horizon = {ratio:.6e}")
    print(f"  |ratio| = {abs(ratio):.6f}")

    # So S(r=rp) should be B^trans / ratio
    # B^trans ~ 1 for (l=m=2, s=-2)
    print(f"\n  Expected S(r=rp) = B^trans * pybhpt_coeff / leading")
    print(f"                   ≈ pybhpt_coeff / leading = {pybhpt_coeff/leading:.6e}")
    print(f"  Actual S(r=rp) from extrapolation = {S_at_horizon:.6e}")


if __name__ == "__main__":
    main()
