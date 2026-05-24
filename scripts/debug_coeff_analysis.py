#!/usr/bin/env python3
"""Quick check: PDE coefficient behavior near horizon and infinity."""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from physical_ansatz.mapping import r_plus
from physical_ansatz.teukolsky_coeffs import coeffs_x
from physical_ansatz.transform_y import (
    transform_coeffs_x_to_y, transform_coeffs_x_to_y_S,
    horizon_regularity_slope, h_factor,
)
from physical_ansatz.prefactor import build_prefactor_primitives, Leaver_prefactors, prefactor_log_derivatives_x
from utils.compute_lambda_usage import compute_lambda


def main():
    a, omega = 0.5, 0.0257
    ell, m, s, M = 2, 2, -2, 1.0
    lam = compute_lambda(a, omega, ell, m, s=s)

    dtype = torch.float64
    cdtype = torch.complex128
    a_t = torch.tensor([a], dtype=dtype)
    omega_t = torch.tensor([omega], dtype=dtype)
    lam_t = torch.tensor([lam], dtype=cdtype)

    # Build y-grid
    rp = float(M + np.sqrt(M**2 - a**2))
    n_pts = 128
    r_grid = np.logspace(np.log10(rp + 0.001), 2.0, n_pts)
    x_grid = rp / r_grid
    y_grid = 2.0 * x_grid - 1.0
    valid = (y_grid >= -1) & (y_grid < 1)
    y_valid = y_grid[valid]
    x_valid = x_grid[valid]
    r_valid = r_grid[valid]

    y_t = torch.tensor(y_valid, dtype=dtype)
    x_t = torch.tensor(x_valid, dtype=dtype)
    r_t = torch.tensor(r_valid, dtype=dtype)
    slope = horizon_regularity_slope(a=a_t, omega=omega_t, lambda_=lam_t, m=m, M=M, s=s)

    # f(y) formulation coefficients
    A2, A1, A0 = coeffs_x(x=x_t, a=a_t, omega=omega_t, m=m, lambda_=lam_t, s=s, M=M)
    B2, B1, B0, rhs = transform_coeffs_x_to_y(
        A2.unsqueeze(0), A1.unsqueeze(0), A0.unsqueeze(0),
        y_t.unsqueeze(0), slope=slope,
    )
    B2_np = np.abs(B2.squeeze(0).detach().numpy())
    B1_np = np.abs(B1.squeeze(0).detach().numpy())
    B0_np = np.abs(B0.squeeze(0).detach().numpy())
    rhs_np = np.abs(rhs.squeeze(0).detach().numpy())

    # S(y) formulation coefficients
    _, rm, _, _, _ = build_prefactor_primitives(r_t, a_t, M=M, need_rs=False)
    P, P_r, P_rr = Leaver_prefactors(r_t, a_t, omega_t, m=m, M=M, s=s, rp=r_plus(a_t, M), rm=rm)
    log_Px, log_Pxx = prefactor_log_derivatives_x(r=r_t, a=a_t, omega=omega_t, m=m, M=M, s=s)
    D2_S, D1_S, D0_S = transform_coeffs_x_to_y_S(
        A2.unsqueeze(0), A1.unsqueeze(0), A0.unsqueeze(0),
        r=r_t.unsqueeze(0), a=a_t, omega=omega_t, m=m, M=M, s=s,
    )
    D2_np = np.abs(D2_S.squeeze(0).detach().numpy())
    D1_np = np.abs(D1_S.squeeze(0).detach().numpy())
    D0_np = np.abs(D0_S.squeeze(0).detach().numpy())

    print(f"a={a}, omega={omega}")
    print(f"y range: [{y_valid[0]:.6f}, {y_valid[-1]:.6f}]")
    print(f"x range: [{x_valid[0]:.6f}, {x_valid[-1]:.6f}]")
    print(f"r range: [{r_valid[0]:.6f}, {r_valid[-1]:.6f}]")

    # Near horizon (first 5 points, y→1)
    print(f"\n=== Near horizon (y→1, x→1) ===")
    for i in range(5):
        try:
            a2v = np.abs(complex(A2[i].detach()))
            a1v = np.abs(complex(A1[i].detach()))
            a0v = np.abs(complex(A0[i].detach()))
        except Exception:
            a2v, a1v, a0v = float('nan'), float('nan'), float('nan')
        print(f"  y={y_valid[i]:.6f}, x={x_valid[i]:.6f}, r={r_valid[i]:.6f}")
        print(f"    |A2|={a2v:.4e}, |A1|={a1v:.4e}, |A0|={a0v:.4e}")
        print(f"    f-PDE: |B2|={B2_np[i]:.4e}, |B1|={B1_np[i]:.4e}, |B0|={B0_np[i]:.4e}, |rhs|={rhs_np[i]:.4e}")
        print(f"    S-PDE: |D2|={D2_np[i]:.4e}, |D1|={D1_np[i]:.4e}, |D0|={D0_np[i]:.4e}")

    # Near infinity (last 5 points, y→-1)
    print(f"\n=== Near infinity (y→-1, x→0) ===")
    for i in range(1, 6):
        idx = -i
        try:
            a2v = np.abs(complex(A2[idx].detach()))
            a1v = np.abs(complex(A1[idx].detach()))
            a0v = np.abs(complex(A0[idx].detach()))
        except Exception:
            a2v, a1v, a0v = float('nan'), float('nan'), float('nan')
        print(f"  y={y_valid[idx]:.6f}, x={x_valid[idx]:.6f}, r={r_valid[idx]:.6f}")
        print(f"    |A2|={a2v:.4e}, |A1|={a1v:.4e}, |A0|={a0v:.4e}")
        print(f"    f-PDE: |B2|={B2_np[idx]:.4e}, |B1|={B1_np[idx]:.4e}, |B0|={B0_np[idx]:.4e}, |rhs|={rhs_np[idx]:.4e}")
        print(f"    S-PDE: |D2|={D2_np[idx]:.4e}, |D1|={D1_np[idx]:.4e}, |D0|={D0_np[idx]:.4e}")

    # Ratio analysis
    print(f"\n=== Coefficient ratios ===")
    b2_b0_ratio = B2_np / (B0_np + 1e-30)
    b1_b0_ratio = B1_np / (B0_np + 1e-30)
    d2_d0_ratio = D2_np / (D0_np + 1e-30)
    print(f"f-PDE: |B2|/|B0|: min={np.min(b2_b0_ratio):.4e}, max={np.max(b2_b0_ratio):.4e}, "
          f"near_horizon={b2_b0_ratio[-1]:.4e}, near_inf={b2_b0_ratio[0]:.4e}")
    print(f"f-PDE: |B1|/|B0|: min={np.min(b1_b0_ratio):.4e}, max={np.max(b1_b0_ratio):.4e}, "
          f"near_horizon={b1_b0_ratio[-1]:.4e}, near_inf={b1_b0_ratio[0]:.4e}")
    print(f"S-PDE: |D2|/|D0|: min={np.min(d2_d0_ratio):.4e}, max={np.max(d2_d0_ratio):.4e}, "
          f"near_horizon={d2_d0_ratio[-1]:.4e}, near_inf={d2_d0_ratio[0]:.4e}")

    # f(1) consistency condition
    print(f"\n=== f(1) consistency check ===")
    A2_1 = complex(A2[0].detach())
    A1_1 = complex(A1[0].detach())
    A0_1 = complex(A0[0].detach())
    slope_1 = complex(slope.squeeze().detach())
    f1_expected = (-slope_1*(A2_1 + A1_1) - A0_1) / (2*A2_1*slope_1)
    print(f"  A2(1) = {A2_1:.6e}")
    print(f"  A1(1) = {A1_1:.6e}")
    print(f"  A0(1) = {A0_1:.6e}")
    print(f"  slope = {slope_1:.6e}")
    print(f"  Expected f(1) = rhs(1)/B0(1) = {f1_expected:.6e}")
    print(f"  |f1_expected| = {abs(f1_expected):.6e}")

    # At infinity: what happens to the coefficients?
    print(f"\n=== Infinity (y=-1) analysis ===")
    A2_inf = complex(A2[-1].detach())
    A1_inf = complex(A1[-1].detach())
    A0_inf = complex(A0[-1].detach())
    B2_inf = complex(B2[0, -1].detach())
    B1_inf = complex(B1[0, -1].detach())
    B0_inf = complex(B0[0, -1].detach())
    rhs_inf = complex(rhs[0, -1].detach())
    print(f"  f-PDE at y≈-1: B2={B2_inf:.6e}, B1={B1_inf:.6e}, B0={B0_inf:.6e}, rhs={rhs_inf:.6e}")
    print(f"  |B2|/|B0| = {abs(B2_inf)/(abs(B0_inf)+1e-30):.4e}")
    print(f"  |B1|/|B0| = {abs(B1_inf)/(abs(B0_inf)+1e-30):.4e}")

    # How many points have |B2|/|B0| < threshold?
    for thresh in [1e-6, 1e-4, 1e-2, 0.1, 1.0]:
        count = np.sum(b2_b0_ratio < thresh)
        print(f"  Points with |B2|/|B0| < {thresh}: {count}/{len(y_valid)}")


if __name__ == "__main__":
    main()
