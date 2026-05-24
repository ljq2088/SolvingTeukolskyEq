#!/usr/bin/env python3
"""Check: what PDE residual does the TRUE pybhpt solution get?"""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import json, numpy as np, torch
from model.autoencoder_pinn import AutoencoderTeukolskyPINN
from physical_ansatz.transform_y import (
    horizon_regularity_slope, h_factor, h1_factor, g_factor,
)
from physical_ansatz.prefactor import Leaver_prefactors, build_prefactor_primitives
from physical_ansatz.mapping import r_plus
from physical_ansatz.teukolsky_coeffs import coeffs_x
from physical_ansatz.transform_y import transform_coeffs_x_to_y
from utils.compute_lambda_usage import compute_lambda
from domain.atlas_builder import load_atlas, map_to_chart


def main():
    device = torch.device("cuda")
    M, ell, m_mode, s_val = 1.0, 2, 2, -2

    atlas = load_atlas("outputs/domain/atlas_l2_m2_logw.json")
    comp = atlas.components[0]
    patch_file = json.load(open("outputs/domain/patch_cover_l2_m2_logw.json"))
    p0 = [p for p in patch_file["patches"] if p["patch_id"] == 0][0]
    a_c = float(p0["a_center"])
    omega_c = float(p0["omega_center"])
    lam_c = compute_lambda(a_c, omega_c, ell, m_mode, s=s_val)

    dtype = torch.float64
    cdtype = torch.complex128
    a_t = torch.tensor([a_c], dtype=dtype)
    omega_t = torch.tensor([omega_c], dtype=dtype)
    lam_t = torch.tensor([lam_c], dtype=cdtype)

    rp = float(M + np.sqrt(M**2 - a_c**2))

    # Build y-grid
    r_grid = np.logspace(np.log10(rp + 0.001), 2.0, 128)
    x_grid = rp / r_grid
    y_grid = 2.0 * x_grid - 1.0
    valid_v = (y_grid >= -1) & (y_grid < 1)
    y_valid = y_grid[valid_v]
    r_valid = r_grid[valid_v]

    # Get pybhpt true solution
    from pybhpt_usage.compute_solution import compute_pybhpt_solution
    _, R_ref = compute_pybhpt_solution(a_c, omega_c, ell=2, m=m_mode, r_grid=r_valid, timeout=30.0)
    R_ref = np.asarray(R_ref, dtype=np.complex128)

    # Compute S_true and f_true
    r_t = torch.tensor(r_valid, dtype=dtype)
    rp_t = r_plus(a_t, M)
    _, rm_t, _, _, _ = build_prefactor_primitives(r_t, a_t, M=M, need_rs=False)
    P, _, _ = Leaver_prefactors(r_t, a_t, omega_t, m=m_mode, M=M, s=s_val, rp=rp_t, rm=rm_t)
    h2 = h_factor(a_t, omega_t, m=m_mode, M=M, s=s_val)
    S_true = torch.tensor(R_ref, dtype=cdtype) / (P * h2)

    slope = horizon_regularity_slope(a=a_t, omega=omega_t, lambda_=lam_t, m=m_mode, M=M, s=s_val)

    x_v = torch.tensor(x_grid[valid_v], dtype=dtype)
    y_v = torch.tensor(y_valid, dtype=dtype)
    h1, _, _ = h1_factor(x_v)
    g, _, _ = g_factor(x_v, slope.squeeze())
    g = g.squeeze()
    h1 = h1.squeeze()
    S_true_sq = S_true.squeeze()
    f_true = (S_true_sq - g - 1.0) / (g * h1)

    # Compute f_y, f_yy via finite differences on y-grid
    f_np = f_true.detach().numpy()
    y_np = y_valid

    # Central differences
    f_y_np = np.zeros_like(f_np)
    f_yy_np = np.zeros_like(f_np)
    for i in range(1, len(y_np) - 1):
        dy = y_np[i+1] - y_np[i-1]
        f_y_np[i] = (f_np[i+1] - f_np[i-1]) / dy
        f_yy_np[i] = (f_np[i+1] - 2*f_np[i] + f_np[i-1]) / ((y_np[i+1] - y_np[i]) * (y_np[i] - y_np[i-1]))
    # Forward/backward at endpoints
    f_y_np[0] = (f_np[1] - f_np[0]) / (y_np[1] - y_np[0])
    f_y_np[-1] = (f_np[-1] - f_np[-2]) / (y_np[-1] - y_np[-2])
    f_yy_np[0] = f_yy_np[1]
    f_yy_np[-1] = f_yy_np[-2]

    # Compute PDE coefficients
    A2, A1, A0 = coeffs_x(x=x_v, a=a_t, omega=omega_t, m=m_mode, lambda_=lam_t, s=s_val, M=M)
    B2, B1, B0, rhs = transform_coeffs_x_to_y(
        A2.unsqueeze(0), A1.unsqueeze(0), A0.unsqueeze(0),
        y_v.unsqueeze(0), slope=slope,
    )

    B2_np = B2.squeeze(0).detach().numpy()
    B1_np = B1.squeeze(0).detach().numpy()
    B0_np = B0.squeeze(0).detach().numpy()
    rhs_np = rhs.squeeze(0).detach().numpy()

    resid = B2_np * f_yy_np + B1_np * f_y_np + B0_np * f_np - rhs_np
    pointwise = np.abs(resid)**2

    print(f"=== PDE residual for TRUE pybhpt solution ===")
    print(f"  mean |res|^2: {np.mean(pointwise):.6e}")
    print(f"  median:       {np.median(pointwise):.6e}")
    print(f"  max:          {np.max(pointwise):.6e}")
    print(f"  min:          {np.min(pointwise):.6e}")

    # Also compute normalized residual (coeff mode)
    scale_coeff = np.abs(B2_np)**2 + np.abs(B1_np)**2 + np.abs(B0_np)**2 + np.abs(rhs_np)**2 + 1e-12
    pointwise_norm = pointwise / scale_coeff
    print(f"\n  mean (coeff-norm): {np.mean(pointwise_norm):.6e}")
    print(f"  median (coeff-norm): {np.median(pointwise_norm):.6e}")
    print(f"  max (coeff-norm): {np.max(pointwise_norm):.6e}")

    # Highlight where the residual is large
    worst_idx = np.argmax(pointwise)
    print(f"\n  Worst point: y={y_np[worst_idx]:.4f}, |res|^2={pointwise[worst_idx]:.4e}")
    print(f"    |B2|={np.abs(B2_np[worst_idx]):.4e}, |B1|={np.abs(B1_np[worst_idx]):.4e}, "
          f"|B0|={np.abs(B0_np[worst_idx]):.4e}, |rhs|={np.abs(rhs_np[worst_idx]):.4e}")

    # Show residual vs y
    top5 = np.argsort(pointwise)[-5:][::-1]
    print(f"\n  Top-5 worst points:")
    for idx in top5:
        print(f"    y={y_np[idx]:.4f}, |res|^2={pointwise[idx]:.4e}, "
              f"|res|={np.sqrt(pointwise[idx]):.4e}")


if __name__ == "__main__":
    main()
