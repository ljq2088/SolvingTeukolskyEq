#!/usr/bin/env python3
"""Test integral consistency and anchor-based losses on true pybhpt solution."""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import json, numpy as np, torch
from physical_ansatz.transform_y import (
    horizon_regularity_slope, h_factor, h1_factor, g_factor,
    compose_reduced_shape_from_f,
)
from physical_ansatz.prefactor import Leaver_prefactors, build_prefactor_primitives
from physical_ansatz.mapping import r_plus
from physical_ansatz.teukolsky_coeffs import coeffs_x
from physical_ansatz.transform_y import transform_coeffs_x_to_y
from physical_ansatz.residual_pinn import compute_integral_consistency_loss
from utils.compute_lambda_usage import compute_lambda
from domain.atlas_builder import load_atlas


def main():
    device = torch.device("cpu")
    M, m_mode, s_val = 1.0, 2, -2
    dtype = torch.float64
    cdtype = torch.complex128

    atlas = load_atlas("outputs/domain/atlas_l2_m2_logw.json")
    patch_file = json.load(open("outputs/domain/patch_cover_l2_m2_logw.json"))
    p0 = [p for p in patch_file["patches"] if p["patch_id"] == 0][0]
    a_c = float(p0["a_center"])
    omega_c = float(p0["omega_center"])
    lam_c = compute_lambda(a_c, omega_c, 2, m_mode, s=s_val)
    rp = float(M + np.sqrt(M**2 - a_c**2))

    a_t = torch.tensor([a_c], dtype=dtype)
    omega_t = torch.tensor([omega_c], dtype=dtype)
    lam_t = torch.tensor([lam_c], dtype=cdtype)

    # Build y-grid
    r_grid = np.logspace(np.log10(rp + 0.001), 2.0, 128)
    x_grid = rp / r_grid
    y_grid = 2.0 * x_grid - 1.0
    valid = (y_grid >= -1) & (y_grid < 1)
    y_valid = y_grid[valid]
    r_valid = r_grid[valid]
    y_t = torch.tensor(y_valid, dtype=dtype)

    # Get true pybhpt solution
    from pybhpt_usage.compute_solution import compute_pybhpt_solution
    _, R_ref = compute_pybhpt_solution(a_c, omega_c, ell=2, m=m_mode, r_grid=r_valid, timeout=30.0)
    R_ref = np.asarray(R_ref, dtype=np.complex128)

    # Compute true f
    r_t = torch.tensor(r_valid, dtype=dtype)
    rp_t = r_plus(a_t, M)
    _, rm_t, _, _, _ = build_prefactor_primitives(r_t, a_t, M=M, need_rs=False)
    P, _, _ = Leaver_prefactors(r_t, a_t, omega_t, m=m_mode, M=M, s=s_val, rp=rp_t, rm=rm_t)
    h2 = h_factor(a_t, omega_t, m=m_mode, M=M, s=s_val)
    S_true = torch.tensor(R_ref, dtype=cdtype) / (P * h2)

    slope = horizon_regularity_slope(a=a_t, omega=omega_t, lambda_=lam_t, m=m_mode, M=M, s=s_val)
    x_v = torch.tensor(x_grid[valid], dtype=dtype)
    h1_v, _, _ = h1_factor(x_v)
    g_v, _, _ = g_factor(x_v, slope.squeeze())
    f_true = (S_true.squeeze() - g_v.squeeze() - 1.0) / (g_v.squeeze() * h1_v.squeeze())

    # Finite-diff f_y, f_yy
    f_np = f_true.detach().numpy()
    y_np = y_valid
    f_y_np = np.zeros_like(f_np)
    f_yy_np = np.zeros_like(f_np)
    for i in range(1, len(y_np) - 1):
        dy = y_np[i+1] - y_np[i-1]
        f_y_np[i] = (f_np[i+1] - f_np[i-1]) / dy
        f_yy_np[i] = (f_np[i+1] - 2*f_np[i] + f_np[i-1]) / ((y_np[i+1] - y_np[i]) * (y_np[i] - y_np[i-1]))
    f_y_np[0] = (f_np[1] - f_np[0]) / (y_np[1] - y_np[0])
    f_y_np[-1] = (f_np[-1] - f_np[-2]) / (y_np[-1] - y_np[-2])
    f_yy_np[0] = f_yy_np[1]
    f_yy_np[-1] = f_yy_np[-2]

    # Build tensors for integral loss
    f_t = torch.tensor(f_np, dtype=cdtype).unsqueeze(0)
    f_y_t = torch.tensor(f_y_np, dtype=cdtype).unsqueeze(0)
    y_batch = y_t.unsqueeze(0)

    # Compute A2, A1, A0 for the integral loss
    x_int = 0.5 * (y_batch + 1.0)
    A2_list, A1_list, A0_list = [], [], []
    for i in range(1):
        A2_i, A1_i, A0_i = coeffs_x(x=x_int.squeeze(0), a=a_t, omega=omega_t,
                                      m=m_mode, lambda_=lam_t.squeeze(), s=s_val, M=M)
        A2_list.append(A2_i); A1_list.append(A1_i); A0_list.append(A0_i)
    A2_int = torch.stack(A2_list, dim=0)
    A1_int = torch.stack(A1_list, dim=0)
    A0_int = torch.stack(A0_list, dim=0)

    # === Test 1: Integral consistency loss on true f ===
    print("=== Integral consistency loss on TRUE pybhpt solution ===")
    # Direct call with all needed intermediates
    cfg = {"problem": {"M": M, "s": s_val, "m": m_mode}}

    # Compute S and S_y from f via ansatz for integral loss
    h1_x, h1_x_d, _ = h1_factor(x_int)
    g_x, g_x_d, _ = g_factor(x_int, slope)
    W = g_x * h1_x
    W_x = g_x_d * h1_x + g_x * h1_x_d
    G = g_x + 1.0
    G_x = g_x_d

    f_x = 2.0 * f_y_t
    S_from_f = W * f_t + G
    S_y_from_f = (W_x * f_t + W * f_x + G_x) / 2.0  # chain rule: S_y = S_x/2

    for n_anchors in [8, 16, 32]:
        loss = compute_integral_consistency_loss(
            model=None, cfg=cfg,
            a_batch=a_t, omega_batch=omega_t, lambda_batch=lam_t,
            y_interior=y_batch,
            _f=f_t, _f_y=f_y_t, _S=S_from_f, _S_y=S_y_from_f,
            _A2=A2_int, _A1=A1_int, _A0=A0_int, _slope=slope,
            n_anchors=n_anchors,
        )
        print(f"  n_anchors={n_anchors}: loss={loss.item():.6e}")

    # === Test 2: What about the f-PDE residual with proper normalization? ===
    print("\n=== f-PDE residual analysis ===")
    B2, B1, B0, rhs = transform_coeffs_x_to_y(A2_int, A1_int, A0_int, y_batch, slope=slope)
    f_yy_t = torch.tensor(f_yy_np, dtype=cdtype).unsqueeze(0)
    resid = B2 * f_yy_t + B1 * f_y_t + B0 * f_t - rhs
    pointwise = torch.abs(resid)**2

    # Different normalizations
    # 1. Raw
    print(f"  Raw mean |res|^2:        {pointwise.mean().item():.4f}")

    # 2. Term-normalized
    scale_term = 1.0 + torch.abs(B2*f_yy_t)**2 + torch.abs(B1*f_y_t)**2 + torch.abs(B0*f_t)**2 + torch.abs(rhs)**2
    pw_term = pointwise / scale_term.clamp_min(1e-12)
    print(f"  Term-norm mean:          {pw_term.mean().item():.6e}")

    # 3. Coeff-normalized
    scale_coeff = torch.abs(B2)**2 + torch.abs(B1)**2 + torch.abs(B0)**2 + torch.abs(rhs)**2 + 1e-12
    pw_coeff = pointwise / scale_coeff
    print(f"  Coeff-norm mean:         {pw_coeff.mean().item():.6e}")

    # 4. Relative residual: |res|^2 / (|B2*f_yy|^2 + |B1*f_y|^2 + |B0*f|^2 + |rhs|^2)
    scale_rel = torch.abs(B2*f_yy_t)**2 + torch.abs(B1*f_y_t)**2 + torch.abs(B0*f_t)**2 + torch.abs(rhs)**2 + 1e-12
    pw_rel = pointwise / scale_rel
    print(f"  Relative mean:           {pw_rel.mean().item():.6e}")

    # 5. Normalized by max coefficient: |res|^2 / max(|B2|^2, |B1|^2, |B0|^2, |rhs|^2)
    scale_max = torch.max(torch.abs(B2)**2, torch.abs(B1)**2)
    scale_max = torch.max(scale_max, torch.abs(B0)**2)
    scale_max = torch.max(scale_max, torch.abs(rhs)**2)
    scale_max = scale_max + 1e-12
    pw_max = pointwise / scale_max
    print(f"  Max-coeff-norm mean:     {pw_max.mean().item():.6e}")

    # === Test 3: Infinity Robin BC check ===
    print("\n=== Infinity Robin BC check ===")
    from physical_ansatz.infinity_robin import analytic_c_inf, compute_S_and_Sy_at_infinity

    y_inf_idx = -1
    f_inf = f_t[0, y_inf_idx:y_inf_idx+1]
    f_y_inf = f_y_t[0, y_inf_idx:y_inf_idx+1]
    S_inf, Sy_inf = compute_S_and_Sy_at_infinity(f_inf, f_y_inf, slope)
    c_inf = analytic_c_inf(a_t, omega_t, lam_t, m=m_mode, M=M, s=s_val)
    print(f"  At y={y_np[y_inf_idx]:.4f}:")
    print(f"    S={complex(S_inf.reshape(-1)[0].item()):.6e}, S_y={complex(Sy_inf.reshape(-1)[0].item()):.6e}")
    print(f"    c_inf (target S_y/S)={complex(c_inf.reshape(-1)[0].item()):.6e}")
    print(f"    actual S_y/S={complex((Sy_inf.reshape(-1)[0]/S_inf.reshape(-1)[0]).item()):.6e}")

    # === Test 4: Horizon check ===
    print("\n=== Horizon check ===")
    y_h_idx = 0
    f_h = f_t[0, y_h_idx]
    S_at_h = compose_reduced_shape_from_f(f=f_h.unsqueeze(0).unsqueeze(0),
                                            y=torch.tensor([y_np[y_h_idx]], dtype=dtype),
                                            slope=slope)
    print(f"  At y={y_np[y_h_idx]:.4f}: f={complex(f_h.item()):.6e}")
    print(f"    S (from ansatz) = {complex(S_at_h.reshape(-1)[0].item()):.6e}")
    print(f"    Expected S(1) = 1.0")


if __name__ == "__main__":
    main()
