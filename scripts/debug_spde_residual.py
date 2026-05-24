#!/usr/bin/env python3
"""Compare S-PDE residual for true pybhpt solution vs f-PDE-trained model.
Key hypothesis: S-PDE is homogeneous (D2*S_yy + D1*S_y + D0*S = 0) and does NOT
degenerate to 0=0 at the horizon, so the true solution should have lower residual
than wrong smooth solutions.
"""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import json, numpy as np, torch
from model.autoencoder_pinn import AutoencoderTeukolskyPINN
from physical_ansatz.transform_y import (
    horizon_regularity_slope, h_factor, h1_factor, g_factor,
    compose_reduced_shape_from_f,
)
from physical_ansatz.prefactor import Leaver_prefactors, build_prefactor_primitives, prefactor_log_derivatives_x
from physical_ansatz.mapping import r_plus
from physical_ansatz.teukolsky_coeffs import coeffs_x
from utils.compute_lambda_usage import compute_lambda
from domain.atlas_builder import load_atlas, map_to_chart
from physical_ansatz.residual_pinn import compute_f_derivatives_autograd


def compute_spde_residual_for_f(f, f_y, f_yy, y, a, omega, lam, M=1.0, m=2, s=-2):
    """Compute S-PDE residual: D2*S_yy + D1*S_y + D0*S, where S is built from f."""
    x = 0.5 * (y + 1.0)
    slope = horizon_regularity_slope(a=a, omega=omega, lambda_=lam, m=m, M=M, s=s)

    # Build S from f via ansatz
    S = compose_reduced_shape_from_f(f=f, y=y, slope=slope)

    # S derivatives via chain rule
    h1, h1_x, h1_xx = h1_factor(x)
    g, g_x, g_xx = g_factor(x, slope)

    W = g * h1
    W_x = g_x * h1 + g * h1_x
    W_xx = g_xx * h1 + 2.0 * g_x * h1_x + g * h1_xx
    G = g + 1.0
    G_x = g_x
    G_xx = g_xx

    f_x = 2.0 * f_y
    f_xx = 4.0 * f_yy

    S_y_from_f = W_x * f + W * f_x + G_x  # This is S_x, need S_y = S_x * dx/dy = S_x / 2
    # Actually: S_x = W_x*f + W*f_x + G_x, and S_y = S_x * dx/dy = S_x / 2
    # But we already have f_y, f_yy, so let me recompute properly
    # S = W*f + G
    # S_y = W_y*f + W*f_y + G_y
    # W_y = W_x * dx/dy = W_x / 2, same for G
    # S_yy = W_yy*f + 2*W_y*f_y + W*f_yy + G_yy
    # W_yy = W_xx * (dx/dy)^2 = W_xx / 4

    W_y = W_x / 2.0
    W_yy = W_xx / 4.0
    G_y = G_x / 2.0
    G_yy = G_xx / 4.0

    S_y = W_y * f + W * f_y + G_y
    S_yy = W_yy * f + 2.0 * W_y * f_y + W * f_yy + G_yy

    # S-PDE coefficients
    rp = r_plus(a, M)
    r = rp / x
    A2, A1, A0 = coeffs_x(x=x, a=a, omega=omega, m=m, lambda_=lam, s=s, M=M)
    log_Px, log_Pxx = prefactor_log_derivatives_x(r=r, a=a, omega=omega, m=m, M=M, s=s)

    D2 = 4.0 * A2
    D1 = 4.0 * A2 * log_Px + 2.0 * A1
    D0 = A2 * log_Pxx + A1 * log_Px + A0

    residual = D2 * S_yy + D1 * S_y + D0 * S
    return residual, S, D2, D1, D0


def main():
    device = torch.device("cuda")
    M, ell, m_mode, s_val = 1.0, 2, 2, -2
    dtype = torch.float64
    cdtype = torch.complex128

    atlas = load_atlas("outputs/domain/atlas_l2_m2_logw.json")
    comp = atlas.components[0]
    patch_file = json.load(open("outputs/domain/patch_cover_l2_m2_logw.json"))
    p0 = [p for p in patch_file["patches"] if p["patch_id"] == 0][0]
    a_c = float(p0["a_center"])
    omega_c = float(p0["omega_center"])
    lam_c = compute_lambda(a_c, omega_c, ell, m_mode, s=s_val)

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
    y_t = torch.tensor(y_valid, dtype=dtype)

    # === True pybhpt solution ===
    from pybhpt_usage.compute_solution import compute_pybhpt_solution
    _, R_ref = compute_pybhpt_solution(a_c, omega_c, ell=2, m=m_mode, r_grid=r_valid, timeout=30.0)
    R_ref = np.asarray(R_ref, dtype=np.complex128)

    r_t = torch.tensor(r_valid, dtype=dtype)
    rp_t = r_plus(a_t, M)
    _, rm_t, _, _, _ = build_prefactor_primitives(r_t, a_t, M=M, need_rs=False)
    P, _, _ = Leaver_prefactors(r_t, a_t, omega_t, m=m_mode, M=M, s=s_val, rp=rp_t, rm=rm_t)
    h2 = h_factor(a_t, omega_t, m=m_mode, M=M, s=s_val)
    S_true = torch.tensor(R_ref, dtype=cdtype) / (P * h2)

    slope = horizon_regularity_slope(a=a_t, omega=omega_t, lambda_=lam_t, m=m_mode, M=M, s=s_val)
    x_v = torch.tensor(x_grid[valid_v], dtype=dtype)
    h1, _, _ = h1_factor(x_v)
    g, _, _ = g_factor(x_v, slope.squeeze())
    g = g.squeeze()
    h1 = h1.squeeze()
    S_true_sq = S_true.squeeze()
    f_true = (S_true_sq - g - 1.0) / (g * h1)

    # Finite-difference f_y, f_yy for true solution
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

    # Compute S-PDE residual for true solution
    f_t = torch.tensor(f_np, dtype=cdtype).unsqueeze(0)
    f_y_t = torch.tensor(f_y_np, dtype=cdtype).unsqueeze(0)
    f_yy_t = torch.tensor(f_yy_np, dtype=cdtype).unsqueeze(0)
    y_batch = y_t.unsqueeze(0)

    resid_true, S_from_f_true, D2_t, D1_t, D0_t = compute_spde_residual_for_f(
        f_t, f_y_t, f_yy_t, y_batch, a_t, omega_t, lam_t)

    # Also compute S-PDE residual directly from S_true (should match)
    # S derivatives via finite diff
    S_np = S_true_sq.detach().numpy()
    S_y_np = np.zeros_like(S_np)
    S_yy_np = np.zeros_like(S_np)
    for i in range(1, len(y_np) - 1):
        dy = y_np[i+1] - y_np[i-1]
        S_y_np[i] = (S_np[i+1] - S_np[i-1]) / dy
        S_yy_np[i] = (S_np[i+1] - 2*S_np[i] + S_np[i-1]) / ((y_np[i+1] - y_np[i]) * (y_np[i] - y_np[i-1]))
    S_y_np[0] = (S_np[1] - S_np[0]) / (y_np[1] - y_np[0])
    S_y_np[-1] = (S_np[-1] - S_np[-2]) / (y_np[-1] - y_np[-2])
    S_yy_np[0] = S_yy_np[1]
    S_yy_np[-1] = S_yy_np[-2]

    # S-PDE coeffs from S directly
    x_v_t = torch.tensor(x_grid[valid_v], dtype=dtype)
    rp_v = r_plus(a_t, M)
    r_v = rp_v / x_v_t
    A2_d, A1_d, A0_d = coeffs_x(x=x_v_t, a=a_t.squeeze(), omega=omega_t.squeeze(),
                                  m=m_mode, lambda_=lam_t.squeeze(), s=s_val, M=M)
    log_Px_d, log_Pxx_d = prefactor_log_derivatives_x(r=r_v, a=a_t, omega=omega_t, m=m_mode, M=M, s=s_val)
    D2_d = 4.0 * A2_d
    D1_d = 4.0 * A2_d * log_Px_d + 2.0 * A1_d
    D0_d = A2_d * log_Pxx_d + A1_d * log_Px_d + A0_d

    S_y_t = torch.tensor(S_y_np, dtype=cdtype)
    S_yy_t = torch.tensor(S_yy_np, dtype=cdtype)
    S_t = torch.tensor(S_np, dtype=cdtype)
    resid_direct = D2_d * S_yy_t + D1_d * S_y_t + D0_d * S_t

    pointwise_true = np.abs(resid_true.squeeze().detach().numpy())**2
    pointwise_direct = np.abs(resid_direct.detach().numpy())**2

    print("=== S-PDE residual for TRUE pybhpt solution ===")
    print(f"  mean |res|^2 (via f):  {np.mean(pointwise_true):.6e}")
    print(f"  median (via f):        {np.median(pointwise_true):.6e}")
    print(f"  max (via f):           {np.max(pointwise_true):.6e}")
    print(f"  mean |res|^2 (direct): {np.mean(pointwise_direct):.6e}")
    print(f"  median (direct):       {np.median(pointwise_direct):.6e}")
    print(f"  max (direct):          {np.max(pointwise_direct):.6e}")

    # Check S-PDE coefficients at horizon and infinity
    print(f"\n=== S-PDE coefficient analysis ===")
    D2_np = np.asarray(D2_d.detach().numpy(), dtype=np.complex128).ravel()
    D1_np = np.asarray(D1_d.detach().numpy(), dtype=np.complex128).ravel()
    D0_np = np.asarray(D0_d.detach().numpy(), dtype=np.complex128).ravel()
    S_np_flat = np.asarray(S_np, dtype=np.complex128).ravel()
    S_y_flat = np.asarray(S_y_np, dtype=np.complex128).ravel()
    S_yy_flat = np.asarray(S_yy_np, dtype=np.complex128).ravel()
    for label, idx in [("horizon (y≈1)", 0), ("mid (y≈0)", len(y_np)//2), ("infinity (y≈-1)", -1)]:
        print(f"  {label}: y={y_np[idx]:.4f}")
        print(f"    |D2|={abs(D2_np[idx]):.4e}, |D1|={abs(D1_np[idx]):.4e}, |D0|={abs(D0_np[idx]):.4e}")
        print(f"    D2={complex(D2_np[idx])}, D1={complex(D1_np[idx])}, D0={complex(D0_np[idx])}")

    # Check individual S-PDE terms
    print(f"\n=== S-PDE term analysis for true solution ===")
    D2_Syy = (D2_d * S_yy_t).detach().numpy().ravel()
    D1_Sy = (D1_d * S_y_t).detach().numpy().ravel()
    D0_S = (D0_d * S_t).detach().numpy().ravel()
    for label, idx in [("horizon", 0), ("mid", len(y_np)//2), ("infinity", -1)]:
        print(f"  {label} (y={y_np[idx]:.4f}):")
        print(f"    |D2*S_yy|={abs(D2_Syy[idx]):.4e}, |D1*S_y|={abs(D1_Sy[idx]):.4e}, |D0*S|={abs(D0_S[idx]):.4e}")
        print(f"    S={complex(S_np_flat[idx])}, S_y={complex(S_y_flat[idx])}, S_yy={complex(S_yy_flat[idx])}")

    # === f-PDE-trained model ===
    print(f"\n=== Training model with f-PDE loss, checking S-PDE residual ===")
    omega_chart = patch_file["meta"]["omega_chart_mode"]
    u_c, v_c = map_to_chart(comp, a_c, omega_c, omega_chart_mode=omega_chart)

    model = AutoencoderTeukolskyPINN().to(device)
    model.train()
    u_t_d = torch.tensor([u_c], device=device, dtype=dtype)
    v_t_d = torch.tensor([v_c], device=device, dtype=dtype)
    a_t_d = torch.tensor([a_c], device=device, dtype=dtype)
    omega_t_d = torch.tensor([omega_c], device=device, dtype=dtype)
    lam_t_d = torch.tensor([lam_c], device=device, dtype=cdtype)
    y_valid_batch = y_t.unsqueeze(0).to(device)
    y_bd = torch.empty(0, device=device, dtype=dtype)
    cfg = {"problem": {"M": M, "s": s_val, "m": m_mode}}

    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    for step in range(2000):
        opt.zero_grad()
        from physical_ansatz.residual_pinn import pinn_residual_loss
        loss, _ = pinn_residual_loss(model=model, cfg=cfg,
            a_batch=a_t_d, omega_batch=omega_t_d, lambda_batch=lam_t_d,
            y_interior=y_valid_batch, y_boundary=y_bd,
            weight_interior=1.0, weight_boundary=0.0,
            normalize_residual=False, u_batch=u_t_d, v_batch=v_t_d)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    # Compute S-PDE residual for trained model
    model.eval()
    with torch.no_grad():
        f_m, f_y_m, f_yy_m = compute_f_derivatives_autograd(
            model, a_t_d, omega_t_d, y_valid_batch, u_batch=u_t_d, v_batch=v_t_d)
        resid_model, S_model, _, _, _ = compute_spde_residual_for_f(
            f_m, f_y_m, f_yy_m, y_valid_batch, a_t_d, omega_t_d, lam_t_d)

    pointwise_model = np.abs(resid_model.squeeze().cpu().numpy())**2

    print(f"  f-PDE loss after training: {loss.item():.6f}")
    print(f"  S-PDE mean |res|^2: {np.mean(pointwise_model):.6e}")
    print(f"  S-PDE median |res|^2: {np.median(pointwise_model):.6e}")
    print(f"  S-PDE max |res|^2: {np.max(pointwise_model):.6e}")

    print(f"\n=== Comparison ===")
    print(f"  True solution S-PDE mean |res|^2:  {np.mean(pointwise_direct):.6e}")
    print(f"  Model (f-PDE trained) S-PDE mean:  {np.mean(pointwise_model):.6e}")
    ratio = np.mean(pointwise_model) / max(np.mean(pointwise_direct), 1e-30)
    print(f"  Ratio model/true: {ratio:.2f}")
    if ratio > 1:
        print(f"  >>> S-PDE loss correctly identifies model as worse!")
    else:
        print(f"  >>> S-PDE loss still biased against true solution")


if __name__ == "__main__":
    main()
