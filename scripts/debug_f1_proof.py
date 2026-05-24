#!/usr/bin/env python3
"""Proof-of-concept: add true f(1) from pybhpt as constraint. If this fixes training,
the diagnosis is confirmed — the PDE residual alone cannot determine f(1)."""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import json, numpy as np, torch
from model.autoencoder_pinn import AutoencoderTeukolskyPINN
from physical_ansatz.transform_y import (
    compose_reduced_shape_from_f, horizon_regularity_slope,
    h_factor, h1_factor, g_factor,
)
from physical_ansatz.prefactor import Leaver_prefactors, build_prefactor_primitives
from physical_ansatz.mapping import r_plus
from utils.compute_lambda_usage import compute_lambda
from domain.atlas_builder import load_atlas, map_to_chart
from physical_ansatz.residual_pinn import pinn_residual_loss, compute_f_derivatives_autograd


def main():
    device = torch.device("cuda")
    M, ell, m_mode, s_val = 1.0, 2, 2, -2

    atlas = load_atlas("outputs/domain/atlas_l2_m2_logw.json")
    comp = atlas.components[0]
    patch_file = json.load(open("outputs/domain/patch_cover_l2_m2_logw.json"))
    p0 = [p for p in patch_file["patches"] if p["patch_id"] == 0][0]
    a_c = float(p0["a_center"])
    omega_c = float(p0["omega_center"])
    omega_chart = patch_file["meta"]["omega_chart_mode"]
    u_c, v_c = map_to_chart(comp, a_c, omega_c, omega_chart_mode=omega_chart)
    lam_c = compute_lambda(a_c, omega_c, ell, m_mode, s=s_val)

    # Get true f(1) from pybhpt
    rp = float(M + np.sqrt(M**2 - a_c**2))
    from pybhpt_usage.compute_solution import compute_pybhpt_solution
    r_fine = np.logspace(np.log10(rp + 1e-6), np.log10(rp + 0.1), 200)
    r_ref, R_ref = compute_pybhpt_solution(a_c, omega_c, ell=2, m=m_mode, r_grid=r_fine, timeout=30.0)
    R_ref = np.asarray(R_ref, dtype=np.complex128)

    dtype = torch.float64
    cdtype = torch.complex128
    a_t = torch.tensor([a_c], dtype=dtype)
    omega_t = torch.tensor([omega_c], dtype=dtype)
    lam_t = torch.tensor([lam_c], dtype=cdtype)
    r_t = torch.tensor(r_ref, dtype=dtype)

    rp_t = r_plus(a_t, M)
    _, rm_t, _, _, _ = build_prefactor_primitives(r_t, a_t, M=M, need_rs=False)
    P, _, _ = Leaver_prefactors(r_t, a_t, omega_t, m=m_mode, M=M, s=s_val, rp=rp_t, rm=rm_t)
    h2 = h_factor(a_t, omega_t, m=m_mode, M=M, s=s_val)
    S_true = (torch.tensor(R_ref, dtype=cdtype) / (P * h2)).squeeze().detach().numpy()

    slope = horizon_regularity_slope(a=a_t, omega=omega_t, lambda_=lam_t, m=m_mode, M=M, s=s_val)
    x_fine = (float(rp_t) / r_t).numpy()
    y_fine = 2.0 * x_fine - 1.0
    slope_np = slope.squeeze().detach().numpy()
    h1_np, _, _ = h1_factor(torch.tensor(x_fine, dtype=dtype))
    g_np, _, _ = g_factor(torch.tensor(x_fine, dtype=dtype), slope.squeeze())
    h1_np = h1_np.numpy()
    g_np = g_np.squeeze().numpy()
    f_true = (S_true - g_np - 1.0) / (g_np * h1_np)

    # f(1) from the closest point to the horizon
    f1_true = complex(f_true[-1])  # closest to y=1 (largest y)
    y1 = float(y_fine[-1])
    print(f"True f at y={y1:.10f}: {f1_true:.10f}")
    print(f"  |f| = {abs(f1_true):.6f}")

    # Also check: what f does the model learn without constraint?
    print("\n=== Training WITHOUT f(1) constraint (baseline) ===")
    model = AutoencoderTeukolskyPINN().to(device)
    model.train()
    u_t = torch.tensor([u_c], device=device, dtype=dtype)
    v_t = torch.tensor([v_c], device=device, dtype=dtype)

    r_grid = np.logspace(np.log10(rp + 0.001), 2.0, 128)
    r_grid_t = torch.tensor(r_grid, device=device, dtype=dtype)
    x_grid = float(rp_t) / r_grid_t
    y_grid = 2.0 * x_grid - 1.0
    valid = (y_grid >= -1) & (y_grid < 1)
    y_valid = y_grid[valid].unsqueeze(0)

    a_t_d = torch.tensor([a_c], device=device, dtype=dtype)
    omega_t_d = torch.tensor([omega_c], device=device, dtype=dtype)
    lam_t_d = torch.tensor([lam_c], device=device, dtype=cdtype)
    y_bd = torch.empty(0, device=device, dtype=dtype)
    cfg = {"problem": {"M": M, "s": s_val, "m": m_mode}}

    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    for step in range(2000):
        opt.zero_grad()
        loss, _ = pinn_residual_loss(model=model, cfg=cfg,
            a_batch=a_t_d, omega_batch=omega_t_d, lambda_batch=lam_t_d,
            y_interior=y_valid, y_boundary=y_bd,
            weight_interior=1.0, weight_boundary=0.0,
            normalize_residual=False, u_batch=u_t, v_batch=v_t)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    model.eval()
    with torch.no_grad():
        f_pred = model(a_t_d, omega_t_d, y_valid, u=u_t, v=v_t).squeeze(0)
    f1_model = complex(f_pred[-1])
    print(f"  Model f(1) after 2000 steps: {f1_model:.6f}")
    print(f"  |f_model - f_true| at y≈1: {abs(f1_model - f1_true):.6e}")

    # Now train WITH f(1) constraint
    print("\n=== Training WITH f(1) constraint (weight=10) ===")
    model2 = AutoencoderTeukolskyPINN().to(device)
    model2.train()
    opt2 = torch.optim.Adam(model2.parameters(), lr=1e-4)

    y_horizon = torch.tensor([[y1]], device=device, dtype=dtype)
    f1_t = torch.tensor([[complex(f1_true)]], device=device, dtype=cdtype)

    for step in range(2000):
        opt2.zero_grad()
        loss, _ = pinn_residual_loss(model=model2, cfg=cfg,
            a_batch=a_t_d, omega_batch=omega_t_d, lambda_batch=lam_t_d,
            y_interior=y_valid, y_boundary=y_bd,
            weight_interior=1.0, weight_boundary=0.0,
            normalize_residual=False, u_batch=u_t, v_batch=v_t)

        # f(1) constraint
        f_pred_1, _, _ = compute_f_derivatives_autograd(
            model2, a_t_d, omega_t_d, y_horizon, u_batch=u_t, v_batch=v_t)
        loss_f1 = torch.mean(torch.abs(f_pred_1 - f1_t)**2)
        loss = loss + 10.0 * loss_f1

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model2.parameters(), 1.0)
        opt2.step()

        if step % 200 == 0:
            print(f"  step {step}: loss={loss.item():.4f}, f1_loss={loss_f1.item():.6f}")

    model2.eval()
    with torch.no_grad():
        f_pred2 = model2(a_t_d, omega_t_d, y_valid, u=u_t, v=v_t).squeeze(0)
    f1_model2 = complex(f_pred2[-1])
    print(f"  Model f(1) after 2000 steps: {f1_model2:.6f}")

    # Compare with pybhpt
    print("\n=== Benchmark comparison ===")
    from physical_ansatz.residual_pinn import compute_f_derivatives_autograd as cfd

    for label, m in [("baseline", model), ("f1_constrained", model2)]:
        m.eval()
        with torch.no_grad():
            r_grid_t2 = torch.tensor(r_grid, device=device, dtype=dtype)
            x_g = float(rp_t) / r_grid_t2
            y_g = 2.0 * x_g - 1.0
            valid2 = (y_g >= -1) & (y_g < 1)
            y_v = y_g[valid2]
            f_p = m(a_t_d, omega_t_d, y_v, u=u_t, v=v_t)
            sl = horizon_regularity_slope(a=a_t_d, omega=omega_t_d, lambda_=lam_t_d, m=m_mode, M=M, s=s_val)
            shape = compose_reduced_shape_from_f(f=f_p.squeeze(0), y=y_v.squeeze(0), slope=sl.squeeze(0))
            h2_d = h_factor(a_t_d, omega_t_d, m=m_mode, M=M, s=s_val)
            rp_v = r_plus(a_t_d, M)
            r_eff = r_grid_t2[valid2]
            _, rm_v, _, _, _ = build_prefactor_primitives(r_eff, a_t_d, M=M, need_rs=False)
            P_v, _, _ = Leaver_prefactors(r_eff, a_t_d, omega_t_d, m=m_mode, M=M, s=s_val, rp=rp_v, rm=rm_v)
            R_pred = P_v.squeeze(0) * h2_d * shape

        r_mod = r_eff.cpu().numpy()
        R_mod = R_pred.detach().cpu().numpy()
        r_ref2, R_ref2 = compute_pybhpt_solution(a_c, omega_c, ell=2, m=m_mode, r_grid=r_mod, timeout=30.0)
        R_ref2 = np.asarray(R_ref2, dtype=np.complex128)
        R_mod_i = np.interp(r_ref2, r_mod, np.abs(R_mod)) * np.exp(1j*np.interp(r_ref2, r_mod, np.angle(R_mod)))
        mask = np.abs(R_ref2) > 1e-15
        rel_err = np.abs(np.abs(R_mod_i)[mask] - np.abs(R_ref2)[mask]) / np.abs(R_ref2)[mask]
        print(f"  {label}: median_rel_err={np.median(rel_err):.4e}")


if __name__ == "__main__":
    main()
