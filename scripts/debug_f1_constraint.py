#!/usr/bin/env python3
"""Test whether adding an explicit f(1) constraint fixes the single-point PDE training.

Key diagnosis: B2, B1, B0 → 0 at y=1, so the PDE residual is identically 0=0
at the horizon. f(1) is unconstrained. This script adds a loss term that pins
f(1) to the analytic value from the horizon consistency condition.
"""
import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from model.autoencoder_pinn import AutoencoderTeukolskyPINN
from physical_ansatz.transform_y import (
    compose_reduced_shape_from_f,
    horizon_regularity_slope,
    h_factor,
)
from physical_ansatz.prefactor import Leaver_prefactors, build_prefactor_primitives
from physical_ansatz.mapping import r_plus
from utils.compute_lambda_usage import compute_lambda
from domain.atlas_builder import load_atlas, map_to_chart
from physical_ansatz.residual_pinn import pinn_residual_loss, compute_f_derivatives_autograd
from physical_ansatz.teukolsky_coeffs import coeffs_x
from physical_ansatz.transform_y import transform_coeffs_x_to_y


def compute_f1_analytic(a, omega, lam, M=1.0, m=2, s=-2):
    """Compute f(1) from the horizon consistency condition.

    Uses the limit: f(1) = lim_{y→1} rhs(y)/B0(y)

    Numerically stable: evaluate at a very fine grid near y=1 and extrapolate.
    """
    dtype = torch.float64
    cdtype = torch.complex128
    a_t = torch.tensor([a], dtype=dtype)
    omega_t = torch.tensor([omega], dtype=dtype)
    lam_t = torch.tensor([lam], dtype=cdtype)

    # Build fine y-grid near horizon
    rp = float(r_plus(a_t, M))
    eps_vals = np.logspace(-12, -3, 50)
    r_vals = rp + eps_vals
    x_vals = rp / r_vals
    y_vals = 2.0 * x_vals - 1.0
    y_t = torch.tensor(y_vals, dtype=dtype)

    slope = horizon_regularity_slope(a=a_t, omega=omega_t, lambda_=lam_t, m=m, M=M, s=s)

    A2, A1, A0 = coeffs_x(x=0.5*(y_t+1.0), a=a_t, omega=omega_t, m=m, lambda_=lam_t, s=s, M=M)
    B2, B1, B0, rhs = transform_coeffs_x_to_y(
        A2.unsqueeze(0), A1.unsqueeze(0), A0.unsqueeze(0),
        y_t.unsqueeze(0), slope=slope,
    )

    B0_np = B0.squeeze(0).detach().numpy()
    rhs_np = rhs.squeeze(0).detach().numpy()
    f_approx = rhs_np / (B0_np + 1e-30)

    # Sort by y (increasing) for extrapolation
    y_np = y_vals
    sort_idx = np.argsort(y_np)
    y_sorted = y_np[sort_idx]
    f_sorted = f_approx[sort_idx]

    # Extrapolate to y=1 using linear fit on last few points
    # f(y) ≈ f(1) + f'(1)*(y-1), so f(y) vs y is linear near y=1
    n_fit = min(10, len(y_sorted))
    y_fit = y_sorted[-n_fit:]
    f_fit = f_sorted[-n_fit:]
    fit_real = np.polyfit(y_fit, f_fit.real, 1)
    fit_imag = np.polyfit(y_fit, f_fit.imag, 1)
    f1 = complex(np.polyval(fit_real, 1.0), np.polyval(fit_imag, 1.0))

    return f1


def benchmark_against_pybhpt(model, a, omega, u, v, lam, r_grid, M, m, s, device, cfg):
    """Compare model output against pybhpt."""
    dtype = torch.float64
    a_t = torch.tensor([a], device=device, dtype=dtype)
    omega_t = torch.tensor([omega], device=device, dtype=dtype)
    u_t = torch.tensor([u], device=device, dtype=dtype)
    v_t = torch.tensor([v], device=device, dtype=dtype)
    lam_t = torch.tensor([lam], device=device, dtype=torch.complex128)
    r_grid_t = torch.tensor(r_grid, device=device, dtype=dtype)
    rp = r_plus(a_t, M)
    x_grid = rp / r_grid_t
    y_grid = 2.0 * x_grid - 1.0

    with torch.no_grad():
        valid = (y_grid >= -1) & (y_grid < 1)
        y_valid = y_grid[valid]
        f_pred = model(a_t, omega_t, y_valid, u=u_t, v=v_t)
        slope = horizon_regularity_slope(a=a_t, omega=omega_t, lambda_=lam_t, m=m, M=M, s=s)
        shape = compose_reduced_shape_from_f(
            f=f_pred.squeeze(0), y=y_valid.squeeze(0), slope=slope.squeeze(0))
        h2 = h_factor(a_t, omega_t, m=m, M=M, s=s)
        rp_val = r_plus(a_t, M)
        r_eff = r_grid_t[valid]
        _, rm, _, _, _ = build_prefactor_primitives(r_eff, a_t, M=M, need_rs=False)
        P, _, _ = Leaver_prefactors(r_eff, a_t, omega_t, m=m, M=M, s=s, rp=rp_val, rm=rm)
        R_pred = P.squeeze(0) * h2 * shape

    try:
        from pybhpt_usage.compute_solution import compute_pybhpt_solution
        r_mod = r_eff.cpu().numpy()
        R_mod = R_pred.detach().cpu().numpy()
        r_ref, R_ref = compute_pybhpt_solution(a, omega, ell=2, m=m, r_grid=r_mod, timeout=30.0)
        R_ref = np.asarray(R_ref, dtype=np.complex128)
        R_mod_interp = np.interp(r_ref, r_mod, np.abs(R_mod)) * np.exp(
            1j * np.interp(r_ref, r_mod, np.angle(R_mod)))
        R_ref_abs = np.abs(R_ref)
        mask = R_ref_abs > 1e-15
        rel_err = np.abs(np.abs(R_mod_interp)[mask] - R_ref_abs[mask]) / R_ref_abs[mask]
        return float(np.median(rel_err))
    except Exception as e:
        print(f"  [pybhpt error: {e}]")
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--robin", type=float, default=0.0)
    parser.add_argument("--f1-weight", type=float, default=1.0,
                        help="Weight for f(1) consistency constraint")
    parser.add_argument("--n-steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    device = torch.device("cuda")
    M, ell, m_mode, s_val = 1.0, 2, 2, -2

    atlas = load_atlas("outputs/domain/atlas_l2_m2_logw.json")
    comp = atlas.components[0]
    patch_file = json.load(open("outputs/domain/patch_cover_l2_m2_logw.json"))
    p0 = [p for p in patch_file["patches"] if p["patch_id"] == 0][0]
    a_center = float(p0["a_center"])
    omega_center = float(p0["omega_center"])
    omega_chart = patch_file["meta"]["omega_chart_mode"]
    u_center, v_center = map_to_chart(comp, a_center, omega_center, omega_chart_mode=omega_chart)
    lam_center = compute_lambda(a_center, omega_center, ell, m_mode, s=s_val)

    # Compute correct f(1) from horizon consistency
    f1_target = compute_f1_analytic(a_center, omega_center, lam_center, M=M, m=m_mode, s=s_val)
    print(f"Analytic f(1) from horizon consistency: {f1_target:.10f}")
    print(f"  |f(1)| = {abs(f1_target):.6f}")

    tag = f"PDE+f1_constraint(w={args.f1_weight})"
    if args.robin > 0:
        tag += f"+Robin(w={args.robin})"
    print(f"\nSingle-point diagnosis [{tag}]: a={a_center:.4f}, omega={omega_center:.6f}")
    print(f"u={u_center:.6f}, v={v_center:.6f}, lr={args.lr}")

    model = AutoencoderTeukolskyPINN()
    model.to(device)
    model.train()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {n_params}")

    rp = float(1.0 + np.sqrt(1.0 - a_center**2))
    r_grid = np.logspace(np.log10(rp + 0.001), 2.0, 128)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    cfg = {"problem": {"M": M, "s": s_val, "m": m_mode}}

    dtype = torch.float64
    cdtype = torch.complex128
    a_t = torch.tensor([a_center], device=device, dtype=dtype)
    omega_t = torch.tensor([omega_center], device=device, dtype=dtype)
    u_t = torch.tensor([u_center], device=device, dtype=dtype)
    v_t = torch.tensor([v_center], device=device, dtype=dtype)
    lam_t_c = torch.tensor([lam_center], device=device, dtype=cdtype)
    y_boundary = torch.empty(0, device=device, dtype=dtype)

    r_grid_t = torch.tensor(r_grid, device=device, dtype=dtype)
    rp_t = r_plus(a_t, M)
    x_grid = rp_t / r_grid_t
    y_grid = 2.0 * x_grid - 1.0
    valid = (y_grid >= -1) & (y_grid < 1)
    y_valid = y_grid[valid].unsqueeze(0)

    # Point very close to y=1 for f(1) constraint
    y_near_horizon = torch.tensor([[y_valid.max().item()]], device=device, dtype=dtype)
    f1_t = torch.tensor([[complex(f1_target)]], device=device, dtype=cdtype)

    print(f"y_grid: {len(y_valid.squeeze(0))} points, range=[{y_valid.min():.4f}, {y_valid.max():.4f}]")

    hdr = f"{'Step':>8s} {'Loss':>10s} {'PDE':>10s} {'F1':>10s} {'GradN':>10s} {'RelErr':>10s}"
    print(hdr)
    print("-" * 62)

    best_loss = float("inf")

    for step in range(args.n_steps):
        optimizer.zero_grad()

        loss, info = pinn_residual_loss(
            model=model, cfg=cfg,
            a_batch=a_t, omega_batch=omega_t, lambda_batch=lam_t_c,
            y_interior=y_valid, y_boundary=y_boundary,
            weight_interior=1.0, weight_boundary=0.0,
            normalize_residual=False, residual_scale_eps=1e-12,
            return_pointwise=False, u_batch=u_t, v_batch=v_t,
        )

        # f(1) constraint: penalize deviation from analytic f(1)
        loss_f1 = torch.tensor(0.0, device=device, dtype=loss.dtype)
        if args.f1_weight > 0:
            f_pred_1, _, _ = compute_f_derivatives_autograd(
                model, a_t, omega_t, y_near_horizon,
                u_batch=u_t, v_batch=v_t,
            )
            loss_f1 = torch.mean(torch.abs(f_pred_1 - f1_t) ** 2)
            loss = loss + args.f1_weight * loss_f1

        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        loss_val = float(loss)

        if step % 100 == 0:
            model.eval()
            rel_err = benchmark_against_pybhpt(
                model, a_center, omega_center, u_center, v_center,
                lam_center, r_grid, M, m_mode, s_val, device, cfg,
            )
            model.train()
            err_str = f"{rel_err:.4e}" if rel_err is not None else "FAIL"
            print(f"{step:>8d} {loss_val:>10.4f} {info['loss_interior']:>10.4f} "
                  f"{float(loss_f1):>10.4f} {float(gn):>10.2f} {err_str:>10s}")
        elif step % 50 == 0:
            print(f"{step:>8d} {loss_val:>10.4f} {info['loss_interior']:>10.4f} "
                  f"{float(loss_f1):>10.4f} {float(gn):>10.2f}")

        best_loss = min(best_loss, loss_val)

    print(f"\nFinal: best_loss={best_loss:.4f}")

    model.eval()
    final_rel_err = benchmark_against_pybhpt(
        model, a_center, omega_center, u_center, v_center,
        lam_center, r_grid, M, m_mode, s_val, device, cfg,
    )
    print(f"Final rel_err: {final_rel_err}")


if __name__ == "__main__":
    main()
