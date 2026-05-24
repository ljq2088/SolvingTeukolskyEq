#!/usr/bin/env python3
"""Compare model f(y) vs pybhpt-true f(y) to find where PDE residual fails."""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model.autoencoder_pinn import AutoencoderTeukolskyPINN
from physical_ansatz.transform_y import (
    compose_reduced_shape_from_f,
    horizon_regularity_slope,
    h_factor, h1_factor, g_factor,
)
from physical_ansatz.prefactor import Leaver_prefactors, build_prefactor_primitives
from physical_ansatz.mapping import r_plus
from physical_ansatz.teukolsky_coeffs import coeffs_x
from physical_ansatz.transform_y import transform_coeffs_x_to_y
from utils.compute_lambda_usage import compute_lambda


def get_true_f(a, omega, y_grid, lam, M=1.0, m=2, s=-2):
    """Get true f(y) by inverting the ansatz from pybhpt R(r)."""
    from pybhpt_usage.compute_solution import compute_pybhpt_solution

    rp = float(M + np.sqrt(M**2 - a**2))
    x_grid = 0.5 * (y_grid + 1.0)
    r_grid = rp / x_grid

    _, R_ref = compute_pybhpt_solution(a, omega, ell=2, m=m, r_grid=r_grid, timeout=30.0)
    R_ref = np.asarray(R_ref, dtype=np.complex128)

    dtype = torch.float64
    cdtype = torch.complex128
    a_t = torch.tensor([a], dtype=dtype)
    omega_t = torch.tensor([omega], dtype=dtype)
    lam_t = torch.tensor([lam], dtype=cdtype)
    r_t = torch.tensor(r_grid, dtype=dtype)
    y_t = torch.tensor(y_grid, dtype=dtype)

    rp_t = r_plus(a_t, M)
    _, rm_t, _, _, _ = build_prefactor_primitives(r_t, a_t, M=M, need_rs=False)
    P, _, _ = Leaver_prefactors(r_t, a_t, omega_t, m=m, M=M, s=s, rp=rp_t, rm=rm_t)
    h2 = h_factor(a_t, omega_t, m=m, M=M, s=s)
    S = torch.tensor(R_ref, dtype=cdtype) / (P * h2)

    slope = horizon_regularity_slope(a=a_t, omega=omega_t, lambda_=lam_t, m=m, M=M, s=s)
    x = 0.5 * (y_t + 1.0)
    h1, _, _ = h1_factor(x)
    g, _, _ = g_factor(x, slope)

    f = (S - g - 1.0) / (g * h1)
    return f.squeeze().detach().numpy(), S.squeeze().detach().numpy(), slope.detach().numpy()


def main():
    a, omega = 0.5, 0.0257
    ell, m, s = 2, 2, -2
    M = 1.0
    lam = compute_lambda(a, omega, ell, m, s=s)

    # Build y-grid
    rp = float(M + np.sqrt(M**2 - a**2))
    n_pts = 128
    r_grid = np.logspace(np.log10(rp + 0.001), 2.0, n_pts)
    rp_t = r_plus(torch.tensor([a]), M)
    x_grid = float(rp_t) / r_grid
    y_grid = 2.0 * x_grid - 1.0
    valid = (y_grid >= -1) & (y_grid < 1)
    y_valid = y_grid[valid]
    y_t = torch.tensor(y_valid, dtype=torch.float64)

    # Get true f(y)
    print("Computing pybhpt true solution...")
    f_true, S_true, slope = get_true_f(a, omega, y_valid, lam, M=M, m=m, s=s)
    slope_t = torch.tensor(slope, dtype=torch.complex128)

    print(f"y range: [{y_valid[0]:.4f}, {y_valid[-1]:.4f}]")
    print(f"f_true range: |f| ∈ [{np.min(np.abs(f_true)):.4e}, {np.max(np.abs(f_true)):.4e}]")
    print(f"f_true at y≈1: f({y_valid[-1]:.4f}) = {complex(f_true[-1]):.6e}")
    print(f"f_true at y≈-1: f({y_valid[0]:.4f}) = {complex(f_true[0]):.6e}")

    # Train a quick model to see what f(y) it learns
    print("\nTraining model for 2000 steps...")
    device = torch.device("cuda")
    model = AutoencoderTeukolskyPINN()
    model.to(device)
    model.train()

    dtype = torch.float64
    cdtype = torch.complex128
    a_t = torch.tensor([a], device=device, dtype=dtype)
    omega_t = torch.tensor([omega], device=device, dtype=dtype)
    lam_t = torch.tensor([lam], device=device, dtype=cdtype)
    y_valid_t = torch.tensor(y_valid, device=device, dtype=dtype).unsqueeze(0)

    # Map (a,omega) to (u,v) for the autoencoder
    from domain.atlas_builder import load_atlas, map_to_chart
    atlas = load_atlas("outputs/domain/atlas_l2_m2_logw.json")
    comp = atlas.components[0]
    import json
    patch_file = json.load(open("outputs/domain/patch_cover_l2_m2_logw.json"))
    omega_chart = patch_file["meta"]["omega_chart_mode"]
    u_c, v_c = map_to_chart(comp, a, omega, omega_chart_mode=omega_chart)
    u_t = torch.tensor([u_c], device=device, dtype=dtype)
    v_t = torch.tensor([v_c], device=device, dtype=dtype)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    cfg = {"problem": {"M": M, "s": s, "m": m}}

    from physical_ansatz.residual_pinn import pinn_residual_loss
    y_boundary = torch.empty(0, device=device, dtype=dtype)

    # Track f(y) evolution
    f_history = []

    for step in range(2000):
        optimizer.zero_grad()
        loss, info = pinn_residual_loss(
            model=model, cfg=cfg,
            a_batch=a_t, omega_batch=omega_t, lambda_batch=lam_t,
            y_interior=y_valid_t, y_boundary=y_boundary,
            weight_interior=1.0, weight_boundary=0.0,
            normalize_residual=False, residual_scale_eps=1e-12,
            return_pointwise=False, u_batch=u_t, v_batch=v_t,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 200 == 0:
            model.eval()
            with torch.no_grad():
                f_pred = model(a_t, omega_t, y_valid_t, u=u_t, v=v_t)
            model.train()
            f_pred_np = f_pred.squeeze(0).detach().cpu().numpy()
            f_history.append((step, f_pred_np.copy()))
            rel_diff = np.abs(f_pred_np - f_true) / (np.abs(f_true) + 1e-12)
            print(f"  step {step}: loss={loss.item():.4f}, "
                  f"max|f-f_true|/|f_true|={np.max(rel_diff):.4e}, "
                  f"mean|f-f_true|/|f_true|={np.mean(rel_diff):.4e}")

    # Final comparison
    model.eval()
    with torch.no_grad():
        f_pred = model(a_t, omega_t, y_valid_t, u=u_t, v=v_t)
        f_pred_np = f_pred.squeeze(0).detach().cpu().numpy()

        # Also compute S from model f and compare
        S_model = compose_reduced_shape_from_f(
            f=f_pred, y=y_valid_t, slope=slope_t.to(device),
        ).squeeze(0).detach().cpu().numpy()

    # ---- Plots ----
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # f(y) comparison
    ax = axes[0, 0]
    ax.plot(y_valid, np.abs(f_true), 'k-', lw=2, label='pybhpt |f(y)|')
    ax.plot(y_valid, np.abs(f_pred_np), 'r--', lw=2, label='model |f(y)|')
    for step, f_h in f_history:
        ax.plot(y_valid, np.abs(f_h), alpha=0.3, lw=0.5)
    ax.set_xlabel('y')
    ax.set_ylabel('|f(y)|')
    ax.set_title('|f(y)| comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # f(y) relative error
    ax = axes[0, 1]
    rel_err = np.abs(f_pred_np - f_true) / (np.abs(f_true) + 1e-12)
    ax.semilogy(y_valid, rel_err, 'b-', lw=1.5)
    ax.set_xlabel('y')
    ax.set_ylabel('|f_model - f_true| / |f_true|')
    ax.set_title('f(y) Relative Error')
    ax.grid(True, alpha=0.3)
    # Mark y=-1 and y=1
    ax.axvline(-1, color='r', linestyle=':', alpha=0.5, label='infinity')
    ax.axvline(1, color='g', linestyle=':', alpha=0.5, label='horizon')
    ax.legend()

    # Coefficient magnitudes B2, B1, B0 vs y
    ax = axes[1, 0]
    A2, A1, A0 = coeffs_x(x=0.5*(y_t+1.0), a=a_t.cpu(), omega=omega_t.cpu(),
                           m=m, lambda_=lam_t.cpu(), s=s, M=M)
    B2, B1, B0, rhs = transform_coeffs_x_to_y(
        A2.unsqueeze(0), A1.unsqueeze(0), A0.unsqueeze(0),
        y_t.unsqueeze(0), slope=slope_t.cpu(),
    )
    B2_np = np.abs(B2.squeeze(0).detach().numpy())
    B1_np = np.abs(B1.squeeze(0).detach().numpy())
    B0_np = np.abs(B0.squeeze(0).detach().numpy())
    rhs_np = np.abs(rhs.squeeze(0).detach().numpy())
    ax.semilogy(y_valid, B2_np, label='|B2|')
    ax.semilogy(y_valid, B1_np, label='|B1|')
    ax.semilogy(y_valid, B0_np, label='|B0|')
    ax.semilogy(y_valid, rhs_np, label='|rhs|')
    ax.set_xlabel('y')
    ax.set_ylabel('Coefficient magnitude')
    ax.set_title('PDE Coefficients vs y')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # S(x) comparison
    ax = axes[1, 1]
    x_plot = 0.5 * (y_valid + 1.0)
    ax.plot(x_plot, np.abs(S_true), 'k-', lw=2, label='pybhpt |S(x)|')
    ax.plot(x_plot, np.abs(S_model), 'r--', lw=2, label='model |S(x)|')
    ax.set_xlabel('x = r+/r')
    ax.set_ylabel('|S(x)|')
    ax.set_title('|S(x)| comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle(f'f(y) Diagnosis: a={a}, omega={omega}', fontsize=13)
    fig.tight_layout()
    fig.savefig("outputs/debug_fy_comparison.png", dpi=150)
    plt.close(fig)
    print("\nSaved to outputs/debug_fy_comparison.png")

    # Print key diagnostics
    print(f"\n=== Key diagnostics ===")
    print(f"|f_model - f_true|/|f_true|: mean={np.mean(rel_err):.4e}, "
          f"max={np.max(rel_err):.4e}, at y={y_valid[np.argmax(rel_err)]:.4f}")
    print(f"|S_model - S_true|/|S_true|: mean={np.mean(np.abs(S_model - S_true)/(np.abs(S_true)+1e-12)):.4e}")

    # Check: does f_model satisfy the ODE at y=1?
    f_model_1 = complex(f_pred_np[-1])
    f_true_1 = complex(f_true[-1])
    A2_1 = float(A2[-1].detach())
    A1_1 = float(A1[-1].detach())
    A0_1 = float(A0[-1].detach())
    slope_val = slope_t.detach()
    expected_f1 = (-slope_val*(A2_1 + A1_1) - A0_1) / (2*A2_1*slope_val)
    expected_f1 = complex(expected_f1)
    print(f"\nf(1) from model: {f_model_1:.6e}")
    print(f"f(1) from pybhpt: {f_true_1:.6e}")
    print(f"f(1) from consistency: {expected_f1:.6e}")

    # Where does the residual actually constrain the solution?
    print(f"\n|B2|/|B0| ratio: min={np.min(B2_np/(B0_np+1e-12)):.4e}, "
          f"max={np.max(B2_np/(B0_np+1e-12)):.4e}")
    print(f"Region where |B2| < |B0|*0.01: {np.sum(B2_np < B0_np*0.01)}/{len(y_valid)} points")


if __name__ == "__main__":
    main()
