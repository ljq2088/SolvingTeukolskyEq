#!/usr/bin/env python3
"""Single-point PDE diagnostic: train model on one (a,omega) to see if PDE residual converges.

Usage:
  python scripts/debug_single_point.py              # pure PDE
  python scripts/debug_single_point.py --robin 1.0  # PDE + Robin BC
  python scripts/debug_single_point.py --robin 1.0 --n-steps 5000
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
from physical_ansatz.infinity_robin import (
    analytic_c_inf, infinity_robin_loss, compute_S_and_Sy_at_infinity,
)


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
        shape = compose_reduced_shape_from_f(f=f_pred.squeeze(0), y=y_valid.squeeze(0), slope=slope.squeeze(0))
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
        R_mod_interp = np.interp(r_ref, r_mod, np.abs(R_mod)) * np.exp(1j * np.interp(r_ref, r_mod, np.angle(R_mod)))
        R_ref_abs = np.abs(R_ref)
        mask = R_ref_abs > 1e-15
        rel_err = np.abs(np.abs(R_mod_interp)[mask] - R_ref_abs[mask]) / R_ref_abs[mask]
        return float(np.median(rel_err))
    except Exception as e:
        print(f"  [pybhpt error: {e}]")
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--robin", type=float, default=0.0, help="Infinity Robin weight (0 = disabled)")
    parser.add_argument("--n-steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--normalize", action="store_true", help="Enable residual normalization")
    parser.add_argument("--normalize-mode", type=str, default="term",
                        choices=["term", "coeff"], help="Normalization mode")
    args = parser.parse_args()

    device = torch.device("cuda")
    M, ell, m_mode, s_val = 1.0, 2, 2, -2

    # Load atlas + patch for (u,v) mapping
    atlas = load_atlas("outputs/domain/atlas_l2_m2_logw.json")
    comp = atlas.components[0]
    patch_file = json.load(open("outputs/domain/patch_cover_l2_m2_logw.json"))
    p0 = [p for p in patch_file["patches"] if p["patch_id"] == 0][0]
    a_center = float(p0["a_center"])
    omega_center = float(p0["omega_center"])
    omega_chart = patch_file["meta"]["omega_chart_mode"]
    u_center, v_center = map_to_chart(comp, a_center, omega_center, omega_chart_mode=omega_chart)
    lam_center = compute_lambda(a_center, omega_center, ell, m_mode, s=s_val)

    norm_tag = f"+Norm({args.normalize_mode})" if args.normalize else ""
    tag = f"PDE{norm_tag}+Robin(w={args.robin})" if args.robin > 0 else f"PDE{norm_tag} only"
    print(f"Single-point diagnosis [{tag}]: a={a_center:.4f}, omega={omega_center:.6f}, lambda={lam_center:.10f}")
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

    # Fixed Chebyshev y-grid
    r_grid_t = torch.tensor(r_grid, device=device, dtype=dtype)
    rp_t = r_plus(a_t, M)
    x_grid = rp_t / r_grid_t
    y_grid = 2.0 * x_grid - 1.0
    valid = (y_grid >= -1) & (y_grid < 1)
    y_valid = y_grid[valid].unsqueeze(0)

    # Prepare y=-1 point for Robin loss
    y_inf = torch.tensor([[-1.0]], device=device, dtype=dtype)

    print(f"y_grid: {len(y_valid.squeeze(0))} points, range=[{y_valid.min():.4f}, {y_valid.max():.4f}]")
    hdr = f"{'Step':>8s} {'Loss':>10s} {'PDE':>10s} {'Robin':>10s} {'GradN':>10s} {'RelErr':>10s}" if args.robin > 0 else \
          f"{'Step':>8s} {'Loss':>10s} {'GradN':>10s} {'RelErr':>10s}"
    print(hdr)
    print("-" * (52 if args.robin > 0 else 42))

    best_loss = float("inf")
    n_steps = args.n_steps

    for step in range(n_steps):
        optimizer.zero_grad()

        loss, info = pinn_residual_loss(
            model=model, cfg=cfg,
            a_batch=a_t, omega_batch=omega_t, lambda_batch=lam_t_c,
            y_interior=y_valid, y_boundary=y_boundary,
            weight_interior=1.0, weight_boundary=0.0,
            normalize_residual=args.normalize, residual_scale_eps=1e-12,
            return_pointwise=False, u_batch=u_t, v_batch=v_t,
            normalize_mode=args.normalize_mode,
        )

        loss_robin_val = 0.0
        if args.robin > 0:
            # Compute Robin loss at y=-1
            f_inf, fy_inf, _ = compute_f_derivatives_autograd(
                model, a_t, omega_t, y_inf, u_batch=u_t, v_batch=v_t,
            )
            f_inf = f_inf.squeeze(-1)
            fy_inf = fy_inf.squeeze(-1)
            slope_inf = horizon_regularity_slope(
                a=a_t, omega=omega_t, lambda_=lam_t_c, m=m_mode, M=M, s=s_val,
            )
            S_inf, Sy_inf = compute_S_and_Sy_at_infinity(f_inf, fy_inf, slope_inf)
            if not torch.is_complex(S_inf):
                S_inf = S_inf.to(dtype=cdtype)
            if not torch.is_complex(Sy_inf):
                Sy_inf = Sy_inf.to(dtype=cdtype)
            c_inf = analytic_c_inf(a_t, omega_t, lam_t_c, m=m_mode, M=M, s=s_val)
            loss_robin = infinity_robin_loss(S_inf, Sy_inf, c_inf).mean()
            loss_robin_val = float(loss_robin)
            if torch.isfinite(loss_robin):
                loss = loss + args.robin * loss_robin

        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        loss_val = float(loss)

        if step % 100 == 0:
            model.eval()
            rel_err = benchmark_against_pybhpt(
                model, a_center, omega_center, u_center, v_center,
                lam_center, r_grid, M, m_mode, s_val, device, cfg
            )
            model.train()
            err_str = f"{rel_err:.4e}" if rel_err is not None else "FAIL"
            if args.robin > 0:
                print(f"{step:>8d} {loss_val:>10.4f} {info['loss_interior']:>10.4f} {loss_robin_val:>10.4f} {float(gn):>10.2f} {err_str:>10s}")
            else:
                print(f"{step:>8d} {loss_val:>10.4f} {float(gn):>10.2f} {err_str:>10s}")
        elif step % 10 == 0:
            if args.robin > 0:
                print(f"{step:>8d} {loss_val:>10.4f} {info['loss_interior']:>10.4f} {loss_robin_val:>10.4f} {float(gn):>10.2f}")
            else:
                print(f"{step:>8d} {loss_val:>10.4f} {float(gn):>10.2f}")

        best_loss = min(best_loss, loss_val)

    print(f"\nFinal: best_loss={best_loss:.4f}")

    model.eval()
    final_rel_err = benchmark_against_pybhpt(
        model, a_center, omega_center, u_center, v_center,
        lam_center, r_grid, M, m_mode, s_val, device, cfg
    )
    print(f"Final rel_err: {final_rel_err}")


if __name__ == "__main__":
    main()
