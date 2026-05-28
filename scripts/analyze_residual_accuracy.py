#!/usr/bin/env python3
"""Compute Teukolsky R-equation normalized residual for a trained Stage-1 model.

Normalized residual = |F| / max(|term_2nd|, |term_1st|, |term_0th|)

where:
    F = Delta * R_rr  +  (s+1) * Delta_r * R_r  +  V * R
    term_2nd = Delta * R_rr
    term_1st = (s+1) * Delta_r * R_r
    term_0th = V * R

Saves an a-omega-r 3D error matrix plus 2D slices for analysis.

Usage:
  python scripts/analyze_residual_accuracy.py \
    --run-dir outputs/stage1_retrain/20260523_032201_patch_000_comp_0_u_0.500_v_0.603 \
    --step 20000 \
    --device cuda
"""
import argparse, sys, os
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import torch
import numpy as np
import yaml


def load_config(run_dir):
    config_path = Path(run_dir) / "config_snapshot.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_model(checkpoint_path, config, device):
    from model.autoencoder_pinn import AutoencoderTeukolskyPINN
    mc = config.get("model", {})
    model = AutoencoderTeukolskyPINN(
        hidden_dims=mc.get("hidden_dims", [128, 128, 128, 128]),
        activation=mc.get("activation", "silu"),
        fourier_num_freqs=mc.get("fourier_num_freqs", 2),
        fourier_base_scale=mc.get("fourier_base_scale", 1.0),
        param_embed_dim=mc.get("param_embed_dim", 64),
        use_film=mc.get("use_film", True),
        use_residual=mc.get("use_residual", True),
    )
    ckpt = torch.load(checkpoint_path, map_location=device)
    sd = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    model.load_state_dict(sd, strict=False)
    model.to(device)
    model.eval()
    return model


def build_r_grid(y, rp):
    """Convert y in [-1,1] to r: x = (y+1)/2, r = rp / x."""
    x = 0.5 * (y + 1.0)
    return rp / x


def compute_R_and_derivatives(model, a, omega, lambda_, y_grid, M=1.0, s=-2, m=2):
    """
    Compute R(r), R_r(r), R_rr(r) on a y-grid via autograd.

    Full chain: f(y) -> S(x) = g*(h1*f+1)+1 -> R(r) = P(r)*h2*S(x)

    Returns: R, R_r, R_rr  each (N,) complex
    """
    from physical_ansatz.transform_y import (
        horizon_regularity_slope,
        compose_reduced_shape_from_f,
        h_factor,
    )
    from physical_ansatz.mapping import r_plus, dx_dr_from_x, d2x_dr2_from_x
    from physical_ansatz.teukolsky_coeffs import r_from_x, delta, delta_r, V_of_r, Leaver_prefactors, build_prefactor_primitives

    device = y_grid.device
    dtype = y_grid.dtype
    rp = r_plus(a, M)

    y_leaf = y_grid.unsqueeze(0).detach().clone().requires_grad_(True)

    # Forward pass
    f = model(a.unsqueeze(0), omega.unsqueeze(0), y_leaf)
    slope = horizon_regularity_slope(a=a.unsqueeze(0), omega=omega.unsqueeze(0),
                                     lambda_=lambda_.unsqueeze(0), m=m, M=M, s=s)
    S = compose_reduced_shape_from_f(f, y_leaf, slope)  # (1, N)

    x = 0.5 * (y_leaf + 1.0)
    r = r_from_x(x, rp)

    # Prefactor P(r) and h2
    r_flat = r.squeeze(0)
    a_val = a.unsqueeze(0)
    omega_val = omega.unsqueeze(0)

    rp_pref, rm_pref, _, _, _ = build_prefactor_primitives(r_flat, a_val, M=M, need_rs=False)
    P, P_r, P_rr = Leaver_prefactors(r_flat, a_val, omega_val, m, M, s,
                                      rp=rp_pref, rm=rm_pref)
    h2 = h_factor(a_val, omega_val, m, M=M, s=s)

    R = P * h2 * S.squeeze(0)  # (N,)

    # r-derivatives via autograd on y
    # R_r = R_y * dy/dr,  R_rr = R_yy*(dy/dr)^2 + R_y*d2y/dr2
    # dy/dr = dx/dr * dy/dx, and since x=(y+1)/2, dy/dx = 2, so dy/dr = 2 * dx/dr
    # But dx/dr depends on x. Let's use the chain rule properly.

    R_y_re = torch.autograd.grad(R.real.sum(), y_leaf, create_graph=True, retain_graph=True)[0]
    R_y_im = torch.autograd.grad(R.imag.sum(), y_leaf, create_graph=True, retain_graph=True)[0]
    R_y = torch.complex(R_y_re, R_y_im).squeeze(0)  # (N,)

    R_yy_re = torch.autograd.grad(R_y_re.sum(), y_leaf, create_graph=True, retain_graph=True)[0]
    R_yy_im = torch.autograd.grad(R_y_im.sum(), y_leaf, create_graph=True, retain_graph=True)[0]
    R_yy = torch.complex(R_yy_re, R_yy_im).squeeze(0)  # (N,)

    # Convert y-derivatives to r-derivatives
    # y = 2x - 1, x = r_+/r
    # dy/dr = dy/dx * dx/dr = 2 * (-r_+/r^2) = -2*r_+/r^2
    # d2y/dr2 = 4*r_+/r^3
    dy_dr = -2.0 * rp / (r_flat ** 2)
    d2y_dr2 = 4.0 * rp / (r_flat ** 3)

    R_r = R_y * dy_dr
    R_rr = R_yy * (dy_dr ** 2) + R_y * d2y_dr2

    return R.detach(), R_r.detach(), R_rr.detach()


def compute_normalized_residual(R, R_r, R_rr, r, a, omega, lambda_, M=1.0, s=-2, m=2):
    """
    Compute Teukolsky residual F and normalized error.

    F = Delta * R_rr  +  (s+1) * Delta_r * R_r  +  V * R

    Normalized = |F| / max(|Delta*R_rr|, |(s+1)*Delta_r*R_r|, |V*R|)
    """
    from physical_ansatz.teukolsky_coeffs import delta, delta_r, V_of_r

    Delta = delta(r, a, M)
    Delta_r = delta_r(r, M)
    V = V_of_r(r, a, omega, m, s, lambda_, M)

    term2 = Delta * R_rr
    term1 = (s + 1.0) * Delta_r * R_r
    term0 = V * R

    F = term2 + term1 + term0

    mag_F = torch.abs(F)
    mag_term2 = torch.abs(term2)
    mag_term1 = torch.abs(term1)
    mag_term0 = torch.abs(term0)

    denom = torch.max(torch.stack([mag_term2, mag_term1, mag_term0], dim=0), dim=0).values
    normalized = mag_F / (denom + 1e-16)

    return F.detach(), normalized.detach(), {
        "term2_abs": mag_term2.detach(),
        "term1_abs": mag_term1.detach(),
        "term0_abs": mag_term0.detach(),
        "denom": denom.detach(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--step", type=int, default=20000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-a", type=int, default=20, help="Number of a grid points")
    parser.add_argument("--n-omega", type=int, default=20, help="Number of omega grid points")
    parser.add_argument("--n-r", type=int, default=200, help="Number of r (y) points for residual eval")
    parser.add_argument("--output-subdir", default="residual_analysis",
                        help="Subdirectory under run_dir for output")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = torch.float64
    cdtype = torch.complex128

    run_dir = Path(args.run_dir)
    ckpt_path = run_dir / "checkpoints" / f"step_{args.step:06d}.pt"
    if not ckpt_path.exists():
        print(f"Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    print(f"Loading config from {run_dir / 'config_snapshot.yaml'}")
    config = load_config(run_dir)
    physics_cfg = config.get("physics", config)
    M = float(physics_cfg["problem"].get("M", 1.0))
    s = int(physics_cfg["problem"].get("s", -2))
    m = int(physics_cfg["problem"].get("m", 2))

    print(f"Loading model from {ckpt_path}")
    model = load_model(str(ckpt_path), config, device)
    model.eval()

    # Patch 0 parameter bounds
    a_min, a_max = 0.206, 0.794
    omega_min, omega_max = 0.008523, 0.0777

    # Build r-grid via Chebyshev on y for better resolution at boundaries.
    # Clamp slightly to avoid x=0 (r=inf, y=-1) and x=1 (r=r+, y=+1).
    n_r = args.n_r
    y_eps = 1e-4
    k = torch.arange(n_r, device=device, dtype=dtype)
    y_grid = -torch.cos(torch.pi * k / (n_r - 1))  # Chebyshev on [-1, 1]
    y_grid = torch.clamp(y_grid, -1.0 + y_eps, 1.0 - y_eps)

    # a and omega grids
    a_vals = torch.linspace(a_min, a_max, args.n_a, device=device, dtype=dtype)
    omega_vals = torch.linspace(omega_min, omega_max, args.n_omega, device=device, dtype=dtype)

    print(f"\nGrid: a({args.n_a}) x omega({args.n_omega}) x r({n_r})")
    print(f"a: [{a_min:.4f}, {a_max:.4f}]")
    print(f"omega: [{omega_min:.6f}, {omega_max:.6f}]")

    # Compute lambda for all (a, omega) pairs
    from physical_ansatz.residual import get_lambda_from_cfg, AuxCache
    cache = AuxCache()

    # Allocate arrays
    # normalized_residual: (n_a, n_omega, n_r)
    normalized = np.full((args.n_a, args.n_omega, n_r), np.nan)
    # Also store r grid (converted from y) for the first (a, omega) - varies slightly with a
    r_grids = np.full((args.n_a, n_r), np.nan)

    total = args.n_a * args.n_omega
    count = 0

    for i, a_val in enumerate(a_vals):
        # r-grid depends on a (through r_+)
        from physical_ansatz.mapping import r_plus
        rp_val = r_plus(a_val.unsqueeze(0), M)
        x = 0.5 * (y_grid + 1.0)
        r_grid = (rp_val / x).squeeze()
        r_grids[i, :] = r_grid.cpu().numpy()

        for j, omega_val in enumerate(omega_vals):
            lam = get_lambda_from_cfg(physics_cfg, cache,
                                      torch.tensor(float(a_val)),
                                      torch.tensor(float(omega_val)))
            lambda_val = torch.tensor(lam, device=device, dtype=cdtype)

            R, R_r, R_rr = compute_R_and_derivatives(
                model, a_val, omega_val, lambda_val, y_grid,
                M=M, s=s, m=m,
            )
            F, norm_res, details = compute_normalized_residual(
                R, R_r, R_rr, r_grid, a_val, omega_val, lambda_val,
                M=M, s=s, m=m,
            )

            with torch.no_grad():
                normalized[i, j, :] = norm_res.cpu().numpy()
            count += 1

            if count % 50 == 0 or count == total:
                max_err = np.nanmax(normalized[:i+1, :j+1, :])
                mean_err = np.nanmean(normalized[:i+1, :j+1, :])
                print(f"  [{count}/{total}] a={float(a_val):.4f}, omega={float(omega_val):.6f}, "
                      f"max_norm_res={float(norm_res.max()):.4e}, running_max={max_err:.4e}, running_mean={mean_err:.4e}")

    # ---- Save results ----
    out_dir = run_dir / args.output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving to {out_dir}")

    np.savez_compressed(
        out_dir / "residual_matrix.npz",
        normalized=normalized,
        a_vals=a_vals.cpu().numpy(),
        omega_vals=omega_vals.cpu().numpy(),
        r_grids=r_grids,
        y_grid=y_grid.cpu().numpy(),
    )

    # ---- Analysis ----
    print(f"\n{'='*60}")
    print("Residual Accuracy Analysis")
    print(f"{'='*60}")
    print(f"Overall max normalized error: {np.nanmax(normalized):.4e}")
    print(f"Overall mean normalized error: {np.nanmean(normalized):.4e}")
    print(f"Overall median normalized error: {np.nanmedian(normalized):.4e}")

    # Percentiles
    for pct in [50, 90, 95, 99]:
        val = np.nanpercentile(normalized, pct)
        print(f"  {pct}th percentile: {val:.4e}")

    # Analysis by parameter region
    # Average over r: (n_a, n_omega)
    mean_over_r = np.nanmean(normalized, axis=2)
    max_over_r = np.nanmax(normalized, axis=2)

    # Worst (a, omega) combinations
    worst_idx = np.unravel_index(np.nanargmax(max_over_r), max_over_r.shape)
    print(f"\nWorst (a, omega): a={a_vals[worst_idx[0]]:.4f}, "
          f"omega={omega_vals[worst_idx[1]]:.6f}, max_norm_err={max_over_r[worst_idx]:.4e}")

    # Save summary arrays
    np.savez_compressed(
        out_dir / "summary_2d.npz",
        mean_over_r=mean_over_r,
        max_over_r=max_over_r,
        a_vals=a_vals.cpu().numpy(),
        omega_vals=omega_vals.cpu().numpy(),
    )

    # ---- Generate summary text ----
    summary_path = out_dir / "analysis_summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"Residual Accuracy Analysis\n")
        f.write(f"{'='*60}\n")
        f.write(f"Run dir: {run_dir}\n")
        f.write(f"Checkpoint: step_{args.step:06d}.pt\n")
        f.write(f"Grid: a({args.n_a}) x omega({args.n_omega}) x r({n_r})\n\n")
        f.write(f"Normalized residual = |F| / max(|Delta*R_rr|, |(s+1)*Delta_r*R_r|, |V*R|)\n\n")
        f.write(f"Overall stats:\n")
        f.write(f"  max:  {np.nanmax(normalized):.4e}\n")
        f.write(f"  mean: {np.nanmean(normalized):.4e}\n")
        f.write(f"  median: {np.nanmedian(normalized):.4e}\n")
        for pct in [50, 90, 95, 99]:
            f.write(f"  {pct}th: {np.nanpercentile(normalized, pct):.4e}\n")
        f.write(f"\nWorst (a, omega): a={a_vals[worst_idx[0]]:.4f}, "
                f"omega={omega_vals[worst_idx[1]]:.6f}, max_err={max_over_r[worst_idx]:.4e}\n")

    print(f"\nSummary written to {summary_path}")

    # ---- Quick ASCII heatmap of mean_over_r ----
    print(f"\nlog10(mean normalized residual) over (a, omega):")
    log_mean = np.log10(np.clip(mean_over_r, 1e-16, None))
    header = "omega\\a  " + "".join(f"{float(a_vals[k]):.2f}  " for k in range(0, args.n_a, 4))
    print(header)
    for j in range(args.n_omega - 1, -1, -2):
        row = f"w={float(omega_vals[j]):.5f} " + "".join(
            f"{log_mean[k, j]:4.1f} " for k in range(0, args.n_a, 4)
        )
        print(row)

    print("\nDone.")


if __name__ == "__main__":
    main()
