#!/usr/bin/env python3
"""
Debug script: verify F_y = dF/dy residual coefficient derivation.

Checks:
  1. Algebraic consistency:  |F_y^{autograd} - F_y^{coeff}| < 1e-6
  2. Near-infinity coefficient behaviour: D2→0, but D2_y+D1 stays O(1)
  3. Shape, dtype, NaN/Inf checks

Usage:
  python scripts/debug_fy_residual_coeffs.py \
    --config config/autoencoder_stage1_retrain_fy.yaml \
    --checkpoint outputs/stage1_retrain/saved_best_v3.pt \
    --patch-id 0 \
    --device cuda
"""
import argparse
import sys
import yaml
from pathlib import Path

# Ensure project root is on path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import torch
import numpy as np

# ---------------------------------------------------------------------------
# Helpers: load model
# ---------------------------------------------------------------------------

def build_model(config, device):
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
    return model.to(device)


def load_model(checkpoint_path, config, device):
    model = build_model(config, device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" in ckpt:
        sd = ckpt["model_state_dict"]
    elif "state_dict" in ckpt:
        sd = ckpt["state_dict"]
    else:
        sd = ckpt
    model.load_state_dict(sd, strict=False)
    model.eval()
    return model


def load_config(config_path):
    """Load config matching load_pinn_full_config semantics."""
    p = Path(config_path).resolve()
    with open(p) as f:
        raw_cfg = yaml.safe_load(f)

    include_cfg = raw_cfg.get("include", {})
    physics_rel = include_cfg.get("physics", raw_cfg.get("physics_config"))
    if physics_rel is None:
        raise ValueError("Missing physics config: use include.physics or physics_config")

    physics_path = (p.parent.parent / physics_rel).resolve()
    with open(physics_path) as f:
        physics_cfg = yaml.safe_load(f)

    train_cfg = {
        k: v for k, v in raw_cfg.items()
        if k not in {"include", "physics_config"}
    }

    return {
        "physics": physics_cfg,
        **train_cfg,
    }


# ---------------------------------------------------------------------------
# Main debug
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Debug F_y residual coefficients")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--patch-id", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-fy", type=int, default=32,
                        help="Number of F_y collocation points")
    parser.add_argument("--n-test-params", type=int, default=4,
                        help="Number of random (a,omega) pairs to test")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = torch.float64
    cdtype = torch.complex128

    config = load_config(args.config)
    physics_cfg = config.get("physics", config)

    print(f"[debug] Loading model from {args.checkpoint}")
    model = load_model(args.checkpoint, config, device)

    M_phys = float(physics_cfg["problem"].get("M", 1.0))
    s_phys = int(physics_cfg["problem"].get("s", -2))
    m_phys = int(physics_cfg["problem"].get("m", 2))

    # Get patch bounds from config (patch 0)
    patch = config["atlas_training"].get("patch", None)
    if patch is None:
        patch_cfg = config.get("sampling", {}).get("parameter_batch", {})
        a_min, a_max = 0.2, 0.8
        omega_min, omega_max = 0.01, 0.08
    else:
        a_min, a_max = 0.2, 0.8
        omega_min, omega_max = 0.01, 0.08

    # Load domain info from atlas files if available
    domain_dir = Path("outputs/atlas_multipatch_phase4_sgdr/domains")
    if domain_dir.exists():
        domain_files = sorted(domain_dir.glob("patch_*.yaml"))
        if args.patch_id < len(domain_files):
            with open(domain_files[args.patch_id]) as f:
                domain_cfg = yaml.safe_load(f)
            a_min = domain_cfg.get("a_min", a_min)
            a_max = domain_cfg.get("a_max", a_max)
            omega_min = domain_cfg.get("omega_min", omega_min)
            omega_max = domain_cfg.get("omega_max", omega_max)
            print(f"[debug] Patch {args.patch_id}: a∈[{a_min},{a_max}], ω∈[{omega_min},{omega_max}]")

    # Generate F_y collocation points
    from physical_ansatz.residual_derivative_y import chebyshev_half_points
    y_fy = chebyshev_half_points(args.n_fy, y_min=-0.9999, y_max=0.9999,
                                  device=device, dtype=dtype)
    print(f"[debug] F_y collocation points: {args.n_fy}")
    print(f"  y range: [{y_fy[0].item():.6f}, {y_fy[-1].item():.6f}]")

    # Generate random test parameters
    rng = np.random.RandomState(42)
    a_vals = rng.uniform(a_min, a_max, args.n_test_params)
    omega_vals = rng.uniform(omega_min, omega_max, args.n_test_params)

    a_batch = torch.tensor(a_vals, device=device, dtype=dtype)
    omega_batch = torch.tensor(omega_vals, device=device, dtype=dtype)

    # Compute lambda for each (a, omega) pair
    from physical_ansatz.residual import get_lambda_from_cfg, AuxCache
    lambda_list = []
    cache = AuxCache()
    for i in range(args.n_test_params):
        lam = get_lambda_from_cfg(physics_cfg, cache, torch.tensor(float(a_vals[i])), torch.tensor(float(omega_vals[i])))
        lambda_list.append(lam)
    lambda_batch = torch.tensor(lambda_list, device=device, dtype=dtype)

    print(f"\n[debug] Test parameters ({args.n_test_params} pairs):")
    for i in range(args.n_test_params):
        print(f"  [{i}] a={a_vals[i]:.4f}, ω={omega_vals[i]:.6f}, λ={lambda_list[i]:.8f}")

    # =========================================================================
    # Check 1: compute dF/dy via autograd and compare with coefficient formula
    # =========================================================================
    print("\n" + "=" * 70)
    print("Check 1: Algebraic consistency |F_y^{autograd} - F_y^{coeff}|")
    print("=" * 70)

    from physical_ansatz.residual_derivative_y import (
        compute_S_derivatives_order3,
        compute_coeff_derivatives_y,
        compute_Fy_residual,
    )
    from physical_ansatz.transform_y import (
        horizon_regularity_slope,
        compose_reduced_shape_from_f,
    )
    from physical_ansatz.teukolsky_coeffs import coeffs_x

    max_abs_errors = []
    max_rel_errors = []

    y_fy_2d = y_fy.unsqueeze(0).expand(args.n_test_params, -1)  # (B, N)

    for i in range(args.n_test_params):
        a_i = a_batch[i:i+1]
        omega_i = omega_batch[i:i+1]
        lambda_i = lambda_batch[i:i+1]
        y_i = y_fy_2d[i:i+1].clone().detach().requires_grad_(True)

        # --- Compute F(y) via coefficient formula ---
        x_i = (y_i + 1.0) / 2.0

        A2_i, A1_i, A0_i = coeffs_x(
            x=x_i, a=a_i, omega=omega_i,
            m=m_phys, lambda_=lambda_i, s=s_phys, M=M_phys,
        )
        # coeffs_x already returns P-extracted S(x)-equation coefficients.
        # Pure x→y: S_x=2*S_y, S_xx=4*S_yy → D2=4*A2, D1=2*A1, D0=A0.
        D2_i = 4.0 * A2_i
        D1_i = 2.0 * A1_i
        D0_i = A0_i

        # S and derivatives (only need up to S_yy for F, then autograd for F_y)
        # Use pre-computed S, S_y, S_yy, S_yyy from the module
        S_i, Sy_i, Syy_i, Syyy_i = compute_S_derivatives_order3(
            model, a_i, omega_i, y_i.squeeze(0),
            lambda_batch=lambda_i, M=M_phys, s=s_phys, m=m_phys,
        )

        # F(y) = D2*S_yy + D1*S_y + D0*S
        F_i = D2_i * Syy_i + D1_i * Sy_i + D0_i * S_i

        # dF/dy via autograd
        # Need to re-compute F with y as leaf for autograd
        y_i_grad = y_fy.clone().detach().requires_grad_(True)
        y_i_grad_2d = y_i_grad.unsqueeze(0)

        x_g = (y_i_grad_2d + 1.0) / 2.0

        A2_g, A1_g, A0_g = coeffs_x(
            x=x_g, a=a_i, omega=omega_i,
            m=m_phys, lambda_=lambda_i, s=s_phys, M=M_phys,
        )
        D2_g = 4.0 * A2_g
        D1_g = 2.0 * A1_g
        D0_g = A0_g

        f_g = model(a_i, omega_i, y_i_grad_2d)
        slope_g = horizon_regularity_slope(
            a=a_i, omega=omega_i, lambda_=lambda_i, m=m_phys, M=M_phys, s=s_phys,
        )
        S_g = compose_reduced_shape_from_f(f_g, y_i_grad_2d, slope_g)

        Sy_re = torch.autograd.grad(S_g.real.sum(), y_i_grad, create_graph=True, retain_graph=True)[0]
        Sy_im = torch.autograd.grad(S_g.imag.sum(), y_i_grad, create_graph=True, retain_graph=True)[0]
        Sy_g = torch.complex(Sy_re, Sy_im).unsqueeze(0)

        Syy_re = torch.autograd.grad(Sy_re.sum(), y_i_grad, create_graph=True, retain_graph=True)[0]
        Syy_im = torch.autograd.grad(Sy_im.sum(), y_i_grad, create_graph=True, retain_graph=True)[0]
        Syy_g = torch.complex(Syy_re, Syy_im).unsqueeze(0)

        F_g = D2_g * Syy_g + D1_g * Sy_g + D0_g * S_g

        Fy_autograd_re = torch.autograd.grad(F_g.real.sum(), y_i_grad, create_graph=False, retain_graph=True)[0]
        Fy_autograd_im = torch.autograd.grad(F_g.imag.sum(), y_i_grad, create_graph=False, retain_graph=True)[0]
        Fy_autograd = torch.complex(Fy_autograd_re, Fy_autograd_im).unsqueeze(0)

        # --- Compute F_y via coefficient formula ---
        _, _, _, D2_y, D1_y, D0_y = compute_coeff_derivatives_y(
            a_i, omega_i, lambda_i, y_fy,
            M=M_phys, s=s_phys, m=m_phys,
        )
        Fy_coeff = compute_Fy_residual(S_i, Sy_i, Syy_i, Syyy_i, D2_i, D1_i, D0_i, D2_y, D1_y, D0_y)

        abs_err = (torch.abs(Fy_autograd - Fy_coeff)).max().item()
        rel_err = (torch.abs(Fy_autograd - Fy_coeff) / (torch.abs(Fy_autograd) + 1e-30)).max().item()

        max_abs_errors.append(abs_err)
        max_rel_errors.append(rel_err)

        status = "PASS" if abs_err < 1e-6 and rel_err < 1e-5 else "FAIL"
        print(f"  [{i}] max_abs_err={abs_err:.2e}  max_rel_err={rel_err:.2e}  [{status}]")

    print(f"\n  Summary: max_abs_error = {max(max_abs_errors):.2e}, max_rel_error = {max(max_rel_errors):.2e}")

    if max(max_abs_errors) > 1e-6 or max(max_rel_errors) > 1e-5:
        print("  *** CHECK 1 FAILED: coefficient derivation may be wrong ***")
        sys.exit(1)
    else:
        print("  Check 1 PASSED")

    # =========================================================================
    # Check 2: Near-infinity coefficient behaviour
    # =========================================================================
    print("\n" + "=" * 70)
    print("Check 2: Near-infinity coefficient behaviour (D2→0, D2_y+D1 stays O(1))")
    print("=" * 70)

    # Use a single representative (a, omega) pair
    a_test = torch.tensor([float(a_vals[0])], device=device, dtype=dtype)
    omega_test = torch.tensor([float(omega_vals[0])], device=device, dtype=dtype)
    lambda_test = torch.tensor([lambda_list[0]], device=device, dtype=dtype)

    # Dense y grid approaching infinity
    y_near_inf = torch.linspace(-0.9999, -0.9, 20, device=device, dtype=dtype)
    D2_inf, D1_inf, D0_inf, D2_y_inf, D1_y_inf, D0_y_inf = compute_coeff_derivatives_y(
        a_test, omega_test, lambda_test, y_near_inf,
        M=M_phys, s=s_phys, m=m_phys,
    )

    print(f"\n  a={float(a_test):.4f}, ω={float(omega_test):.6f}")
    print(f"  {'y':>10s}  {'|D2|':>12s}  {'|D1|':>12s}  {'|D2_y+D1|':>12s}  {'|D2|/|D2_y+D1|':>16s}")
    print(f"  {'-'*10}  {'-'*12}  {'-'*12}  {'-'*12}  {'-'*16}")

    for j in range(len(y_near_inf)):
        yv = float(y_near_inf[j])
        d2v = float(torch.abs(D2_inf[0, j]))
        d1v = float(torch.abs(D1_inf[0, j]))
        d2yd1v = float(torch.abs(D2_y_inf[0, j] + D1_inf[0, j]))
        ratio = d2v / (d2yd1v + 1e-30)
        print(f"  {yv:10.6f}  {d2v:12.4e}  {d1v:12.4e}  {d2yd1v:12.4e}  {ratio:16.4e}")

    # Verify D2→0 near infinity
    d2_near_inf = float(torch.abs(D2_inf[0, 0]))
    d2_mid = float(torch.abs(D2_inf[0, -1]))
    d2yd1_near_inf = float(torch.abs(D2_y_inf[0, 0] + D1_inf[0, 0]))

    print(f"\n  At y→-1: |D2|={d2_near_inf:.4e}, |D2_y+D1|={d2yd1_near_inf:.4e}")
    print(f"  D2 decreases toward infinity: {d2_near_inf < d2_mid} (|D2| near inf < |D2| at y=-0.9)")
    print(f"  D2_y+D1 stays O(1): {d2yd1_near_inf > 0.01}")

    if d2yd1_near_inf < 1e-4:
        print("  *** WARNING: D2_y+D1 is very small near infinity, F_y may not help ***")
    else:
        print("  Check 2 PASSED: F_y provides non-vanishing S_yy constraint near infinity")

    # =========================================================================
    # Check 3: Shape, dtype, NaN/Inf
    # =========================================================================
    print("\n" + "=" * 70)
    print("Check 3: Shape, dtype, NaN/Inf sanity")
    print("=" * 70)

    S, Sy, Syy, Syyy = compute_S_derivatives_order3(
        model, a_batch, omega_batch, y_fy,
        lambda_batch=lambda_batch, M=M_phys, s=s_phys, m=m_phys,
    )

    D2, D1, D0, D2_y, D1_y, D0_y = compute_coeff_derivatives_y(
        a_batch, omega_batch, lambda_batch, y_fy,
        M=M_phys, s=s_phys, m=m_phys,
    )

    Fy = compute_Fy_residual(S, Sy, Syy, Syyy, D2, D1, D0, D2_y, D1_y, D0_y)

    B, N = args.n_test_params, args.n_fy
    checks = [
        ("S", S, (B, N)),
        ("S_y", Sy, (B, N)),
        ("S_yy", Syy, (B, N)),
        ("S_yyy", Syyy, (B, N)),
        ("D2", D2, (B, N)),
        ("D1", D1, (B, N)),
        ("D0", D0, (B, N)),
        ("D2_y", D2_y, (B, N)),
        ("D1_y", D1_y, (B, N)),
        ("D0_y", D0_y, (B, N)),
        ("Fy", Fy, (B, N)),
    ]

    all_ok = True
    for name, tensor, expected_shape in checks:
        shape_ok = tensor.shape == expected_shape
        complex_ok = tensor.is_complex()
        finite_ok = torch.all(torch.isfinite(tensor))
        nan_free = not torch.any(torch.isnan(tensor))
        inf_free = not torch.any(torch.isinf(tensor))

        issues = []
        if not shape_ok:
            issues.append(f"shape {tensor.shape} != {expected_shape}")
        if not complex_ok:
            issues.append("not complex")
        if not finite_ok:
            issues.append("has non-finite values")
        if not nan_free:
            issues.append("has NaN")
        if not inf_free:
            issues.append("has Inf")

        status = "OK" if not issues else "FAIL: " + ", ".join(issues)
        if issues:
            all_ok = False
        print(f"  {name:8s}  shape={str(tensor.shape):12s}  {status}")

    if not all_ok:
        print("\n  *** CHECK 3 FAILED ***")
        sys.exit(1)
    else:
        print("  Check 3 PASSED")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("ALL CHECKS PASSED")
    print("=" * 70)
    print(f"  Model checkpoint: {args.checkpoint}")
    print(f"  F_y points: {args.n_fy}")
    print(f"  Test (a,omega) pairs: {args.n_test_params}")
    print(f"  max |F_y^{autograd} - F_y^{coeff}|: {max(max_abs_errors):.2e}")
    print(f"  Near-infinity |D2_y+D1|: {d2yd1_near_inf:.4e}")


if __name__ == "__main__":
    main()
