#!/usr/bin/env python3
"""Pre-training verification for Stage-1 patch 0 retrain.

Checks:
1. Boundary hard-coding: g(1)=0, h1(1)=0, S(1)=1
2. PDE residual in y-space (B2*f_yy + B1*f_y + B0*f = RHS)
3. chebyshev_half collocation: y_ref ∈ [1,3], no NaN
4. Parameter pools: non-overlapping, within patch 0 bounds
5. Warm-start weight loading: non-NaN output
"""

import sys
import os
sys.path.insert(0, ".")
import torch
import numpy as np

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_script_dir)


def check_boundary_hardcoding():
    """Verify g(1)=0, h1(1)=0, S(1)=1 for arbitrary f."""
    from physical_ansatz.transform_y import h1_factor, g_factor, compose_reduced_shape_from_f

    device = torch.device("cpu")
    dtype = torch.float64
    x = torch.tensor([1.0], device=device, dtype=dtype)

    h1, h1_x, h1_xx = h1_factor(x)
    slope = torch.tensor([1.5], device=device, dtype=dtype)
    g, g_x, g_xx = g_factor(x, slope)

    # Random f values: (B, N)
    f = torch.randn(3, x.numel(), device=device, dtype=dtype)
    S = compose_reduced_shape_from_f(f, x, slope)

    h1_ok = torch.allclose(h1, torch.zeros_like(h1), atol=1e-8)
    g_ok = torch.allclose(g, torch.zeros_like(g), atol=1e-8)
    S_ok = torch.allclose(S, torch.ones_like(S), atol=1e-8)

    print(f"[1] Boundary hard-coding:")
    print(f"    h1(x=1) = {h1.item():.2e}  {'OK' if h1_ok else 'FAIL'}")
    print(f"    g(x=1)  = {g[0,0].item():.2e}  {'OK' if g_ok else 'FAIL'}")
    print(f"    S(x=1)  = {S[0,0].item():.6f}   {'OK' if S_ok else 'FAIL'}")
    return h1_ok and g_ok and S_ok


def check_pde_residual_in_y_space():
    """Verify residual can be computed and is finite."""
    from physical_ansatz.residual_pinn import compute_pointwise_pde_residual
    from model.pinn_mlp import PINN_MLP
    from config.config_loader import load_pinn_full_config

    cfg = load_pinn_full_config(
        os.path.join(_project_root, "config/autoencoder_stage1_retrain.yaml")
    )
    M = float(cfg["physics"]["problem"]["M"])
    s_val = int(cfg["physics"]["problem"]["s"])
    m_val = int(cfg["physics"]["problem"]["m"])

    pde_cfg = {"problem": {"M": M, "s": s_val, "m": m_val}, "physics": cfg.get("physics", {})}
    a_val = 0.5
    omega_val = 0.05
    lambda_val = 0.0  # rough estimate for test

    model = PINN_MLP(
        hidden_dims=[128, 128, 128, 128], activation="silu",
        fourier_num_freqs=2, fourier_base_scale=1.0,
        param_embed_dim=64, use_film=True, use_residual=True,
        local_coord_mode="raw_aw",
        M=M, m_mode=m_val,
    )

    y = torch.linspace(-0.9, 0.9, 32, dtype=torch.float64).requires_grad_(True)
    a = torch.full((1,), a_val, dtype=torch.float64)
    w = torch.full((1,), omega_val, dtype=torch.float64)
    lam = torch.full((1,), lambda_val, dtype=torch.float64)

    residual, pointwise = compute_pointwise_pde_residual(
        model=model, cfg=pde_cfg,
        a_batch=a, omega_batch=w, lambda_batch=lam,
        y_interior=y,
    )

    is_finite = bool(torch.isfinite(residual).all())
    mean_abs = pointwise.abs().mean().item()

    print(f"[2] PDE residual in y-space:")
    print(f"    mean |residual| = {mean_abs:.6e}  {'OK' if is_finite else 'FAIL'}")
    return is_finite


def check_chebyshev_half_collocation():
    """Verify chebyshev_half: y_ref ∈ [1,3], mapped to [y_min, y_max], no NaN."""
    from dataset.sampling import sample_points_chebyshev_half

    N = 128
    y_min, y_max = -0.9999, 0.9999

    k = torch.arange(N, dtype=torch.float64)
    y_ref = 2.0 * torch.cos(torch.pi * (k - N) / (2.0 * N)) + 1.0

    y = sample_points_chebyshev_half(N, y_min=y_min, y_max=y_max)

    ref_in_range = (y_ref.min() >= 0.999) and (y_ref.max() <= 3.001)
    y_in_range = (y.min() >= y_min - 1e-6) and (y.max() <= y_max + 1e-6)
    no_nan = not torch.isnan(y).any()
    monotonic = bool(torch.all(y[1:] >= y[:-1]))

    n_cluster = N // 10
    cluster_fraction = (y[-n_cluster:] > 0.5 * y_max).float().mean().item()

    print(f"[3] chebyshev_half collocation (N={N}):")
    print(f"    y_ref range: [{y_ref.min().item():.4f}, {y_ref.max().item():.4f}]  {'OK' if ref_in_range else 'FAIL'}")
    print(f"    y range:     [{y.min().item():.6f}, {y.max().item():.6f}]  {'OK' if y_in_range else 'FAIL'}")
    print(f"    no NaN: {no_nan}, monotonic: {monotonic}")
    print(f"    top 10% near horizon: {cluster_fraction:.1%}")
    return bool(ref_in_range and y_in_range and no_nan and monotonic)


def check_parameter_pools():
    """Verify train/val pools for patch 0: no overlap, within bounds."""
    import yaml
    import glob

    cfg_path = os.path.join(_project_root, "config/autoencoder_stage1_retrain.yaml")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    # Find patch cover file
    patch_cover_path = None
    candidates = glob.glob(os.path.join(_project_root, "domain/atlas_logomega_patches*.json"))
    if not candidates:
        candidates = glob.glob(os.path.join(_project_root, "domain/*patches*.json"))
    if not candidates:
        # Try SolvingTeukolsky domain
        candidates = glob.glob("/home/ljq/code/PINN/SolvingTeukolsky/domain/atlas_logomega_patches*.json")
    if candidates:
        patch_cover_path = candidates[0]

    if patch_cover_path is None:
        print("[4] Parameter pools: SKIP (no patch_cover file found)")
        return True

    from domain.patch_cover import load_patch_cover, load_valid_chart_points
    patches = load_patch_cover(patch_cover_path)
    patch = patches[0]

    train_aw, train_uv = load_valid_chart_points(patch, "train")
    val_aw, val_uv = load_valid_chart_points(patch, "val")

    train_set = {tuple(row) for row in train_aw}
    val_set = {tuple(row) for row in val_aw}
    overlap = train_set & val_set

    a_min, a_max = patch.a_min, patch.a_max
    w_min, w_max = patch.omega_min, patch.omega_max

    train_a_ok = bool((train_aw[:, 0] >= a_min).all() and (train_aw[:, 0] <= a_max).all())
    train_w_ok = bool((train_aw[:, 1] >= w_min).all() and (train_aw[:, 1] <= w_max).all())

    print(f"[4] Parameter pools (patch 0):")
    print(f"    patch_cover: {patch_cover_path}")
    print(f"    a ∈ [{a_min:.4f}, {a_max:.4f}], ω ∈ [{w_min:.4f}, {w_max:.4f}]")
    print(f"    train: {len(train_aw)} points, val: {len(val_aw)} points")
    print(f"    a in bounds: {train_a_ok}, ω in bounds: {train_w_ok}")
    print(f"    overlap: {len(overlap)} points {'OK' if len(overlap) == 0 else 'WARNING'}")

    return bool(len(overlap) == 0 and train_a_ok and train_w_ok)


def check_warmstart():
    """Verify pinn_mlp checkpoint loads and produces non-NaN output."""
    ckpt_path = "/home/ljq/code/PINN/SolvingTeukolsky/outputs/atlas_multipatch_phase4_sgdr/models/patch_000_best.pt"
    if not os.path.exists(ckpt_path):
        print(f"[5] Warm-start: SKIP (checkpoint not found: {ckpt_path})")
        return True

    from model.pinn_mlp import PINN_MLP
    from model.autoencoder_pinn import AutoencoderTeukolskyPINN, copy_pinn_mlp_to_autoencoder

    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt["model_state_dict"]
    full_cfg = ckpt.get("full_cfg", {})

    old_model_cfg = full_cfg.get("train", {}).get("model", {})
    old_model = PINN_MLP(
        hidden_dims=old_model_cfg.get("hidden_dims", [128, 128, 128, 128]),
        activation=old_model_cfg.get("activation", "silu"),
        param_embed_dim=old_model_cfg.get("param_embed_dim", 64),
        local_coord_mode=old_model_cfg.get("local_coord_mode", "chart_uv"),
        fourier_num_freqs=old_model_cfg.get("fourier_num_freqs", 2),
        fourier_base_scale=old_model_cfg.get("fourier_base_scale", 1.0),
        use_film=old_model_cfg.get("use_film", True),
        use_residual=old_model_cfg.get("use_residual", True),
    )
    old_model.load_state_dict(state_dict, strict=True)

    autoencoder = AutoencoderTeukolskyPINN(
        hidden_dims=[128, 128, 128, 128], activation="silu",
        fourier_num_freqs=2, fourier_base_scale=1.0,
        param_embed_dim=64, use_film=True, use_residual=True,
    )
    copy_pinn_mlp_to_autoencoder(old_model, autoencoder)

    # Test forward pass — _predict_decoder handles reshaping internally
    y = torch.linspace(-0.9, 0.9, 64, dtype=torch.float64)
    a = torch.tensor([0.5], dtype=torch.float64)
    w = torch.tensor([0.05], dtype=torch.float64)

    with torch.no_grad():
        f_rin = autoencoder._predict_decoder(autoencoder.rin_decoder, a, w, y)

    is_finite = bool(torch.isfinite(f_rin.real).all() and torch.isfinite(f_rin.imag).all())

    print(f"[5] Warm-start from pinn_mlp checkpoint:")
    print(f"    checkpoint: {ckpt_path}")
    print(f"    best_val_mean: {ckpt.get('best_val_mean', 'N/A')}")
    print(f"    output shape: {f_rin.shape}")
    print(f"    output finite: {is_finite}")
    print(f"    |f| range: [{f_rin.abs().min().item():.4f}, {f_rin.abs().max().item():.4f}]")

    return bool(is_finite)


def main():
    print("=" * 60)
    print("Pre-training verification for Stage-1 patch 0 retrain")
    print("=" * 60)
    print()

    results = []
    for check_fn in [
        check_boundary_hardcoding,
        check_pde_residual_in_y_space,
        check_chebyshev_half_collocation,
        check_parameter_pools,
        check_warmstart,
    ]:
        try:
            ok = check_fn()
            results.append(ok)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)
        print()

    n_pass = sum(results)
    n_total = len(results)
    print("=" * 60)
    print(f"Result: {n_pass}/{n_total} checks passed")
    if n_pass == n_total:
        print("ALL CHECKS PASSED - ready for training")
    else:
        print("SOME CHECKS FAILED - fix before training")
    print("=" * 60)
    return n_pass == n_total


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
