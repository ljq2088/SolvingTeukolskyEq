#!/usr/bin/env python3
"""Diagnose per-loss gradient contributions.

Computes each loss term separately and measures its gradient norm
w.r.t. trainable model parameters. Helps identify which loss terms
are actually driving learning vs. just contributing to total loss value.

Usage:
  python scripts/diagnose_loss_gradients.py \\
    --config config/autoencoder_stage1_fy_normalized.yaml \\
    --checkpoint outputs/stage1_fy_normalized/.../checkpoints/latest_model.pt \\
    --patch-id 0 --device cuda
"""
import argparse, json, sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import torch
import numpy as np


def load_config(config_path):
    import yaml
    p = Path(config_path).resolve()
    with open(p) as f:
        raw = yaml.safe_load(f)
    include = raw.get("include", {})
    physics_rel = include.get("physics")
    if physics_rel:
        physics_path = (p.parent.parent / physics_rel).resolve()
        with open(physics_path) as f:
            physics_cfg = yaml.safe_load(f)
    else:
        physics_cfg = {}
    train_cfg = {k: v for k, v in raw.items() if k not in {"include", "physics_config"}}
    return {"physics": physics_cfg, **train_cfg}


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


def param_grad_norm(model):
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.data.norm(2).item() ** 2
    return total ** 0.5


def param_grad_norms_per_module(model):
    norms = {}
    for name, p in model.named_parameters():
        if p.grad is not None and p.requires_grad:
            # Group by top-level module
            module = name.split(".")[0]
            if module not in norms:
                norms[module] = 0.0
            norms[module] += p.grad.data.norm(2).item() ** 2
    return {k: v ** 0.5 for k, v in norms.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--patch-id", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-a-omega", type=int, default=4)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = torch.float64

    config = load_config(args.config)
    physics_cfg = config.get("physics", config)
    atlas_cfg = config.get("atlas_training", {})

    M = float(physics_cfg["problem"].get("M", 1.0))
    s = int(physics_cfg["problem"].get("s", -2))
    m = int(physics_cfg["problem"].get("m", 2))

    print(f"Loading model from {args.checkpoint}")
    model = load_model(args.checkpoint, config, device)
    model.train()  # enable grad computation

    # Get patch bounds
    domain_dir = Path("outputs/atlas_multipatch_phase4_sgdr/domains")
    a_min, a_max = 0.2, 0.8
    omega_min, omega_max = 0.01, 0.08
    if domain_dir.exists():
        domain_files = sorted(domain_dir.glob("patch_*.yaml"))
        if args.patch_id < len(domain_files):
            import yaml
            with open(domain_files[args.patch_id]) as f:
                dc = yaml.safe_load(f)
            a_min, a_max = dc.get("a_min", a_min), dc.get("a_max", a_max)
            omega_min, omega_max = dc.get("omega_min", omega_min), dc.get("omega_max", omega_max)

    # Sample (a, omega)
    rng = np.random.RandomState(1234)
    a_vals = rng.uniform(a_min, a_max, args.n_a_omega)
    omega_vals = rng.uniform(omega_min, omega_max, args.n_a_omega)

    a_batch = torch.tensor(a_vals, device=device, dtype=dtype)
    omega_batch = torch.tensor(omega_vals, device=device, dtype=dtype)

    from physical_ansatz.residual import get_lambda_from_cfg, AuxCache
    cache = AuxCache()
    lambda_list = []
    for i in range(args.n_a_omega):
        lam = get_lambda_from_cfg(physics_cfg, cache,
                                   torch.tensor(float(a_vals[i])),
                                   torch.tensor(float(omega_vals[i])))
        lambda_list.append(lam)
    lambda_batch = torch.tensor(lambda_list, device=device, dtype=dtype)

    # Collocation points
    from physical_ansatz.residual_derivative_y import chebyshev_half_points
    from physical_ansatz.transform_y import horizon_regularity_slope

    n_int = atlas_cfg.get("n_interior", 256)
    y_int = chebyshev_half_points(n_int, y_min=-0.9999, y_max=0.9999, device=device, dtype=dtype)

    fy_cfg = atlas_cfg.get("fy_residual", {})
    fy_enabled = fy_cfg.get("enabled", False)
    n_fy = int(fy_cfg.get("n_points", 32))
    fy_weight = float(fy_cfg.get("weight", 0.01))
    y_fy = chebyshev_half_points(n_fy, y_min=-0.9999, y_max=0.9999, device=device, dtype=dtype)

    inf_robin_weight = float(atlas_cfg.get("infinity_robin_weight", 0.0))

    # =========================================================================
    # Compute each loss term and its gradient norm
    # =========================================================================
    print(f"\n{'='*70}")
    print(f"Per-loss gradient analysis (B={args.n_a_omega}, n_int={n_int}, n_fy={n_fy})")
    print(f"a range: [{a_min:.4f}, {a_max:.4f}], omega range: [{omega_min:.6f}, {omega_max:.6f}]")
    print(f"{'='*70}")

    results = {}

    for i in range(args.n_a_omega):
        print(f"\n  [{i}] a={a_vals[i]:.4f}, ω={omega_vals[i]:.6f}")
        ai = a_batch[i:i+1]
        oi = omega_batch[i:i+1]
        li = lambda_batch[i:i+1]

        # --- PDE loss (interior) ---
        model.zero_grad()
        y_leaf = y_int.unsqueeze(0).detach().clone().requires_grad_(True)
        f = model(ai, oi, y_leaf)
        slope = horizon_regularity_slope(a=ai, omega=oi, lambda_=li, m=m, M=M, s=s)

        from physical_ansatz.transform_y import compose_reduced_shape_from_f
        S = compose_reduced_shape_from_f(f, y_leaf, slope)

        from physical_ansatz.residual import compute_f_residual_autograd
        pointwise, _, _ = compute_f_residual_autograd(
            model, ai, oi, y_int, u_batch=None, v_batch=None,
            lambda_batch=li, M=M, s=s, m=m,
            slope=slope, S_val=S, y_leaf=y_leaf,
        )
        loss_pde = pointwise.mean()
        loss_pde.backward(retain_graph=True)
        grad_pde = param_grad_norm(model)
        grad_pde_by_mod = param_grad_norms_per_module(model)
        model.zero_grad()

        # --- Infinity Robin loss ---
        grad_inf = 0.0
        grad_inf_by_mod = {}
        if inf_robin_weight > 0:
            from physical_ansatz.infinity_robin import (
                analytic_c_inf, infinity_robin_loss, compute_S_and_Sy_at_infinity,
            )
            y_inf = torch.full((1, 1), -1.0, device=device, dtype=dtype)
            y_inf_leaf = y_inf.detach().clone().requires_grad_(True)
            f_inf = model(ai, oi, y_inf_leaf)
            fy_re = torch.autograd.grad(f_inf.real.sum(), y_inf_leaf, create_graph=True, retain_graph=True)[0]
            fy_im = torch.autograd.grad(f_inf.imag.sum(), y_inf_leaf, create_graph=True, retain_graph=True)[0]
            fy_inf = torch.complex(fy_re, fy_im)
            S_inf, Sy_inf = compute_S_and_Sy_at_infinity(f_inf.squeeze(-1), fy_inf.squeeze(-1), slope)
            cdtype = torch.complex128
            c_inf = analytic_c_inf(ai, oi, li.to(dtype=cdtype), m=m, M=M, s=s)
            loss_inf = infinity_robin_loss(S_inf, Sy_inf, c_inf).mean()

            loss_inf.backward(retain_graph=True)
            grad_inf = param_grad_norm(model)
            grad_inf_by_mod = param_grad_norms_per_module(model)
            model.zero_grad()

        # --- F_y residual loss ---
        grad_fy = 0.0
        grad_fy_by_mod = {}
        if fy_enabled:
            from physical_ansatz.residual_derivative_y import (
                compute_S_derivatives_order3, compute_coeff_derivatives_y,
                compute_Fy_residual,
            )
            y_fy_2d = y_fy.unsqueeze(0).expand(1, -1).detach().clone().requires_grad_(True)
            f_fy = model(ai, oi, y_fy_2d)
            S_fy = compose_reduced_shape_from_f(f_fy, y_fy_2d, slope)
            Sy_re = torch.autograd.grad(S_fy.real.sum(), y_fy_2d, create_graph=True, retain_graph=True)[0]
            Sy_im = torch.autograd.grad(S_fy.imag.sum(), y_fy_2d, create_graph=True, retain_graph=True)[0]
            Sy_fy = torch.complex(Sy_re, Sy_im)
            Syy_re = torch.autograd.grad(Sy_re.sum(), y_fy_2d, create_graph=True, retain_graph=True)[0]
            Syy_im = torch.autograd.grad(Sy_im.sum(), y_fy_2d, create_graph=True, retain_graph=True)[0]
            Syy_fy = torch.complex(Syy_re, Syy_im)
            Syyy_re = torch.autograd.grad(Syy_re.sum(), y_fy_2d, create_graph=True, retain_graph=True)[0]
            Syyy_im = torch.autograd.grad(Syy_im.sum(), y_fy_2d, create_graph=True, retain_graph=True)[0]
            Syyy_fy = torch.complex(Syyy_re, Syyy_im)

            D2, D1, D0, D2_y, D1_y, D0_y = compute_coeff_derivatives_y(
                ai, oi, li, y_fy, M=M, s=s, m=m,
            )
            Fy = compute_Fy_residual(S_fy, Sy_fy, Syy_fy, Syyy_fy, D2, D1, D0, D2_y, D1_y, D0_y)
            loss_fy = torch.mean(torch.abs(Fy) ** 2)
            weighted_fy = fy_weight * loss_fy
            weighted_fy.backward(retain_graph=True)
            grad_fy_weighted = param_grad_norm(model)
            grad_fy_by_mod_weighted = param_grad_norms_per_module(model)
            model.zero_grad()

            # Also compute unweighted gradient
            loss_fy.backward(retain_graph=True)
            grad_fy = param_grad_norm(model)
            grad_fy_by_mod = param_grad_norms_per_module(model)
            model.zero_grad()

            fy_raw = float(loss_fy.detach().cpu().item())
            fy_mag = float(torch.abs(Fy).mean().detach().cpu().item())
            print(f"    F_y raw loss: {fy_raw:.2f}, |F_y| mean: {fy_mag:.2f}")
            print(f"    F_y weighted grad: {grad_fy_weighted:.2e}")
            print(f"    F_y unweighted grad: {grad_fy:.2e}")
            print(f"    F_y weighted grad/module: {json.dumps({k: f'{v:.2e}' for k,v in grad_fy_by_mod_weighted.items()})}")
        else:
            fy_raw = 0.0
            grad_fy_weighted = 0.0

        # Total combined gradient
        model.zero_grad()
        y_all = y_int.unsqueeze(0).detach().clone().requires_grad_(True)
        f_all = model(ai, oi, y_all)
        S_all = compose_reduced_shape_from_f(f_all, y_all, slope)
        pointwise_all, _, _ = compute_f_residual_autograd(
            model, ai, oi, y_int, None, None, li, M=M, s=s, m=m,
            slope=slope, S_val=S_all, y_leaf=y_all,
        )
        total = pointwise_all.mean()
        if fy_enabled:
            total = total + fy_weight * loss_fy
        if inf_robin_weight > 0:
            total = total + inf_robin_weight * loss_inf
        total.backward()
        grad_total = param_grad_norm(model)
        grad_total_by_mod = param_grad_norms_per_module(model)
        model.zero_grad()

        print(f"    PDE loss: {float(loss_pde.detach().cpu().item()):.4f}, grad: {grad_pde:.2e}")
        print(f"    Robin loss grad: {grad_inf:.2e}")
        print(f"    Total grad: {grad_total:.2e}")
        print(f"    Total grad/module: {json.dumps({k: f'{v:.2e}' for k,v in grad_total_by_mod.items()})}")
        print(f"    F_y_contrib: {grad_fy_weighted/grad_total*100:.1f}% of total grad")

        # Store results
        results[i] = {
            "a": float(a_vals[i]), "omega": float(omega_vals[i]),
            "grad_pde": grad_pde, "grad_total": grad_total,
            "grad_fy_weighted": grad_fy_weighted, "grad_fy": grad_fy,
            "grad_inf": grad_inf,
            "pde_loss": float(loss_pde), "fy_raw": fy_raw,
        }

    # Summary
    print(f"\n{'='*70}")
    print("Summary (mean over samples)")
    print(f"{'='*70}")
    for key in ["grad_pde", "grad_total", "grad_fy_weighted", "grad_fy", "grad_inf"]:
        vals = [r[key] for r in results.values()]
        if all(v == 0 for v in vals):
            continue
        print(f"  {key:20s}: {np.mean(vals):.2e}")
    fy_contribs = [r["grad_fy_weighted"]/max(r["grad_total"],1e-30)*100 for r in results.values()]
    if fy_enabled:
        print(f"  F_y grad contribution: {np.mean(fy_contribs):.1f}%")


if __name__ == "__main__":
    main()
