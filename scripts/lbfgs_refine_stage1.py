#!/usr/bin/env python3
"""L-BFGS refinement from a Stage-1 checkpoint with fixed random collocation.

Usage:
  python scripts/lbfgs_refine_stage1.py \
    --checkpoint outputs/stage1_pinn_random_v3/.../checkpoints/step_025600.pt \
    --config config/autoencoder_stage1_pinn_random_v3.yaml \
    --patch-id 0 --device cuda --steps 500
"""
import argparse, sys, os, math, json
from pathlib import Path
from datetime import datetime

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import torch
import numpy as np
import yaml
from tqdm import trange

from model.autoencoder_pinn import AutoencoderTeukolskyPINN
from physical_ansatz.residual_pinn import pinn_residual_loss
from physical_ansatz.transform_y import horizon_regularity_slope, compose_reduced_shape_from_f
from config.config_loader import load_pinn_full_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--patch-id", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--n-interior", type=int, default=512, help="Number of Chebyshev collocation points")
    parser.add_argument("--n-param-batches", type=int, default=1, help="Number of fixed parameter batches (1 = no cycling)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--f-reg-weight", type=float, default=0.01, help="L2 regularization on |f(y)|^2 to prevent trivial solution")
    parser.add_argument("--normalize-residual", action="store_true", help="Use normalized residual loss")
    parser.add_argument("--normalize-mode", default="term", choices=["term", "coeff", "max_term"], help="Residual normalization mode")
    parser.add_argument("--s-step-weight", type=float, default=0.0, help="Relative trust-region weight against previous-step S")
    parser.add_argument("--s-step-eps", type=float, default=1.0e-12, help="Epsilon for relative previous-step S loss")
    parser.add_argument("--output-root", default="outputs/stage1_lbfgs_refine")
    parser.add_argument("--val-every", type=int, default=10)
    parser.add_argument("--val-n-points", type=int, default=512)
    parser.add_argument("--val-param-samples", type=int, default=48)
    parser.add_argument("--viz-a", type=float, default=0.5, help="Reference a for viz")
    parser.add_argument("--viz-omega", type=float, default=0.02573, help="Reference omega for viz")
    parser.add_argument("--viz-u", type=float, default=0.5, help="Reference u for viz")
    parser.add_argument("--viz-v", type=float, default=0.6026, help="Reference v for viz")
    parser.add_argument("--viz-every", type=int, default=50, help="Generate reference viz every N steps")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = torch.float64

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    # Load config
    full_cfg = load_pinn_full_config(args.config)
    physics_cfg = full_cfg["physics"]
    cfg = full_cfg["train"]
    mc = cfg.get("model", {})
    atlas_cfg = cfg.get("atlas_training", {})
    val_cfg = cfg.get("training", {}).get("validation", {})

    # Build model
    model = AutoencoderTeukolskyPINN(
        hidden_dims=mc.get("hidden_dims", [128, 128, 128, 128]),
        activation=mc.get("activation", "silu"),
        fourier_num_freqs=mc.get("fourier_num_freqs", 2),
        fourier_base_scale=mc.get("fourier_base_scale", 1.0),
        param_embed_dim=mc.get("param_embed_dim", 64),
        use_film=mc.get("use_film", True),
        use_residual=mc.get("use_residual", True),
    )
    ckpt = torch.load(str(ckpt_path), map_location=device)
    sd = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    model.load_state_dict(sd, strict=True)
    model.to(device)
    model.eval()
    print(f"Loaded checkpoint: {ckpt_path} (step={ckpt.get('step', '?')}, best_val={ckpt.get('best_val_mean', '?')})")
    print(f"Model params: {sum(p.numel() for p in model.parameters())}")

    # Setup output directory
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_root) / f"{ts}_patch_{args.patch_id:03d}_lbfgs"
    ckpt_dir = run_dir / "checkpoints"
    log_dir = run_dir / "logs"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {run_dir}")

    # Save config snapshot
    with open(run_dir / "config_snapshot.yaml", "w") as f:
        yaml.dump(full_cfg, f)

    # ---- Fixed random y-grid ----
    y_eps = 1e-4
    n_int = args.n_interior
    y_interior = torch.rand(n_int, device=device, dtype=dtype) * (1.0 - 2*y_eps) - (1.0 - y_eps)
    y_interior = y_interior.unsqueeze(0)  # (1, N)
    print(f"Random y-grid: {n_int} points, range [{float(y_interior.min()):.6f}, {float(y_interior.max()):.6f}]")

    # ---- Fixed parameter batches ----
    # Patch 0 bounds (from config)
    a_min, a_max = 0.206, 0.794
    omega_min, omega_max = 0.008523, 0.0777
    print(f"Patch {args.patch_id}: a=[{a_min:.4f}, {a_max:.4f}], omega=[{omega_min:.6f}, {omega_max:.6f}]")

    # Generate fixed parameter batches using Sobol for good coverage
    sobol_engine = torch.quasirandom.SobolEngine(dimension=2, scramble=True, seed=12345)

    fixed_param_batches = []
    for i in range(args.n_param_batches):
        uv = sobol_engine.draw(args.batch_size).to(device=device, dtype=dtype)
        u_raw = uv[:, 0]
        v_raw = uv[:, 1]
        a_batch = a_min + (a_max - a_min) * u_raw
        omega_batch = omega_min + (omega_max - omega_min) * v_raw
        u_batch = u_raw
        v_batch = v_raw

        # Compute lambda for each (a, omega)
        from physical_ansatz.residual import get_lambda_from_cfg, AuxCache
        cache = AuxCache()
        lambda_vals = []
        for ai, oi in zip(a_batch.cpu().numpy(), omega_batch.cpu().numpy()):
            lam = get_lambda_from_cfg(physics_cfg, cache, torch.tensor(float(ai)), torch.tensor(float(oi)))
            lambda_vals.append(lam)
        lambda_batch = torch.tensor(np.array(lambda_vals), device=device, dtype=torch.complex128)

        fixed_param_batches.append((a_batch, omega_batch, u_batch, v_batch, lambda_batch))
        print(f"  Batch {i}: a=[{float(a_batch.min()):.4f}, {float(a_batch.max()):.4f}], "
              f"omega=[{float(omega_batch.min()):.6f}, {float(omega_batch.max()):.6f}]")

    def compute_reduced_shape(a_b, omega_b, u_b, v_b, lambda_b, y_b):
        problem_cfg = physics_cfg["problem"]
        M = float(problem_cfg.get("M", 1.0))
        m = int(problem_cfg.get("m", 2))
        s = int(problem_cfg.get("s", -2))
        f_vals = model(a_b, omega_b, y_b, u=u_b, v=v_b)
        slope = horizon_regularity_slope(
            a=a_b,
            omega=omega_b,
            lambda_=lambda_b,
            m=m,
            M=M,
            s=s,
        )
        return compose_reduced_shape_from_f(f=f_vals, y=y_b, slope=slope)

    y_train_expanded_by_batch = [
        y_interior.expand(batch[0].shape[0], -1)
        for batch in fixed_param_batches
    ]

    prev_s_batches = []
    if args.s_step_weight > 0:
        model.eval()
        with torch.no_grad():
            for batch, y_b in zip(fixed_param_batches, y_train_expanded_by_batch):
                a_b, omega_b, u_b, v_b, lambda_b = batch
                prev_s_batches.append(
                    compute_reduced_shape(a_b, omega_b, u_b, v_b, lambda_b, y_b).detach()
                )
        model.train()
        print(f"S-step trust region enabled: weight={args.s_step_weight:g}, eps={args.s_step_eps:g}")

    # ---- L-BFGS optimizer ----
    optimizer = torch.optim.LBFGS(
        model.parameters(),
        lr=args.lr,
        max_iter=5,
        max_eval=8,
        tolerance_grad=1e-8,
        tolerance_change=1e-14,
        history_size=50,
        line_search_fn="strong_wolfe",
    )

    # ---- Validation data (fixed set for consistent comparison) ----
    n_val_params = args.val_param_samples
    uv_val = sobol_engine.draw(n_val_params).to(device=device, dtype=dtype)
    u_val = uv_val[:, 0]
    v_val = uv_val[:, 1]
    a_val = a_min + (a_max - a_min) * u_val
    omega_val = omega_min + (omega_max - omega_min) * v_val

    from physical_ansatz.residual import get_lambda_from_cfg, AuxCache
    cache_val = AuxCache()
    lambda_val_list = []
    for ai, oi in zip(a_val.cpu().numpy(), omega_val.cpu().numpy()):
        lam = get_lambda_from_cfg(physics_cfg, cache_val, torch.tensor(float(ai)), torch.tensor(float(oi)))
        lambda_val_list.append(lam)
    lambda_val = torch.tensor(np.array(lambda_val_list), device=device, dtype=torch.complex128)

    y_val = torch.rand(args.val_n_points, device=device, dtype=dtype) * (1.0 - 2*y_eps) - (1.0 - y_eps)
    y_val = y_val.unsqueeze(0)

    def validate():
        model.eval()
        total_loss = 0.0
        total_reg = 0.0
        case_losses = []
        for i in range(n_val_params):
            ai = a_val[i:i+1]
            oi = omega_val[i:i+1]
            li = lambda_val[i:i+1]
            ui = u_val[i:i+1]
            vi = v_val[i:i+1]
            y_empty = torch.empty(0, device=device, dtype=dtype)
            loss_i, _ = pinn_residual_loss(
                model=model, cfg=physics_cfg,
                a_batch=ai, omega_batch=oi, lambda_batch=li,
                y_interior=y_val, y_boundary=y_empty,
                weight_interior=1.0, weight_boundary=0.0,
                normalize_residual=args.normalize_residual,
                residual_scale_eps=1.0e-12,
                return_pointwise=False,
                normalize_mode=args.normalize_mode,
                u_batch=ui, v_batch=vi,
            )
            reg_i = 0.0
            if args.f_reg_weight > 0:
                with torch.no_grad():
                    f_v = model(ai, oi, y_val, u=ui, v=vi)
                    reg_i = float(args.f_reg_weight * f_v.abs().pow(2).mean())
            total_loss += float(loss_i) + reg_i
            case_losses.append(float(loss_i) + reg_i)
            total_reg += reg_i
        model.train()
        avg = total_loss / n_val_params
        return avg, case_losses, total_reg / n_val_params

    # ---- Initial validation ----
    init_val, init_cases, init_reg = validate()
    best_val = init_val
    best_step = 0
    print(f"\nInitial val_mean: {init_val:.6f} (reg={init_reg:.6f})")
    print(f"Initial val_worst: {max(init_cases):.6f}")

    # ---- Initial viz (verify model loaded correctly) ----
    fig_dir = run_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    _generate_viz(model, physics_cfg, ckpt, args, fig_dir, device, dtype)
    print(f"\nStarting L-BFGS ({args.steps} steps)...\n")

    history = []
    step_count = [0]
    pbar = trange(args.steps, desc="lbfgs", dynamic_ncols=True)

    for _ in pbar:
        batch_idx = step_count[0] % args.n_param_batches
        a_b, omega_b, u_b, v_b, lambda_b = fixed_param_batches[batch_idx]
        y_b = y_train_expanded_by_batch[batch_idx]
        y_empty = torch.empty(0, device=device, dtype=dtype)
        closure_stats = {}

        def closure():
            optimizer.zero_grad()
            loss_pde, _ = pinn_residual_loss(
                model=model, cfg=physics_cfg,
                a_batch=a_b, omega_batch=omega_b, lambda_batch=lambda_b,
                y_interior=y_b, y_boundary=y_empty,
                weight_interior=1.0, weight_boundary=0.0,
                normalize_residual=args.normalize_residual,
                residual_scale_eps=1.0e-12,
                return_pointwise=False,
                normalize_mode=args.normalize_mode,
                u_batch=u_b, v_batch=v_b,
            )
            # f(y) regularization: prevent trivial solution (f→huge near horizon where B2→0)
            if args.f_reg_weight > 0:
                f_vals = model(a_b, omega_b, y_b, u=u_b, v=v_b)
                loss_reg = args.f_reg_weight * (f_vals.abs().pow(2).mean())
            else:
                loss_reg = torch.zeros((), device=device, dtype=loss_pde.dtype)
            if args.s_step_weight > 0:
                s_vals = compute_reduced_shape(a_b, omega_b, u_b, v_b, lambda_b, y_b)
                s_prev = prev_s_batches[batch_idx]
                loss_s_step = (
                    (s_vals - s_prev).abs().pow(2)
                    / (s_prev.abs().pow(2) + args.s_step_eps)
                ).mean()
            else:
                loss_s_step = torch.zeros((), device=device, dtype=loss_pde.dtype)
            total_loss = loss_pde + loss_reg + args.s_step_weight * loss_s_step
            closure_stats["loss_pde"] = float(loss_pde.detach().cpu().item())
            closure_stats["loss_reg"] = float(loss_reg.detach().cpu().item())
            closure_stats["loss_s_step"] = float(loss_s_step.detach().cpu().item())
            total_loss.backward()
            return total_loss

        loss_val = optimizer.step(closure)
        step_count[0] += 1
        step = step_count[0]

        if args.s_step_weight > 0:
            model.eval()
            with torch.no_grad():
                prev_s_batches[batch_idx] = compute_reduced_shape(
                    a_b, omega_b, u_b, v_b, lambda_b, y_b
                ).detach()
            model.train()

        if step % args.val_every == 0 or step == 1:
            val_mean, val_cases, val_reg = validate()
            if val_mean < best_val:
                best_val = val_mean
                best_step = step
                torch.save({
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "best_val_mean": best_val,
                    "val_cases": val_cases,
                    "source_checkpoint": str(ckpt_path),
                }, ckpt_dir / "best_model.pt")
                improved = "*"
            else:
                improved = ""

            history.append({
                "step": step,
                "train_loss": float(loss_val),
                "val_mean": val_mean,
                "val_worst": max(val_cases),
                "val_reg": val_reg,
                "best_val": best_val,
                "loss_pde": closure_stats.get("loss_pde"),
                "loss_reg": closure_stats.get("loss_reg"),
                "loss_s_step": closure_stats.get("loss_s_step"),
            })

            with open(log_dir / "history.jsonl", "a") as f:
                f.write(json.dumps(history[-1]) + "\n")

            pbar.set_postfix_str(
                f"loss={float(loss_val):.4f} pde={closure_stats.get('loss_pde', float('nan')):.4g} "
                f"sstep={closure_stats.get('loss_s_step', 0.0):.2e} val={val_mean:.4f} "
                f"reg={val_reg:.4f} best={best_val:.4f}{improved}"
            )

            if step % 100 == 0:
                torch.save({
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "best_val_mean": best_val,
                    "source_checkpoint": str(ckpt_path),
                }, ckpt_dir / f"step_{step:06d}.pt")

            if args.viz_every > 0 and step % args.viz_every == 0:
                fig_dir = run_dir / "figures"
                fig_dir.mkdir(exist_ok=True)
                _generate_viz(model, physics_cfg, {"step": step}, args, fig_dir, device, dtype)

    # ---- Final validation ----
    final_val, final_cases, final_reg = validate()
    print(f"\n{'='*50}")
    print(f"L-BFGS complete: {args.steps} steps")
    print(f"Initial val: {init_val:.6f}")
    print(f"Final val:   {final_val:.6f}")
    print(f"Best val:    {best_val:.6f} (step {best_step})")
    print(f"Output:      {run_dir}")

    # Save final model
    torch.save({
        "step": args.steps,
        "model_state_dict": model.state_dict(),
        "best_val_mean": best_val,
        "initial_val": init_val,
        "final_val": final_val,
        "source_checkpoint": str(ckpt_path),
    }, ckpt_dir / "final_model.pt")

    # Summary
    with open(run_dir / "summary.json", "w") as f:
        json.dump({
            "source_checkpoint": str(ckpt_path),
            "lbfgs_steps": args.steps,
            "lr": args.lr,
            "f_reg_weight": args.f_reg_weight,
            "normalize_residual": args.normalize_residual,
            "normalize_mode": args.normalize_mode,
            "s_step_weight": args.s_step_weight,
            "s_step_eps": args.s_step_eps,
            "initial_val": init_val,
            "final_val": final_val,
            "best_val": best_val,
            "best_step": best_step,
            "n_interior": args.n_interior,
            "n_param_batches": args.n_param_batches,
            "batch_size": args.batch_size,
        }, f, indent=2)

    # ---- Generate reference visualization (same as trainer.visualize_reference) ----
    print("\nGenerating reference visualization...")
    fig_dir = run_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    _generate_viz(model, physics_cfg, ckpt, args, fig_dir, device, dtype)

    print("Done.")


def _generate_viz(model, physics_cfg, ckpt, args, fig_dir, device, dtype):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from physical_ansatz.residual import get_lambda_from_cfg, AuxCache
    from physical_ansatz.prefactor import Leaver_prefactors, build_prefactor_primitives, r_plus
    from physical_ansatz.transform_y import h_factor, horizon_regularity_slope, compose_reduced_shape_from_f
    from dataset.sampling import sample_points_chebyshev_grid

    problem_cfg = physics_cfg["problem"]
    M = float(problem_cfg.get("M", 1.0))
    l = int(problem_cfg.get("l", 2))
    m = int(problem_cfg.get("m", 2))
    s = int(problem_cfg.get("s", -2))

    step = ckpt.get("step", "?")

    # ---- Pick reference sample: nearest val-pool point to patch center (matching trainer) ----
    patch_center = ckpt.get("patch_center", {})
    pc_u = float(patch_center.get("u", args.viz_u))
    pc_v = float(patch_center.get("v", args.viz_v))

    latest_val = ckpt.get("latest_val_metrics", {})
    case_metrics = latest_val.get("case_metrics", [])

    if case_metrics:
        best_dist = float("inf")
        best_case = None
        for case in case_metrics:
            du = float(case["u"]) - pc_u
            dv = float(case["v"]) - pc_v
            d2 = du * du + dv * dv
            if d2 < best_dist:
                best_dist = d2
                best_case = case
        if best_case is not None:
            a_t = torch.tensor(float(best_case["a"]), device=device, dtype=dtype)
            omega_t = torch.tensor(float(best_case["omega"]), device=device, dtype=dtype)
            u_t = torch.tensor(float(best_case["u"]), device=device, dtype=dtype)
            v_t = torch.tensor(float(best_case["v"]), device=device, dtype=dtype)
        else:
            a_t = torch.tensor(args.viz_a, device=device, dtype=dtype)
            omega_t = torch.tensor(args.viz_omega, device=device, dtype=dtype)
            u_t = torch.tensor(args.viz_u, device=device, dtype=dtype)
            v_t = torch.tensor(args.viz_v, device=device, dtype=dtype)
    else:
        a_t = torch.tensor(args.viz_a, device=device, dtype=dtype)
        omega_t = torch.tensor(args.viz_omega, device=device, dtype=dtype)
        u_t = torch.tensor(args.viz_u, device=device, dtype=dtype)
        v_t = torch.tensor(args.viz_v, device=device, dtype=dtype)

    patch_id = ckpt.get("patch_id", args.patch_id)

    cache = AuxCache()
    lam = get_lambda_from_cfg(physics_cfg, cache, a_t, omega_t)
    lam_c = lam.detach().clone() if torch.is_tensor(lam) else torch.tensor(lam)
    lam_c = lam_c.to(device=device, dtype=torch.complex128)
    h2 = h_factor(a_t, omega_t, m=m, M=M, s=s)

    rp = r_plus(a_t, M)
    r_min = max(2.0, float(rp.detach().cpu().item()) + 1e-4)
    r_max = 1000.0
    n_pts = 400

    # Left: uniform r
    r_grid = torch.linspace(r_min, r_max, n_pts, device=device, dtype=dtype)
    x_from_r = rp / r_grid
    y_from_r = 2.0 * x_from_r - 1.0

    # Right: cheb y
    y_min_c = 2.0 * float(rp.detach().cpu().item()) / r_max - 1.0
    y_max_c = 2.0 * float(rp.detach().cpu().item()) / r_min - 1.0
    y_grid_cheb = sample_points_chebyshev_grid(n_points=n_pts, y_min=y_min_c, y_max=y_max_c, device=device, dtype=dtype)
    x_from_y = 0.5 * (y_grid_cheb + 1.0)
    r_from_y = rp / x_from_y

    model.eval()
    with torch.no_grad():
        # On r-grid
        f_r = model(a_t.unsqueeze(0), omega_t.unsqueeze(0), y_from_r, u=u_t.unsqueeze(0), v=v_t.unsqueeze(0)).squeeze(0)
        slope_r = horizon_regularity_slope(a=a_t.unsqueeze(0), omega=omega_t.unsqueeze(0), lambda_=lam_c.unsqueeze(0), m=m, M=M, s=s).squeeze(0)
        shape_r = compose_reduced_shape_from_f(f=f_r, y=y_from_r, slope=slope_r)
        rp_r, rm_r, _, _, _ = build_prefactor_primitives(r_grid, a_t, M=M, need_rs=False)
        P_r, _, _ = Leaver_prefactors(r_grid, a_t, omega_t, m=m, M=M, s=s, rp=rp_r, rm=rm_r)
        R_pred_r = P_r * h2 * shape_r

        # On y-grid
        f_y = model(a_t.unsqueeze(0), omega_t.unsqueeze(0), y_grid_cheb, u=u_t.unsqueeze(0), v=v_t.unsqueeze(0)).squeeze(0)
        slope_y = horizon_regularity_slope(a=a_t.unsqueeze(0), omega=omega_t.unsqueeze(0), lambda_=lam_c.unsqueeze(0), m=m, M=M, s=s).squeeze(0)
        shape_y = compose_reduced_shape_from_f(f=f_y, y=y_grid_cheb, slope=slope_y)
        rp_y, rm_y, _, _, _ = build_prefactor_primitives(r_from_y, a_t, M=M, need_rs=False)
        P_y, _, _ = Leaver_prefactors(r_from_y, a_t, omega_t, m=m, M=M, s=s, rp=rp_y, rm=rm_y)
        R_pred_y = P_y * h2 * shape_y

    r_np = r_grid.detach().cpu().numpy()
    y_np = y_grid_cheb.detach().cpu().numpy()
    R_pred_r_np = R_pred_r.detach().cpu().numpy()
    shape_pred_y_np = shape_y.detach().cpu().numpy()

    # Benchmark (pybhpt)
    a_scalar = float(a_t)
    omega_scalar = float(omega_t)
    benchmark_available = False
    benchmark_status = "benchmark=off"
    R_ref_r_np = None
    shape_ref_y_np = None

    try:
        from pybhpt_usage.compute_solution import compute_pybhpt_solution
        r_np_sorted = np.sort(r_np)
        r_y_np = r_from_y.detach().cpu().numpy()
        r_y_sorted = np.sort(r_y_np)
        order_r = np.argsort(r_np)
        inv_order_r = np.empty_like(order_r)
        inv_order_r[order_r] = np.arange(order_r.size)
        order_y = np.argsort(r_y_np)
        inv_order_y = np.empty_like(order_y)
        inv_order_y[order_y] = np.arange(order_y.size)

        _, R_ref_sorted = compute_pybhpt_solution(a_scalar, omega_scalar, ell=l, m=m,
                                                    r_grid=r_np_sorted, timeout=30.0)
        R_ref_r_np = np.asarray(R_ref_sorted, dtype=np.complex128)[inv_order_r]
        _, R_ref_y_sorted = compute_pybhpt_solution(a_scalar, omega_scalar, ell=l, m=m,
                                                      r_grid=r_y_sorted, timeout=30.0)
        R_ref_y_np_back = np.asarray(R_ref_y_sorted, dtype=np.complex128)[inv_order_y]
        shape_ref_y_np = R_ref_y_np_back / (P_y.detach().cpu().numpy() * complex(h2.detach().cpu().item()))
        benchmark_available = True
        rel_err = np.abs(R_pred_r_np - R_ref_r_np) / (np.abs(R_ref_r_np) + 1e-14)
        benchmark_status = f"benchmark=pybhpt medR={np.median(rel_err):.2e} maxR={np.max(rel_err):.2e}"
    except Exception as e:
        benchmark_status = f"benchmark=pybhpt-failed: {e}"

    fig, axes = plt.subplots(3, 2, figsize=(10, 10), sharex=False)

    for ax_row, label, pred_data, ref_data in [
        (0, "Re(R)", np.real(R_pred_r_np), np.real(R_ref_r_np) if benchmark_available else None),
        (1, "Im(R)", np.imag(R_pred_r_np), np.imag(R_ref_r_np) if benchmark_available else None),
        (2, "|R|", np.abs(R_pred_r_np), np.abs(R_ref_r_np) if benchmark_available else None),
    ]:
        axes[ax_row, 0].plot(r_np, pred_data, label=f"Pred {label}", lw=1.6)
        if ref_data is not None:
            axes[ax_row, 0].plot(r_np, ref_data, "--", label=f"ref {label}", lw=1.0)
        axes[ax_row, 0].set_ylabel(label)
        axes[ax_row, 0].legend()
        axes[ax_row, 0].grid(alpha=0.3)

    axes[2, 0].set_xlabel("r")

    for ax_row, label, pred_data, ref_data in [
        (0, "Re(S)", np.real(shape_pred_y_np), np.real(shape_ref_y_np) if benchmark_available else None),
        (1, "Im(S)", np.imag(shape_pred_y_np), np.imag(shape_ref_y_np) if benchmark_available else None),
        (2, "|S|", np.abs(shape_pred_y_np), np.abs(shape_ref_y_np) if benchmark_available else None),
    ]:
        axes[ax_row, 1].plot(y_np, pred_data, label=f"Pred {label}", lw=1.6)
        if ref_data is not None:
            axes[ax_row, 1].plot(y_np, ref_data, "--", label=f"ref {label}", lw=1.0)
        axes[ax_row, 1].set_ylabel(label)
        axes[ax_row, 1].legend()
        axes[ax_row, 1].grid(alpha=0.3)

    axes[2, 1].set_xlabel("y")

    fig.suptitle(
        f"patch={patch_id}, step={step}, a={a_scalar:.6f}, "
        f"omega={omega_scalar:.6f}, {benchmark_status}",
        fontsize=12
    )
    fig.tight_layout()
    save_path = fig_dir / f"step_{step}_ref.png"
    fig.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Viz saved: {save_path}")


if __name__ == "__main__":
    main()
