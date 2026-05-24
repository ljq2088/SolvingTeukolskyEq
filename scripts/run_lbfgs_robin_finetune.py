#!/usr/bin/env python3
"""L-BFGS Robin BC fine-tuning for Stage-1 autoencoder checkpoint.

Loads a PDE-converged checkpoint, then runs L-BFGS with ONLY the
infinity Robin loss + proximal parameter penalty. The proximal term
|theta - theta_0|^2 keeps parameters close to the PDE solution while
L-BFGS line search safely improves the Robin boundary condition.

Usage:
  python scripts/run_lbfgs_robin_finetune.py \
    --checkpoint outputs/.../checkpoints/step_010000.pt \
    --steps 30 --prox-weight 1e-4 --device cuda
"""
import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from trainer.atlas_patch_trainer import AtlasPatchTrainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str,
                        default="config/autoencoder_stage1_rin_train.yaml")
    parser.add_argument("--patch-id", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--prox-weight", type=float, default=1e-4,
                        help="Proximal penalty weight (higher = more conservative)")
    parser.add_argument("--probe-json", type=str,
                        default="outputs/domain/probe_l2_m2.json")
    parser.add_argument("--atlas-json", type=str,
                        default="outputs/domain/atlas_l2_m2_logw.json")
    parser.add_argument("--patch-json", type=str,
                        default="outputs/domain/patch_cover_l2_m2_logw.json")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device)

    # Output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path("outputs/lbfgs_robin_finetune") / timestamp
    else:
        out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build trainer (loads model from checkpoint internally)
    trainer = AtlasPatchTrainer(
        cfg_path=args.config,
        patch_id=args.patch_id,
        device=str(device),
        probe_json=args.probe_json,
        atlas_json=args.atlas_json,
        patch_json=args.patch_json,
        resume_checkpoint=args.checkpoint,
        output_root=str(out_dir.parent),
        model_type="autoencoder",
    )

    # Disable pybhpt benchmark (global_step=10000 triggers it on every validate(),
    # which would be impractically slow)
    trainer.val_benchmark_backend = "none"

    print(f"\nPre-LBFGS validation ...")
    val_pre = trainer.validate()
    print(f"  val_mean={val_pre['val_mean']:.4f}, "
          f"val_worst={val_pre['val_worst']:.4f}")

    # Measure Robin loss before
    from physical_ansatz.infinity_robin import (
        analytic_c_inf, infinity_robin_loss, compute_S_and_Sy_at_infinity,
    )
    from physical_ansatz.residual_pinn import compute_f_derivatives_autograd
    from physical_ansatz.transform_y import horizon_regularity_slope

    M_phys = float(trainer.physics_cfg["problem"].get("M", 1.0))
    s_phys = int(trainer.physics_cfg["problem"].get("s", -2))
    m_phys = int(trainer.physics_cfg["problem"].get("m", 2))
    output_type = getattr(trainer.model, "output_type", "f")

    # Measure pre-LBFGS Robin loss
    a_b, omega_b, u_b, v_b = trainer.sample_param_batch()
    lambda_b = trainer.resolve_aux_batch(a_b, omega_b)
    y_inf = torch.full((1,), -1.0, device=device, dtype=trainer.dtype)
    y_inf_batch = y_inf.unsqueeze(0).expand(a_b.shape[0], 1)

    # Measure pre-LBFGS Robin loss (no no_grad - compute_f_derivatives needs autograd)
    if output_type == "S":
        S_inf, Sy_inf, _ = compute_f_derivatives_autograd(
            trainer.model, a_b, omega_b, y_inf_batch,
            u_batch=u_b, v_batch=v_b,
        )
        S_inf = S_inf.squeeze(-1)
        Sy_inf = Sy_inf.squeeze(-1)
    else:
        f_inf, fy_inf, _ = compute_f_derivatives_autograd(
            trainer.model, a_b, omega_b, y_inf_batch,
            u_batch=u_b, v_batch=v_b,
        )
        f_inf = f_inf.squeeze(-1)
        fy_inf = fy_inf.squeeze(-1)
        slope_inf = horizon_regularity_slope(
            a=a_b, omega=omega_b, lambda_=lambda_b,
            m=m_phys, M=M_phys, s=s_phys,
        )
        S_inf, Sy_inf = compute_S_and_Sy_at_infinity(f_inf, fy_inf, slope_inf)

    cdtype = torch.complex128 if a_b.dtype == torch.float64 else torch.complex64
    if not torch.is_complex(S_inf):
        S_inf = S_inf.to(dtype=cdtype)
        Sy_inf = Sy_inf.to(dtype=cdtype)
    lambda_c = lambda_b.to(dtype=cdtype) if not torch.is_complex(lambda_b) else lambda_b

    c_inf = analytic_c_inf(
        a_b, omega_b, lambda_c, m=m_phys, M=M_phys, s=s_phys,
    )
    robin_pre = infinity_robin_loss(S_inf, Sy_inf, c_inf).detach().mean().item()

    print(f"  pre-LBFGS Robin loss: {robin_pre:.6e}")

    # Run L-BFGS Robin fine-tuning
    print(f"\nRunning L-BFGS Robin fine-tuning: {args.steps} steps, "
          f"prox_weight={args.prox_weight:.1e} ...")
    trainer._train_lbfgs_robin(
        steps=args.steps,
        prox_weight=args.prox_weight,
        val_every=5,
    )

    # Post-LBFGS validation
    print(f"\nPost-LBFGS validation ...")
    val_post = trainer.validate()
    print(f"  val_mean={val_post['val_mean']:.4f}, "
          f"val_worst={val_post['val_worst']:.4f}")

    # Measure Robin loss after
    trainer.model.eval()
    a_b2, omega_b2, u_b2, v_b2 = trainer.sample_param_batch()
    lambda_b2 = trainer.resolve_aux_batch(a_b2, omega_b2)
    y_inf_b2 = y_inf.unsqueeze(0).expand(a_b2.shape[0], 1)

    # Measure Robin loss after
    trainer.model.eval()
    a_b2, omega_b2, u_b2, v_b2 = trainer.sample_param_batch()
    lambda_b2 = trainer.resolve_aux_batch(a_b2, omega_b2)
    y_inf_b2 = y_inf.unsqueeze(0).expand(a_b2.shape[0], 1)

    if output_type == "S":
        S_inf2, Sy_inf2, _ = compute_f_derivatives_autograd(
            trainer.model, a_b2, omega_b2, y_inf_b2,
            u_batch=u_b2, v_batch=v_b2,
        )
        S_inf2 = S_inf2.squeeze(-1)
        Sy_inf2 = Sy_inf2.squeeze(-1)
    else:
        f_inf2, fy_inf2, _ = compute_f_derivatives_autograd(
            trainer.model, a_b2, omega_b2, y_inf_b2,
            u_batch=u_b2, v_batch=v_b2,
        )
        f_inf2 = f_inf2.squeeze(-1)
        fy_inf2 = fy_inf2.squeeze(-1)
        slope_inf2 = horizon_regularity_slope(
            a=a_b2, omega=omega_b2, lambda_=lambda_b2,
            m=m_phys, M=M_phys, s=s_phys,
        )
        S_inf2, Sy_inf2 = compute_S_and_Sy_at_infinity(f_inf2, fy_inf2, slope_inf2)

    if not torch.is_complex(S_inf2):
        S_inf2 = S_inf2.to(dtype=cdtype)
        Sy_inf2 = Sy_inf2.to(dtype=cdtype)
    lambda_c2 = lambda_b2.to(dtype=cdtype) if not torch.is_complex(lambda_b2) else lambda_b2

    c_inf2 = analytic_c_inf(
        a_b2, omega_b2, lambda_c2, m=m_phys, M=M_phys, s=s_phys,
    )
    robin_post = infinity_robin_loss(S_inf2, Sy_inf2, c_inf2).detach().mean().item()

    # Save final model
    trainer._save_checkpoint(out_dir / "final_model.pt", val_metrics=val_post)

    results = {
        "checkpoint": args.checkpoint,
        "steps": args.steps,
        "prox_weight": args.prox_weight,
        "val_pre_mean": float(val_pre["val_mean"]),
        "val_post_mean": float(val_post["val_mean"]),
        "robin_pre": float(robin_pre),
        "robin_post": float(robin_post),
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults:")
    print(f"  Robin loss: {robin_pre:.6e} -> {robin_post:.6e}")
    print(f"  Val mean:   {val_pre['val_mean']:.4f} -> {val_post['val_mean']:.4f}")
    print(f"Output: {out_dir}")

    trainer.close()
    print("Done.")


if __name__ == "__main__":
    main()
