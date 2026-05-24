#!/usr/bin/env python3
"""Conservative adaptive collocation + L-BFGS PDE fine-tuning for Stage-1.

Strategy:
  - No Robin loss (infinity_robin_weight=0)
  - Start from pure Chebyshev collocation (identical to checkpoint training)
  - Gradually introduce residual-based sampling, keep >=70% Chebyshev
  - Adam with conservative adaptive collocation
  - L-BFGS with pure PDE residual loss for final refinement
  - Evaluate ONLY by pybhpt relative error

Usage:
  python scripts/run_adaptive_lbfgs_pde.py \
    --checkpoint outputs/.../checkpoints/step_010000.pt \
    --adam-steps 200 --lbfgs-steps 50 --device cuda
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


def run_benchmark(trainer, tag: str):
    """Run pybhpt benchmark and print results."""
    results = []
    trainer.model.eval()
    for i in range(trainer.val_meta["a"].shape[0]):
        bm = trainer._benchmark_single_case(
            a_val=trainer.val_meta["a"][i],
            omega_val=trainer.val_meta["omega"][i],
            u_val=trainer.val_meta["u"][i],
            v_val=trainer.val_meta["v"][i],
            lambda_val=trainer.val_meta["lambda"][i],
        )
        a_v = float(trainer.val_meta["a"][i].cpu().item())
        w_v = float(trainer.val_meta["omega"][i].cpu().item())
        results.append({
            "a": a_v, "omega": w_v,
            "median_rel_err": bm["median_rel_err_R"],
            "max_rel_err": bm["max_rel_err_R"],
        })
        print(f"  [{tag}] a={a_v:.4f} omega={w_v:.4f}: "
              f"median={bm['median_rel_err_R']:.4%} max={bm['max_rel_err_R']:.4%}")

    medians = sorted([r["median_rel_err"] for r in results])
    median_of_medians = medians[len(medians) // 2]
    print(f"  [{tag}] median-of-medians: {median_of_medians:.4%}")
    return results, median_of_medians


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str,
                        default="config/autoencoder_stage1_rin_train.yaml")
    parser.add_argument("--patch-id", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--adam-steps", type=int, default=200,
                        help="Adam steps with adaptive collocation (0 = skip)")
    parser.add_argument("--lbfgs-steps", type=int, default=50,
                        help="L-BFGS PDE refinement steps")
    parser.add_argument("--probe-json", type=str,
                        default="outputs/domain/probe_l2_m2.json")
    parser.add_argument("--atlas-json", type=str,
                        default="outputs/domain/atlas_l2_m2_logw.json")
    parser.add_argument("--patch-json", type=str,
                        default="outputs/domain/patch_cover_l2_m2_logw.json")
    parser.add_argument("--output-dir", type=str, default=None)
    # Adaptive collocation params
    parser.add_argument("--warmup-epochs", type=int, default=20)
    parser.add_argument("--transition-epochs", type=int, default=30)
    parser.add_argument("--uniform-frac-final", type=float, default=0.7)
    parser.add_argument("--resample-every", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=5e-5)
    args = parser.parse_args()

    device = torch.device(args.device)

    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path("outputs/adaptive_lbfgs_pde") / timestamp
    else:
        out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save args
    with open(out_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2, default=str)

    # Build trainer
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

    # Override config settings for this experiment
    trainer.lr = args.lr
    for param_group in trainer.optimizer.param_groups:
        param_group["lr"] = args.lr

    # Enable conservative adaptive collocation
    trainer.adaptive_enabled = True
    trainer.adaptive_warmup_epochs = args.warmup_epochs
    trainer.adaptive_transition_epochs = args.transition_epochs
    trainer.adaptive_uniform_frac_final = args.uniform_frac_final
    trainer.adaptive_resample_every = args.resample_every
    trainer.adaptive_temperature = args.temperature
    # Ensure eval grid is initialized
    if trainer._adaptive_eval_y is None:
        from dataset.sampling import sample_points_chebyshev_grid
        trainer.adaptive_eval_n = 256
        trainer.adaptive_param_batch = 4
        trainer._adaptive_eval_y = sample_points_chebyshev_grid(
            n_points=256, y_min=-0.99, y_max=0.99,
            device=trainer.device, dtype=trainer.dtype,
        )
    trainer._adaptive_current_y = None
    trainer._adaptive_epoch = -1

    # Disable pybhpt benchmark during training (too slow per validate())
    trainer.val_benchmark_backend = "none"

    # Run pre-training benchmark
    print("\n=== Pre-training pybhpt benchmark ===")
    bench_pre, pre_median = run_benchmark(trainer, "pre")

    # Adam phase with conservative adaptive collocation
    if args.adam_steps > 0:
        print(f"\n=== Adam phase: {args.adam_steps} steps, "
              f"lr={args.lr}, adaptive collocation ===")
        trainer.train(steps=trainer.global_step + args.adam_steps)

        # Post-Adam benchmark
        print("\n=== Post-Adam pybhpt benchmark ===")
        bench_adam, adam_median = run_benchmark(trainer, "adam")
    else:
        bench_adam = bench_pre
        adam_median = pre_median

    # L-BFGS PDE refinement
    if args.lbfgs_steps > 0:
        print(f"\n=== L-BFGS PDE refinement: {args.lbfgs_steps} steps ===")
        # Disable adaptive during L-BFGS (use fixed Chebyshev for deterministic optimization)
        trainer.adaptive_enabled = False
        trainer._adaptive_current_y = None
        trainer._train_lbfgs(steps=args.lbfgs_steps)

        # Post-LBFGS benchmark
        print("\n=== Post-LBFGS pybhpt benchmark ===")
        bench_final, final_median = run_benchmark(trainer, "lbfgs")
    else:
        bench_final = bench_adam
        final_median = adam_median

    # Save final model
    final_val = trainer.validate()
    trainer._save_checkpoint(out_dir / "final_model.pt", val_metrics=final_val)

    # Save results
    results = {
        "checkpoint": args.checkpoint,
        "adam_steps": args.adam_steps,
        "lbfgs_steps": args.lbfgs_steps,
        "lr": args.lr,
        "warmup_epochs": args.warmup_epochs,
        "transition_epochs": args.transition_epochs,
        "uniform_frac_final": args.uniform_frac_final,
        "resample_every": args.resample_every,
        "temperature": args.temperature,
        "pre_median_rel_err": float(pre_median),
        "adam_median_rel_err": float(adam_median),
        "final_median_rel_err": float(final_median),
        "bench_pre": bench_pre,
        "bench_adam": bench_adam,
        "bench_final": bench_final,
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n=== Summary ===")
    print(f"  Pre:      median-rel-err = {pre_median:.4%}")
    if args.adam_steps > 0:
        print(f"  Post-Adam: median-rel-err = {adam_median:.4%}")
    print(f"  Final:     median-rel-err = {final_median:.4%}")
    print(f"Output: {out_dir}")

    trainer.close()
    print("Done.")


if __name__ == "__main__":
    main()
