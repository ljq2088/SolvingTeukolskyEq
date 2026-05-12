"""
Train ChebCoeffNet on poorly-performing atlas patches.

Uses Chebyshev coefficient network with Adam warmup + L-BFGS refinement.
Targets patches 5, 6, 7 (high-omega) and optionally 2, 8 (extreme a).

No anchor loss — pure PDE residual training.
"""
from __future__ import annotations
import sys
sys.path.append("/home/ljq/code/PINN/SolvingTeukolsky")

import argparse
from pathlib import Path

from trainer.atlas_patch_trainer import AtlasPatchTrainer

CFG = "config/pinn_config.yaml"
PROBE = "outputs/domain/probe_l2_m2.json"
ATLAS = "outputs/domain/atlas_l2_m2.json"
PATCH = "outputs/domain/patch_cover_l2_m2.json"
OUTPUT_ROOT = "outputs/atlas_cheb_retrain"

# Priority: patches 5 (high-omega center), 7 (low-a high-omega), 6 (high-a high-omega)
PATCHES = [5, 7, 6, 2, 8]


def main():
    parser = argparse.ArgumentParser(description="Train ChebCoeffNet on bad atlas patches.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--adam-steps", type=int, default=5000)
    parser.add_argument("--lbfgs-steps", type=int, default=200)
    parser.add_argument("--cheb-N", type=int, default=64)
    parser.add_argument("--patches", type=str, default=None,
                        help="Comma-separated patch IDs (default: 5,7,6,2,8)")
    parser.add_argument("--output-root", type=str, default=OUTPUT_ROOT)
    parser.add_argument("--anchor-enabled", action="store_true", default=False)
    args = parser.parse_args()

    target_ids = None
    if args.patches:
        target_ids = {int(x.strip()) for x in args.patches.split(",")}

    plan = [pid for pid in PATCHES if target_ids is None or pid in target_ids]

    print(f"[cheb-train] {len(plan)} patches: {plan}")
    print(f"  Adam steps: {args.adam_steps}, L-BFGS steps: {args.lbfgs_steps}")
    print(f"  Cheb N: {args.cheb_N}, Anchor: {args.anchor_enabled}")

    for patch_id in plan:
        print("=" * 80)
        print(f"[cheb-train] patch {patch_id}")
        print("=" * 80)

        trainer = AtlasPatchTrainer(
            cfg_path=CFG,
            probe_json=PROBE,
            atlas_json=ATLAS,
            patch_json=PATCH,
            patch_id=patch_id,
            device=args.device,
            anchor_enabled=args.anchor_enabled,
            n_anchor_y=4,
            verbose=True,
            output_root=args.output_root,
            init_checkpoint=None,
            init_load_optimizer=False,
            model_type="cheb",
            cheb_N=args.cheb_N,
        )

        # Override steps
        trainer.steps_default = args.adam_steps
        trainer.atlas_train_cfg["lbfgs_steps"] = args.lbfgs_steps

        print(f"  Patch center : (u,v)=({trainer.patch.u_center:.6f}, {trainer.patch.v_center:.6f})")
        print(f"  Phys center  : (a,omega)=({trainer.ref_sample['a'].item():.6f}, {trainer.ref_sample['omega'].item():.6f})")
        print(f"  Train pool   : {len(trainer.train_aw)}")
        print(f"  Val pool     : {len(trainer.val_aw)}")
        print(f"  Model params : {sum(p.numel() for p in trainer.model.parameters()):,}")

        try:
            result = trainer.train(steps=args.adam_steps)
            print(f"[cheb-train] patch {patch_id} done:")
            print(f"  run_dir       = {result['run_dir']}")
            print(f"  best_val_mean = {result['best_val_mean']:.6e}")
            if result.get("final_val") is not None:
                fv = result["final_val"]
                print(f"  final_val_mean = {fv['val_mean']:.6e}")
                print(f"  final_val_worst = {fv['val_worst']:.6e}")
        finally:
            trainer.close()

    print("[cheb-train] all done")


if __name__ == "__main__":
    main()
