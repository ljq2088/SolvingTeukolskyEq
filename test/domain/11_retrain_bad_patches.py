"""
Retrain poorly-performing atlas patches identified by the 9-patch benchmark.

Patches ranked worst-to-best by median relative error vs GSN:
  5 (median 4.22, max 13.66), 7 (3.15, 16.51), 6 (1.24, 46.49) — critical
  8 (0.55, 1.00), 2 (0.53, 1.00), 3 (0.22, 27.50), 1 (0.21, 1.00) — moderate

Strategy: anchor enabled, warmstart from best neighbor, more steps.
"""
from __future__ import annotations
import sys
sys.path.append("/home/ljq/code/PINN/SolvingTeukolsky")

import argparse
from pathlib import Path

from trainer.atlas_patch_trainer import AtlasPatchTrainer

# Base paths from the existing training run
MODEL_DIR = Path("/home/ljq/code/PINN/SolvingTeukolsky/outputs/atlas_multipatch_train_coarse040_retrain_rel1e4/models")
CFG = "config/pinn_config.yaml"
PROBE = "outputs/domain/probe_l2_m2.json"
ATLAS = "outputs/domain/atlas_l2_m2.json"
PATCH = "outputs/domain/patch_cover_l2_m2.json"
OUTPUT_ROOT = "outputs/atlas_retrain_bad_patches"

# Retrain plan: (patch_id, warmstart_model, description)
# Warmstart from best available neighbors:
#   patch_000_best — center patch, best overall
#   patch_004_best — right-center, decent
#   patch_003_best — left-center, okay
RETRAIN_PLAN = [
    (5, "patch_000_best.pt", "high-omega center"),
    (7, "patch_003_best.pt", "low-a high-omega"),
    (6, "patch_004_best.pt", "high-a high-omega"),
    (2, "patch_000_best.pt", "low-omega center"),
    (8, "patch_004_best.pt", "high-a low-omega"),
    (1, "patch_003_best.pt", "low-a low-omega"),
    (3, "patch_000_best.pt", "low-a center"),
]


def main():
    parser = argparse.ArgumentParser(description="Retrain bad atlas patches.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--anchor-enabled", action="store_true", default=False)
    parser.add_argument("--n-anchor-y", type=int, default=4)
    parser.add_argument("--verbose", action="store_true", default=True)
    parser.add_argument("--patches", type=str, default=None,
                        help="Comma-separated patch IDs to retrain (default: all in plan)")
    parser.add_argument("--output-root", type=str, default=OUTPUT_ROOT)
    args = parser.parse_args()

    target_ids = None
    if args.patches:
        target_ids = {int(x.strip()) for x in args.patches.split(",")}

    plan = [(pid, ws, desc) for pid, ws, desc in RETRAIN_PLAN
            if target_ids is None or pid in target_ids]

    print(f"[retrain] {len(plan)} patches to retrain")
    for pid, ws, desc in plan:
        print(f"  patch {pid}: warmstart={ws} ({desc})")

    for patch_id, warmstart_file, desc in plan:
        warmstart_path = MODEL_DIR / warmstart_file
        if not warmstart_path.exists():
            print(f"[retrain] WARNING: warmstart {warmstart_path} not found for patch {patch_id}, cold start")
            warmstart_path = None

        print("=" * 80)
        print(f"[retrain] patch {patch_id} ({desc})")
        print(f"  warmstart: {warmstart_path}")
        print("=" * 80)

        trainer = AtlasPatchTrainer(
            cfg_path=CFG,
            probe_json=PROBE,
            atlas_json=ATLAS,
            patch_json=PATCH,
            patch_id=patch_id,
            device=args.device,
            anchor_enabled=args.anchor_enabled,
            n_anchor_y=args.n_anchor_y,
            verbose=args.verbose,
            output_root=args.output_root,
            init_checkpoint=str(warmstart_path) if warmstart_path else None,
            init_load_optimizer=False,
        )

        print(f"  Patch center : (u,v)=({trainer.patch.u_center:.6f}, {trainer.patch.v_center:.6f})")
        print(f"  Phys center  : (a,omega)=({trainer.ref_sample['a'].item():.6f}, {trainer.ref_sample['omega'].item():.6f})")
        print(f"  Patch pool   : {len(trainer.patch_aw)}")
        print(f"  Train pool   : {len(trainer.train_aw)}")
        print(f"  Val pool     : {len(trainer.val_aw)}")
        print(f"  Anchor       : {trainer.anchor_enabled}")
        print(f"  n_anchor_y   : {trainer.n_anchor_y}")

        try:
            result = trainer.train(steps=args.steps)
            print(f"[retrain] patch {patch_id} done:")
            print(f"  run_dir       = {result['run_dir']}")
            print(f"  best_val_mean = {result['best_val_mean']:.6e}")
            if result.get("final_val") is not None:
                fv = result["final_val"]
                print(f"  final_val_mean = {fv['val_mean']:.6e}")
                print(f"  final_val_worst = {fv['val_worst']:.6e}")
        finally:
            trainer.close()

    print("[retrain] all done")


if __name__ == "__main__":
    main()
