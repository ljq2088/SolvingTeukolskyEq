from __future__ import annotations
import sys

sys.path.append("/home/ljq/code/PINN/SolvingTeukolsky")
import argparse
from pathlib import Path

from trainer.atlas_patch_trainer import AtlasPatchTrainer


def main():
    parser = argparse.ArgumentParser(description="Full training on one atlas patch.")
    parser.add_argument("--cfg", type=str, required=True)
    parser.add_argument("--probe-json", type=str, required=True)
    parser.add_argument("--atlas-json", type=str, required=True)
    parser.add_argument("--patch-json", type=str, required=True)
    parser.add_argument("--patch-id", type=int, required=True)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--anchor-enabled", action="store_true")
    parser.add_argument("--n-anchor-y", type=int, default=4)
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    trainer = AtlasPatchTrainer(
        cfg_path=args.cfg,
        probe_json=args.probe_json,
        atlas_json=args.atlas_json,
        patch_json=args.patch_json,
        patch_id=args.patch_id,
        device=args.device,
        anchor_enabled=args.anchor_enabled,
        n_anchor_y=args.n_anchor_y,
        verbose=args.verbose,
    )

    print("=" * 80)
    print(f"Patch id       : {trainer.patch.patch_id}")
    print(f"Component id   : {trainer.patch.component_id}")
    print(f"Patch center   : (u,v)=({trainer.patch.u_center:.6f}, {trainer.patch.v_center:.6f})")
    print(f"Patch phys ctr : (a,omega)=({trainer.ref_sample['a'].item():.6f}, {trainer.ref_sample['omega'].item():.6f})")
    print(f"Patch pool size: {len(trainer.patch_aw)}")
    print(f"Train pool size: {len(trainer.train_aw)}")
    print(f"Val pool size  : {len(trainer.val_aw)}")
    print(f"Anchor enabled : {trainer.anchor_enabled}")
    print(f"n_anchor_y     : {trainer.n_anchor_y}")
    print("=" * 80)

    result = trainer.train(steps=args.steps)

    print("=" * 80)
    print(f"run_dir       = {result['run_dir']}")
    print(f"best_val_mean = {result['best_val_mean']:.6e}")
    if result["final_val"] is not None:
        print(f"final_val     = {result['final_val']}")
    print("=" * 80)


if __name__ == "__main__":
    main()