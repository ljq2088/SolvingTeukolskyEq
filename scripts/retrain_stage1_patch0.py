#!/usr/bin/env python3
"""Clean retrain of Stage-1 autoencoder on a single patch.

Saves full state (model + optimizer + patch metadata) for resumability.
Generates reference visualizations every viz_every steps.

Usage:
  python scripts/retrain_stage1_patch0.py --patch-id 0 --device cuda
  python scripts/retrain_stage1_patch0.py --patch-id 0 --device cuda --steps 500
  python scripts/retrain_stage1_patch0.py --patch-id 0 --device cuda \\
      --resume-checkpoint outputs/stage1_retrain/.../checkpoints/latest_model.pt \\
      --resume-run-dir outputs/stage1_retrain/...
"""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch

FALLBACK_DOMAIN_DIR = PROJECT_ROOT / "outputs" / "domain"


def main():
    parser = argparse.ArgumentParser(description="Stage-1 autoencoder clean retrain")
    parser.add_argument("--config", type=str,
                        default="config/autoencoder_stage1_retrain.yaml")
    parser.add_argument("--domain-dir", type=str, default=None)
    parser.add_argument("--probe-json", type=str, default=None)
    parser.add_argument("--atlas-json", type=str, default=None)
    parser.add_argument("--patch-json", type=str, default=None)
    parser.add_argument("--patch-id", type=int, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--output-root", type=str, default=None)
    parser.add_argument("--resume-checkpoint", type=str, default=None)
    parser.add_argument("--resume-run-dir", type=str, default=None)
    parser.add_argument("--init-pinn-checkpoint", type=str, default=None)
    args = parser.parse_args()

    # ---- Resolve domain files ----
    _DOMAIN_FILE_MAP = {
        "probe": "probe_l2_m2.json",
        "atlas": "atlas_l2_m2_logw.json",
        "patch": "patch_cover_l2_m2_logw.json",
    }

    def _resolve(name):
        attr = f"{name}_json"
        if getattr(args, attr) is not None:
            return getattr(args, attr)
        fname = _DOMAIN_FILE_MAP[name]
        if args.domain_dir is not None:
            return str(Path(args.domain_dir) / fname)
        p = FALLBACK_DOMAIN_DIR / fname
        if p.exists():
            return str(p)
        return None

    probe_json = _resolve("probe")
    atlas_json = _resolve("atlas")
    patch_json = _resolve("patch")
    if any(x is None for x in [probe_json, atlas_json, patch_json]):
        print("ERROR: missing domain files.")
        sys.exit(1)

    print(f"probe_json: {probe_json}")
    print(f"atlas_json: {atlas_json}")
    print(f"patch_json: {patch_json}")

    # ---- Build trainer ----
    from trainer.atlas_patch_trainer import AtlasPatchTrainer

    trainer = AtlasPatchTrainer(
        cfg_path=args.config,
        probe_json=probe_json,
        atlas_json=atlas_json,
        patch_json=patch_json,
        patch_id=args.patch_id,
        device=args.device,
        anchor_enabled=False,
        output_root=args.output_root,
        model_type="autoencoder",
        resume_checkpoint=args.resume_checkpoint,
        resume_run_dir=args.resume_run_dir,
    )

    model = trainer.model
    n_total = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("=" * 60)
    print(f"model_type:     {trainer.model_type}")
    print(f"patch_id:       {args.patch_id}")
    print(f"device:         {args.device}")
    print(f"dtype:          {trainer.dtype}")
    print(f"Total params:   {n_total}")
    print(f"Trainable:      {n_trainable}")
    print()

    modules = [
        ("encoder", model.encoder),
        ("rin_decoder", model.rin_decoder),
        ("amplitude_net", model.amplitude_net),
        ("up_decoder", model.up_decoder),
        ("down_decoder", model.down_decoder),
    ]
    for name, mod in modules:
        t = sum(p.numel() for p in mod.parameters())
        tr = sum(p.numel() for p in mod.parameters() if p.requires_grad)
        frozen_mark = "" if tr > 0 else " [FROZEN]"
        print(f"  {name:20s}  {tr:>8d}/{t:<8d} trainable{frozen_mark}")

    print(f"run_dir:        {trainer.run_dir}")
    steps = args.steps if args.steps is not None else trainer.steps_default
    print(f"steps:          {steps}")
    print("=" * 60)

    # ---- Train ----
    print(f"\nStarting Stage-1 retrain: {steps} steps...")
    result = trainer.train(steps=steps)

    print()
    print("=" * 60)
    print("Training complete.")
    print(f"  run_dir:       {result['run_dir']}")
    print(f"  best_val_mean: {result['best_val_mean']:.6f}")
    if result.get("final_val"):
        fv = result["final_val"]
        print(f"  final_val_mean: {fv.get('val_mean', 'N/A')}")
        print(f"  final_val_worst: {fv.get('val_worst', 'N/A')}")
    print(f"  best_model:    {result['run_dir']}/checkpoints/best_model.pt")
    print("=" * 60)
    print("\nRun eval_dense_grid.py to compute a-omega-r error heatmap:")
    print(f"  python scripts/eval_dense_grid.py \\")
    print(f"    --checkpoint {result['run_dir']}/checkpoints/best_model.pt \\")
    print(f"    --config config/autoencoder_stage1_retrain.yaml \\")
    print(f"    --patch-id {args.patch_id} \\")
    print(f"    --device {args.device}")


if __name__ == "__main__":
    main()
