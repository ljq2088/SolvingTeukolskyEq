#!/usr/bin/env python3
"""
Stage-1 autoencoder production training script (single patch).

Trains only encoder + rin_decoder on R_in PDE residual,
with amplitude_net / up_decoder / down_decoder frozen.

Usage:
  python scripts/train_autoencoder_stage1_atlas.py \\
    --config config/autoencoder_stage1_rin_train.yaml \\
    --domain-dir outputs/domain \\
    --patch-id 0 \\
    --device cuda \\
    --steps 100 \\
    --verbose

  # With anchor enabled:
  python scripts/train_autoencoder_stage1_atlas.py ... --anchor

  # With init checkpoint (weight migration from old PINN_MLP):
  python scripts/train_autoencoder_stage1_atlas.py ... \\
    --init-pinn-checkpoint path/to/pinn_checkpoint.pt

  # Resume from previous autoencoder checkpoint:
  python scripts/train_autoencoder_stage1_atlas.py ... \\
    --resume-checkpoint path/to/latest_model.pt \\
    --resume-run-dir path/to/run_dir
"""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

FALLBACK_DOMAIN_DIR = Path("/home/ljq/code/PINN/SolvingTeukolsky/outputs/domain")

import torch


def main():
    parser = argparse.ArgumentParser(description="Stage-1 autoencoder training (single patch)")
    parser.add_argument("--config", type=str, required=True,
                        help="Config YAML path")
    parser.add_argument("--domain-dir", type=str, default=None,
                        help="Directory containing probe/atlas/patch JSON files")
    parser.add_argument("--probe-json", type=str, default=None)
    parser.add_argument("--atlas-json", type=str, default=None)
    parser.add_argument("--patch-json", type=str, default=None)
    parser.add_argument("--patch-id", type=int, required=True,
                        help="Patch ID to train")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--steps", type=int, default=None,
                        help="Override max training steps")
    parser.add_argument("--output-root", type=str, default=None,
                        help="Override output directory")
    parser.add_argument("--init-pinn-checkpoint", type=str, default=None,
                        help="Path to old PINN_MLP checkpoint for weight migration")
    parser.add_argument("--init-load-optimizer", action="store_true",
                        help="Load optimizer state from init checkpoint")
    parser.add_argument("--resume-checkpoint", type=str, default=None,
                        help="Path to autoencoder checkpoint to resume from")
    parser.add_argument("--resume-run-dir", type=str, default=None,
                        help="Run directory for resume (needed for history)")
    parser.add_argument("--anchor", action="store_true", default=False,
                        help="Enable pybhpt anchor supervision")
    parser.add_argument("--n-anchor-y", type=int, default=4,
                        help="Number of anchor y-points")
    parser.add_argument("--verbose", action="store_true", default=False)
    args = parser.parse_args()

    # ---- Resolve domain files ----
    _DOMAIN_FILE_MAP = {
        "probe": "probe_l2_m2.json",
        "atlas": "atlas_l2_m2.json",
        "patch": "patch_cover_l2_m2.json",
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
            print(f"[warn] using fallback domain dir: {FALLBACK_DOMAIN_DIR}")
            return str(p)
        return None

    probe_json = _resolve("probe")
    atlas_json = _resolve("atlas")
    patch_json = _resolve("patch")
    if any(x is None for x in [probe_json, atlas_json, patch_json]):
        print("ERROR: missing domain files. Use --domain-dir or explicit --probe-json/--atlas-json/--patch-json")
        sys.exit(1)

    # ---- Build config overrides ----
    cfg_path = args.config
    if not Path(cfg_path).exists():
        print(f"ERROR: config not found: {cfg_path}")
        sys.exit(1)

    # If --init-pinn-checkpoint given, inject into config for trainer
    if args.init_pinn_checkpoint:
        import yaml
        with open(cfg_path, "r") as f:
            cfg_data = yaml.safe_load(f)
        if "model" not in cfg_data:
            cfg_data["model"] = {}
        cfg_data["model"]["init_from_pinn_checkpoint"] = args.init_pinn_checkpoint
        # Write temp config
        tmp_cfg = Path("/tmp/autoencoder_stage1_rin_train_tmp.yaml")
        with open(tmp_cfg, "w") as f:
            yaml.safe_dump(cfg_data, f)
        cfg_path = str(tmp_cfg)
        print(f"[init] injected init_from_pinn_checkpoint into temp config: {tmp_cfg}")

    # ---- Build trainer ----
    from trainer.atlas_patch_trainer import AtlasPatchTrainer

    output_root = args.output_root

    trainer = AtlasPatchTrainer(
        cfg_path=cfg_path,
        probe_json=probe_json,
        atlas_json=atlas_json,
        patch_json=patch_json,
        patch_id=args.patch_id,
        device=args.device,
        anchor_enabled=args.anchor,
        n_anchor_y=args.n_anchor_y,
        verbose=args.verbose,
        output_root=output_root,
        model_type="autoencoder",
        resume_checkpoint=args.resume_checkpoint,
        resume_run_dir=args.resume_run_dir,
    )

    model = trainer.model
    n_total = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # ---- Print model info ----
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
    print(f"steps:          {args.steps if args.steps else trainer.steps_default}")
    print("=" * 60)

    # ---- Train ----
    steps = args.steps if args.steps is not None else trainer.steps_default
    print(f"\nStarting Stage-1 training: {steps} steps...")
    result = trainer.train(steps=steps)

    print()
    print("=" * 60)
    print("Training complete.")
    print(f"  run_dir:       {result['run_dir']}")
    print(f"  best_val_mean: {result['best_val_mean']}")
    if result.get("final_val"):
        print(f"  final_val_mean: {result['final_val'].get('val_mean', 'N/A')}")
    print("=" * 60)


if __name__ == "__main__":
    main()
