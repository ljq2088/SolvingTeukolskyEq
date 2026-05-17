#!/usr/bin/env python3
"""Export pre-Stage-4 bundle: copy Stage-1/Stage-2 artifacts with metadata.

Creates a self-contained rollback point before any Stage-4 training.
Does NOT overwrite existing Stage-1/Stage-2 outputs.

Usage:
  python scripts/export_pre_stage4_bundle.py \
    --stage1-artifact outputs/stage1_artifacts/patch_000_logw_v2 \
    --stage2-checkpoint outputs/autoencoder_stage2_amplitude_train/20260515_192625_stage2_amplitude/checkpoints/best_model.pt \
    --output-dir outputs/pre_stage4_bundle/patch_000_logw_v2
"""
import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def get_git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()
    except Exception:
        return "unknown"


def main():
    parser = argparse.ArgumentParser(description="Export pre-Stage-4 bundle")
    parser.add_argument("--stage1-artifact", type=str, required=True)
    parser.add_argument("--stage2-checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--patch-id", type=str, default="patch_000_logw_v2")
    args = parser.parse_args()

    stage1_dir = Path(args.stage1_artifact)
    stage2_ckpt = Path(args.stage2_checkpoint)
    out_dir = Path(args.output_dir)

    if not stage1_dir.exists():
        raise FileNotFoundError(f"Stage-1 artifact not found: {stage1_dir}")
    if not stage2_ckpt.exists():
        raise FileNotFoundError(f"Stage-2 checkpoint not found: {stage2_ckpt}")

    if out_dir.exists():
        print(f"[export] Output dir already exists: {out_dir}")
        print(f"[export] Remove it first if you want to re-export.")
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)

    git_commit = get_git_commit()

    # Copy Stage-1 best model
    stage1_best = stage1_dir / "stage1_best_model.pt"
    if stage1_best.exists():
        shutil.copy2(stage1_best, out_dir / "stage1_best_model.pt")
        print(f"[export] Copied {stage1_best.name}")
    else:
        print(f"[export] WARNING: {stage1_best} not found")

    # Copy Stage-2 best model
    shutil.copy2(stage2_ckpt, out_dir / "stage2_best_model.pt")
    print(f"[export] Copied stage2_best_model.pt")

    # Symlink or note rpred_cache
    rpred_cache = stage1_dir / "rpred_cache.npz"
    if rpred_cache.exists():
        shutil.copy2(rpred_cache, out_dir / "rpred_cache.npz")
        print(f"[export] Copied rpred_cache.npz")

    # Copy metadata
    meta_src = stage1_dir / "metadata.json"
    if meta_src.exists():
        shutil.copy2(meta_src, out_dir / "stage1_metadata.json")

    # Write bundle metadata
    metadata = {
        "git_commit": git_commit,
        "exported_at": datetime.now().isoformat(),
        "patch_id": args.patch_id,
        "omega_chart_mode": "log",
        "stage1_artifact_source": str(stage1_dir.resolve()),
        "stage2_checkpoint_source": str(stage2_ckpt.resolve()),
        "scattering_convention": "B_inc*u_down*A_down + B_ref*u_up*A_up",
        "stage3_status": "LEARNED_DECODERS_BYPASSED",
        "stage4_note": (
            "Stage-3 learned u_up/u_down decoders are NOT used in Stage-4. "
            "Stage-4 uses spectral method basis cache (fixed u_up/u_down). "
            "Only RinDecoder and AmplitudeNet are fine-tuned at very low LR. "
            "If Stage-4 results degrade, roll back to this bundle."
        ),
        "rollback_path": str(out_dir.resolve()),
        "rollback_files": [
            "stage1_best_model.pt",
            "stage2_best_model.pt",
            "rpred_cache.npz",
        ],
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Rollback instructions
    rollback = f"""# Stage-4 Rollback Instructions

Bundle: {out_dir}

## When to roll back
- If Stage-4 consistency error increases instead of decreases
- If RinDecoder or AmplitudeNet drift > 1e-2
- If NaN/Inf in Stage-4 loss

## How to roll back
Use the models in this directory:
- **Stage-1 Rin PINN**: stage1_best_model.pt
- **Stage-2 AmplitudeNet**: stage2_best_model.pt
- **Parameter pool**: rpred_cache.npz
- **Spectral basis**: use the fixed spectral_basis_cache (independent of Stage-4)

Stage-4 does NOT modify up_decoder, down_decoder, or the encoder.
Up/down decoders are bypassed — spectral basis cache is used instead.

## Git state
Commit: {git_commit}
"""
    with open(out_dir / "rollback_instructions.md", "w") as f:
        f.write(rollback)

    print(f"\n[export] Bundle saved to {out_dir}")
    print(f"         stage1_best_model.pt")
    print(f"         stage2_best_model.pt")
    print(f"         metadata.json")
    print(f"         rollback_instructions.md")
    print(f"\n[export] Git commit: {git_commit}")


if __name__ == "__main__":
    main()
