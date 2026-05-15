#!/usr/bin/env python3
"""Summarize a Stage-3 u-basis training run from summary.json.

Usage:
  python scripts/summarize_stage3_ubasis.py --run-dir <run_dir>
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Summarize Stage-3 u-basis training run")
    parser.add_argument("--run-dir", type=str, required=True,
                        help="Path to run directory containing summary.json")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    summary_path = run_dir / "summary.json"

    if not summary_path.exists():
        # Try history.jsonl
        hist_path = run_dir / "history.jsonl"
        if hist_path.exists():
            history = []
            with open(hist_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        history.append(json.loads(line))
        else:
            print(f"ERROR: Neither summary.json nor history.jsonl found in {run_dir}")
            sys.exit(1)
    else:
        with open(summary_path) as f:
            data = json.load(f)
        history = data.get("history", [])

    if not history:
        print("No history entries found.")
        sys.exit(1)

    # Extract metrics
    epochs = [h["epoch"] for h in history]
    loss_ups = [h["val_loss_up"] for h in history]
    loss_downs = [h["val_loss_down"] for h in history]
    val_losses = [h["val_loss"] for h in history]
    train_losses = [h.get("train_loss", float("nan")) for h in history]
    lrs = [h.get("lr", float("nan")) for h in history]
    gn_ups = [h.get("grad_norm_up", float("nan")) for h in history]
    gn_downs = [h.get("grad_norm_down", float("nan")) for h in history]

    # Find minima
    idx_up_min = np.argmin(loss_ups)
    idx_down_min = np.argmin(loss_downs)
    idx_val_min = np.argmin(val_losses)

    # Detect rebound: did the best loss happen well before the final epoch?
    total_epochs = len(history)
    rebound_up = idx_up_min < total_epochs - 2  # best is not among last 2
    rebound_down = idx_down_min < total_epochs - 2

    # Best checkpoint
    best_ckpt = run_dir / "checkpoints" / "best_model.pt"

    print("=" * 60)
    print("Stage-3 u-basis Training Summary")
    print("=" * 60)
    print(f"  run_dir:        {run_dir}")
    print(f"  total epochs:   {total_epochs}")
    print()

    print("--- Best (lowest val_loss) ---")
    print(f"  epoch:          {epochs[idx_val_min]}")
    print(f"  val_loss:       {val_losses[idx_val_min]:.6e}")
    print(f"  val_loss_up:    {loss_ups[idx_val_min]:.6e}")
    print(f"  val_loss_down:  {loss_downs[idx_val_min]:.6e}")
    print(f"  lr:             {lrs[idx_val_min]:.2e}")
    print()

    print("--- loss_up ---")
    print(f"  min:            {loss_ups[idx_up_min]:.6e}  (epoch {epochs[idx_up_min]})")
    print(f"  final:          {loss_ups[-1]:.6e}")
    print(f"  initial:        {loss_ups[0]:.6e}")
    print(f"  reduction:      {loss_ups[0] / max(loss_ups[idx_up_min], 1e-30):.2f}x")
    print(f"  rebound:        {'YES' if rebound_up else 'no'}")
    print()

    print("--- loss_down ---")
    print(f"  min:            {loss_downs[idx_down_min]:.6e}  (epoch {epochs[idx_down_min]})")
    print(f"  final:          {loss_downs[-1]:.6e}")
    print(f"  initial:        {loss_downs[0]:.6e}")
    print(f"  reduction:      {loss_downs[0] / max(loss_downs[idx_down_min], 1e-30):.2f}x")
    print(f"  rebound:        {'YES' if rebound_down else 'no'}")
    print()

    print("--- Gradient norms ---")
    print(f"  gn_up range:    [{min(gn_ups):.2e}, {max(gn_ups):.2e}]")
    print(f"  gn_down range:  [{min(gn_downs):.2e}, {max(gn_downs):.2e}]")
    print()

    print("--- Learning rate ---")
    print(f"  start:          {lrs[0]:.2e}")
    print(f"  end:            {lrs[-1]:.2e}")
    lr_changes = sum(1 for i in range(1, len(lrs)) if lrs[i] != lrs[i-1])
    print(f"  changes:        {lr_changes}")
    print()

    print("--- Best checkpoint ---")
    if best_ckpt.exists():
        print(f"  {best_ckpt}")
    else:
        print("  NOT FOUND")
    print()

    # Save markdown summary
    md_path = run_dir / "stage3_summary.md"
    with open(md_path, "w") as f:
        f.write(f"# Stage-3 u-basis Training Summary\n\n")
        f.write(f"- **run_dir**: `{run_dir}`\n")
        f.write(f"- **total epochs**: {total_epochs}\n\n")

        f.write(f"## Best (lowest val_loss)\n\n")
        f.write(f"- epoch: {epochs[idx_val_min]}\n")
        f.write(f"- val_loss: {val_losses[idx_val_min]:.6e}\n")
        f.write(f"- val_loss_up: {loss_ups[idx_val_min]:.6e}\n")
        f.write(f"- val_loss_down: {loss_downs[idx_val_min]:.6e}\n\n")

        f.write(f"## loss_up\n\n")
        f.write(f"- min: {loss_ups[idx_up_min]:.6e} (epoch {epochs[idx_up_min]})\n")
        f.write(f"- final: {loss_ups[-1]:.6e}\n")
        f.write(f"- reduction: {loss_ups[0] / max(loss_ups[idx_up_min], 1e-30):.2f}x\n")
        f.write(f"- rebound: {'YES' if rebound_up else 'no'}\n\n")

        f.write(f"## loss_down\n\n")
        f.write(f"- min: {loss_downs[idx_down_min]:.6e} (epoch {epochs[idx_down_min]})\n")
        f.write(f"- final: {loss_downs[-1]:.6e}\n")
        f.write(f"- reduction: {loss_downs[0] / max(loss_downs[idx_down_min], 1e-30):.2f}x\n")
        f.write(f"- rebound: {'YES' if rebound_down else 'no'}\n\n")

        f.write(f"## Gradient norms\n\n")
        f.write(f"- gn_up: [{min(gn_ups):.2e}, {max(gn_ups):.2e}]\n")
        f.write(f"- gn_down: [{min(gn_downs):.2e}, {max(gn_downs):.2e}]\n\n")

        f.write(f"## Learning rate\n\n")
        f.write(f"- start: {lrs[0]:.2e}\n")
        f.write(f"- end: {lrs[-1]:.2e}\n")
        f.write(f"- changes: {lr_changes}\n")

    print(f"Summary saved to {md_path}")


if __name__ == "__main__":
    main()
