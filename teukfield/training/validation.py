from __future__ import annotations

import json
from pathlib import Path

import torch

from teukfield.data.samplers import sample_patch, sample_y
from teukfield.losses.residuals import strong_reduced_residual


def validate_random(model, cfg: dict, output_dir: str, device: str = "cpu"):
    model.eval()
    with torch.enable_grad():
        a, logw = sample_patch(cfg["validation"]["random_param_points"], cfg, device=device)
        y = sample_y(a.numel(), cfg["validation"]["random_y_points"], device=device)
        loss, metrics = strong_reduced_residual(model, y, a, logw)
    summary = {"residual_loss": float(loss.detach().cpu()), **metrics}
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "validation_summary.json").write_text(json.dumps(summary, indent=2))
    return summary

