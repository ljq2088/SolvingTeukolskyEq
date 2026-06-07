from __future__ import annotations

import json
from pathlib import Path

import torch

from teukfield.data.samplers import sample_patch, sample_y
from teukfield.losses.residuals import strong_reduced_residual, weak_reduced_residual_placeholder
from teukfield.physics.angular import lambda_spheroidal


def validate_random(model, cfg: dict, output_dir: str, device: str = "cpu", include_weak: bool = False):
    model.eval()
    with torch.enable_grad():
        a, logw = sample_patch(cfg["validation"]["random_param_points"], cfg, device=device)
        y = sample_y(a.numel(), cfg["validation"]["random_y_points"], device=device)
        omega = torch.pow(torch.tensor(10.0, dtype=a.dtype, device=a.device), logw)
        lambda_ = lambda_spheroidal(a, omega, ell=cfg["physics"]["l"], m=cfg["physics"]["m"], s=cfg["physics"]["s"])
        loss, metrics = strong_reduced_residual(model, y, a, logw, lambda_=lambda_)
        if include_weak:
            weak_loss, weak_metrics = weak_reduced_residual_placeholder(model, y, a, logw, lambda_=lambda_)
            metrics.update(weak_metrics)
            metrics["weak_loss"] = float(weak_loss.detach().cpu())
    summary = {"residual_loss": float(loss.detach().cpu()), **metrics}
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "validation_summary.json").write_text(json.dumps(summary, indent=2))
    return summary
