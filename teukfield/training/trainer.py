from __future__ import annotations

import json
from pathlib import Path

import torch

from teukfield.data.samplers import sample_patch, sample_y
from teukfield.losses.total_loss import teukfield_total_loss
from teukfield.training.schedule import stage_weights


def build_batch(cfg: dict, device: str = "cpu") -> dict[str, torch.Tensor]:
    sampling = cfg["sampling"]
    a, logw = sample_patch(sampling["param_batch"], cfg, device=device)
    y_res = sample_y(a.numel(), sampling["residual_y_points"], device=device)
    z_inner = torch.linspace(0.75, 0.98, int(sampling["inner_points"]), dtype=torch.float64, device=device)
    z_outer = torch.linspace(0.02, 0.35, int(sampling["outer_points"]), dtype=torch.float64, device=device)
    return {"a": a, "logw": logw, "y_res": y_res, "z_inner": z_inner, "z_outer": z_outer}


def train_dry_run(model, cfg: dict, steps: int, output_dir: str, device: str = "cpu", spectral_adapter=None, stage: int = 1):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["optimizer"]["adamw_lr"]))
    rows = []
    for step in range(1, steps + 1):
        batch = build_batch(cfg, device)
        weights = stage_weights(stage, step, steps)
        if spectral_adapter is None:
            weights = {**weights, "inner": 0.0, "outer": 0.0}
        opt.zero_grad(set_to_none=True)
        loss, metrics = teukfield_total_loss(model, batch, weights, spectral_adapter=spectral_adapter, device=device)
        loss.backward()
        opt.step()
        row = {"step": step, "loss": float(loss.detach().cpu()), **{f"w_{k}": v for k, v in weights.items()}, **metrics}
        rows.append(row)
        print(json.dumps(row), flush=True)
    (out_dir / "dry_run_metrics.json").write_text(json.dumps(rows, indent=2))
    torch.save(model.state_dict(), out_dir / "teukfield_ampnet_dry.pt")
    return rows
