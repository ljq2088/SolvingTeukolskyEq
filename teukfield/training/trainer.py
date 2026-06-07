from __future__ import annotations

import json
from pathlib import Path

import torch

from teukfield.data.samplers import sample_patch, sample_y
from teukfield.losses.total_loss import dry_total_loss


def train_dry_run(model, cfg: dict, steps: int, output_dir: str, device: str = "cpu"):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["optimizer"]["adamw_lr"]))
    rows = []
    for step in range(1, steps + 1):
        a, logw = sample_patch(cfg["sampling"]["param_batch"], cfg, device=device)
        y = sample_y(a.numel(), cfg["sampling"]["residual_y_points"], device=device)
        opt.zero_grad(set_to_none=True)
        loss, metrics = dry_total_loss(model, y, a, logw)
        loss.backward()
        opt.step()
        row = {"step": step, "loss": float(loss.detach().cpu()), **metrics}
        rows.append(row)
        print(json.dumps(row), flush=True)
    (out_dir / "dry_run_metrics.json").write_text(json.dumps(rows, indent=2))
    torch.save(model.state_dict(), out_dir / "teukfield_ampnet_dry.pt")
    return rows

