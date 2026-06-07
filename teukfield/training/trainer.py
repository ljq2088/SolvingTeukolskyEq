from __future__ import annotations

import json
from pathlib import Path

import torch

from teukfield.data.samplers import sample_patch, sample_y
from teukfield.physics.angular import lambda_spheroidal
from teukfield.losses.total_loss import teukfield_total_loss
from teukfield.training.schedule import stage_weights


def build_batch(cfg: dict, device: str = "cpu", cache: dict[str, torch.Tensor] | None = None) -> dict[str, torch.Tensor]:
    sampling = cfg["sampling"]
    if cache is not None:
        idx = torch.randint(cache["a"].numel(), (int(sampling["param_batch"]),), device=device)
        y_res = sample_y(idx.numel(), sampling["residual_y_points"], device=device)
        return {
            "a": cache["a"][idx],
            "logw": cache["logw"][idx],
            "lambda": cache["lambda"][idx],
            "y_res": y_res,
            "z_inner": cache["z_inner"],
            "z_outer": cache["z_outer"],
            "R_inner_spec": cache["R_inner_spec"][idx],
            "A_down_u": cache["A_down_u"][idx],
            "A_up_u": cache["A_up_u"][idx],
        }
    a, logw = sample_patch(sampling["param_batch"], cfg, device=device)
    omega = torch.pow(torch.tensor(10.0, dtype=a.dtype, device=a.device), logw)
    lambda_ = lambda_spheroidal(a, omega, ell=cfg["physics"]["l"], m=cfg["physics"]["m"], s=cfg["physics"]["s"])
    y_res = sample_y(a.numel(), sampling["residual_y_points"], device=device)
    z_inner = torch.linspace(0.75, 0.98, int(sampling["inner_points"]), dtype=torch.float64, device=device)
    z_outer = torch.linspace(0.02, 0.35, int(sampling["outer_points"]), dtype=torch.float64, device=device)
    return {"a": a, "logw": logw, "lambda": lambda_, "y_res": y_res, "z_inner": z_inner, "z_outer": z_outer}


def build_boundary_cache(cfg: dict, spectral_adapter, cache_size: int, device: str = "cpu") -> dict[str, torch.Tensor] | None:
    if spectral_adapter is None or cache_size <= 0:
        return None
    sampling = cfg["sampling"]
    a, logw = sample_patch(cache_size, cfg, device=device)
    omega = torch.pow(torch.tensor(10.0, dtype=a.dtype, device=a.device), logw)
    lambda_ = lambda_spheroidal(a, omega, ell=cfg["physics"]["l"], m=cfg["physics"]["m"], s=cfg["physics"]["s"])
    z_inner = torch.linspace(0.75, 0.98, int(sampling["inner_points"]), dtype=torch.float64, device=device)
    z_outer = torch.linspace(0.02, 0.35, int(sampling["outer_points"]), dtype=torch.float64, device=device)
    A_down_u, A_up_u = spectral_adapter.torch_outer_batch(a, logw, z_outer, device)
    return {
        "a": a,
        "logw": logw,
        "lambda": lambda_,
        "z_inner": z_inner,
        "z_outer": z_outer,
        "R_inner_spec": spectral_adapter.torch_inner_batch(a, logw, z_inner, device),
        "A_down_u": A_down_u,
        "A_up_u": A_up_u,
    }


def train_dry_run(
    model,
    cfg: dict,
    steps: int,
    output_dir: str,
    device: str = "cpu",
    spectral_adapter=None,
    stage: int = 1,
    boundary_cache_size: int = 0,
):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["optimizer"]["adamw_lr"]))
    cache = build_boundary_cache(cfg, spectral_adapter, boundary_cache_size, device)
    rows = []
    for step in range(1, steps + 1):
        batch = build_batch(cfg, device, cache=cache)
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
    (out_dir / "training_metrics.json").write_text(json.dumps(rows, indent=2))
    (out_dir / "dry_run_metrics.json").write_text(json.dumps(rows, indent=2))
    torch.save(model.state_dict(), out_dir / "teukfield_ampnet.pt")
    torch.save(model.state_dict(), out_dir / "teukfield_ampnet_dry.pt")
    return rows
