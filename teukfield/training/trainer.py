from __future__ import annotations

import json
from pathlib import Path

import torch

from teukfield.data.samplers import sample_patch, sample_y
from teukfield.data.anchors import torch_pybhpt_R_anchor_batch
from teukfield.physics.angular import lambda_spheroidal
from teukfield.losses.total_loss import teukfield_total_loss
from teukfield.training.schedule import stage_weights


def build_batch(
    cfg: dict,
    device: str = "cpu",
    cache: dict[str, torch.Tensor] | None = None,
    disable_ls_amp_anchor: bool = False,
) -> dict[str, torch.Tensor]:
    sampling = cfg["sampling"]
    if cache is not None:
        if "a" in cache:
            idx = torch.randint(cache["a"].numel(), (int(sampling["param_batch"]),), device=device)
            a_main = cache["a"][idx]
            logw_main = cache["logw"][idx]
            lambda_main = cache["lambda"][idx]
        else:
            a_main, logw_main = sample_patch(sampling["param_batch"], cfg, device=device)
            omega_main = torch.pow(torch.tensor(10.0, dtype=a_main.dtype, device=a_main.device), logw_main)
            lambda_main = lambda_spheroidal(
                a_main, omega_main, ell=cfg["physics"]["l"], m=cfg["physics"]["m"], s=cfg["physics"]["s"]
            )
            idx = None
        y_res = sample_y(a_main.numel(), sampling["residual_y_points"], device=device)
        batch = {"a": a_main, "logw": logw_main, "lambda": lambda_main, "y_res": y_res, "disable_ls_amp_anchor": disable_ls_amp_anchor}
        if idx is not None and "z_inner" in cache:
            batch.update(
                {
                    "z_inner": cache["z_inner"],
                    "z_outer": cache["z_outer"],
                    "z_R_anchor": cache["z_R_anchor"],
                    "R_inner_spec": cache["R_inner_spec"][idx],
                    "A_down_u": cache["A_down_u"][idx],
                    "A_up_u": cache["A_up_u"][idx],
                    "R_anchor_spec": cache["R_anchor_spec"][idx],
                    "B_inc_anchor": cache["B_inc_anchor"][idx],
                    "B_ref_anchor": cache["B_ref_anchor"][idx],
                }
            )
        if "R_outer_pybhpt" in cache:
            batch["R_outer_pybhpt"] = cache["R_outer_pybhpt"][idx]
        if "R_anchor_pybhpt" in cache:
            idx_R = torch.randint(cache["a_R_anchor"].numel(), (int(sampling["param_batch"]),), device=device)
            batch.update(
                {
                    "a_R_anchor": cache["a_R_anchor"][idx_R],
                    "logw_R_anchor": cache["logw_R_anchor"][idx_R],
                    "lambda_R_anchor": cache["lambda_R_anchor"][idx_R],
                    "R_anchor_spec": cache["R_anchor_pybhpt"][idx_R],
                    "z_R_anchor": cache["z_R_anchor_pybhpt"],
                }
            )
        return batch
    a, logw = sample_patch(sampling["param_batch"], cfg, device=device)
    omega = torch.pow(torch.tensor(10.0, dtype=a.dtype, device=a.device), logw)
    lambda_ = lambda_spheroidal(a, omega, ell=cfg["physics"]["l"], m=cfg["physics"]["m"], s=cfg["physics"]["s"])
    y_res = sample_y(a.numel(), sampling["residual_y_points"], device=device)
    z_inner = torch.linspace(
        float(sampling.get("inner_z_min", 0.75)),
        float(sampling.get("inner_z_max", 0.98)),
        int(sampling["inner_points"]),
        dtype=torch.float64,
        device=device,
    )
    z_outer = torch.linspace(
        float(sampling.get("outer_z_min", 0.005)),
        float(sampling.get("outer_z_max", 0.05)),
        int(sampling["outer_points"]),
        dtype=torch.float64,
        device=device,
    )
    return {"a": a, "logw": logw, "lambda": lambda_, "y_res": y_res, "z_inner": z_inner, "z_outer": z_outer}


def build_boundary_cache(
    cfg: dict,
    spectral_adapter,
    cache_size: int,
    device: str = "cpu",
    include_pybhpt_outer_amp: bool = False,
    pybhpt_timeout: float = 60.0,
) -> dict[str, torch.Tensor] | None:
    if spectral_adapter is None or cache_size <= 0:
        return None
    sampling = cfg["sampling"]
    a, logw = sample_patch(cache_size, cfg, device=device)
    omega = torch.pow(torch.tensor(10.0, dtype=a.dtype, device=a.device), logw)
    lambda_ = lambda_spheroidal(a, omega, ell=cfg["physics"]["l"], m=cfg["physics"]["m"], s=cfg["physics"]["s"])
    z_inner = torch.linspace(
        float(sampling.get("inner_z_min", 0.75)),
        float(sampling.get("inner_z_max", 0.98)),
        int(sampling["inner_points"]),
        dtype=torch.float64,
        device=device,
    )
    z_outer = torch.linspace(
        float(sampling.get("outer_z_min", 0.005)),
        float(sampling.get("outer_z_max", 0.05)),
        int(sampling["outer_points"]),
        dtype=torch.float64,
        device=device,
    )
    z_R_anchor = torch.linspace(
        float(sampling.get("R_anchor_z_min", 0.05)),
        float(sampling.get("R_anchor_z_max", 0.90)),
        int(sampling.get("anchor_points", 32)),
        dtype=torch.float64,
        device=device,
    )
    A_down_u, A_up_u = spectral_adapter.torch_outer_batch(a, logw, z_outer, device)
    B_inc_anchor, B_ref_anchor = spectral_adapter.torch_amplitude_anchor_batch(a, logw, device)
    cache = {
        "a": a,
        "logw": logw,
        "lambda": lambda_,
        "z_inner": z_inner,
        "z_outer": z_outer,
        "z_R_anchor": z_R_anchor,
        "R_inner_spec": spectral_adapter.torch_inner_batch(a, logw, z_inner, device),
        "A_down_u": A_down_u,
        "A_up_u": A_up_u,
        "R_anchor_spec": spectral_adapter.torch_R_anchor_batch(a, logw, z_R_anchor, device),
        "B_inc_anchor": B_inc_anchor,
        "B_ref_anchor": B_ref_anchor,
    }
    if include_pybhpt_outer_amp:
        cache["R_outer_pybhpt"] = torch_pybhpt_R_anchor_batch(a, logw, z_outer, device, timeout=pybhpt_timeout)
    return cache


def build_pybhpt_R_anchor_cache(
    cfg: dict,
    cache_size: int,
    device: str = "cpu",
    timeout: float = 60.0,
) -> dict[str, torch.Tensor] | None:
    if cache_size <= 0:
        return None
    sampling = cfg["sampling"]
    a, logw = sample_patch(cache_size, cfg, device=device)
    omega = torch.pow(torch.tensor(10.0, dtype=a.dtype, device=a.device), logw)
    lambda_ = lambda_spheroidal(a, omega, ell=cfg["physics"]["l"], m=cfg["physics"]["m"], s=cfg["physics"]["s"])
    z_anchor = torch.linspace(
        float(sampling.get("pybhpt_R_anchor_z_min", sampling.get("R_anchor_z_min", 0.01))),
        float(sampling.get("pybhpt_R_anchor_z_max", sampling.get("R_anchor_z_max", 0.90))),
        int(sampling.get("pybhpt_anchor_points", sampling.get("anchor_points", 32))),
        dtype=torch.float64,
        device=device,
    )
    return {
        "a_R_anchor": a,
        "logw_R_anchor": logw,
        "lambda_R_anchor": lambda_,
        "z_R_anchor_pybhpt": z_anchor,
        "R_anchor_pybhpt": torch_pybhpt_R_anchor_batch(a, logw, z_anchor, device, timeout=timeout),
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
    pybhpt_R_anchor_cache_size: int = 0,
    pybhpt_outer_amp_anchor: bool = False,
    disable_ls_amp_anchor: bool = False,
    pybhpt_timeout: float = 60.0,
    lr: float | None = None,
    log_every: int = 1,
    checkpoint_every: int = 0,
):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr if lr is not None else cfg["optimizer"]["adamw_lr"]))
    cache = build_boundary_cache(
        cfg,
        spectral_adapter,
        boundary_cache_size,
        device,
        include_pybhpt_outer_amp=pybhpt_outer_amp_anchor,
        pybhpt_timeout=pybhpt_timeout,
    )
    pybhpt_cache = build_pybhpt_R_anchor_cache(cfg, pybhpt_R_anchor_cache_size, device, timeout=pybhpt_timeout)
    if cache is None:
        cache = pybhpt_cache
    elif pybhpt_cache is not None:
        cache.update(pybhpt_cache)
    rows = []
    for step in range(1, steps + 1):
        batch = build_batch(cfg, device, cache=cache, disable_ls_amp_anchor=disable_ls_amp_anchor)
        weights = stage_weights(stage, step, steps)
        if spectral_adapter is None:
            weights = {**weights, "inner": 0.0, "outer": 0.0}
        opt.zero_grad(set_to_none=True)
        loss, metrics = teukfield_total_loss(model, batch, weights, spectral_adapter=spectral_adapter, device=device)
        loss.backward()
        opt.step()
        row = {"step": step, "loss": float(loss.detach().cpu()), **{f"w_{k}": v for k, v in weights.items()}, **metrics}
        rows.append(row)
        if step == 1 or step == steps or step % max(1, int(log_every)) == 0:
            print(json.dumps(row), flush=True)
        if checkpoint_every > 0 and step % int(checkpoint_every) == 0:
            (out_dir / "training_metrics.json").write_text(json.dumps(rows, indent=2))
            torch.save(model.state_dict(), out_dir / "teukfield_ampnet.pt")
            torch.save(model.state_dict(), out_dir / f"teukfield_ampnet_step_{step:06d}.pt")
    (out_dir / "training_metrics.json").write_text(json.dumps(rows, indent=2))
    (out_dir / "dry_run_metrics.json").write_text(json.dumps(rows, indent=2))
    torch.save(model.state_dict(), out_dir / "teukfield_ampnet.pt")
    torch.save(model.state_dict(), out_dir / "teukfield_ampnet_dry.pt")
    return rows
