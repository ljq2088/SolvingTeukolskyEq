from __future__ import annotations

import torch

from teukfield.losses.boundary_consistency import inner_consistency_loss, outer_consistency_loss
from teukfield.losses.residuals import strong_reduced_residual, weak_reduced_residual_placeholder


def dry_total_loss(model, y: torch.Tensor, a: torch.Tensor, logw: torch.Tensor, residual_weight: float = 1.0e-3):
    loss_res, metrics = strong_reduced_residual(model, y, a, logw)
    return residual_weight * loss_res, metrics


def teukfield_total_loss(
    model,
    batch: dict[str, torch.Tensor],
    weights: dict[str, float],
    spectral_adapter=None,
    device: str | torch.device = "cpu",
) -> tuple[torch.Tensor, dict[str, float]]:
    loss = torch.zeros((), dtype=torch.float64, device=device)
    metrics: dict[str, float] = {}

    if weights.get("residual", 0.0) > 0.0:
        loss_res, res_metrics = strong_reduced_residual(model, batch["y_res"], batch["a"], batch["logw"])
        loss = loss + float(weights["residual"]) * loss_res
        metrics.update(res_metrics)
        metrics["loss_res"] = float(loss_res.detach().cpu())

    if spectral_adapter is not None and weights.get("inner", 0.0) > 0.0:
        y_inner = 2.0 * batch["z_inner"] - 1.0
        out_inner = model(y_inner.unsqueeze(0).expand(batch["a"].numel(), -1), batch["a"], batch["logw"])
        target_inner = spectral_adapter.torch_inner_batch(batch["a"], batch["logw"], batch["z_inner"], device)
        loss_inner = inner_consistency_loss(out_inner, target_inner)
        loss = loss + float(weights["inner"]) * loss_inner
        metrics["loss_inner"] = float(loss_inner.detach().cpu())

    if spectral_adapter is not None and weights.get("outer", 0.0) > 0.0:
        y_outer = 2.0 * batch["z_outer"] - 1.0
        out_outer = model(y_outer.unsqueeze(0).expand(batch["a"].numel(), -1), batch["a"], batch["logw"])
        A_down_u, A_up_u = spectral_adapter.torch_outer_batch(batch["a"], batch["logw"], batch["z_outer"], device)
        loss_outer = outer_consistency_loss(out_outer, A_down_u, A_up_u)
        loss = loss + float(weights["outer"]) * loss_outer
        metrics["loss_outer"] = float(loss_outer.detach().cpu())

    if weights.get("weak", 0.0) > 0.0:
        loss_weak, weak_metrics = weak_reduced_residual_placeholder(model, batch["y_res"], batch["a"], batch["logw"])
        loss = loss + float(weights["weak"]) * loss_weak
        metrics.update(weak_metrics)
        metrics["loss_weak"] = float(loss_weak.detach().cpu())

    metrics["loss_total"] = float(loss.detach().cpu())
    return loss, metrics
