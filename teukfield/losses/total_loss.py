from __future__ import annotations

import torch

from teukfield.losses.boundary_consistency import inner_consistency_loss, outer_consistency_loss
from teukfield.losses.amplitude_losses import amplitude_anchor_loss
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
    lambda_ = batch.get("lambda")
    out_outer = None
    A_down_u = None
    A_up_u = None

    if weights.get("residual", 0.0) > 0.0:
        loss_res, res_metrics = strong_reduced_residual(model, batch["y_res"], batch["a"], batch["logw"], lambda_=lambda_)
        loss = loss + float(weights["residual"]) * loss_res
        metrics.update(res_metrics)
        metrics["loss_res"] = float(loss_res.detach().cpu())

    if spectral_adapter is not None and weights.get("inner", 0.0) > 0.0:
        y_inner = 2.0 * batch["z_inner"] - 1.0
        out_inner = model(y_inner.unsqueeze(0).expand(batch["a"].numel(), -1), batch["a"], batch["logw"], lambda_=lambda_)
        target_inner = batch.get("R_inner_spec")
        if target_inner is None:
            target_inner = spectral_adapter.torch_inner_batch(batch["a"], batch["logw"], batch["z_inner"], device)
        loss_inner = inner_consistency_loss(out_inner, target_inner)
        loss = loss + float(weights["inner"]) * loss_inner
        metrics["loss_inner"] = float(loss_inner.detach().cpu())

    if spectral_adapter is not None and weights.get("outer", 0.0) > 0.0:
        y_outer = 2.0 * batch["z_outer"] - 1.0
        out_outer = model(y_outer.unsqueeze(0).expand(batch["a"].numel(), -1), batch["a"], batch["logw"], lambda_=lambda_)
        A_down_u = batch.get("A_down_u")
        A_up_u = batch.get("A_up_u")
        if A_down_u is None or A_up_u is None:
            A_down_u, A_up_u = spectral_adapter.torch_outer_batch(batch["a"], batch["logw"], batch["z_outer"], device)
        loss_outer = outer_consistency_loss(out_outer, A_down_u, A_up_u)
        loss = loss + float(weights["outer"]) * loss_outer
        metrics["loss_outer"] = float(loss_outer.detach().cpu())

    if weights.get("amp_anchor", 0.0) > 0.0 and batch.get("R_outer_pybhpt") is not None:
        if out_outer is None:
            y_outer = 2.0 * batch["z_outer"] - 1.0
            out_outer = model(y_outer.unsqueeze(0).expand(batch["a"].numel(), -1), batch["a"], batch["logw"], lambda_=lambda_)
        if A_down_u is None or A_up_u is None:
            A_down_u = batch.get("A_down_u")
            A_up_u = batch.get("A_up_u")
            if A_down_u is None or A_up_u is None:
                A_down_u, A_up_u = spectral_adapter.torch_outer_batch(batch["a"], batch["logw"], batch["z_outer"], device)
        B_inc = out_outer["B_inc"].reshape(-1, 1)
        B_ref = out_outer["B_ref"].reshape(-1, 1)
        combo = B_inc * A_down_u + B_ref * A_up_u
        R_outer_ref = batch["R_outer_pybhpt"]
        loss_pybhpt_outer_amp = torch.mean(
            (torch.abs(combo - R_outer_ref) / torch.clamp(torch.abs(R_outer_ref), min=1.0e-30)) ** 2
        )
        loss = loss + float(weights["amp_anchor"]) * loss_pybhpt_outer_amp
        metrics["loss_pybhpt_outer_amp"] = float(loss_pybhpt_outer_amp.detach().cpu())

    if weights.get("R_anchor", 0.0) > 0.0:
        z_anchor = batch.get("z_R_anchor")
        R_anchor = batch.get("R_anchor_spec")
        a_anchor = batch.get("a_R_anchor", batch["a"])
        logw_anchor = batch.get("logw_R_anchor", batch["logw"])
        lambda_anchor = batch.get("lambda_R_anchor", lambda_)
        if z_anchor is None:
            z_anchor = batch.get("z_anchor")
        if R_anchor is None:
            if spectral_adapter is None:
                raise ValueError("R_anchor requires either batch R_anchor_spec or spectral_adapter.")
            R_anchor = spectral_adapter.torch_R_anchor_batch(a_anchor, logw_anchor, z_anchor, device)
        y_anchor = 2.0 * z_anchor - 1.0
        out_anchor = model(y_anchor.unsqueeze(0).expand(a_anchor.numel(), -1), a_anchor, logw_anchor, lambda_=lambda_anchor)
        loss_R_anchor = torch.mean((torch.abs(out_anchor["R"] - R_anchor) / torch.clamp(torch.abs(R_anchor), min=1.0e-30)) ** 2)
        loss = loss + float(weights["R_anchor"]) * loss_R_anchor
        metrics["loss_R_anchor"] = float(loss_R_anchor.detach().cpu())

    if spectral_adapter is not None and weights.get("amp_anchor", 0.0) > 0.0 and not batch.get("disable_ls_amp_anchor", False):
        B_inc_ref = batch.get("B_inc_anchor")
        B_ref_ref = batch.get("B_ref_anchor")
        if B_inc_ref is None or B_ref_ref is None:
            B_inc_ref, B_ref_ref = spectral_adapter.torch_amplitude_anchor_batch(batch["a"], batch["logw"], device)
        y_amp = torch.zeros((batch["a"].numel(), 1), dtype=batch["a"].dtype, device=device)
        out_amp = model(y_amp, batch["a"], batch["logw"], lambda_=lambda_)
        loss_amp_anchor = amplitude_anchor_loss(out_amp, B_inc_ref, B_ref_ref)
        loss = loss + float(weights["amp_anchor"]) * loss_amp_anchor
        metrics["loss_ls_amp_anchor"] = float(loss_amp_anchor.detach().cpu())

    if weights.get("weak", 0.0) > 0.0:
        loss_weak, weak_metrics = weak_reduced_residual_placeholder(model, batch["y_res"], batch["a"], batch["logw"], lambda_=lambda_)
        loss = loss + float(weights["weak"]) * loss_weak
        metrics.update(weak_metrics)
        metrics["loss_weak"] = float(loss_weak.detach().cpu())

    metrics["loss_total"] = float(loss.detach().cpu())
    return loss, metrics
