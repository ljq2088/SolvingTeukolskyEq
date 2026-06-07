from __future__ import annotations

import torch

from teukfield.losses.residuals import strong_reduced_residual


def dry_total_loss(model, y: torch.Tensor, a: torch.Tensor, logw: torch.Tensor, residual_weight: float = 1.0e-3):
    loss_res, metrics = strong_reduced_residual(model, y, a, logw)
    return residual_weight * loss_res, metrics

