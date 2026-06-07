from __future__ import annotations

import torch


def relative_complex_mse(pred: torch.Tensor, target: torch.Tensor, scale: torch.Tensor | None = None) -> torch.Tensor:
    if scale is None:
        scale = torch.clamp(torch.abs(target), min=1.0e-30)
    return torch.mean((torch.abs(pred - target) / torch.clamp(scale, min=1.0e-30)) ** 2)


def inner_consistency_loss(model_out: dict, R_inner_spec: torch.Tensor) -> torch.Tensor:
    return relative_complex_mse(model_out["R"], R_inner_spec)


def outer_consistency_loss(model_out: dict, A_down_u: torch.Tensor, A_up_u: torch.Tensor) -> torch.Tensor:
    B_inc = model_out["B_inc"].reshape(-1, 1)
    B_ref = model_out["B_ref"].reshape(-1, 1)
    target = B_inc * A_down_u + B_ref * A_up_u
    scale = torch.abs(B_inc * A_down_u) + torch.abs(B_ref * A_up_u)
    return relative_complex_mse(model_out["R"], target, scale)

