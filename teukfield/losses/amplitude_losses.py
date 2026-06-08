from __future__ import annotations

import torch


def phase_safe_complex_relerr(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mag = (torch.log(torch.clamp(torch.abs(pred), min=1e-300)) - torch.log(torch.clamp(torch.abs(target), min=1e-300))) ** 2
    phase = 1.0 - torch.real(pred / torch.clamp(torch.abs(pred), min=1e-300) * torch.conj(target / torch.clamp(torch.abs(target), min=1e-300)))
    return torch.mean(mag + phase)


def amplitude_anchor_loss(model_out: dict[str, torch.Tensor], B_inc_ref: torch.Tensor, B_ref_ref: torch.Tensor) -> torch.Tensor:
    loss_inc = phase_safe_complex_relerr(model_out["B_inc"].reshape_as(B_inc_ref), B_inc_ref)
    loss_ref = phase_safe_complex_relerr(model_out["B_ref"].reshape_as(B_ref_ref), B_ref_ref)
    return 0.5 * (loss_inc + loss_ref)
