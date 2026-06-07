from __future__ import annotations

import torch

from teukfield.physics.reduced_equation import reduced_coefficients_y


def complex_grad(value: torch.Tensor, x: torch.Tensor, create_graph: bool = True) -> torch.Tensor:
    real = torch.autograd.grad(value.real.sum(), x, create_graph=create_graph, retain_graph=True)[0]
    imag = torch.autograd.grad(value.imag.sum(), x, create_graph=create_graph, retain_graph=True)[0]
    return torch.complex(real, imag)


def strong_reduced_residual(model, y: torch.Tensor, a: torch.Tensor, logw: torch.Tensor, lambda_: torch.Tensor | None = None):
    y_leaf = y.detach().clone().requires_grad_(True)
    out = model(y_leaf, a, logw, lambda_=lambda_)
    S = out["S"]
    Sy = complex_grad(S, y_leaf)
    Syy = complex_grad(Sy, y_leaf)
    omega = torch.pow(torch.tensor(10.0, dtype=a.dtype, device=a.device), logw)
    D2, D1, D0 = reduced_coefficients_y(y_leaf, a, omega, out["lambda"], m=model.m, s=model.s)
    res = D2 * Syy + D1 * Sy + D0 * S
    den = torch.clamp(torch.maximum(torch.maximum(torch.abs(D2 * Syy), torch.abs(D1 * Sy)), torch.abs(D0 * S)), min=1.0e-300)
    rel = torch.abs(res) / den
    return torch.mean(rel * rel), {"res_rel_median": float(torch.median(rel.detach()).cpu())}


def weak_reduced_residual_placeholder(
    model,
    y: torch.Tensor,
    a: torch.Tensor,
    logw: torch.Tensor,
    lambda_: torch.Tensor | None = None,
    n_tests: int = 4,
) -> tuple[torch.Tensor, dict[str, float]]:
    """First milestone weak-residual interface.

    This computes moment residuals against low-order polynomial test functions
    on the supplied collocation points. It is intentionally lightweight; the
    next iteration can replace the mean rule by Chebyshev/Gauss quadrature.
    """
    y_leaf = y.detach().clone().requires_grad_(True)
    out = model(y_leaf, a, logw, lambda_=lambda_)
    S = out["S"]
    Sy = complex_grad(S, y_leaf)
    Syy = complex_grad(Sy, y_leaf)
    omega = torch.pow(torch.tensor(10.0, dtype=a.dtype, device=a.device), logw)
    D2, D1, D0 = reduced_coefficients_y(y_leaf, a, omega, out["lambda"], m=model.m, s=model.s)
    res = D2 * Syy + D1 * Sy + D0 * S
    moments = []
    yy = y_leaf.to(torch.complex128)
    for q in range(n_tests):
        phi = yy**q
        moments.append(torch.mean(res * phi, dim=-1))
    stacked = torch.stack(moments, dim=-1)
    scale = torch.clamp(torch.mean(torch.abs(D2 * Syy) + torch.abs(D1 * Sy) + torch.abs(D0 * S), dim=-1), min=1.0e-300)
    rel = torch.abs(stacked) / scale.unsqueeze(-1)
    return torch.mean(rel * rel), {"weak_rel_median": float(torch.median(rel.detach()).cpu())}
