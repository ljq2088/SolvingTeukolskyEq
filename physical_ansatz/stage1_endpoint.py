"""Stage-1 infinity endpoint helper: compute S(-1), S_y(-1) and Robin residual.

Thin wrapper that composes the model forward pass, ansatz construction,
and analytic c_inf into a single diagnostic call.
"""
import torch
from physical_ansatz.infinity_robin import (
    analytic_c_inf,
    infinity_robin_residual,
    compute_S_and_Sy_at_infinity,
)
from physical_ansatz.transform_y import horizon_regularity_slope
from physical_ansatz.residual_pinn import compute_f_derivatives_autograd


def compute_stage1_S_and_Sy_at_infinity(
    model,
    a,
    omega,
    u=None,
    v=None,
    lambda_=None,
    m=2,
    M=1.0,
    s=-2,
):
    """Compute S(-1), S_y(-1), c_inf, and Robin residual for a batch.

    Args:
        model: AutoencoderTeukolskyPINN, eval mode
        a: (B,) real tensor
        omega: (B,) real tensor
        u: optional (B,) real tensor
        v: optional (B,) real tensor
        lambda_: optional (B,) complex tensor — if None, caller must pass
        m, M, s: Teukolsky parameters

    Returns:
        dict with keys:
            S_inf: (B,) complex — S at y=-1
            Sy_inf: (B,) complex — dS/dy at y=-1
            c_inf: (B,) complex — analytic Robin slope
            robin_residual: (B,) complex — Sy - c_inf*S
            robin_rel: (B,) float — relative Robin residual
    """
    B = a.shape[0]
    device = a.device
    dtype = a.dtype
    cdtype = torch.complex128 if dtype == torch.float64 else torch.complex64

    y_inf = torch.full((B, 1), -1.0, device=device, dtype=dtype)

    f, fy, _ = compute_f_derivatives_autograd(
        model, a, omega, y_inf, u_batch=u, v_batch=v,
    )
    f_val = f.squeeze(-1)
    # Workaround for autograd bug: compute_f_derivatives_autograd returns
    # fy of shape (B, B, N) instead of (B, N) due to grad w.r.t. all y_points.
    # Extract only the diagonal (sample i depends only on y_points[i]).
    fy_sq = fy.squeeze(-1)
    if fy_sq.ndim > 1 and fy_sq.shape[0] == fy_sq.shape[1]:
        fy_val = fy_sq.diagonal()
    else:
        fy_val = fy_sq

    slope = horizon_regularity_slope(a=a, omega=omega, lambda_=lambda_, m=m, M=M, s=s)

    S_inf, Sy_inf = compute_S_and_Sy_at_infinity(f_val, fy_val, slope)

    lambda_c = lambda_.to(dtype=cdtype) if not torch.is_complex(lambda_) else lambda_
    c_inf = analytic_c_inf(a, omega, lambda_c, m=m, M=M, s=s)

    robin_res = infinity_robin_residual(S_inf, Sy_inf, c_inf)
    robin_rel = torch.abs(robin_res) / (torch.abs(Sy_inf) + torch.abs(c_inf * S_inf) + 1e-12)

    return {
        "S_inf": S_inf,
        "Sy_inf": Sy_inf,
        "c_inf": c_inf,
        "robin_residual": robin_res,
        "robin_rel": robin_rel,
    }
