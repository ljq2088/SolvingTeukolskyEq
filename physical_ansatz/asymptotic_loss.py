from __future__ import annotations

import torch

from .prefactor import Leaver_prefactors, build_prefactor_primitives, r_star
from .transform_y import (
    h_factor,
    horizon_regularity_slope,
    compose_reduced_shape_from_f,
)


def compute_infinity_asymptotic_loss(
    model,
    cfg: dict,
    a_batch: torch.Tensor,
    omega_batch: torch.Tensor,
    lambda_batch: torch.Tensor,
    u_batch: torch.Tensor,
    v_batch: torch.Tensor,
    Binc_spec: torch.Tensor,
    Bref_spec: torch.Tensor,
    r_points,
    beta: float = 1.0,
    relative: bool = True,
    eps: float = 1.0e-12,
):
    """
    Near-infinity asymptotic loss with per-parameter spectral baseline and
    log-space multiplicative correction for large B dynamic range.

    B_inc_net = B_inc_spec * exp(log|dB_inc| + i*arg_inc)
    B_ref_net = B_ref_spec * exp(log|dB_ref| + i*arg_ref)

    R_inf = B_inc_net * r^{-1} * exp(-i*omega*r_*) + B_ref_net * r^3 * exp(+i*omega*r_*)
    S_inf = R_inf / (P * h2)

    Loss compares S_pred with S_inf at large r, weighted toward infinity (y=-1).

    Returns:
        loss_inf: asymptotic S-matching loss
        loss_B:  regularization on raw amp_head output (log|dB|->0, phase->0)
        info:    python-float diagnostics
    """
    device = a_batch.device
    dtype = a_batch.dtype
    cdtype = torch.complex128 if dtype == torch.float64 else torch.complex64

    M = float(cfg["problem"].get("M", 1.0))
    s = int(cfg["problem"].get("s", -2))
    m = int(cfg["problem"].get("m", 2))

    B = int(a_batch.shape[0])

    r_grid = torch.as_tensor(r_points, device=device, dtype=dtype)
    if r_grid.ndim != 1:
        raise ValueError("r_points must be a 1D list/tensor.")

    Nr = int(r_grid.numel())

    # ---- Build batchwise r, x, y grids ----
    rp = torch.sqrt(torch.clamp(M * M - a_batch * a_batch, min=0.0)) + M  # (B,)
    r = r_grid.unsqueeze(0).expand(B, Nr)                                 # (B, Nr)
    x_grid = rp.unsqueeze(-1) / r                                          # (B, Nr)
    y_grid = 2.0 * x_grid - 1.0                                            # (B, Nr)

    # ---- S_pred from network ----
    f_pred = model(a_batch, omega_batch, y_grid, u=u_batch, v=v_batch)
    slope = horizon_regularity_slope(
        a=a_batch, omega=omega_batch, lambda_=lambda_batch, m=m, M=M, s=s,
    )
    S_pred = compose_reduced_shape_from_f(f=f_pred, y=y_grid, slope=slope)

    # ---- Multiplicative log-space corrections ----
    dB_inc, dB_ref = model.predict_asymptotic_delta(
        a_batch, omega_batch, u=u_batch, v=v_batch,
    )

    Binc_spec = Binc_spec.to(device=device, dtype=cdtype)
    Bref_spec = Bref_spec.to(device=device, dtype=cdtype)

    Binc_net = Binc_spec * dB_inc    # (B,)
    Bref_net = Bref_spec * dB_ref    # (B,)

    # ---- Asymptotic R_inf (vectorized) ----
    rs = r_star(r, a_batch.unsqueeze(-1), M=M)      # (B, Nr)
    omega_b = omega_batch.unsqueeze(-1)              # (B, 1)

    R_inf = (
        Binc_net.unsqueeze(-1) * r.to(cdtype).pow(-1.0) * torch.exp(-1j * omega_b.to(cdtype) * rs.to(cdtype))
        + Bref_net.unsqueeze(-1) * r.to(cdtype).pow(3.0) * torch.exp(+1j * omega_b.to(cdtype) * rs.to(cdtype))
    )

    # ---- S_inf = R_inf / (P * h2) (vectorized) ----
    rp_pref, rm_pref, _, _, _ = build_prefactor_primitives(r, a_batch.unsqueeze(-1), M=M, need_rs=False)
    P, _, _ = Leaver_prefactors(
        r=r, a=a_batch.unsqueeze(-1), omega=omega_batch.unsqueeze(-1),
        m=m, M=M, s=s, rp=rp_pref, rm=rm_pref,
    )
    h2 = h_factor(a_batch, omega_batch, m=m, M=M, s=s).unsqueeze(-1)   # (B, 1)
    S_inf = R_inf / (P.to(cdtype) * h2.to(cdtype))

    # ---- Weight: closer to y=-1 gets higher weight ----
    weights = x_grid.clamp_min(eps).pow(-float(beta))
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(eps)

    # ---- Case-based relative squared error ----
    err2 = torch.abs(S_pred - S_inf.detach()) ** 2

    if relative:
        scale = torch.sum(weights * torch.abs(S_inf.detach()) ** 2, dim=-1).clamp_min(eps)
        loss_inf_case = torch.sum(weights * err2, dim=-1) / scale
    else:
        loss_inf_case = torch.sum(weights * err2, dim=-1)

    loss_inf = loss_inf_case.mean()

    # ---- Log-space regularization: penalize deviation from dB=(1,0) ----
    a_1 = a_batch if a_batch.ndim == 2 else a_batch.unsqueeze(-1)
    omega_1 = omega_batch if omega_batch.ndim == 2 else omega_batch.unsqueeze(-1)
    u_1 = u_batch if u_batch.ndim == 2 else u_batch.unsqueeze(-1)
    v_1 = v_batch if v_batch.ndim == 2 else v_batch.unsqueeze(-1)
    feats_reg, _, _ = model._build_param_features(a=a_1, omega=omega_1, u=u_1, v=v_1)
    raw_out = model.amp_head(feats_reg)  # (B, 4): [log|dB_inc|, arg_inc, log|dB_ref|, arg_ref]
    loss_B = torch.mean(raw_out[:, 0] ** 2 + raw_out[:, 1] ** 2 + raw_out[:, 2] ** 2 + raw_out[:, 3] ** 2)

    info = {
        "loss_inf": float(loss_inf.detach().cpu().item()),
        "loss_B": float(loss_B.detach().cpu().item()),
        "mean_abs_Sinf": float(torch.mean(torch.abs(S_inf)).detach().cpu().item()),
        "mean_abs_Spred_inf": float(torch.mean(torch.abs(S_pred)).detach().cpu().item()),
        "mean_abs_dBinc": float(torch.mean(torch.abs(dB_inc.detach())).cpu().item()),
        "mean_abs_dBref": float(torch.mean(torch.abs(dB_ref.detach())).cpu().item()),
    }
    return loss_inf, loss_B, info
