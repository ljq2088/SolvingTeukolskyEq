from __future__ import annotations

import torch

from .prefactor import Leaver_prefactors, build_prefactor_primitives, r_star
from .transform_y import (
    h_factor,
    horizon_regularity_slope,
    compose_reduced_shape_from_f,
)


def compute_amplitude_teacher_loss(
    Binc_pred: torch.Tensor,
    Bref_pred: torch.Tensor,
    amp_raw: dict,
    Binc_spec: torch.Tensor,
    Bref_spec: torch.Tensor,
    eps: float = 1.0e-12,
):
    """Supervise absolute amplitude predictor with spectral coefficients.

    Magnitude: log-amplitude loss.
    Phase: unit-complex phase loss (avoids 2pi wrapping).
    """
    dtype = amp_raw["rho_inc"].dtype
    cdtype = Binc_pred.dtype

    Binc_spec = Binc_spec.to(dtype=cdtype, device=Binc_pred.device)
    Bref_spec = Bref_spec.to(dtype=cdtype, device=Bref_pred.device)

    abs_inc = torch.abs(Binc_spec).clamp_min(eps)
    abs_ref = torch.abs(Bref_spec).clamp_min(eps)

    rho_inc_t = torch.log(abs_inc).to(dtype=dtype)
    rho_ref_t = torch.log(abs_ref).to(dtype=dtype)

    mag_loss = torch.mean(
        (amp_raw["rho_inc"] - rho_inc_t) ** 2
        + (amp_raw["rho_ref"] - rho_ref_t) ** 2
    )

    phase_inc_t = Binc_spec / abs_inc.to(dtype=cdtype)
    phase_ref_t = Bref_spec / abs_ref.to(dtype=cdtype)

    phase_inc_p = torch.exp(1j * amp_raw["phi_inc"]).to(dtype=cdtype)
    phase_ref_p = torch.exp(1j * amp_raw["phi_ref"]).to(dtype=cdtype)

    phase_loss = torch.mean(
        torch.abs(phase_inc_p - phase_inc_t) ** 2
        + torch.abs(phase_ref_p - phase_ref_t) ** 2
    )

    rel_inc = torch.mean(torch.abs(Binc_pred - Binc_spec) / abs_inc.clamp_min(eps))
    rel_ref = torch.mean(torch.abs(Bref_pred - Bref_spec) / abs_ref.clamp_min(eps))

    loss_amp = mag_loss + phase_loss

    info = {
        "loss_amp": float(loss_amp.detach().cpu().item()),
        "loss_amp_mag": float(mag_loss.detach().cpu().item()),
        "loss_amp_phase": float(phase_loss.detach().cpu().item()),
        "amp_rel_Binc": float(rel_inc.detach().cpu().item()),
        "amp_rel_Bref": float(rel_ref.detach().cpu().item()),
    }
    return loss_amp, info


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
    amplitude_mode: str = "spectral_correction",
    hybrid_eta: float = 0.0,
):
    """Near-infinity asymptotic loss supporting three amplitude modes.

    spectral_correction: B_used = B_spec * dB
    learned_absolute:    B_used = B_pred
    hybrid_warmstart:    B_used = (1-eta)*B_spec + eta*B_pred

    Returns: loss_inf, loss_B, loss_amp, info
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

    # ---- Determine B_net based on amplitude mode ----
    amplitude_mode = str(amplitude_mode).lower()
    if amplitude_mode not in ("spectral_correction", "learned_absolute", "hybrid_warmstart"):
        raise ValueError(f"Unsupported amplitude_mode={amplitude_mode}")

    Binc_spec = Binc_spec.to(device=device, dtype=cdtype)
    Bref_spec = Bref_spec.to(device=device, dtype=cdtype)

    loss_B = torch.zeros((), device=device, dtype=dtype)
    loss_amp = torch.zeros((), device=device, dtype=dtype)
    amp_info = {}

    if amplitude_mode == "spectral_correction":
        dB_inc, dB_ref = model.predict_asymptotic_delta(
            a_batch, omega_batch, u=u_batch, v=v_batch
        )
        Binc_net = Binc_spec * dB_inc
        Bref_net = Bref_spec * dB_ref

        a_1 = a_batch if a_batch.ndim == 2 else a_batch.unsqueeze(-1)
        omega_1 = omega_batch if omega_batch.ndim == 2 else omega_batch.unsqueeze(-1)
        u_1 = u_batch if u_batch.ndim == 2 else u_batch.unsqueeze(-1)
        v_1 = v_batch if v_batch.ndim == 2 else v_batch.unsqueeze(-1)
        feats_reg, _, _ = model._build_param_features(a=a_1, omega=omega_1, u=u_1, v=v_1)
        raw_out = model.amp_head(feats_reg)
        loss_B = torch.mean(raw_out[:, 0] ** 2 + raw_out[:, 1] ** 2 + raw_out[:, 2] ** 2 + raw_out[:, 3] ** 2)

        amp_info = {
            "mean_abs_dBinc": float(torch.mean(torch.abs(dB_inc.detach())).cpu().item()),
            "mean_abs_dBref": float(torch.mean(torch.abs(dB_ref.detach())).cpu().item()),
        }

    else:
        if not hasattr(model, "predict_asymptotic_amplitudes"):
            raise AttributeError("Model must implement predict_asymptotic_amplitudes.")

        Binc_pred, Bref_pred, amp_raw = model.predict_asymptotic_amplitudes(
            a_batch, omega_batch, u=u_batch, v=v_batch
        )

        loss_amp, amp_info = compute_amplitude_teacher_loss(
            Binc_pred=Binc_pred,
            Bref_pred=Bref_pred,
            amp_raw=amp_raw,
            Binc_spec=Binc_spec,
            Bref_spec=Bref_spec,
            eps=eps,
        )

        if amplitude_mode == "learned_absolute":
            Binc_net = Binc_pred
            Bref_net = Bref_pred
        else:  # hybrid_warmstart
            eta = float(hybrid_eta)
            eta = max(0.0, min(1.0, eta))
            Binc_net = (1.0 - eta) * Binc_spec + eta * Binc_pred
            Bref_net = (1.0 - eta) * Bref_spec + eta * Bref_pred

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

    info = {
        "loss_inf": float(loss_inf.detach().cpu().item()),
        "loss_B": float(loss_B.detach().cpu().item()),
        "mean_abs_Sinf": float(torch.mean(torch.abs(S_inf)).detach().cpu().item()),
        "mean_abs_Spred_inf": float(torch.mean(torch.abs(S_pred)).detach().cpu().item()),
    }
    info.update(amp_info)
    return loss_inf, loss_B, loss_amp, info
