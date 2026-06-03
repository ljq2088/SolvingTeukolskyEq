#!/usr/bin/env python3
"""Center-patch Chebyshev-coefficient PINN for S=R_in/P, s=-2.

Coordinate convention in this prototype follows the current experiment request:
    y = -1  horizon,  y = +1  infinity.

The project coefficient code uses x=r_+/r, hence x=(1-y)/2 here.  We train a
small parameter patch around the middle of (a, log10 omega):
    a ~= 0.5, log10(omega) ~= -1.5.

The ansatz hard-enforces:
    S(-1)=1,
    S_y(-1)=alpha_H,
    S_y(+1)=alpha_inf*S(+1),
where alpha is obtained from the degenerate endpoint equation D1*S_y+D0*S=0.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.compute_lambda_usage import compute_lambda  # noqa: E402
from utils.matlcheb import cheb  # noqa: E402
from physical_ansatz.prefactor import Leaver_prefactors, build_prefactor_primitives, r_plus  # noqa: E402
from physical_ansatz.transform_y import h_factor  # noqa: E402


CDTYPE = torch.complex128
RDTYPE = torch.float64


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data: dict) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def cheb_terms(y: torch.Tensor, n_terms: int) -> torch.Tensor:
    terms = [torch.ones_like(y)]
    if n_terms == 1:
        return torch.stack(terms, dim=-1)
    terms.append(y)
    for _ in range(2, n_terms):
        terms.append(2.0 * y * terms[-1] - terms[-2])
    return torch.stack(terms, dim=-1)


def endpoint_alpha(a: torch.Tensor, omega: torch.Tensor, lam: torch.Tensor, *, side: str, m: int = 2, s: int = -2) -> torch.Tensor:
    if side == "horizon":
        eps = torch.tensor([1e-6, 2e-6, 5e-6, 1e-5, 2e-5, 5e-5], device=a.device, dtype=RDTYPE)
        x = 1.0 - eps.unsqueeze(0)
        fit_var = eps
    elif side == "infinity":
        eps = torch.tensor([1e-6, 2e-6, 5e-6, 1e-5, 2e-5, 5e-5], device=a.device, dtype=RDTYPE)
        x = eps.unsqueeze(0)
        fit_var = eps
    else:
        raise ValueError(side)
    B = a.shape[0]
    x = x.expand(B, -1)
    A2, A1, A0 = coeffs_x_local(x, a, omega, lam, m=m, s=s)
    # y=(1-2x), so D1=-2*A1 and alpha_y=-D0/D1=A0/(2*A1)
    vals = A0 / (2.0 * A1)
    # Quadratic extrapolation to endpoint for each batch element.
    X = torch.stack([torch.ones_like(fit_var), fit_var, fit_var * fit_var], dim=-1).to(device=a.device, dtype=RDTYPE)
    pinv = torch.linalg.pinv(X).to(CDTYPE)
    coeff = pinv @ vals.transpose(0, 1)
    return coeff[0]


def coeffs_y(a: torch.Tensor, omega: torch.Tensor, lam: torch.Tensor, y: torch.Tensor, *, m: int = 2, s: int = -2):
    x = 0.5 * (1.0 - y)
    A2, A1, A0 = coeffs_x_local(x, a, omega, lam, m=m, s=s)
    return 4.0 * A2, -2.0 * A1, A0


def coeffs_x_local(x: torch.Tensor, a: torch.Tensor, omega: torch.Tensor, lam: torch.Tensor, *, m: int = 2, s: int = -2):
    a_b = a.unsqueeze(-1).to(RDTYPE)
    omega_b = omega.unsqueeze(-1).to(CDTYPE)
    lam_b = lam.unsqueeze(-1).to(CDTYPE)
    rp = 1.0 + torch.sqrt(torch.clamp(1.0 - a_b * a_b, min=0.0))
    rm = 1.0 - torch.sqrt(torch.clamp(1.0 - a_b * a_b, min=0.0))
    r = rp / x
    delta = r * r - 2.0 * r + a_b * a_b
    delta_r = 2.0 * r - 2.0
    K = (r * r + a_b * a_b) * omega_b - a_b * m
    V = (K * K - 2.0j * s * (r - 1.0) * K) / delta + 4.0j * s * omega_b * r - lam_b
    sigma_p = (2.0 * omega_b * rp - m * a_b) / (rp - rm)
    pp = -s - 1.0j * sigma_p
    pm = -1.0 - s + 2.0j * omega_b + 1.0j * sigma_p
    drp = r - rp
    drm = r - rm
    g = pp / drp + pm / drm + 1.0j * omega_b
    gp = -pp / (drp * drp) - pm / (drm * drm)
    dx_dr = -(x * x) / rp
    d2x_dr2 = 2.0 * x**3 / (rp * rp)
    A2 = delta * dx_dr * dx_dr
    A1 = delta * (2.0 * dx_dr * g + d2x_dr2) + (s + 1.0) * delta_r * dx_dr
    A0 = V + (s + 1.0) * delta_r * g + delta * (g * g + gp)
    return A2.to(CDTYPE), A1.to(CDTYPE), A0.to(CDTYPE)


def coeffs_y_np(a: float, omega: float, lam: complex, y: np.ndarray, *, m: int = 2, s: int = -2):
    with torch.no_grad():
        a_t = torch.tensor([a], dtype=RDTYPE)
        omega_t = torch.tensor([omega], dtype=RDTYPE)
        lam_t = torch.tensor([lam], dtype=CDTYPE)
        y_t = torch.tensor(y, dtype=RDTYPE).unsqueeze(0)
        D2, D1, D0 = coeffs_y(a_t, omega_t, lam_t, y_t, m=m, s=s)
    return (
        D2.squeeze(0).detach().cpu().numpy(),
        D1.squeeze(0).detach().cpu().numpy(),
        D0.squeeze(0).detach().cpu().numpy(),
    )


def endpoint_alpha_np(a: float, omega: float, lam: complex, *, side: str, m: int = 2, s: int = -2) -> complex:
    with torch.no_grad():
        a_t = torch.tensor([a], dtype=RDTYPE)
        omega_t = torch.tensor([omega], dtype=RDTYPE)
        lam_t = torch.tensor([lam], dtype=CDTYPE)
        alpha = endpoint_alpha(a_t, omega_t, lam_t, side=side, m=m, s=s)
    return complex(alpha.detach().cpu().numpy()[0])


def solve_spectral_ansatz_output(
    a: float,
    logw: float,
    *,
    n_cheb: int,
    n_colloc: int,
    ell: int,
    m: int,
    s: int,
    ridge: float = 1.0e-12,
) -> tuple[np.ndarray, dict]:
    """Solve directly for the network output vector using full-interval Chebyshev collocation.

    Unknown vector is exactly the model output layout:
        [S(+1), bubble*T_0 coefficient, ..., bubble*T_{n_cheb-1} coefficient].
    The Hermite ansatz enforces S(-1), S_y(-1), and S_y(+1)-alpha_inf*S(+1).
    Interior collocation rows solve the homogeneous S equation in least squares.
    """
    omega = float(10.0 ** logw)
    lam = complex(compute_lambda(float(a), omega, ell, m, s=s))
    D, y = cheb(n_colloc)
    D2mat = D @ D
    alpha_h = endpoint_alpha_np(a, omega, lam, side="horizon", m=m, s=s)
    alpha_i = endpoint_alpha_np(a, omega, lam, side="infinity", m=m, s=s)

    t = 0.5 * (y + 1.0)
    h00 = 2.0 * t**3 - 3.0 * t**2 + 1.0
    h10 = t**3 - 2.0 * t**2 + t
    h01 = -2.0 * t**3 + 3.0 * t**2
    h11 = t**3 - t**2
    base0 = h00 + h10 * (2.0 * alpha_h)
    basis_c = h01 + h11 * (2.0 * alpha_i)

    terms = [np.ones_like(y), y.copy()]
    for _ in range(2, n_cheb):
        terms.append(2.0 * y * terms[-1] - terms[-2])
    T = np.stack(terms[:n_cheb], axis=1)
    bubble = (t * (1.0 - t)) ** 2
    basis_coeff = bubble[:, None] * T
    basis = np.column_stack([basis_c, basis_coeff]).astype(np.complex128)

    base0_y = D @ base0
    base0_yy = D2mat @ base0
    basis_y = D @ basis
    basis_yy = D2mat @ basis

    interior = np.arange(1, n_colloc)
    D2c, D1c, D0c = coeffs_y_np(a, omega, lam, y[interior], m=m, s=s)
    rhs = -(D2c * base0_yy[interior] + D1c * base0_y[interior] + D0c * base0[interior])
    mat = D2c[:, None] * basis_yy[interior, :] + D1c[:, None] * basis_y[interior, :] + D0c[:, None] * basis[interior, :]

    row_norm = np.maximum(np.linalg.norm(mat, axis=1), np.abs(rhs))
    row_norm = np.where(row_norm > 1.0e-300, row_norm, 1.0)
    mat_s = mat / row_norm[:, None]
    rhs_s = rhs / row_norm
    if ridge > 0:
        reg = math.sqrt(ridge) * np.eye(n_cheb + 1, dtype=np.complex128)
        mat_s = np.vstack([mat_s, reg])
        rhs_s = np.concatenate([rhs_s, np.zeros(n_cheb + 1, dtype=np.complex128)])
    sol, residuals, rank, sing = np.linalg.lstsq(mat_s, rhs_s, rcond=None)
    full_S = base0 + basis @ sol
    full_S_y = base0_y + basis_y @ sol
    full_S_yy = base0_yy + basis_yy @ sol
    D2all, D1all, D0all = coeffs_y_np(a, omega, lam, y[1:-1], m=m, s=s)
    res = D2all * full_S_yy[1:-1] + D1all * full_S_y[1:-1] + D0all * full_S[1:-1]
    den = np.maximum.reduce([
        np.abs(D2all * full_S_yy[1:-1]),
        np.abs(D1all * full_S_y[1:-1]),
        np.abs(D0all * full_S[1:-1]),
    ])
    rel = np.abs(res) / np.maximum(den, 1.0e-300)
    diag = {
        "a": float(a),
        "omega": omega,
        "logw": float(logw),
        "rank": int(rank),
        "cond": float(sing[0] / max(sing[-1], 1.0e-300)) if sing.size else float("nan"),
        "rel_median": float(np.median(rel)),
        "rel_mean": float(np.mean(rel)),
        "rel_max": float(np.max(rel)),
        "tail": float(np.max(np.abs(sol[-min(4, n_cheb):])) / max(np.max(np.abs(sol)), 1.0e-300)),
    }
    return sol.astype(np.complex128), diag


class ParamToCheb(nn.Module):
    def __init__(self, n_cheb: int, width: int, depth: int):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = 2
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, width))
            layers.append(nn.SiLU())
            in_dim = width
        layers.append(nn.Linear(in_dim, 2 * (n_cheb + 1)))
        self.net = nn.Sequential(*layers)
        self.n_cheb = n_cheb

    def forward(self, a: torch.Tensor, logw: torch.Tensor):
        # Center/scale this local patch.
        inp = torch.stack([(a - 0.5) / 0.05, (logw + 1.5) / 0.15], dim=-1).to(RDTYPE)
        out = self.net(inp)
        c_val = out[:, 0] + 1j * out[:, 1]
        coeff_raw = out[:, 2:].reshape(a.shape[0], self.n_cheb, 2)
        coeff = coeff_raw[..., 0] + 1j * coeff_raw[..., 1]
        return c_val.to(CDTYPE), coeff.to(CDTYPE)


def hermite_base(y: torch.Tensor, c_val: torch.Tensor, alpha_h: torch.Tensor, alpha_i: torch.Tensor):
    t = 0.5 * (y + 1.0)
    h00 = 2 * t**3 - 3 * t**2 + 1
    h10 = t**3 - 2 * t**2 + t
    h01 = -2 * t**3 + 3 * t**2
    h11 = t**3 - t**2
    m0 = 2.0 * alpha_h.unsqueeze(-1)
    m1 = 2.0 * alpha_i.unsqueeze(-1) * c_val.unsqueeze(-1)
    return h00 + h10 * m0 + h01 * c_val.unsqueeze(-1) + h11 * m1


def model_S(model: ParamToCheb, a: torch.Tensor, logw: torch.Tensor, y: torch.Tensor, lam: torch.Tensor):
    omega = (10.0 ** logw).to(RDTYPE)
    c_val, coeff = model(a, logw)
    alpha_h = endpoint_alpha(a, omega, lam, side="horizon")
    alpha_i = endpoint_alpha(a, omega, lam, side="infinity")
    base = hermite_base(y, c_val, alpha_h, alpha_i)
    T = cheb_terms(y, model.n_cheb).to(CDTYPE)
    residual_shape = torch.sum(coeff.unsqueeze(1) * T, dim=-1)
    t = 0.5 * (y + 1.0)
    bubble = (t * (1.0 - t)) ** 2
    return base + bubble * residual_shape


def sample_batch(batch: int, n_y: int, *, a_half: float, logw_half: float, device: str):
    a = 0.5 + a_half * (2.0 * torch.rand(batch, device=device, dtype=RDTYPE) - 1.0)
    logw = -1.5 + logw_half * (2.0 * torch.rand(batch, device=device, dtype=RDTYPE) - 1.0)
    # Mix Chebyshev-ish deterministic points with random interior points.
    theta = torch.linspace(0.0, math.pi, n_y, device=device, dtype=RDTYPE)
    y_base = torch.cos(theta).flip(0)
    jitter = 0.02 * (2.0 * torch.rand(batch, n_y, device=device, dtype=RDTYPE) - 1.0)
    y = torch.clamp(y_base.unsqueeze(0) + jitter, -0.999, 0.999)
    y.requires_grad_(True)
    return a, logw, y


def fixed_grid_batch(n_param_side: int, n_y: int, *, a_half: float, logw_half: float, device: str):
    aa = torch.linspace(0.5 - a_half, 0.5 + a_half, n_param_side, device=device, dtype=RDTYPE)
    ll = torch.linspace(-1.5 - logw_half, -1.5 + logw_half, n_param_side, device=device, dtype=RDTYPE)
    A, L = torch.meshgrid(aa, ll, indexing="ij")
    a = A.reshape(-1)
    logw = L.reshape(-1)
    theta = torch.linspace(0.0, math.pi, n_y, device=device, dtype=RDTYPE)
    y_line = torch.cos(theta).flip(0).clamp(-0.999, 0.999)
    y = y_line.unsqueeze(0).expand(a.shape[0], -1).clone().detach().requires_grad_(True)
    return a, logw, y


def lambda_batch(a: torch.Tensor, logw: torch.Tensor, *, ell: int, m: int, s: int):
    vals = []
    for av, lv in zip(a.detach().cpu().numpy(), logw.detach().cpu().numpy()):
        vals.append(compute_lambda(float(av), float(10.0 ** lv), ell, m, s=s))
    return torch.tensor(vals, dtype=CDTYPE, device=a.device)


def residual_loss(model: ParamToCheb, a: torch.Tensor, logw: torch.Tensor, y: torch.Tensor, lam: torch.Tensor, *, m: int, s: int):
    omega = (10.0 ** logw).to(RDTYPE)
    S = model_S(model, a, logw, y, lam)
    ones = torch.ones_like(S.real)
    Sy_re = torch.autograd.grad(S.real, y, grad_outputs=ones, create_graph=True, retain_graph=True)[0]
    Sy_im = torch.autograd.grad(S.imag, y, grad_outputs=ones, create_graph=True, retain_graph=True)[0]
    Syy_re = torch.autograd.grad(Sy_re, y, grad_outputs=ones, create_graph=True, retain_graph=True)[0]
    Syy_im = torch.autograd.grad(Sy_im, y, grad_outputs=ones, create_graph=True)[0]
    Sy = Sy_re.to(CDTYPE) + 1.0j * Sy_im.to(CDTYPE)
    Syy = Syy_re.to(CDTYPE) + 1.0j * Syy_im.to(CDTYPE)
    D2, D1, D0 = coeffs_y(a, omega, lam, y, m=m, s=s)
    res = D2 * Syy + D1 * Sy + D0 * S
    den = torch.maximum(torch.maximum(torch.abs(D2 * Syy), torch.abs(D1 * Sy)), torch.abs(D0 * S)).clamp_min(1e-24)
    rel = torch.abs(res) / den
    tail = torch.mean(torch.abs(model(a, logw)[1][:, -4:]) ** 2)
    return torch.mean(rel * rel), {
        "rel_mean": float(rel.detach().mean().cpu()),
        "rel_median": float(rel.detach().median().cpu()),
        "rel_max": float(rel.detach().max().cpu()),
        "tail": float(tail.detach().cpu()),
    }, tail


def evaluate(model, *, device, args, n_param=9, n_y=128):
    aa = torch.linspace(0.5 - args.a_half_width, 0.5 + args.a_half_width, n_param, device=device, dtype=RDTYPE)
    ll = torch.linspace(-1.5 - args.logw_half_width, -1.5 + args.logw_half_width, n_param, device=device, dtype=RDTYPE)
    rows = []
    for a in aa:
        for logw in ll:
            y = torch.linspace(-0.999, 0.999, n_y, device=device, dtype=RDTYPE).unsqueeze(0)
            y.requires_grad_(True)
            ab = a.reshape(1)
            lb = logw.reshape(1)
            lam = lambda_batch(ab, lb, ell=args.ell, m=args.m, s=args.s)
            _, info, _ = residual_loss(model, ab, lb, y, lam, m=args.m, s=args.s)
            rows.append({"a": float(a.cpu()), "omega": float((10.0**logw).cpu()), **info})
    return rows


def make_summary(args, log_rows: list[dict], eval_rows: list[dict], benchmark_rows: list[dict], step: int) -> dict:
    summary = {
        "args": vars(args),
        "step": int(step),
        "n_train_log_rows": len(log_rows),
        "n_eval_rows": len(eval_rows),
        "n_benchmark_rows": len(benchmark_rows),
    }
    if log_rows:
        summary["latest_train"] = log_rows[-1]
    if eval_rows:
        summary.update({
            "eval_rel_median_median": float(np.median([r["rel_median"] for r in eval_rows])),
            "eval_rel_mean_median": float(np.median([r["rel_mean"] for r in eval_rows])),
            "eval_rel_max_max": float(np.max([r["rel_max"] for r in eval_rows])),
        })
    ok_bench = [r for r in benchmark_rows if r.get("status") == "ok"]
    if ok_bench:
        summary.update({
            "benchmark_median_rel_err_R": float(np.median([r["medR"] for r in ok_bench])),
            "benchmark_max_rel_err_R": float(np.max([r["maxR"] for r in ok_bench])),
            "latest_benchmark": ok_bench[-1],
        })
    return summary


def save_checkpoint(run_dir: Path, model: ParamToCheb, args, log_rows: list[dict], eval_rows: list[dict], benchmark_rows: list[dict], step: int) -> None:
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "args": vars(args),
        "step": int(step),
        "train_log": log_rows,
        "eval_rows": eval_rows,
        "benchmark_rows": benchmark_rows,
    }
    torch.save(payload, ckpt_dir / "latest.pt")
    torch.save(payload, ckpt_dir / f"step_{int(step):06d}.pt")


def plot_training_curves(run_dir: Path, log_rows: list[dict]) -> None:
    if not log_rows:
        return
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = run_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    steps = np.asarray([r["step"] for r in log_rows], dtype=float)
    fig, ax = plt.subplots(figsize=(8, 5))
    for key in ["loss", "rel_median", "rel_mean", "rel_max", "tail"]:
        if key in log_rows[0]:
            vals = np.asarray([r.get(key, np.nan) for r in log_rows], dtype=float)
            ax.semilogy(steps, vals, label=key)
    ax.set_xlabel("step")
    ax.set_ylabel("value")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "training_curves.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_pybhpt_benchmark(model: ParamToCheb, run_dir: Path, args, *, device: str, step: int) -> dict:
    """Generate the old 3x2 pybhpt comparison figure for the patch center."""
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pybhpt_usage.compute_solution import compute_pybhpt_solution

    fig_dir = run_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    a_scalar = 0.5
    omega_scalar = 10.0 ** -1.5
    a_t = torch.tensor([a_scalar], device=device, dtype=RDTYPE)
    logw_t = torch.tensor([math.log10(omega_scalar)], device=device, dtype=RDTYPE)
    omega_t = torch.tensor([omega_scalar], device=device, dtype=RDTYPE)
    lam = lambda_batch(a_t, logw_t, ell=args.ell, m=args.m, s=args.s)
    h2 = h_factor(a_t, omega_t, m=args.m, M=1.0, s=args.s)

    rp = r_plus(a_t, 1.0)
    rp_scalar = float(rp.detach().cpu().item())
    r_min = max(2.0, rp_scalar + 1e-4)
    r_max = 1000.0
    n_pts = 400

    r_grid = torch.linspace(r_min, r_max, n_pts, device=device, dtype=RDTYPE)
    old_y_from_r = 2.0 * rp / r_grid - 1.0
    new_y_from_r = -old_y_from_r

    old_y_min = 2.0 * rp_scalar / r_max - 1.0
    old_y_max = 2.0 * rp_scalar / r_min - 1.0
    theta = torch.linspace(0.0, math.pi, n_pts, device=device, dtype=RDTYPE)
    old_y_grid = 0.5 * (old_y_max + old_y_min) + 0.5 * (old_y_max - old_y_min) * torch.cos(theta)
    old_y_grid = torch.sort(old_y_grid).values
    x_from_old_y = 0.5 * (old_y_grid + 1.0)
    r_from_old_y = rp / x_from_old_y
    new_y_grid = -old_y_grid

    with torch.no_grad():
        S_pred_r = model_S(model, a_t, logw_t, new_y_from_r.unsqueeze(0), lam).squeeze(0)
        S_pred_y = model_S(model, a_t, logw_t, new_y_grid.unsqueeze(0), lam).squeeze(0)
        rp_r, rm_r, _, _, _ = build_prefactor_primitives(r_grid, a_t, M=1.0, need_rs=False)
        P_r, _, _ = Leaver_prefactors(r_grid, a_t, omega_t, m=args.m, M=1.0, s=args.s, rp=rp_r, rm=rm_r)
        rp_y, rm_y, _, _, _ = build_prefactor_primitives(r_from_old_y, a_t, M=1.0, need_rs=False)
        P_y, _, _ = Leaver_prefactors(r_from_old_y, a_t, omega_t, m=args.m, M=1.0, s=args.s, rp=rp_y, rm=rm_y)
        R_pred_r = P_r * h2 * S_pred_r

    r_np = r_grid.detach().cpu().numpy()
    old_y_np = old_y_grid.detach().cpu().numpy()
    r_y_np = r_from_old_y.detach().cpu().numpy()
    R_pred_r_np = R_pred_r.detach().cpu().numpy()
    S_pred_y_np = S_pred_y.detach().cpu().numpy()

    benchmark_available = False
    benchmark_status = "benchmark=pybhpt-failed"
    R_ref_r_np = None
    S_ref_y_np = None
    row = {
        "step": int(step),
        "a": a_scalar,
        "omega": omega_scalar,
        "status": "failed",
        "medR": np.nan,
        "maxR": np.nan,
    }
    try:
        r_np_sorted = np.sort(r_np)
        order_r = np.argsort(r_np)
        inv_order_r = np.empty_like(order_r)
        inv_order_r[order_r] = np.arange(order_r.size)
        r_y_sorted = np.sort(r_y_np)
        order_y = np.argsort(r_y_np)
        inv_order_y = np.empty_like(order_y)
        inv_order_y[order_y] = np.arange(order_y.size)

        _, R_ref_sorted = compute_pybhpt_solution(a_scalar, omega_scalar, ell=args.ell, m=args.m, r_grid=r_np_sorted, timeout=30.0)
        R_ref_r_np = np.asarray(R_ref_sorted, dtype=np.complex128)[inv_order_r]
        _, R_ref_y_sorted = compute_pybhpt_solution(a_scalar, omega_scalar, ell=args.ell, m=args.m, r_grid=r_y_sorted, timeout=30.0)
        R_ref_y_np = np.asarray(R_ref_y_sorted, dtype=np.complex128)[inv_order_y]
        S_ref_y_np = R_ref_y_np / (P_y.detach().cpu().numpy() * complex(h2.detach().cpu().item()))
        rel_err = np.abs(R_pred_r_np - R_ref_r_np) / (np.abs(R_ref_r_np) + 1e-14)
        row.update({"status": "ok", "medR": float(np.median(rel_err)), "maxR": float(np.max(rel_err))})
        benchmark_status = f"benchmark=pybhpt medR={row['medR']:.2e} maxR={row['maxR']:.2e}"
        benchmark_available = True
    except Exception as exc:
        row["error"] = str(exc)
        benchmark_status = f"benchmark=pybhpt-failed: {exc}"

    fig, axes = plt.subplots(3, 2, figsize=(10, 10), sharex=False)
    for ax_row, label, pred_data, ref_data in [
        (0, "Re(R)", np.real(R_pred_r_np), np.real(R_ref_r_np) if benchmark_available else None),
        (1, "Im(R)", np.imag(R_pred_r_np), np.imag(R_ref_r_np) if benchmark_available else None),
        (2, "|R|", np.abs(R_pred_r_np), np.abs(R_ref_r_np) if benchmark_available else None),
    ]:
        axes[ax_row, 0].plot(r_np, pred_data, label=f"Pred {label}", lw=1.6)
        if ref_data is not None:
            axes[ax_row, 0].plot(r_np, ref_data, "--", label=f"ref {label}", lw=1.0)
        axes[ax_row, 0].set_ylabel(label)
        axes[ax_row, 0].legend()
        axes[ax_row, 0].grid(alpha=0.3)
    axes[2, 0].set_xlabel("r")

    for ax_row, label, pred_data, ref_data in [
        (0, "Re(S)", np.real(S_pred_y_np), np.real(S_ref_y_np) if benchmark_available else None),
        (1, "Im(S)", np.imag(S_pred_y_np), np.imag(S_ref_y_np) if benchmark_available else None),
        (2, "|S|", np.abs(S_pred_y_np), np.abs(S_ref_y_np) if benchmark_available else None),
    ]:
        axes[ax_row, 1].plot(old_y_np, pred_data, label=f"Pred {label}", lw=1.6)
        if ref_data is not None:
            axes[ax_row, 1].plot(old_y_np, ref_data, "--", label=f"ref {label}", lw=1.0)
        axes[ax_row, 1].set_ylabel(label)
        axes[ax_row, 1].legend()
        axes[ax_row, 1].grid(alpha=0.3)
    axes[2, 1].set_xlabel("y")

    fig.suptitle(
        f"patch=local-center, step={step}, a={a_scalar:.6f}, "
        f"omega={omega_scalar:.6f}, {benchmark_status}",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(fig_dir / f"step_{int(step):06d}_ref.png", dpi=160, bbox_inches="tight")
    fig.savefig(fig_dir / "latest_ref.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    return row


def save_artifacts(run_dir: Path, model: ParamToCheb, args, log_rows: list[dict], eval_rows: list[dict], benchmark_rows: list[dict], step: int, *, checkpoint: bool) -> None:
    write_csv(run_dir / "train_log.csv", log_rows)
    write_csv(run_dir / "eval_residuals.csv", eval_rows)
    write_csv(run_dir / "pybhpt_benchmark.csv", benchmark_rows)
    write_json(run_dir / "summary.json", make_summary(args, log_rows, eval_rows, benchmark_rows, step))
    plot_training_curves(run_dir, log_rows)
    if checkpoint:
        save_checkpoint(run_dir, model, args, log_rows, eval_rows, benchmark_rows, step)


def complex_output_to_real_vector(z: np.ndarray) -> np.ndarray:
    out = np.empty(2 * z.shape[0], dtype=np.float64)
    out[0::2] = z.real
    out[1::2] = z.imag
    return out


def spectral_initialize_model(model: ParamToCheb, args, run_dir: Path, *, device: str) -> list[dict]:
    center_out, center_diag = solve_spectral_ansatz_output(
        0.5,
        -1.5,
        n_cheb=args.n_cheb,
        n_colloc=args.spectral_init_colloc,
        ell=args.ell,
        m=args.m,
        s=args.s,
        ridge=args.spectral_init_ridge,
    )
    last = None
    for module in reversed(model.net):
        if isinstance(module, nn.Linear):
            last = module
            break
    if last is None:
        raise RuntimeError("ParamToCheb has no Linear output layer")
    with torch.no_grad():
        last.weight.zero_()
        last.bias.copy_(torch.tensor(complex_output_to_real_vector(center_out), dtype=RDTYPE, device=device))

    rows = [{**center_diag, "kind": "center_bias"}]
    if args.spectral_pretrain_steps <= 0:
        write_csv(run_dir / "spectral_init.csv", rows)
        return rows

    aa = np.linspace(0.5 - args.a_half_width, 0.5 + args.a_half_width, args.spectral_init_param_side)
    ll = np.linspace(-1.5 - args.logw_half_width, -1.5 + args.logw_half_width, args.spectral_init_param_side)
    inputs = []
    targets = []
    for av in aa:
        for lv in ll:
            sol, diag = solve_spectral_ansatz_output(
                float(av),
                float(lv),
                n_cheb=args.n_cheb,
                n_colloc=args.spectral_init_colloc,
                ell=args.ell,
                m=args.m,
                s=args.s,
                ridge=args.spectral_init_ridge,
            )
            rows.append({**diag, "kind": "pretrain_target"})
            inputs.append((float(av), float(lv)))
            targets.append(complex_output_to_real_vector(sol))
    write_csv(run_dir / "spectral_init.csv", rows)

    a_train = torch.tensor([p[0] for p in inputs], dtype=RDTYPE, device=device)
    logw_train = torch.tensor([p[1] for p in inputs], dtype=RDTYPE, device=device)
    target = torch.tensor(np.stack(targets), dtype=RDTYPE, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.spectral_pretrain_lr, weight_decay=0.0)
    pre_rows = []
    for step in range(1, args.spectral_pretrain_steps + 1):
        c_val, coeff = model(a_train, logw_train)
        pred = torch.empty_like(target)
        pred[:, 0] = c_val.real
        pred[:, 1] = c_val.imag
        pred[:, 2::2] = coeff.real
        pred[:, 3::2] = coeff.imag
        scale = target.abs().detach().mean(dim=0).clamp_min(1.0)
        loss = torch.mean(((pred - target) / scale) ** 2)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step == 1 or step % max(1, args.spectral_pretrain_steps // 10) == 0:
            row = {"step": step, "loss": float(loss.detach().cpu())}
            pre_rows.append(row)
            print(f"spectral-pretrain={step} loss={row['loss']:.3e}", flush=True)
    write_csv(run_dir / "spectral_pretrain_log.csv", pre_rows)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--n-y", type=int, default=96)
    parser.add_argument("--n-cheb", type=int, default=24)
    parser.add_argument("--width", type=int, default=96)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--lbfgs-steps", type=int, default=0)
    parser.add_argument("--lbfgs-param-side", type=int, default=5)
    parser.add_argument("--a-half-width", type=float, default=0.03)
    parser.add_argument("--logw-half-width", type=float, default=0.12)
    parser.add_argument("--ell", type=int, default=2)
    parser.add_argument("--m", type=int, default=2)
    parser.add_argument("--s", type=int, default=-2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", default="outputs/center_patch_cheb_pinn")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--viz-every", type=int, default=500)
    parser.add_argument("--spectral-init", action="store_true")
    parser.add_argument("--spectral-init-colloc", type=int, default=80)
    parser.add_argument("--spectral-init-param-side", type=int, default=5)
    parser.add_argument("--spectral-init-ridge", type=float, default=1e-12)
    parser.add_argument("--spectral-pretrain-steps", type=int, default=0)
    parser.add_argument("--spectral-pretrain-lr", type=float, default=2e-3)
    args = parser.parse_args()

    torch.set_default_dtype(RDTYPE)
    run_dir = Path(args.output_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "figures").mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "config.json", vars(args))
    device = args.device
    model = ParamToCheb(args.n_cheb, args.width, args.depth).to(device=device, dtype=RDTYPE)
    spectral_rows = []
    if args.spectral_init:
        print("Running full-interval spectral output initialization...", flush=True)
        spectral_rows = spectral_initialize_model(model, args, run_dir, device=device)
        print(
            "spectral-init center "
            f"rel_med={spectral_rows[0]['rel_median']:.3e} "
            f"rel_mean={spectral_rows[0]['rel_mean']:.3e} "
            f"rel_max={spectral_rows[0]['rel_max']:.3e}",
            flush=True,
        )
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-6)
    log_rows = []
    eval_rows = []
    benchmark_rows = []
    if args.spectral_init:
        benchmark_rows.append(plot_pybhpt_benchmark(model, run_dir, args, device=device, step=0))
        eval_rows = evaluate(model, device=device, args=args)
        save_artifacts(run_dir, model, args, log_rows, eval_rows, benchmark_rows, 0, checkpoint=True)
    for step in range(1, args.steps + 1):
        a, logw, y = sample_batch(args.batch, args.n_y, a_half=args.a_half_width, logw_half=args.logw_half_width, device=device)
        lam = lambda_batch(a, logw, ell=args.ell, m=args.m, s=args.s)
        loss_res, info, tail = residual_loss(model, a, logw, y, lam, m=args.m, s=args.s)
        loss = loss_res + 1e-4 * tail
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step()
        if step == 1 or step % args.log_every == 0:
            row = {"phase": "adam", "step": step, "loss": float(loss.detach().cpu()), **info}
            log_rows.append(row)
            print(f"step={step} loss={row['loss']:.3e} rel_med={row['rel_median']:.3e} rel_mean={row['rel_mean']:.3e} rel_max={row['rel_max']:.3e}", flush=True)
        if step == 1 or step % args.save_every == 0:
            save_artifacts(run_dir, model, args, log_rows, eval_rows, benchmark_rows, step, checkpoint=True)
        if step % args.eval_every == 0:
            eval_rows = evaluate(model, device=device, args=args)
            save_artifacts(run_dir, model, args, log_rows, eval_rows, benchmark_rows, step, checkpoint=False)
        if step == 1 or step % args.viz_every == 0:
            benchmark_rows.append(plot_pybhpt_benchmark(model, run_dir, args, device=device, step=step))
            save_artifacts(run_dir, model, args, log_rows, eval_rows, benchmark_rows, step, checkpoint=True)

    if args.lbfgs_steps > 0:
        a_fix, logw_fix, y_fix = fixed_grid_batch(
            args.lbfgs_param_side,
            args.n_y,
            a_half=args.a_half_width,
            logw_half=args.logw_half_width,
            device=device,
        )
        lam_fix = lambda_batch(a_fix, logw_fix, ell=args.ell, m=args.m, s=args.s)
        lbfgs = torch.optim.LBFGS(
            model.parameters(),
            lr=0.5,
            max_iter=1,
            history_size=50,
            line_search_fn="strong_wolfe",
        )
        for k in range(1, args.lbfgs_steps + 1):
            def closure():
                lbfgs.zero_grad(set_to_none=True)
                loss_res, _, tail = residual_loss(model, a_fix, logw_fix, y_fix, lam_fix, m=args.m, s=args.s)
                loss = loss_res + 1e-4 * tail
                loss.backward()
                return loss

            loss = lbfgs.step(closure)
            if k == 1 or k % 20 == 0:
                loss_res, info, tail = residual_loss(model, a_fix, logw_fix, y_fix, lam_fix, m=args.m, s=args.s)
                global_step = args.steps + k
                row = {"phase": "lbfgs", "step": global_step, "loss": float((loss_res + 1e-4 * tail).detach().cpu()), **info}
                log_rows.append(row)
                print(f"lbfgs={k} loss={row['loss']:.3e} rel_med={row['rel_median']:.3e} rel_mean={row['rel_mean']:.3e} rel_max={row['rel_max']:.3e}", flush=True)
            global_step = args.steps + k
            if k == 1 or global_step % args.save_every == 0:
                save_artifacts(run_dir, model, args, log_rows, eval_rows, benchmark_rows, global_step, checkpoint=True)
            if k % args.eval_every == 0:
                eval_rows = evaluate(model, device=device, args=args)
                save_artifacts(run_dir, model, args, log_rows, eval_rows, benchmark_rows, global_step, checkpoint=False)
            if k == 1 or k % args.viz_every == 0:
                benchmark_rows.append(plot_pybhpt_benchmark(model, run_dir, args, device=device, step=global_step))
                save_artifacts(run_dir, model, args, log_rows, eval_rows, benchmark_rows, global_step, checkpoint=True)

    final_step = args.steps + args.lbfgs_steps
    eval_rows = evaluate(model, device=device, args=args)
    benchmark_rows.append(plot_pybhpt_benchmark(model, run_dir, args, device=device, step=final_step))
    torch.save({"model": model.state_dict(), "args": vars(args), "step": final_step}, run_dir / "model.pt")
    save_artifacts(run_dir, model, args, log_rows, eval_rows, benchmark_rows, final_step, checkpoint=True)
    summary = make_summary(args, log_rows, eval_rows, benchmark_rows, final_step)
    print(f"Saved to {run_dir}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
