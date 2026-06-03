#!/usr/bin/env python3
"""Spectral-PINN test on the horizon-side local domain.

This is not a direct linear spectral solve.  The trainable variables are the
complex values of S=R_in/(P*h) on Chebyshev-Gauss-Lobatto nodes.  Derivatives
are computed by Chebyshev differentiation matrices and the loss is the relative
Teukolsky residual plus horizon boundary losses.

Physical coordinate:
    y = 2 r_+ / r - 1, horizon y=+1, infinity y=-1.
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

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.test_horizon_local_spectral import (  # noqa: E402
    coeffs_physical_y_np,
    horizon_slope_physical_y,
    pybhpt_reference_S,
    solve_horizon_local,
)
from scripts.train_center_patch_cheb_pinn import compute_lambda  # noqa: E402
from utils.matlcheb import cheb  # noqa: E402


CDTYPE = torch.complex128
RDTYPE = torch.float64


def build_domain(y_match: float, n: int, device: str):
    D_ref, t = cheb(n)
    y_left = float(y_match)
    y_right = 1.0
    half = 0.5 * (y_right - y_left)
    mid = 0.5 * (y_right + y_left)
    y = mid + half * t
    D1 = D_ref / half
    D2 = D1 @ D1
    return (
        torch.tensor(y, dtype=RDTYPE, device=device),
        torch.tensor(D1, dtype=RDTYPE, device=device),
        torch.tensor(D2, dtype=RDTYPE, device=device),
    )


def make_initial_values(y: torch.Tensor, slope_h: complex, mode: str):
    if mode == "zero":
        S0 = torch.ones_like(y, dtype=CDTYPE)
    elif mode == "taylor":
        # y[0]=1.  Linear horizon Taylor profile.
        slope = torch.tensor(slope_h, dtype=CDTYPE, device=y.device)
        S0 = 1.0 + slope * (y.to(CDTYPE) - 1.0)
    elif mode == "random":
        S0 = torch.ones_like(y, dtype=CDTYPE)
        S0 = S0 + 1e-2 * (torch.randn_like(y).to(CDTYPE) + 1j * torch.randn_like(y).to(CDTYPE))
    else:
        raise ValueError(mode)
    return S0


def spectral_pinn_loss(
    S_param,
    D1,
    D2,
    D2c,
    D1c,
    D0c,
    slope_h,
    *,
    bc_weight: float,
    endpoint_weight: float,
    loss_mode: str,
    linear_scale: torch.Tensor | None,
):
    S = S_param
    Sy = D1.to(CDTYPE) @ S
    Syy = D2.to(CDTYPE) @ S
    idx = torch.arange(1, S.shape[0], device=S.device)
    res = D2c * Syy[idx] + D1c * Sy[idx] + D0c * S[idx]
    den = torch.maximum(
        torch.maximum(torch.abs(D2c * Syy[idx]), torch.abs(D1c * Sy[idx])),
        torch.abs(D0c * S[idx]),
    ).clamp_min(1e-30)
    rel = torch.abs(res) / den
    if loss_mode == "relative":
        residual_loss = torch.mean(rel * rel)
    elif loss_mode == "linear":
        if linear_scale is None:
            raise ValueError("linear_scale is required for loss_mode='linear'")
        scaled = res / linear_scale.to(CDTYPE)
        residual_loss = torch.mean(torch.abs(scaled) ** 2)
    else:
        raise ValueError(loss_mode)
    slope = torch.tensor(slope_h, dtype=CDTYPE, device=S.device)
    bc_val = torch.abs(S[0] - 1.0) ** 2
    bc_der = torch.abs(Sy[0] - slope) ** 2
    # Keep match-side values bounded; this avoids Adam exploiting huge values
    # when relative residual terms are locally insensitive.
    endpoint_reg = torch.abs(S[-1]) ** 2
    loss = residual_loss + bc_weight * (bc_val + bc_der) + endpoint_weight * endpoint_reg
    info = {
        "loss": float(loss.detach().cpu()),
        "residual_loss": float(residual_loss.detach().cpu()),
        "rel_median": float(rel.detach().median().cpu()),
        "rel_mean": float(rel.detach().mean().cpu()),
        "rel_max": float(rel.detach().max().cpu()),
        "bc_val": float(bc_val.detach().cpu()),
        "bc_der": float(bc_der.detach().cpu()),
        "match_abs": float(torch.abs(S[-1]).detach().cpu()),
    }
    return loss, info


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_linear_scale(D1, D2, D2c, D1c, D0c):
    idx = torch.arange(1, D1.shape[0], device=D1.device)
    op = D2c[:, None] * D2[idx, :].to(CDTYPE) + D1c[:, None] * D1[idx, :].to(CDTYPE)
    eye_cols = idx
    op[torch.arange(idx.numel(), device=D1.device), eye_cols] += D0c
    return torch.linalg.norm(op, dim=1).clamp_min(1.0)


def plot_result(run_dir: Path, y_np, S_np, S_ref_np, rel_S, log_rows, summary):
    fig_dir = run_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
    axes[0].plot(y_np, S_np.real, label="spectral-PINN Re(S)")
    axes[0].plot(y_np, S_ref_np.real, "--", label="pybhpt Re(S)")
    axes[0].set_ylabel("Re(S)")
    axes[0].grid(alpha=0.3)
    axes[0].legend()
    axes[1].plot(y_np, S_np.imag, label="spectral-PINN Im(S)")
    axes[1].plot(y_np, S_ref_np.imag, "--", label="pybhpt Im(S)")
    axes[1].set_ylabel("Im(S)")
    axes[1].grid(alpha=0.3)
    axes[1].legend()
    axes[2].semilogy(y_np, rel_S, label="|S-S_ref|/|S_ref|")
    axes[2].set_xlabel("physical y=2r+/r-1")
    axes[2].set_ylabel("rel err")
    axes[2].grid(alpha=0.3)
    axes[2].legend()
    fig.suptitle(
        f"spectral-PINN horizon local, medS={summary['rel_S_median_vs_pybhpt']:.2e}, "
        f"medRes={summary['final_rel_median']:.2e}"
    )
    fig.tight_layout()
    fig.savefig(fig_dir / "horizon_spectral_pinn_vs_pybhpt.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    if log_rows:
        fig, ax = plt.subplots(figsize=(8, 5))
        steps = np.array([r["step"] for r in log_rows], dtype=float)
        for key in ["loss", "residual_loss", "rel_median", "rel_mean", "rel_max", "bc_val", "bc_der"]:
            ax.semilogy(steps, [r[key] for r in log_rows], label=key)
        ax.set_xlabel("step")
        ax.set_ylabel("value")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(fig_dir / "training_curves.png", dpi=160, bbox_inches="tight")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", type=float, default=0.5)
    parser.add_argument("--omega", type=float, default=10.0**-1.5)
    parser.add_argument("--ell", type=int, default=2)
    parser.add_argument("--m", type=int, default=2)
    parser.add_argument("--s", type=int, default=-2)
    parser.add_argument("--y-match", type=float, default=0.6)
    parser.add_argument("--n", type=int, default=80)
    parser.add_argument("--adam-steps", type=int, default=2000)
    parser.add_argument("--lbfgs-steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--bc-weight", type=float, default=1e4)
    parser.add_argument("--endpoint-weight", type=float, default=0.0)
    parser.add_argument("--loss-mode", choices=["relative", "linear"], default="relative")
    parser.add_argument("--init", choices=["taylor", "zero", "random", "direct"], default="taylor")
    parser.add_argument("--init-noise", type=float, default=0.0)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--horizon-exclude", type=float, default=1e-3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", default="outputs/horizon_local_spectral_pinn")
    args = parser.parse_args()

    torch.set_default_dtype(RDTYPE)
    run_dir = Path(args.output_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    lam = complex(compute_lambda(args.a, args.omega, args.ell, args.m, s=args.s))
    slope_h = horizon_slope_physical_y(args.a, args.omega, lam, m=args.m, s=args.s)
    y, D1, D2 = build_domain(args.y_match, args.n, args.device)
    D2_np, D1_np, D0_np = coeffs_physical_y_np(
        args.a, args.omega, lam, y.detach().cpu().numpy()[1:], m=args.m, s=args.s
    )
    D2c = torch.tensor(D2_np, dtype=CDTYPE, device=args.device)
    D1c = torch.tensor(D1_np, dtype=CDTYPE, device=args.device)
    D0c = torch.tensor(D0_np, dtype=CDTYPE, device=args.device)
    linear_scale = build_linear_scale(D1, D2, D2c, D1c, D0c) if args.loss_mode == "linear" else None

    if args.init == "direct":
        direct = solve_horizon_local(
            a=args.a,
            omega=args.omega,
            ell=args.ell,
            m=args.m,
            s=args.s,
            y_match=args.y_match,
            n=args.n,
        )
        S_init = torch.tensor(direct["S"], dtype=CDTYPE, device=args.device)
    else:
        S_init = make_initial_values(y, slope_h, args.init)
    if args.init_noise > 0.0:
        noise = torch.randn_like(S_init.real).to(CDTYPE) + 1j * torch.randn_like(S_init.real).to(CDTYPE)
        S_init = S_init * (1.0 + args.init_noise * noise)
    S_param = torch.nn.Parameter(S_init.clone())
    opt = torch.optim.Adam([S_param], lr=args.lr)
    log_rows = []

    for step in range(1, args.adam_steps + 1):
        loss, info = spectral_pinn_loss(
            S_param,
            D1,
            D2,
            D2c,
            D1c,
            D0c,
            slope_h,
            bc_weight=args.bc_weight,
            endpoint_weight=args.endpoint_weight,
            loss_mode=args.loss_mode,
            linear_scale=linear_scale,
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step == 1 or step % args.log_every == 0:
            row = {"phase": "adam", "step": step, **info}
            log_rows.append(row)
            print(
                f"step={step} loss={row['loss']:.3e} rel_med={row['rel_median']:.3e} "
                f"rel_mean={row['rel_mean']:.3e} rel_max={row['rel_max']:.3e}",
                flush=True,
            )

    if args.lbfgs_steps > 0:
        lbfgs = torch.optim.LBFGS([S_param], lr=0.5, max_iter=1, history_size=50, line_search_fn="strong_wolfe")
        for k in range(1, args.lbfgs_steps + 1):
            def closure():
                lbfgs.zero_grad(set_to_none=True)
                loss, _ = spectral_pinn_loss(
                    S_param, D1, D2, D2c, D1c, D0c, slope_h,
                    bc_weight=args.bc_weight,
                    endpoint_weight=args.endpoint_weight,
                    loss_mode=args.loss_mode,
                    linear_scale=linear_scale,
                )
                loss.backward()
                return loss

            lbfgs.step(closure)
            if k == 1 or k % max(1, args.log_every // 5) == 0:
                _, info = spectral_pinn_loss(
                    S_param, D1, D2, D2c, D1c, D0c, slope_h,
                    bc_weight=args.bc_weight,
                    endpoint_weight=args.endpoint_weight,
                    loss_mode=args.loss_mode,
                    linear_scale=linear_scale,
                )
                row = {"phase": "lbfgs", "step": args.adam_steps + k, **info}
                log_rows.append(row)
                print(
                    f"lbfgs={k} loss={row['loss']:.3e} rel_med={row['rel_median']:.3e} "
                    f"rel_mean={row['rel_mean']:.3e} rel_max={row['rel_max']:.3e}",
                    flush=True,
                )

    _, final_info = spectral_pinn_loss(
        S_param,
        D1,
        D2,
        D2c,
        D1c,
        D0c,
        slope_h,
        bc_weight=args.bc_weight,
        endpoint_weight=args.endpoint_weight,
        loss_mode=args.loss_mode,
        linear_scale=linear_scale,
    )
    y_np = y.detach().cpu().numpy()
    S_np = S_param.detach().cpu().numpy()
    r_np, _, S_ref_np, compare_mask = pybhpt_reference_S(
        a=args.a,
        omega=args.omega,
        ell=args.ell,
        m=args.m,
        s=args.s,
        y=y_np,
        horizon_exclude=args.horizon_exclude,
    )
    valid = compare_mask & np.isfinite(S_ref_np.real) & np.isfinite(S_ref_np.imag)
    rel_S = np.full(y_np.shape, np.nan, dtype=float)
    rel_S[valid] = np.abs(S_np[valid] - S_ref_np[valid]) / np.maximum(np.abs(S_ref_np[valid]), 1e-14)

    profile_rows = []
    for yi, ri, si, sr, err in zip(y_np, r_np, S_np, S_ref_np, rel_S):
        profile_rows.append({
            "y": float(yi),
            "r": float(ri),
            "S_re": float(si.real),
            "S_im": float(si.imag),
            "S_ref_re": float(sr.real) if np.isfinite(sr.real) else np.nan,
            "S_ref_im": float(sr.imag) if np.isfinite(sr.imag) else np.nan,
            "rel_S": float(err) if np.isfinite(err) else np.nan,
        })
    write_csv(run_dir / "train_log.csv", log_rows)
    write_csv(run_dir / "profile.csv", profile_rows)
    torch.save({"S": S_param.detach().cpu(), "args": vars(args), "lambda": lam}, run_dir / "spectral_pinn.pt")

    summary = {
        "args": vars(args),
        "lambda": [lam.real, lam.imag],
        "horizon_slope_physical_y": [slope_h.real, slope_h.imag],
        "final_loss": final_info["loss"],
        "final_residual_loss": final_info["residual_loss"],
        "final_rel_median": final_info["rel_median"],
        "final_rel_mean": final_info["rel_mean"],
        "final_rel_max": final_info["rel_max"],
        "final_bc_val": final_info["bc_val"],
        "final_bc_der": final_info["bc_der"],
        "n_pybhpt_compare": int(np.sum(valid)),
        "rel_S_median_vs_pybhpt": float(np.nanmedian(rel_S)),
        "rel_S_mean_vs_pybhpt": float(np.nanmean(rel_S)),
        "rel_S_max_vs_pybhpt": float(np.nanmax(rel_S)),
    }
    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    plot_result(run_dir, y_np, S_np, S_ref_np, rel_S, log_rows, summary)
    print(f"Saved to {run_dir}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
