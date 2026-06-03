#!/usr/bin/env python3
"""Local horizon-side spectral solve for S=R_in/(P*h), s=-2.

Physical compact coordinate:
    y = 2 r_+ / r - 1
so infinity is y=-1 and the horizon is y=+1.
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

from physical_ansatz.prefactor import Leaver_prefactors, build_prefactor_primitives, r_plus  # noqa: E402
from physical_ansatz.transform_y import h_factor  # noqa: E402
from pybhpt_usage.compute_solution import compute_pybhpt_solution  # noqa: E402
from scripts.train_center_patch_cheb_pinn import (  # noqa: E402
    CDTYPE,
    RDTYPE,
    coeffs_x_local,
    compute_lambda,
    endpoint_alpha,
)
from utils.matlcheb import cheb  # noqa: E402


def coeffs_physical_y_np(a: float, omega: float, lam: complex, y: np.ndarray, *, m: int, s: int):
    """Coefficients for physical y=2x-1 with x=r_+/r."""
    with torch.no_grad():
        a_t = torch.tensor([a], dtype=RDTYPE)
        omega_t = torch.tensor([omega], dtype=RDTYPE)
        lam_t = torch.tensor([lam], dtype=CDTYPE)
        y_t = torch.tensor(y, dtype=RDTYPE).unsqueeze(0)
        x_t = 0.5 * (y_t + 1.0)
        A2, A1, A0 = coeffs_x_local(x_t, a_t, omega_t, lam_t, m=m, s=s)
    return (
        (4.0 * A2).squeeze(0).detach().cpu().numpy(),
        (2.0 * A1).squeeze(0).detach().cpu().numpy(),
        A0.squeeze(0).detach().cpu().numpy(),
    )


def horizon_slope_physical_y(a: float, omega: float, lam: complex, *, m: int, s: int) -> complex:
    """Return S_y at y=+1 in physical y=2x-1."""
    with torch.no_grad():
        a_t = torch.tensor([a], dtype=RDTYPE)
        omega_t = torch.tensor([omega], dtype=RDTYPE)
        lam_t = torch.tensor([lam], dtype=CDTYPE)
        alpha_reversed = endpoint_alpha(a_t, omega_t, lam_t, side="horizon", m=m, s=s)
    return -complex(alpha_reversed.detach().cpu().numpy()[0])


def solve_horizon_local(*, a: float, omega: float, ell: int, m: int, s: int, y_match: float, n: int):
    lam = complex(compute_lambda(a, omega, ell, m, s=s))
    D_ref, t = cheb(n)
    y_left = float(y_match)
    y_right = 1.0
    half = 0.5 * (y_right - y_left)
    mid = 0.5 * (y_right + y_left)
    y = mid + half * t
    D1 = D_ref / half
    D2 = D1 @ D1

    A = np.zeros((n + 1, n + 1), dtype=np.complex128)
    b = np.zeros(n + 1, dtype=np.complex128)

    slope_h = horizon_slope_physical_y(a, omega, lam, m=m, s=s)
    # cheb ordering maps index 0 -> y=1 horizon, index n -> y=y_match.
    A[0, 0] = 1.0
    b[0] = 1.0
    A[1, :] = D1[0, :]
    b[1] = slope_h

    idx = np.arange(2, n + 1)
    D2c, D1c, D0c = coeffs_physical_y_np(a, omega, lam, y[idx], m=m, s=s)
    A[idx, :] = D2c[:, None] * D2[idx, :] + D1c[:, None] * D1[idx, :]
    A[idx, idx] += D0c

    row = np.linalg.norm(A, axis=1)
    row = np.where(row > 1.0e-300, row, 1.0)
    Ar = A / row[:, None]
    br = b / row
    col = np.linalg.norm(Ar, axis=0)
    col = np.where(col > 1.0e-300, col, 1.0)
    As = Ar / col[None, :]
    u_scaled = np.linalg.solve(As, br)
    S = u_scaled / col
    Sy = D1 @ S
    Syy = D2 @ S

    D2all, D1all, D0all = coeffs_physical_y_np(a, omega, lam, y[1:], m=m, s=s)
    res = D2all * Syy[1:] + D1all * Sy[1:] + D0all * S[1:]
    den = np.maximum.reduce([
        np.abs(D2all * Syy[1:]),
        np.abs(D1all * Sy[1:]),
        np.abs(D0all * S[1:]),
    ])
    rel_res = np.abs(res) / np.maximum(den, 1.0e-300)
    return {
        "y": y,
        "S": S,
        "Sy": Sy,
        "Syy": Syy,
        "lam": lam,
        "slope_h": slope_h,
        "cond_scaled": float(np.linalg.cond(As)),
        "rel_res_median": float(np.median(rel_res)),
        "rel_res_mean": float(np.mean(rel_res)),
        "rel_res_max": float(np.max(rel_res)),
    }


def pybhpt_reference_S(*, a: float, omega: float, ell: int, m: int, s: int, y: np.ndarray, horizon_exclude: float):
    a_t = torch.tensor([a], dtype=RDTYPE)
    omega_t = torch.tensor([omega], dtype=RDTYPE)
    rp = float(r_plus(a_t, 1.0).detach().cpu().item())
    y_ref = np.asarray(y, dtype=float)
    safe = y_ref <= 1.0 - horizon_exclude
    r = 2.0 * rp / (y_ref + 1.0)
    R = np.full(y_ref.shape, np.nan + 1.0j * np.nan, dtype=np.complex128)
    if np.any(safe):
        r_safe = r[safe]
        order = np.argsort(r_safe)
        inv = np.empty_like(order)
        inv[order] = np.arange(order.size)
        _, R_sorted = compute_pybhpt_solution(a, omega, ell=ell, m=m, r_grid=r_safe[order], timeout=60.0)
        R[safe] = np.asarray(R_sorted, dtype=np.complex128)[inv]

    r_t = torch.tensor(r, dtype=RDTYPE)
    rp_t, rm_t, _, _, _ = build_prefactor_primitives(r_t, a_t, M=1.0, need_rs=False)
    P, _, _ = Leaver_prefactors(r_t, a_t, omega_t, m=m, M=1.0, s=s, rp=rp_t, rm=rm_t)
    h = h_factor(a_t, omega_t, m=m, M=1.0, s=s)
    S_ref = R / (P.detach().cpu().numpy() * complex(h.detach().cpu().item()))
    endpoint_mask = y_ref > 1.0 - 1.0e-14
    S_ref[endpoint_mask] = 1.0 + 0.0j
    return r, R, S_ref, safe | endpoint_mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", type=float, default=0.5)
    parser.add_argument("--omega", type=float, default=10.0**-1.5)
    parser.add_argument("--ell", type=int, default=2)
    parser.add_argument("--m", type=int, default=2)
    parser.add_argument("--s", type=int, default=-2)
    parser.add_argument("--y-match", type=float, default=0.6)
    parser.add_argument("--n", type=int, default=80)
    parser.add_argument("--horizon-exclude", type=float, default=1e-3)
    parser.add_argument("--output-dir", default="outputs/horizon_local_spectral")
    args = parser.parse_args()

    run_dir = Path(args.output_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    fig_dir = run_dir / "figures"
    run_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    sol = solve_horizon_local(
        a=args.a,
        omega=args.omega,
        ell=args.ell,
        m=args.m,
        s=args.s,
        y_match=args.y_match,
        n=args.n,
    )
    y = sol["y"]
    S = sol["S"]
    r, R_ref, S_ref, compare_mask = pybhpt_reference_S(
        a=args.a,
        omega=args.omega,
        ell=args.ell,
        m=args.m,
        s=args.s,
        y=y,
        horizon_exclude=args.horizon_exclude,
    )
    rel_S = np.full(y.shape, np.nan, dtype=float)
    valid = compare_mask & np.isfinite(S_ref.real) & np.isfinite(S_ref.imag)
    rel_S[valid] = np.abs(S[valid] - S_ref[valid]) / np.maximum(np.abs(S_ref[valid]), 1.0e-14)

    rows = []
    for yi, ri, si, sr, err in zip(y, r, S, S_ref, rel_S):
        rows.append({
            "y": float(yi),
            "r": float(ri),
            "S_re": float(si.real),
            "S_im": float(si.imag),
            "S_ref_re": float(sr.real),
            "S_ref_im": float(sr.imag),
            "rel_S": float(err),
        })
    with open(run_dir / "profile.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "args": vars(args),
        "lambda": [sol["lam"].real, sol["lam"].imag],
        "horizon_slope_physical_y": [sol["slope_h"].real, sol["slope_h"].imag],
        "cond_scaled": sol["cond_scaled"],
        "rel_res_median": sol["rel_res_median"],
        "rel_res_mean": sol["rel_res_mean"],
        "rel_res_max": sol["rel_res_max"],
        "n_pybhpt_compare": int(np.sum(valid)),
        "rel_S_median_vs_pybhpt": float(np.nanmedian(rel_S)),
        "rel_S_mean_vs_pybhpt": float(np.nanmean(rel_S)),
        "rel_S_max_vs_pybhpt": float(np.nanmax(rel_S)),
    }
    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
    axes[0].plot(y, S.real, label="spectral Re(S)")
    axes[0].plot(y, S_ref.real, "--", label="pybhpt Re(S)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[0].set_ylabel("Re(S)")
    axes[1].plot(y, S.imag, label="spectral Im(S)")
    axes[1].plot(y, S_ref.imag, "--", label="pybhpt Im(S)")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    axes[1].set_ylabel("Im(S)")
    axes[2].semilogy(y, rel_S, label="|S-S_ref|/|S_ref|")
    axes[2].legend()
    axes[2].grid(alpha=0.3)
    axes[2].set_ylabel("rel err")
    axes[2].set_xlabel("physical y=2r+/r-1")
    fig.suptitle(
        f"horizon local spectral: a={args.a:.3f}, omega={args.omega:.6g}, "
        f"y_match={args.y_match:.3f}, medS={summary['rel_S_median_vs_pybhpt']:.2e}"
    )
    fig.tight_layout()
    fig.savefig(fig_dir / "horizon_local_S_vs_pybhpt.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved to {run_dir}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
