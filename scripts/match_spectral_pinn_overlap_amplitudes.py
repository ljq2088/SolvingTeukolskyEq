#!/usr/bin/env python3
"""Fit amplitudes from Spectral-PINN values on an overlap interval.

No derivative matching is used.  On physical y in [y1, y2],

    u_in(y) A_in(y) = B_inc u_down(y) A_down(y)
                     + B_ref u_up(y) A_up(y).

The two complex amplitudes are obtained by column-scaled least squares over
many overlap points.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
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

from scripts.match_spectral_pinn_y0_amplitudes import (  # noqa: E402
    complex_to_pair,
    get_mma_kernel_path,
    load_horizon_model,
    load_infinity_model,
    predict_coeff,
    rel_err,
    run_mma_script,
)
from physical_ansatz.prefactor import Leaver_prefactors, build_prefactor_primitives  # noqa: E402
from physical_ansatz.transform_y import h_factor  # noqa: E402
from utils.amplitude import A_down, A_in, A_up, r_of_z  # noqa: E402
from utils.mode import KerrMode  # noqa: E402


RDTYPE = torch.float64


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def cheb_eval(coeff: np.ndarray, y: np.ndarray, y_left: float, y_right: float) -> np.ndarray:
    mid = 0.5 * (y_left + y_right)
    half = 0.5 * (y_right - y_left)
    xi = (y - mid) / half
    if np.any(xi < -1.0 - 1e-12) or np.any(xi > 1.0 + 1e-12):
        raise ValueError(f"Evaluation y outside domain [{y_left}, {y_right}]")
    xi = np.clip(xi, -1.0, 1.0)
    theta = np.arccos(xi)
    k = np.arange(coeff.shape[0])
    V = np.cos(np.outer(theta, k))
    return V @ coeff


def branch_R_values(mode: KerrMode, basis: str, u: np.ndarray, y: np.ndarray) -> np.ndarray:
    z = 0.5 * (y + 1.0)
    r = r_of_z(z, mode)
    if basis == "in":
        r_t = torch.tensor(r, dtype=RDTYPE)
        a_t = torch.tensor([mode.a], dtype=RDTYPE)
        omega_t = torch.tensor([mode.omega], dtype=RDTYPE)
        rp_t, rm_t, _, _, _ = build_prefactor_primitives(r_t, a_t, M=mode.M, need_rs=False)
        P, _, _ = Leaver_prefactors(
            r_t,
            a_t,
            omega_t,
            m=mode.m,
            M=mode.M,
            s=mode.s,
            rp=rp_t,
            rm=rm_t,
        )
        h = h_factor(a_t, omega_t, m=mode.m, M=mode.M, s=mode.s)
        A = P.detach().cpu().numpy() * complex(h.detach().cpu().item())
    elif basis == "down":
        A = A_down(r, mode)
    elif basis == "up":
        A = A_up(r, mode)
    else:
        raise ValueError(basis)
    return A * u


def scaled_complex_lstsq(mat: np.ndarray, rhs: np.ndarray) -> tuple[np.ndarray, float, float]:
    col_scale = np.linalg.norm(mat, axis=0)
    col_scale = np.where(col_scale > 1e-300, col_scale, 1.0)
    mat_s = mat / col_scale[None, :]
    coef_s, *_ = np.linalg.lstsq(mat_s, rhs, rcond=None)
    coef = coef_s / col_scale
    resid = mat @ coef - rhs
    return coef, float(np.linalg.cond(mat_s)), float(np.linalg.norm(resid) / max(np.linalg.norm(rhs), 1e-300))


def fit_one(
    *,
    a: float,
    omega: float,
    ell: int,
    m: int,
    s: int,
    y1: float,
    y2: float,
    n_match: int,
    horizon_model,
    horizon_cfg: dict,
    down_model,
    down_cfg: dict,
    up_model,
    up_cfg: dict,
    device: str,
    fit_space: str,
) -> dict:
    logw = float(np.log10(omega))
    mode = KerrMode(M=1.0, a=a, omega=omega, ell=ell, m=m, s=s)
    c_in = predict_coeff(horizon_model, a, logw, device)
    c_down = predict_coeff(down_model, a, logw, device)
    c_up = predict_coeff(up_model, a, logw, device)

    y = np.linspace(y1, y2, n_match, dtype=np.float64)
    u_in = cheb_eval(c_in, y, float(horizon_cfg["y_match"]), 1.0)
    u_down = cheb_eval(c_down, y, -1.0, float(down_cfg["y_right"]))
    u_up = cheb_eval(c_up, y, -1.0, float(up_cfg["y_right"]))

    R_in = branch_R_values(mode, "in", u_in, y)
    R_down = branch_R_values(mode, "down", u_down, y)
    R_up = branch_R_values(mode, "up", u_up, y)
    if fit_space == "raw":
        mat = np.stack([R_down, R_up], axis=1)
        rhs = R_in
    elif fit_space == "down-ratio":
        mat = np.stack([np.ones_like(R_down), R_up / R_down], axis=1)
        rhs = R_in / R_down
    else:
        raise ValueError(f"Unknown fit_space: {fit_space}")
    coef, cond_scaled, rel_res = scaled_complex_lstsq(mat, rhs)
    B_inc, B_ref = complex(coef[0]), complex(coef[1])
    fitted_rhs = mat @ coef
    fitted = B_inc * R_down + B_ref * R_up
    point_rel = np.abs(fitted - R_in) / np.maximum(np.abs(R_in), 1e-300)
    return {
        "a": a,
        "omega": omega,
        "logw": logw,
        "lambda": mode.lambda_value,
        "B_inc": B_inc,
        "B_ref": B_ref,
        "B_trans": 1.0 + 0.0j,
        "y": y,
        "R_in": R_in,
        "R_fit": fitted,
        "R_down": R_down,
        "R_up": R_up,
        "fit_space": fit_space,
        "raw_cond": float(np.linalg.cond(mat)),
        "scaled_cond": cond_scaled,
        "fit_rel_res": rel_res,
        "fit_point_rel_median": float(np.median(point_rel)),
        "fit_point_rel_max": float(np.max(point_rel)),
    }


def save_plot(run_dir: Path, result: dict):
    fig_dir = run_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    y = result["y"]
    rel = np.abs(result["R_fit"] - result["R_in"]) / np.maximum(np.abs(result["R_in"]), 1e-300)
    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
    axes[0].plot(y, result["R_in"].real, label="R_in Spectral-PINN")
    axes[0].plot(y, result["R_fit"].real, "--", label="B_inc down + B_ref up")
    axes[0].set_ylabel("Re(R)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[1].plot(y, result["R_in"].imag, label="R_in Spectral-PINN")
    axes[1].plot(y, result["R_fit"].imag, "--", label="fit")
    axes[1].set_ylabel("Im(R)")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    axes[2].semilogy(y, rel)
    axes[2].set_xlabel("physical y")
    axes[2].set_ylabel("pointwise rel fit")
    axes[2].grid(alpha=0.3)
    fig.suptitle(
        f"value-only overlap fit: rel_med={result['fit_point_rel_median']:.2e}, "
        f"scaled_cond={result['scaled_cond']:.2e}"
    )
    fig.tight_layout()
    fig.savefig(fig_dir / "overlap_value_fit.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon-run", required=True)
    parser.add_argument("--infinity-run", required=True)
    parser.add_argument("--infinity-up-run", default=None)
    parser.add_argument("--a", type=float, default=0.5)
    parser.add_argument("--omega", type=float, default=10.0 ** -1.5)
    parser.add_argument("--ell", type=int, default=2)
    parser.add_argument("--m", type=int, default=2)
    parser.add_argument("--s", type=int, default=-2)
    parser.add_argument("--y1", type=float, default=-0.1)
    parser.add_argument("--y2", type=float, default=0.1)
    parser.add_argument("--n-match", type=int, default=101)
    parser.add_argument("--fit-space", choices=["raw", "down-ratio"], default="raw")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--skip-mma", action="store_true")
    parser.add_argument("--kernel-path", default=None)
    parser.add_argument("--mma-wl", default="mma/Radial_Function.wl")
    parser.add_argument("--mma-wl-win", default="F:/EMRI/Radial_flow/Radial_Function.wl")
    parser.add_argument("--mma-timeout", type=float, default=120.0)
    parser.add_argument("--output-dir", default="outputs/spectral_pinn_overlap_amplitude_match")
    args = parser.parse_args()

    torch.set_default_dtype(RDTYPE)
    run_dir = Path(args.output_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    horizon_model, horizon_cfg = load_horizon_model(PROJECT_ROOT / args.horizon_run, args.device)
    down_model, down_cfg = load_infinity_model(PROJECT_ROOT / args.infinity_run, "down", args.device)
    up_run = args.infinity_up_run or args.infinity_run
    up_model, up_cfg = load_infinity_model(PROJECT_ROOT / up_run, "up", args.device)

    result = fit_one(
        a=args.a,
        omega=args.omega,
        ell=args.ell,
        m=args.m,
        s=args.s,
        y1=args.y1,
        y2=args.y2,
        n_match=args.n_match,
        horizon_model=horizon_model,
        horizon_cfg=horizon_cfg,
        down_model=down_model,
        down_cfg=down_cfg,
        up_model=up_model,
        up_cfg=up_cfg,
        device=args.device,
        fit_space=args.fit_space,
    )
    save_plot(run_dir, result)

    rows = [{
        "a": result["a"],
        "omega": result["omega"],
        "logw": result["logw"],
        "y1": args.y1,
        "y2": args.y2,
        "n_match": args.n_match,
        "fit_space": args.fit_space,
        "B_inc_re": result["B_inc"].real,
        "B_inc_im": result["B_inc"].imag,
        "B_ref_re": result["B_ref"].real,
        "B_ref_im": result["B_ref"].imag,
        "raw_cond": result["raw_cond"],
        "scaled_cond": result["scaled_cond"],
        "fit_rel_res": result["fit_rel_res"],
        "fit_point_rel_median": result["fit_point_rel_median"],
        "fit_point_rel_max": result["fit_point_rel_max"],
    }]

    mma_status = "skipped"
    mma_result = None
    kernel_path = None
    if not args.skip_mma:
        kernel_path = get_mma_kernel_path(args.kernel_path)
        if kernel_path is None:
            mma_status = "kernel-not-found"
        else:
            wl_path = args.mma_wl_win if kernel_path.lower().endswith(".exe") else str((PROJECT_ROOT / args.mma_wl).resolve())
            mma_map, mma_status = run_mma_script(
                [(args.a, args.omega)],
                kernel_path=kernel_path,
                wl_path=wl_path,
                work_dir=run_dir,
                timeout=args.mma_timeout,
            )
            mma_result, case_status = mma_map.get((args.a, args.omega), (None, "missing"))
            mma_status = f"{mma_status}:{case_status}"
            if mma_result is not None:
                rows[0].update({
                    "mma_B_inc_re": mma_result["B_inc"].real,
                    "mma_B_inc_im": mma_result["B_inc"].imag,
                    "mma_B_ref_re": mma_result["B_ref"].real,
                    "mma_B_ref_im": mma_result["B_ref"].imag,
                    "relerr_B_inc": rel_err(result["B_inc"], mma_result["B_inc"]),
                    "relerr_B_ref": rel_err(result["B_ref"], mma_result["B_ref"]),
                })
    write_csv(run_dir / "amplitude_compare.csv", rows)
    summary = {
        "args": vars(args),
        "run_dir": str(run_dir),
        "inputs": {
            "horizon_run": str(PROJECT_ROOT / args.horizon_run),
            "infinity_run_down": str(PROJECT_ROOT / args.infinity_run),
            "infinity_run_up": str(PROJECT_ROOT / up_run),
        },
        "spectral_pinn": {
            "B_inc": complex_to_pair(result["B_inc"]),
            "B_ref": complex_to_pair(result["B_ref"]),
            "B_trans": complex_to_pair(result["B_trans"]),
            "raw_cond": result["raw_cond"],
            "scaled_cond": result["scaled_cond"],
            "fit_rel_res": result["fit_rel_res"],
            "fit_point_rel_median": result["fit_point_rel_median"],
            "fit_point_rel_max": result["fit_point_rel_max"],
        },
        "mma": {
            "status": mma_status,
            "kernel_path": kernel_path,
            "B_inc": complex_to_pair(mma_result["B_inc"]) if mma_result else None,
            "B_ref": complex_to_pair(mma_result["B_ref"]) if mma_result else None,
            "B_trans": complex_to_pair(mma_result["B_trans"]) if mma_result else None,
            "relerr_B_inc": rel_err(result["B_inc"], mma_result["B_inc"]) if mma_result else None,
            "relerr_B_ref": rel_err(result["B_ref"], mma_result["B_ref"]) if mma_result else None,
        },
    }
    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
