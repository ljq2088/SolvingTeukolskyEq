#!/usr/bin/env python3
"""Fit an up-branch tensor-product Chebyshev decoder over (a, log10 omega).

The output is the full Chebyshev-in-y coefficient vector.  This is a spectral
decoder in parameter space: if the coefficient field is analytic on the patch,
the tensor-product coefficients should decay geometrically as well.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_patch_spectral_decoder import branch_specs, cheb_eval_matrix, solve_teacher  # noqa: E402
from utils.matlcheb import cheb  # noqa: E402
from utils.mode import KerrMode  # noqa: E402


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def cheb_lobatto_physical(center: float, half_width: float, n_side: int):
    _, xi = cheb(n_side - 1)
    values = center + half_width * xi
    return values, xi


def tensor_vandermonde(xi_a: np.ndarray, xi_w: np.ndarray, deg_a: int, deg_w: int):
    Va = np.cos(np.outer(np.arccos(np.clip(xi_a, -1, 1)), np.arange(deg_a + 1)))
    Vw = np.cos(np.outer(np.arccos(np.clip(xi_w, -1, 1)), np.arange(deg_w + 1)))
    rows = []
    for ia in range(len(xi_a)):
        for iw in range(len(xi_w)):
            rows.append(np.kron(Va[ia], Vw[iw]))
    return np.asarray(rows), Va, Vw


def fit_tensor_decoder(coeffs: np.ndarray, xi_a: np.ndarray, xi_w: np.ndarray, deg_a: int, deg_w: int):
    n_a, n_w, n_coeff = coeffs.shape
    A, _, _ = tensor_vandermonde(xi_a, xi_w, deg_a, deg_w)
    y = coeffs.reshape(n_a * n_w, n_coeff)
    scale = np.linalg.norm(A, axis=0)
    scale = np.where(scale > 1e-300, scale, 1.0)
    x, *_ = np.linalg.lstsq(A / scale[None, :], y, rcond=None)
    x = x / scale[:, None]
    return x.reshape(deg_a + 1, deg_w + 1, n_coeff)


def eval_tensor_decoder(decoder: np.ndarray, xi_a: np.ndarray, xi_w: np.ndarray):
    deg_a = decoder.shape[0] - 1
    deg_w = decoder.shape[1] - 1
    A, _, _ = tensor_vandermonde(xi_a, xi_w, deg_a, deg_w)
    return (A @ decoder.reshape((deg_a + 1) * (deg_w + 1), decoder.shape[2])).reshape(len(xi_a), len(xi_w), decoder.shape[2])


def build_coeff_grid(args):
    specs = branch_specs(args)
    spec = specs["up"]
    a_vals, xi_a = cheb_lobatto_physical(args.a_center, args.a_half_width, args.n_param_side)
    logw_vals, xi_w = cheb_lobatto_physical(args.logw_center, args.logw_half_width, args.n_param_side)
    coeffs = np.zeros((args.n_param_side, args.n_param_side, args.n + 1), dtype=np.complex128)
    rows = []
    for ia, av in enumerate(a_vals):
        for iw, lw in enumerate(logw_vals):
            mode = KerrMode(M=1.0, a=float(av), omega=10.0 ** float(lw), ell=args.ell, m=args.m, s=args.s)
            coeff, metrics = solve_teacher(mode, spec, args.n, args.grid_kind)
            coeffs[ia, iw] = coeff
            rows.append({"a": float(av), "logw": float(lw), "omega": mode.omega, **{f"teacher_{k}": v for k, v in metrics.items()}})
    return a_vals, logw_vals, xi_a, xi_w, coeffs, rows


def rel_errors(pred: np.ndarray, target: np.ndarray, n: int):
    _, xi = cheb(n)
    V = np.cos(np.outer(np.arccos(xi), np.arange(n + 1))).astype(np.complex128)
    pred_u = pred.reshape(-1, n + 1) @ V.T
    target_u = target.reshape(-1, n + 1) @ V.T
    coeff_rel = np.linalg.norm((pred - target).reshape(-1, n + 1), axis=1) / np.maximum(np.linalg.norm(target.reshape(-1, n + 1), axis=1), 1e-300)
    value_rel = np.linalg.norm(pred_u - target_u, axis=1) / np.maximum(np.linalg.norm(target_u, axis=1), 1e-300)
    return coeff_rel, value_rel


def plot(run_dir: Path, args, decoder, coeffs, pred, a_vals, logw_vals):
    fig_dir = run_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    center_idx = (int(np.argmin(abs(a_vals - args.a_center))), int(np.argmin(abs(logw_vals - args.logw_center))))
    teacher = coeffs[center_idx]
    network = pred[center_idx]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].semilogy(np.abs(teacher), "o-", ms=3, lw=1, label="teacher")
    axes[0].semilogy(np.abs(network), "s--", ms=3, lw=1, label="param Cheb decoder")
    axes[0].set_title("up y-Cheb coefficient decay")
    axes[0].set_xlabel("y-Cheb mode k")
    axes[0].set_ylabel("|c_k|")
    axes[0].grid(alpha=0.3)
    axes[0].legend()
    axes[1].semilogy(np.abs(network - teacher), "o-", ms=3, lw=1)
    axes[1].set_title("|decoder-teacher|")
    axes[1].set_xlabel("y-Cheb mode k")
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "up_param_cheb_vs_teacher_coeff_decay_center.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    modal_energy = np.max(np.abs(decoder), axis=2)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(np.log10(np.maximum(modal_energy, 1e-300)), origin="lower", aspect="auto")
    ax.set_title("log10 max_k |parameter Cheb coeff|")
    ax.set_xlabel("logw Cheb mode")
    ax.set_ylabel("a Cheb mode")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(fig_dir / "up_parameter_cheb_coeff_decay.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--a-center", type=float, default=0.5)
    parser.add_argument("--logw-center", type=float, default=-1.5)
    parser.add_argument("--a-half-width", type=float, default=0.12)
    parser.add_argument("--logw-half-width", type=float, default=0.25)
    parser.add_argument("--n-param-side", type=int, default=9)
    parser.add_argument("--deg-a", type=int, default=8)
    parser.add_argument("--deg-logw", type=int, default=8)
    parser.add_argument("--ell", type=int, default=2)
    parser.add_argument("--m", type=int, default=2)
    parser.add_argument("--s", type=int, default=-2)
    parser.add_argument("--y-match", type=float, default=-0.25)
    parser.add_argument("--match-width", type=float, default=0.12)
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--grid-kind", choices=["linear", "anmr", "auto"], default="anmr")
    parser.add_argument("--output-dir", default="outputs/up_param_cheb_decoder")
    args = parser.parse_args()
    run_dir = Path(args.output_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    print(json.dumps({"run_dir": str(run_dir)}, indent=2), flush=True)
    a_vals, logw_vals, xi_a, xi_w, coeffs, rows = build_coeff_grid(args)
    decoder = fit_tensor_decoder(coeffs, xi_a, xi_w, args.deg_a, args.deg_logw)
    pred = eval_tensor_decoder(decoder, xi_a, xi_w)
    coeff_rel, value_rel = rel_errors(pred, coeffs, args.n)
    out_rows = []
    flat = 0
    for ia, av in enumerate(a_vals):
        for iw, lw in enumerate(logw_vals):
            out_rows.append({"a": float(av), "logw": float(lw), "omega": 10.0 ** float(lw), "coeff_rel_l2": float(coeff_rel[flat]), "value_rel_l2": float(value_rel[flat])})
            flat += 1
    write_csv(run_dir / "up_param_cheb_eval.csv", out_rows)
    write_csv(run_dir / "up_teacher_metrics.csv", rows)
    np.savez(run_dir / "up_param_cheb_decoder.npz", decoder=decoder, a_vals=a_vals, logw_vals=logw_vals, coeffs=coeffs, pred=pred)
    plot(run_dir, args, decoder, coeffs, pred, a_vals, logw_vals)
    summary = {
        "run_dir": str(run_dir),
        "coeff_rel_median": float(np.median(coeff_rel)),
        "coeff_rel_max": float(np.max(coeff_rel)),
        "value_rel_median": float(np.median(value_rel)),
        "value_rel_max": float(np.max(value_rel)),
        "teacher_tail_median": float(np.median([r["teacher_tail_rel"] for r in rows])),
        "teacher_tail_max": float(np.max([r["teacher_tail_rel"] for r in rows])),
    }
    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
