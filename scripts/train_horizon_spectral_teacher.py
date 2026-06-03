#!/usr/bin/env python3
"""Train a parameter network for the horizon-side local spectral solution.

The teacher is the local horizon Chebyshev solve on physical
``y=2r_+/r-1`` with y=1 at the horizon.  The student maps
``(a, log10 omega) -> S(y_j)`` on fixed CGL nodes in ``[y_match, 1]``.
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

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.test_horizon_local_spectral import (  # noqa: E402
    coeffs_physical_y_np,
    pybhpt_reference_S,
    solve_horizon_local,
)
from scripts.train_center_patch_cheb_pinn import compute_lambda  # noqa: E402
from utils.matlcheb import cheb  # noqa: E402


RDTYPE = torch.float64
CDTYPE = torch.complex128


class ParamToHorizonValues(nn.Module):
    def __init__(self, n_nodes: int, width: int, depth: int, a_center: float, logw_center: float, a_scale: float, logw_scale: float):
        super().__init__()
        self.n_nodes = n_nodes
        self.a_center = a_center
        self.logw_center = logw_center
        self.a_scale = a_scale
        self.logw_scale = logw_scale
        layers: list[nn.Module] = []
        in_dim = 2
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, width))
            layers.append(nn.SiLU())
            in_dim = width
        layers.append(nn.Linear(in_dim, 2 * n_nodes))
        self.net = nn.Sequential(*layers)

    def forward(self, a: torch.Tensor, logw: torch.Tensor):
        x = torch.stack([
            (a - self.a_center) / self.a_scale,
            (logw - self.logw_center) / self.logw_scale,
        ], dim=-1)
        out = self.net(x)
        return out[:, 0::2].to(CDTYPE) + 1j * out[:, 1::2].to(CDTYPE)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def fixed_y_nodes(y_match: float, n: int):
    _, t = cheb(n)
    return 0.5 * (1.0 + y_match) + 0.5 * (1.0 - y_match) * t


def build_teacher_dataset(args, run_dir: Path):
    aa = np.linspace(args.a_center - args.a_half_width, args.a_center + args.a_half_width, args.n_param_side)
    ll = np.linspace(args.logw_center - args.logw_half_width, args.logw_center + args.logw_half_width, args.n_param_side)
    records = []
    values = []
    for a in aa:
        for logw in ll:
            omega = 10.0 ** float(logw)
            sol = solve_horizon_local(a=float(a), omega=omega, ell=args.ell, m=args.m, s=args.s, y_match=args.y_match, n=args.n)
            records.append({
                "a": float(a),
                "logw": float(logw),
                "omega": omega,
                "teacher_rel_res_median": sol["rel_res_median"],
                "teacher_rel_res_mean": sol["rel_res_mean"],
                "teacher_rel_res_max": sol["rel_res_max"],
                "teacher_cond_scaled": sol["cond_scaled"],
            })
            values.append(sol["S"])
    value_arr = np.stack(values).astype(np.complex128)
    np.savez(run_dir / "teacher_dataset.npz", records=np.array(records, dtype=object), S=value_arr)
    write_csv(run_dir / "teacher_dataset.csv", records)
    return records, value_arr


def residual_metrics_for_profile(a: float, logw: float, S_np: np.ndarray, y_np: np.ndarray, args):
    omega = 10.0 ** float(logw)
    lam = complex(compute_lambda(float(a), omega, args.ell, args.m, s=args.s))
    Dref, _ = cheb(args.n)
    half = 0.5 * (1.0 - args.y_match)
    D1mat = Dref / half
    D2mat = D1mat @ D1mat
    Sy = D1mat @ S_np
    Syy = D2mat @ S_np
    D2c, D1c, D0c = coeffs_physical_y_np(float(a), omega, lam, y_np[1:], m=args.m, s=args.s)
    res = D2c * Syy[1:] + D1c * Sy[1:] + D0c * S_np[1:]
    den = np.maximum.reduce([
        np.abs(D2c * Syy[1:]),
        np.abs(D1c * Sy[1:]),
        np.abs(D0c * S_np[1:]),
    ])
    rel = np.abs(res) / np.maximum(den, 1.0e-300)
    return {
        "rel_res_median": float(np.median(rel)),
        "rel_res_mean": float(np.mean(rel)),
        "rel_res_max": float(np.max(rel)),
    }


def evaluate_model(model, args, run_dir: Path, *, device: str):
    y_np = fixed_y_nodes(args.y_match, args.n)
    rows = []
    plot_cases = []
    aa = np.linspace(args.a_center - args.a_half_width, args.a_center + args.a_half_width, args.eval_param_side)
    ll = np.linspace(args.logw_center - args.logw_half_width, args.logw_center + args.logw_half_width, args.eval_param_side)
    model.eval()
    with torch.no_grad():
        for a in aa:
            for logw in ll:
                a_t = torch.tensor([float(a)], dtype=RDTYPE, device=device)
                logw_t = torch.tensor([float(logw)], dtype=RDTYPE, device=device)
                S_pred = model(a_t, logw_t).squeeze(0).detach().cpu().numpy()
                omega = 10.0 ** float(logw)
                teacher = solve_horizon_local(a=float(a), omega=omega, ell=args.ell, m=args.m, s=args.s, y_match=args.y_match, n=args.n)
                S_teacher = teacher["S"]
                rel_teacher = np.abs(S_pred - S_teacher) / np.maximum(np.abs(S_teacher), 1e-14)
                res_metrics = residual_metrics_for_profile(float(a), float(logw), S_pred, y_np, args)
                _, _, S_ref, mask = pybhpt_reference_S(
                    a=float(a),
                    omega=omega,
                    ell=args.ell,
                    m=args.m,
                    s=args.s,
                    y=y_np,
                    horizon_exclude=args.horizon_exclude,
                )
                valid = mask & np.isfinite(S_ref.real) & np.isfinite(S_ref.imag)
                rel_py = np.abs(S_pred[valid] - S_ref[valid]) / np.maximum(np.abs(S_ref[valid]), 1e-14)
                row = {
                    "a": float(a),
                    "logw": float(logw),
                    "omega": omega,
                    "teacher_rel_median": float(np.median(rel_teacher)),
                    "teacher_rel_max": float(np.max(rel_teacher)),
                    "pybhpt_rel_median": float(np.median(rel_py)),
                    "pybhpt_rel_max": float(np.max(rel_py)),
                    **res_metrics,
                }
                rows.append(row)
                if abs(float(a) - args.a_center) < 1e-12 and abs(float(logw) - args.logw_center) < 1e-12:
                    plot_cases.append((S_pred, S_teacher, S_ref, valid, row))
    write_csv(run_dir / "eval.csv", rows)
    if plot_cases:
        fig_dir = run_dir / "figures"
        fig_dir.mkdir(exist_ok=True)
        S_pred, S_teacher, S_ref, valid, row = plot_cases[0]
        fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
        axes[0].plot(y_np, S_pred.real, label="student Re(S)")
        axes[0].plot(y_np, S_teacher.real, "--", label="teacher Re(S)")
        axes[0].plot(y_np[valid], S_ref[valid].real, ":", label="pybhpt Re(S)")
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        axes[0].set_ylabel("Re(S)")
        axes[1].plot(y_np, S_pred.imag, label="student Im(S)")
        axes[1].plot(y_np, S_teacher.imag, "--", label="teacher Im(S)")
        axes[1].plot(y_np[valid], S_ref[valid].imag, ":", label="pybhpt Im(S)")
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        axes[1].set_ylabel("Im(S)")
        rel_py_full = np.full(y_np.shape, np.nan)
        rel_py_full[valid] = np.abs(S_pred[valid] - S_ref[valid]) / np.maximum(np.abs(S_ref[valid]), 1e-14)
        axes[2].semilogy(y_np, rel_py_full, label="student vs pybhpt")
        axes[2].semilogy(y_np, np.abs(S_pred - S_teacher) / np.maximum(np.abs(S_teacher), 1e-14), label="student vs teacher")
        axes[2].set_xlabel("physical y=2r+/r-1")
        axes[2].set_ylabel("relative error")
        axes[2].legend()
        axes[2].grid(alpha=0.3)
        fig.suptitle(
            f"center eval: py_med={row['pybhpt_rel_median']:.2e}, res_med={row['rel_res_median']:.2e}"
        )
        fig.tight_layout()
        fig.savefig(fig_dir / "center_student_vs_teacher_pybhpt.png", dpi=160, bbox_inches="tight")
        plt.close(fig)
    return rows


def plot_training(run_dir: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fig_dir = run_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    steps = np.array([r["step"] for r in rows], dtype=float)
    fig, ax = plt.subplots(figsize=(8, 5))
    for key in ["loss", "rel_median", "rel_max"]:
        ax.semilogy(steps, [r[key] for r in rows], label=key)
    ax.set_xlabel("step")
    ax.set_ylabel("value")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "training_curves.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--a-center", type=float, default=0.5)
    parser.add_argument("--logw-center", type=float, default=-1.5)
    parser.add_argument("--a-half-width", type=float, default=0.02)
    parser.add_argument("--logw-half-width", type=float, default=0.08)
    parser.add_argument("--n-param-side", type=int, default=9)
    parser.add_argument("--eval-param-side", type=int, default=5)
    parser.add_argument("--ell", type=int, default=2)
    parser.add_argument("--m", type=int, default=2)
    parser.add_argument("--s", type=int, default=-2)
    parser.add_argument("--y-match", type=float, default=0.6)
    parser.add_argument("--n", type=int, default=80)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--dy-weight", type=float, default=0.0)
    parser.add_argument("--dyy-weight", type=float, default=0.0)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--horizon-exclude", type=float, default=1e-3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", default="outputs/horizon_spectral_teacher")
    args = parser.parse_args()

    torch.set_default_dtype(RDTYPE)
    run_dir = Path(args.output_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "figures").mkdir(exist_ok=True)
    with open(run_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    print("Building teacher dataset...", flush=True)
    teacher_records, teacher_values = build_teacher_dataset(args, run_dir)
    Dref, _ = cheb(args.n)
    half = 0.5 * (1.0 - args.y_match)
    D1mat = Dref / half
    D2mat = D1mat @ D1mat
    a_arr = torch.tensor([r["a"] for r in teacher_records], dtype=RDTYPE, device=args.device)
    logw_arr = torch.tensor([r["logw"] for r in teacher_records], dtype=RDTYPE, device=args.device)
    target = torch.tensor(teacher_values, dtype=CDTYPE, device=args.device)
    D1_t = torch.tensor(D1mat, dtype=RDTYPE, device=args.device).to(CDTYPE)
    D2_t = torch.tensor(D2mat, dtype=RDTYPE, device=args.device).to(CDTYPE)
    target_y = target @ D1_t.T
    target_yy = target @ D2_t.T

    model = ParamToHorizonValues(
        args.n + 1,
        args.width,
        args.depth,
        args.a_center,
        args.logw_center,
        max(args.a_half_width, 1e-12),
        max(args.logw_half_width, 1e-12),
    ).to(device=args.device, dtype=RDTYPE)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-8)
    rows = []
    rng = np.random.default_rng(1234)
    for step in range(1, args.steps + 1):
        idx = torch.tensor(rng.choice(target.shape[0], size=min(args.batch, target.shape[0]), replace=False), dtype=torch.long, device=args.device)
        pred = model(a_arr[idx], logw_arr[idx])
        diff = pred - target[idx]
        denom = torch.abs(target[idx]).detach().clamp_min(1.0)
        value_loss = torch.mean((torch.abs(diff) / denom) ** 2)
        pred_y = pred @ D1_t.T
        pred_yy = pred @ D2_t.T
        dy_loss = torch.mean((torch.abs(pred_y - target_y[idx]) / torch.abs(target_y[idx]).detach().clamp_min(1.0)) ** 2)
        dyy_loss = torch.mean((torch.abs(pred_yy - target_yy[idx]) / torch.abs(target_yy[idx]).detach().clamp_min(1.0)) ** 2)
        loss = value_loss + args.dy_weight * dy_loss + args.dyy_weight * dyy_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step == 1 or step % args.log_every == 0:
            rel = torch.abs(diff.detach()) / torch.abs(target[idx]).detach().clamp_min(1e-14)
            row = {
                "step": step,
                "loss": float(loss.detach().cpu()),
                "value_loss": float(value_loss.detach().cpu()),
                "dy_loss": float(dy_loss.detach().cpu()),
                "dyy_loss": float(dyy_loss.detach().cpu()),
                "rel_median": float(rel.median().cpu()),
                "rel_max": float(rel.max().cpu()),
            }
            rows.append(row)
            print(f"step={step} loss={row['loss']:.3e} rel_med={row['rel_median']:.3e} rel_max={row['rel_max']:.3e}", flush=True)
    write_csv(run_dir / "train_log.csv", rows)
    plot_training(run_dir, rows)
    torch.save({"model": model.state_dict(), "args": vars(args)}, run_dir / "model.pt")

    print("Evaluating...", flush=True)
    eval_rows = evaluate_model(model, args, run_dir, device=args.device)
    summary = {
        "args": vars(args),
        "teacher_rel_res_median_median": float(np.median([r["teacher_rel_res_median"] for r in teacher_records])),
        "teacher_rel_res_max_max": float(np.max([r["teacher_rel_res_max"] for r in teacher_records])),
        "eval_teacher_rel_median_median": float(np.median([r["teacher_rel_median"] for r in eval_rows])),
        "eval_teacher_rel_max_max": float(np.max([r["teacher_rel_max"] for r in eval_rows])),
        "eval_pybhpt_rel_median_median": float(np.median([r["pybhpt_rel_median"] for r in eval_rows])),
        "eval_pybhpt_rel_max_max": float(np.max([r["pybhpt_rel_max"] for r in eval_rows])),
        "eval_residual_rel_median_median": float(np.median([r["rel_res_median"] for r in eval_rows])),
        "eval_residual_rel_max_max": float(np.max([r["rel_res_max"] for r in eval_rows])),
    }
    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved to {run_dir}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
