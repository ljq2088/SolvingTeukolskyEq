#!/usr/bin/env python3
"""Train a patch-local up-branch Chebyshev coefficient surrogate.

This is a focused replacement for the plain MLP coefficient head.  It uses:
  * full Chebyshev coefficient output, no artificial truncation;
  * per-mode complex standardization;
  * full-patch training with worst-case penalty;
  * value-space and weighted-tail losses;
  * AnMR teacher grids by default.
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
import torch
from torch import nn

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_patch_spectral_decoder import (  # noqa: E402
    BasisSpec,
    branch_specs,
    cheb_eval_matrix,
    solve_teacher,
    z_to_comp_xi,
)
from utils.matlcheb import cheb  # noqa: E402
from utils.mode import KerrMode  # noqa: E402


RDTYPE = torch.float64
CDTYPE = torch.complex128


class StructuredCoeffNet(nn.Module):
    def __init__(
        self,
        n_coeff: int,
        width: int,
        depth: int,
        a_center: float,
        logw_center: float,
        a_scale: float,
        logw_scale: float,
        fourier_bands: int,
    ):
        super().__init__()
        self.n_coeff = n_coeff
        self.a_center = a_center
        self.logw_center = logw_center
        self.a_scale = a_scale
        self.logw_scale = logw_scale
        self.fourier_bands = fourier_bands
        in_dim = 5 + 4 * fourier_bands
        layers: list[nn.Module] = []
        for _ in range(depth):
            layers += [nn.Linear(in_dim, width), nn.SiLU()]
            in_dim = width
        self.trunk = nn.Sequential(*layers)
        self.coeff_head = nn.Linear(in_dim, 2 * n_coeff)
        self.envelope_head = nn.Sequential(nn.Linear(in_dim, 64), nn.SiLU(), nn.Linear(64, 2))
        nn.init.zeros_(self.coeff_head.weight)
        nn.init.zeros_(self.coeff_head.bias)

    def features(self, a: torch.Tensor, logw: torch.Tensor) -> torch.Tensor:
        xa = (a - self.a_center) / self.a_scale
        xw = (logw - self.logw_center) / self.logw_scale
        feats = [xa, xw, xa * xw, xa * xa, xw * xw]
        for band in range(self.fourier_bands):
            freq = float(2**band) * torch.pi
            feats += [torch.sin(freq * xa), torch.cos(freq * xa), torch.sin(freq * xw), torch.cos(freq * xw)]
        return torch.stack(feats, dim=-1)

    def forward_norm(self, a: torch.Tensor, logw: torch.Tensor) -> torch.Tensor:
        h = self.trunk(self.features(a, logw))
        raw = self.coeff_head(h)
        return raw[:, 0::2].to(CDTYPE) + 1j * raw[:, 1::2].to(CDTYPE)


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_up_dataset(args, run_dir: Path):
    specs = branch_specs(args)
    spec = specs["up"]
    a_vals = np.linspace(args.a_center - args.a_half_width, args.a_center + args.a_half_width, args.n_param_side)
    logw_vals = np.linspace(args.logw_center - args.logw_half_width, args.logw_center + args.logw_half_width, args.n_param_side)
    coeffs = []
    rows = []
    for av in a_vals:
        for lw in logw_vals:
            mode = KerrMode(M=1.0, a=float(av), omega=10.0 ** float(lw), ell=args.ell, m=args.m, s=args.s)
            coeff, metrics = solve_teacher(mode, spec, args.n, args.grid_kind)
            coeffs.append(coeff)
            rows.append({
                "a": float(av),
                "logw": float(lw),
                "omega": mode.omega,
                **{f"teacher_{k}": v for k, v in metrics.items()},
            })
    coeffs_np = np.stack(coeffs).astype(np.complex128)
    np.savez(run_dir / "up_teacher_coeffs.npz", coeffs=coeffs_np, a=np.array([r["a"] for r in rows]), logw=np.array([r["logw"] for r in rows]))
    write_csv(run_dir / "up_teacher_metrics.csv", rows)
    return spec, rows, coeffs_np


def standardize(coeffs: np.ndarray, floor: float):
    mean = coeffs.mean(axis=0)
    scale = np.std(coeffs, axis=0)
    scale = np.maximum(scale, floor)
    return mean.astype(np.complex128), scale.astype(np.float64)


def physical_coeff(norm_coeff: torch.Tensor, mean: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return mean.unsqueeze(0) + norm_coeff * scale.unsqueeze(0).to(CDTYPE)


def sample_losses(pred_coeff, target_coeff, V, k_weights, args):
    coeff_scale = torch.linalg.vector_norm(target_coeff, dim=1).clamp_min(1e-30)
    coeff_rel = torch.linalg.vector_norm(pred_coeff - target_coeff, dim=1) / coeff_scale
    pred_u = pred_coeff @ V.T
    target_u = target_coeff @ V.T
    value_scale = torch.linalg.vector_norm(target_u, dim=1).clamp_min(1e-30)
    value_rel = torch.linalg.vector_norm(pred_u - target_u, dim=1) / value_scale
    tail = torch.sum((torch.abs(pred_coeff - target_coeff) ** 2) * k_weights.unsqueeze(0), dim=1) / torch.sum(
        (torch.abs(target_coeff) ** 2).clamp_min(1e-60) * k_weights.unsqueeze(0), dim=1
    ).clamp_min(1e-30)
    return args.coeff_weight * coeff_rel**2 + args.value_weight * value_rel**2 + args.tail_weight * tail


def evaluate(model, args, run_dir: Path, rows: list[dict], coeffs_np: np.ndarray, mean_np: np.ndarray, scale_np: np.ndarray):
    device = args.device
    _, xi = cheb(args.n)
    V = cheb_eval_matrix(args.n, xi, device).to(CDTYPE)
    a = torch.tensor([r["a"] for r in rows], dtype=RDTYPE, device=device)
    logw = torch.tensor([r["logw"] for r in rows], dtype=RDTYPE, device=device)
    target = torch.tensor(coeffs_np, dtype=CDTYPE, device=device)
    mean = torch.tensor(mean_np, dtype=CDTYPE, device=device)
    scale = torch.tensor(scale_np, dtype=RDTYPE, device=device)
    with torch.no_grad():
        pred = physical_coeff(model.forward_norm(a, logw), mean, scale)
        target_u = target @ V.T
        pred_u = pred @ V.T
        coeff_rel = torch.linalg.vector_norm(pred - target, dim=1) / torch.linalg.vector_norm(target, dim=1).clamp_min(1e-30)
        value_rel = torch.linalg.vector_norm(pred_u - target_u, dim=1) / torch.linalg.vector_norm(target_u, dim=1).clamp_min(1e-30)
    eval_rows = []
    for i, row in enumerate(rows):
        eval_rows.append({
            "a": row["a"],
            "logw": row["logw"],
            "omega": row["omega"],
            "coeff_rel_l2": float(coeff_rel[i].cpu()),
            "value_rel_l2": float(value_rel[i].cpu()),
        })
    write_csv(run_dir / "up_eval.csv", eval_rows)
    return pred.cpu().numpy(), eval_rows


def plot_outputs(args, run_dir: Path, rows: list[dict], coeffs_np: np.ndarray, pred_np: np.ndarray):
    fig_dir = run_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    a_arr = np.array([r["a"] for r in rows])
    logw_arr = np.array([r["logw"] for r in rows])
    center_idx = int(np.argmin(np.abs(a_arr - args.a_center) + np.abs(logw_arr - args.logw_center)))
    teacher = coeffs_np[center_idx]
    pred = pred_np[center_idx]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].semilogy(np.abs(teacher), "o-", ms=3, lw=1, label="teacher")
    axes[0].semilogy(np.abs(pred), "s--", ms=3, lw=1, label="network")
    axes[0].set_title(f"up coeff decay, a={a_arr[center_idx]:.3f}, logw={logw_arr[center_idx]:.3f}")
    axes[0].set_xlabel("Chebyshev mode k")
    axes[0].set_ylabel("|c_k|")
    axes[0].grid(alpha=0.3)
    axes[0].legend()
    axes[1].semilogy(np.abs(pred - teacher), "o-", ms=3, lw=1)
    axes[1].set_title("|network - teacher|")
    axes[1].set_xlabel("Chebyshev mode k")
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "up_network_vs_teacher_coeff_decay_center.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    eval_df = np.genfromtxt(run_dir / "up_eval.csv", delimiter=",", names=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    a_vals = np.unique(eval_df["a"])
    lw_vals = np.unique(eval_df["logw"])
    grid = np.full((len(a_vals), len(lw_vals)), np.nan)
    for row in eval_df:
        ia = int(np.where(a_vals == row["a"])[0][0])
        iw = int(np.where(lw_vals == row["logw"])[0][0])
        grid[ia, iw] = row["value_rel_l2"]
    im = ax.imshow(np.log10(grid), origin="lower", aspect="auto", vmin=-6, vmax=0, extent=[lw_vals.min(), lw_vals.max(), a_vals.min(), a_vals.max()])
    ax.set_title("up log10 value relative L2")
    ax.set_xlabel("log10 omega")
    ax.set_ylabel("a")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(fig_dir / "up_value_rel_error_heatmap.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--a-center", type=float, default=0.5)
    parser.add_argument("--logw-center", type=float, default=-1.5)
    parser.add_argument("--a-half-width", type=float, default=0.12)
    parser.add_argument("--logw-half-width", type=float, default=0.25)
    parser.add_argument("--n-param-side", type=int, default=5)
    parser.add_argument("--ell", type=int, default=2)
    parser.add_argument("--m", type=int, default=2)
    parser.add_argument("--s", type=int, default=-2)
    parser.add_argument("--y-match", type=float, default=-0.25)
    parser.add_argument("--match-width", type=float, default=0.12)
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--grid-kind", choices=["linear", "anmr", "auto"], default="anmr")
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--fourier-bands", type=int, default=5)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--coeff-floor", type=float, default=1e-14)
    parser.add_argument("--coeff-weight", type=float, default=1.0)
    parser.add_argument("--value-weight", type=float, default=10.0)
    parser.add_argument("--tail-weight", type=float, default=0.1)
    parser.add_argument("--worst-weight", type=float, default=1.0)
    parser.add_argument("--tail-power", type=float, default=2.0)
    parser.add_argument("--log-every", type=int, default=250)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", default="outputs/up_spectral_envelope")
    args = parser.parse_args()

    torch.set_default_dtype(RDTYPE)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    run_dir = Path(args.output_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    print(json.dumps({"run_dir": str(run_dir)}, indent=2), flush=True)

    _, rows, coeffs_np = build_up_dataset(args, run_dir)
    mean_np, scale_np = standardize(coeffs_np, args.coeff_floor)
    np.savez(run_dir / "up_normalizer.npz", mean=mean_np, scale=scale_np)
    device = args.device
    model = StructuredCoeffNet(args.n + 1, args.width, args.depth, args.a_center, args.logw_center, args.a_half_width, args.logw_half_width, args.fourier_bands).to(device, dtype=RDTYPE)
    mean = torch.tensor(mean_np, dtype=CDTYPE, device=device)
    scale = torch.tensor(scale_np, dtype=RDTYPE, device=device)
    target = torch.tensor(coeffs_np, dtype=CDTYPE, device=device)
    target_norm = (target - mean.unsqueeze(0)) / scale.unsqueeze(0).to(CDTYPE)
    a = torch.tensor([r["a"] for r in rows], dtype=RDTYPE, device=device)
    logw = torch.tensor([r["logw"] for r in rows], dtype=RDTYPE, device=device)
    _, xi = cheb(args.n)
    V = cheb_eval_matrix(args.n, xi, device).to(CDTYPE)
    k = torch.arange(args.n + 1, dtype=RDTYPE, device=device)
    k_weights = (1.0 + k) ** args.tail_power
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-10)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.steps, 1), eta_min=args.min_lr)
    hist = []
    best = {"value_max": float("inf"), "step": 0}
    for step in range(1, args.steps + 1):
        norm_pred = model.forward_norm(a, logw)
        pred = physical_coeff(norm_pred, mean, scale)
        coeff_norm_loss = torch.mean(torch.abs(norm_pred - target_norm) ** 2)
        losses = sample_losses(pred, target, V, k_weights, args)
        loss = coeff_norm_loss + losses.mean() + args.worst_weight * losses.max()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step()
        scheduler.step()
        if step % args.log_every == 0 or step == 1 or step == args.steps:
            with torch.no_grad():
                pred_u = pred @ V.T
                target_u = target @ V.T
                value_rel = torch.linalg.vector_norm(pred_u - target_u, dim=1) / torch.linalg.vector_norm(target_u, dim=1).clamp_min(1e-30)
                coeff_rel = torch.linalg.vector_norm(pred - target, dim=1) / torch.linalg.vector_norm(target, dim=1).clamp_min(1e-30)
                row = {
                    "step": step,
                    "loss": float(loss.cpu()),
                    "coeff_med": float(coeff_rel.median().cpu()),
                    "coeff_max": float(coeff_rel.max().cpu()),
                    "value_med": float(value_rel.median().cpu()),
                    "value_max": float(value_rel.max().cpu()),
                }
            hist.append(row)
            print(json.dumps(row), flush=True)
            if row["value_max"] < best["value_max"]:
                best = {"value_max": row["value_max"], "step": step}
                torch.save(model.state_dict(), run_dir / "up_structured_coeff_net_best.pt")
    torch.save(model.state_dict(), run_dir / "up_structured_coeff_net_last.pt")
    write_csv(run_dir / "train_history.csv", hist)
    if (run_dir / "up_structured_coeff_net_best.pt").exists():
        model.load_state_dict(torch.load(run_dir / "up_structured_coeff_net_best.pt", map_location=device))
    pred_np, eval_rows = evaluate(model, args, run_dir, rows, coeffs_np, mean_np, scale_np)
    plot_outputs(args, run_dir, rows, coeffs_np, pred_np)
    summary = {
        "run_dir": str(run_dir),
        "best": best,
        "eval": {
            "value_median": float(np.median([r["value_rel_l2"] for r in eval_rows])),
            "value_max": float(np.max([r["value_rel_l2"] for r in eval_rows])),
            "coeff_median": float(np.median([r["coeff_rel_l2"] for r in eval_rows])),
            "coeff_max": float(np.max([r["coeff_rel_l2"] for r in eval_rows])),
        },
    }
    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
