#!/usr/bin/env python3
"""Train horizon-local Chebyshev coefficient surrogate, then residual-refine.

Physical coordinate y=2r_+/r-1.  This experiment uses y in [0, 1].
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

from scripts.test_horizon_local_spectral import coeffs_physical_y_np, pybhpt_reference_S, solve_horizon_local  # noqa: E402
from scripts.train_center_patch_cheb_pinn import compute_lambda  # noqa: E402
from utils.matlcheb import cheb, real_to_cheb  # noqa: E402


RDTYPE = torch.float64
CDTYPE = torch.complex128


class ParamToCoeff(nn.Module):
    def __init__(self, n_coeff: int, width: int, depth: int, a_center: float, logw_center: float, a_scale: float, logw_scale: float, out_coeff: int | None = None):
        super().__init__()
        self.n_coeff = n_coeff
        self.out_coeff = out_coeff or n_coeff
        self.a_center = a_center
        self.logw_center = logw_center
        self.a_scale = a_scale
        self.logw_scale = logw_scale
        layers = []
        in_dim = 2
        for _ in range(depth):
            layers += [nn.Linear(in_dim, width), nn.SiLU()]
            in_dim = width
        layers.append(nn.Linear(in_dim, 2 * self.out_coeff))
        self.net = nn.Sequential(*layers)

    def forward(self, a, logw):
        x = torch.stack([(a - self.a_center) / self.a_scale, (logw - self.logw_center) / self.logw_scale], dim=-1)
        out = self.net(x)
        low = out[:, 0::2].to(CDTYPE) + 1j * out[:, 1::2].to(CDTYPE)
        if self.out_coeff == self.n_coeff:
            return low
        full = torch.zeros((a.shape[0], self.n_coeff), dtype=CDTYPE, device=a.device)
        full[:, : self.out_coeff] = low
        return full


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def cheb_eval_matrix(n: int, device: str):
    _, t = cheb(n)
    k = np.arange(n + 1)
    V = np.cos(np.outer(np.arccos(t), k))
    return torch.tensor(V, dtype=RDTYPE, device=device), t


def build_diff_mats(n: int, y_match: float, device: str):
    D, t = cheb(n)
    half = 0.5 * (1.0 - y_match)
    y = 0.5 * (1.0 + y_match) + half * t
    return (
        torch.tensor(y, dtype=RDTYPE, device=device),
        torch.tensor(D / half, dtype=RDTYPE, device=device),
        torch.tensor((D / half) @ (D / half), dtype=RDTYPE, device=device),
    )


def build_dataset(args, run_dir: Path):
    aa = np.linspace(args.a_center - args.a_half_width, args.a_center + args.a_half_width, args.n_param_side)
    ll = np.linspace(args.logw_center - args.logw_half_width, args.logw_center + args.logw_half_width, args.n_param_side)
    rows, coeffs = [], []
    for a in aa:
        for logw in ll:
            omega = 10.0 ** float(logw)
            sol = solve_horizon_local(a=float(a), omega=omega, ell=args.ell, m=args.m, s=args.s, y_match=args.y_match, n=args.n)
            c = real_to_cheb(sol["S"])
            coeffs.append(c)
            rows.append({
                "a": float(a),
                "logw": float(logw),
                "omega": omega,
                "teacher_res_med": sol["rel_res_median"],
                "teacher_res_max": sol["rel_res_max"],
            })
    coeffs = np.stack(coeffs).astype(np.complex128)
    np.savez(run_dir / "coeff_dataset.npz", coeffs=coeffs, records=np.array(rows, dtype=object))
    write_csv(run_dir / "coeff_dataset.csv", rows)
    return rows, coeffs


def residual_loss_from_coeff(model, a, logw, V, y, D1, D2, args):
    coeff = model(a, logw)
    S = coeff @ V.to(CDTYPE).T
    Sy = S @ D1.to(CDTYPE).T
    Syy = S @ D2.to(CDTYPE).T
    losses = []
    infos = []
    y_np = y.detach().cpu().numpy()
    for i in range(a.shape[0]):
        av = float(a[i].detach().cpu())
        lw = float(logw[i].detach().cpu())
        omega = 10.0 ** lw
        lam = complex(compute_lambda(av, omega, args.ell, args.m, s=args.s))
        D2n, D1n, D0n = coeffs_physical_y_np(av, omega, lam, y_np[1:], m=args.m, s=args.s)
        D2c = torch.tensor(D2n, dtype=CDTYPE, device=a.device)
        D1c = torch.tensor(D1n, dtype=CDTYPE, device=a.device)
        D0c = torch.tensor(D0n, dtype=CDTYPE, device=a.device)
        res = D2c * Syy[i, 1:] + D1c * Sy[i, 1:] + D0c * S[i, 1:]
        den = torch.maximum(torch.maximum(torch.abs(D2c * Syy[i, 1:]), torch.abs(D1c * Sy[i, 1:])), torch.abs(D0c * S[i, 1:])).clamp_min(1e-30)
        rel = torch.abs(res) / den
        bc = torch.abs(S[i, 0] - 1.0) ** 2
        losses.append(torch.mean(rel * rel) + args.bc_weight * bc)
        infos.append((float(rel.detach().median().cpu()), float(rel.detach().max().cpu()), float(bc.detach().cpu())))
    loss = torch.stack(losses).mean()
    return loss, infos


def evaluate(model, args, run_dir: Path, device: str):
    V, _ = cheb_eval_matrix(args.n, device)
    y, D1, D2 = build_diff_mats(args.n, args.y_match, device)
    y_np = y.detach().cpu().numpy()
    rows = []
    aa = np.linspace(args.a_center - args.a_half_width, args.a_center + args.a_half_width, args.eval_param_side)
    ll = np.linspace(args.logw_center - args.logw_half_width, args.logw_center + args.logw_half_width, args.eval_param_side)
    center_plot = None
    with torch.no_grad():
        for av in aa:
            for lw in ll:
                a = torch.tensor([av], dtype=RDTYPE, device=device)
                logw = torch.tensor([lw], dtype=RDTYPE, device=device)
                coeff = model(a, logw)
                S = (coeff @ V.to(CDTYPE).T).squeeze(0).cpu().numpy()
                teacher = solve_horizon_local(a=float(av), omega=10.0**float(lw), ell=args.ell, m=args.m, s=args.s, y_match=args.y_match, n=args.n)
                rel_teacher = np.abs(S - teacher["S"]) / np.maximum(np.abs(teacher["S"]), 1e-14)
                loss, infos = residual_loss_from_coeff(model, a, logw, V, y, D1, D2, args)
                _, _, S_ref, mask = pybhpt_reference_S(a=float(av), omega=10.0**float(lw), ell=args.ell, m=args.m, s=args.s, y=y_np, horizon_exclude=args.horizon_exclude)
                valid = mask & np.isfinite(S_ref.real)
                rel_py = np.abs(S[valid] - S_ref[valid]) / np.maximum(np.abs(S_ref[valid]), 1e-14)
                row = {
                    "a": float(av), "logw": float(lw), "omega": 10.0**float(lw),
                    "teacher_rel_median": float(np.median(rel_teacher)),
                    "teacher_rel_max": float(np.max(rel_teacher)),
                    "pybhpt_rel_median": float(np.median(rel_py)),
                    "pybhpt_rel_max": float(np.max(rel_py)),
                    "res_rel_median": infos[0][0],
                    "res_rel_max": infos[0][1],
                    "bc": infos[0][2],
                }
                rows.append(row)
                if abs(av - args.a_center) < 1e-12 and abs(lw - args.logw_center) < 1e-12:
                    center_plot = (S, teacher["S"], S_ref, valid, row)
    write_csv(run_dir / "eval.csv", rows)
    if center_plot is not None:
        fig_dir = run_dir / "figures"
        fig_dir.mkdir(exist_ok=True)
        S, St, Sr, valid, row = center_plot
        fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
        axes[0].plot(y_np, S.real, label="student")
        axes[0].plot(y_np, St.real, "--", label="teacher")
        axes[0].plot(y_np[valid], Sr[valid].real, ":", label="pybhpt")
        axes[0].set_ylabel("Re(S)"); axes[0].legend(); axes[0].grid(alpha=.3)
        axes[1].plot(y_np, S.imag, label="student")
        axes[1].plot(y_np, St.imag, "--", label="teacher")
        axes[1].plot(y_np[valid], Sr[valid].imag, ":", label="pybhpt")
        axes[1].set_ylabel("Im(S)"); axes[1].legend(); axes[1].grid(alpha=.3)
        axes[2].semilogy(y_np, np.abs(S-St)/np.maximum(np.abs(St),1e-14), label="vs teacher")
        rel_full = np.full_like(y_np, np.nan, dtype=float)
        rel_full[valid] = np.abs(S[valid]-Sr[valid])/np.maximum(np.abs(Sr[valid]),1e-14)
        axes[2].semilogy(y_np, rel_full, label="vs pybhpt")
        axes[2].set_xlabel("physical y"); axes[2].set_ylabel("rel err"); axes[2].legend(); axes[2].grid(alpha=.3)
        fig.suptitle(f"center coeff net: py_med={row['pybhpt_rel_median']:.2e}, res_med={row['res_rel_median']:.2e}")
        fig.tight_layout()
        fig.savefig(fig_dir / "center_coeff_student.png", dpi=160, bbox_inches="tight")
        plt.close(fig)
    return rows


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
    parser.add_argument("--y-match", type=float, default=0.0)
    parser.add_argument("--n", type=int, default=80)
    parser.add_argument("--out-coeff", type=int, default=0, help="Number of low-order coefficients to output; 0 means n+1")
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--coeff-steps", type=int, default=3000)
    parser.add_argument("--res-steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--res-lr", type=float, default=1e-5)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--bc-weight", type=float, default=1e4)
    parser.add_argument("--coeff-rel-floor", type=float, default=1e-10)
    parser.add_argument("--value-weight", type=float, default=1.0)
    parser.add_argument("--dy-weight", type=float, default=1e-4)
    parser.add_argument("--dyy-weight", type=float, default=1e-8)
    parser.add_argument("--tail-weight", type=float, default=0.0)
    parser.add_argument("--tail-start", type=int, default=12)
    parser.add_argument("--tail-decay", type=float, default=0.35)
    parser.add_argument("--init-center-bias", action="store_true")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--horizon-exclude", type=float, default=1e-3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", default="outputs/horizon_spectral_coeff")
    args = parser.parse_args()

    torch.set_default_dtype(RDTYPE)
    run_dir = Path(args.output_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "figures").mkdir(exist_ok=True)
    with open(run_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    print("Building coefficient dataset...", flush=True)
    records, coeff_np = build_dataset(args, run_dir)
    V, _ = cheb_eval_matrix(args.n, args.device)
    y, D1, D2 = build_diff_mats(args.n, args.y_match, args.device)
    a_arr = torch.tensor([r["a"] for r in records], dtype=RDTYPE, device=args.device)
    logw_arr = torch.tensor([r["logw"] for r in records], dtype=RDTYPE, device=args.device)
    target = torch.tensor(coeff_np, dtype=CDTYPE, device=args.device)
    target_values = target @ V.to(CDTYPE).T
    target_y = target_values @ D1.to(CDTYPE).T
    target_yy = target_values @ D2.to(CDTYPE).T
    out_coeff = args.out_coeff if args.out_coeff and args.out_coeff > 0 else args.n + 1
    out_coeff = min(out_coeff, args.n + 1)
    model = ParamToCoeff(args.n + 1, args.width, args.depth, args.a_center, args.logw_center, args.a_half_width, args.logw_half_width, out_coeff=out_coeff).to(args.device, dtype=RDTYPE)
    if args.init_center_bias:
        center_idx = int(np.argmin([abs(r["a"] - args.a_center) + abs(r["logw"] - args.logw_center) for r in records]))
        center_coeff = coeff_np[center_idx, :out_coeff]
        last_linear = None
        for module in reversed(model.net):
            if isinstance(module, nn.Linear):
                last_linear = module
                break
        if last_linear is not None:
            with torch.no_grad():
                last_linear.weight.zero_()
                bias = np.empty(2 * out_coeff, dtype=np.float64)
                bias[0::2] = center_coeff.real
                bias[1::2] = center_coeff.imag
                last_linear.bias.copy_(torch.tensor(bias, dtype=RDTYPE, device=args.device))
    rng = np.random.default_rng(123)
    rows = []
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-8)
    k_idx = torch.arange(args.n + 1, dtype=RDTYPE, device=args.device)
    tail_start = min(max(args.tail_start, 1), args.n)
    tail_mask = k_idx >= tail_start
    tail_floor = torch.exp(-args.tail_decay * (k_idx - tail_start).clamp_min(0.0)).clamp_min(1e-12)
    for step in range(1, args.coeff_steps + 1):
        idx = torch.tensor(rng.choice(target.shape[0], min(args.batch, target.shape[0]), replace=False), dtype=torch.long, device=args.device)
        pred = model(a_arr[idx], logw_arr[idx])
        coeff_slice = slice(0, out_coeff)
        coeff_scale = torch.abs(target[idx, coeff_slice]).detach().clamp_min(args.coeff_rel_floor)
        coeff_loss = torch.mean((torch.abs(pred[:, coeff_slice] - target[idx, coeff_slice]) / coeff_scale) ** 2)
        pred_values = pred @ V.to(CDTYPE).T
        pred_y = pred_values @ D1.to(CDTYPE).T
        pred_yy = pred_values @ D2.to(CDTYPE).T
        value_loss = torch.mean((torch.abs(pred_values - target_values[idx]) / torch.abs(target_values[idx]).detach().clamp_min(1.0)) ** 2)
        dy_loss = torch.mean((torch.abs(pred_y - target_y[idx]) / torch.abs(target_y[idx]).detach().clamp_min(1.0)) ** 2)
        dyy_loss = torch.mean((torch.abs(pred_yy - target_yy[idx]) / torch.abs(target_yy[idx]).detach().clamp_min(1.0)) ** 2)
        low_amp = torch.max(torch.abs(pred[:, :tail_start]).detach(), dim=1, keepdim=True).values.clamp_min(1e-12)
        allowed_tail = low_amp * tail_floor.unsqueeze(0)
        tail_excess = torch.relu(torch.abs(pred) - allowed_tail)
        tail_loss = torch.mean((tail_excess[:, tail_mask] / allowed_tail[:, tail_mask]) ** 2)
        loss = (
            coeff_loss
            + args.value_weight * value_loss
            + args.dy_weight * dy_loss
            + args.dyy_weight * dyy_loss
            + args.tail_weight * tail_loss
        )
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if step == 1 or step % args.log_every == 0:
            rel = torch.abs(pred.detach()-target[idx]) / torch.abs(target[idx]).detach().clamp_min(1e-14)
            rel_low = torch.abs(pred[:, coeff_slice].detach()-target[idx, coeff_slice]) / torch.abs(target[idx, coeff_slice]).detach().clamp_min(1e-14)
            row = {
                "phase": "coeff",
                "step": step,
                "loss": float(loss.detach().cpu()),
                "coeff_loss": float(coeff_loss.detach().cpu()),
                "value_loss": float(value_loss.detach().cpu()),
                "dy_loss": float(dy_loss.detach().cpu()),
                "dyy_loss": float(dyy_loss.detach().cpu()),
                "tail_loss": float(tail_loss.detach().cpu()),
                "rel_median": float(rel.median().cpu()),
                "rel_max": float(rel.max().cpu()),
                "rel_low_median": float(rel_low.median().cpu()),
                "rel_low_max": float(rel_low.max().cpu()),
            }
            rows.append(row); print(f"coeff step={step} loss={row['loss']:.3e} rel_med={row['rel_median']:.3e} rel_max={row['rel_max']:.3e}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.res_lr, weight_decay=0.0)
    for step in range(1, args.res_steps + 1):
        idx = torch.tensor(rng.choice(target.shape[0], min(args.batch, target.shape[0]), replace=False), dtype=torch.long, device=args.device)
        loss, infos = residual_loss_from_coeff(model, a_arr[idx], logw_arr[idx], V, y, D1, D2, args)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if step == 1 or step % args.log_every == 0:
            row = {"phase": "residual", "step": args.coeff_steps + step, "loss": float(loss.detach().cpu()), "rel_median": float(np.median([x[0] for x in infos])), "rel_max": float(np.max([x[1] for x in infos]))}
            rows.append(row); print(f"res step={step} loss={row['loss']:.3e} rel_med={row['rel_median']:.3e} rel_max={row['rel_max']:.3e}", flush=True)

    write_csv(run_dir / "train_log.csv", rows)
    torch.save({"model": model.state_dict(), "args": vars(args), "out_coeff": out_coeff}, run_dir / "model.pt")
    eval_rows = evaluate(model, args, run_dir, args.device)
    summary = {
        "args": vars(args),
        "teacher_res_median_median": float(np.median([r["teacher_res_med"] for r in records])),
        "teacher_res_max_max": float(np.max([r["teacher_res_max"] for r in records])),
        "eval_teacher_rel_median_median": float(np.median([r["teacher_rel_median"] for r in eval_rows])),
        "eval_teacher_rel_max_max": float(np.max([r["teacher_rel_max"] for r in eval_rows])),
        "eval_pybhpt_rel_median_median": float(np.median([r["pybhpt_rel_median"] for r in eval_rows])),
        "eval_pybhpt_rel_max_max": float(np.max([r["pybhpt_rel_max"] for r in eval_rows])),
        "eval_res_rel_median_median": float(np.median([r["res_rel_median"] for r in eval_rows])),
        "eval_res_rel_max_max": float(np.max([r["res_rel_max"] for r in eval_rows])),
    }
    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved to {run_dir}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
