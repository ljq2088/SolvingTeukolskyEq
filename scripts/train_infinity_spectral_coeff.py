#!/usr/bin/env python3
"""Train infinity-side Chebyshev coefficient surrogates for up/down bases.

Physical coordinate y=2r_+/r-1.  This experiment uses y in [-1, 0].
For outer bases:
    u_down = R_down / A_down,  A_down = r^-1 exp(-i omega r*)
    u_up   = R_up   / A_up,    A_up   = r^3  exp(+i omega r*)
and u(-1)=1 with derivative from the degenerate ODE.
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

from utils.amplitude import boundary_du_exact, coeffs_numeric  # noqa: E402
from utils.mode import KerrMode  # noqa: E402
from utils.matlcheb import cheb, real_to_cheb  # noqa: E402


RDTYPE = torch.float64
CDTYPE = torch.complex128


class ParamToCoeff(nn.Module):
    def __init__(self, n_coeff: int, out_coeff: int, width: int, depth: int, a_center: float, logw_center: float, a_scale: float, logw_scale: float):
        super().__init__()
        self.n_coeff = n_coeff
        self.out_coeff = out_coeff
        self.a_center = a_center
        self.logw_center = logw_center
        self.a_scale = a_scale
        self.logw_scale = logw_scale
        layers = []
        in_dim = 2
        for _ in range(depth):
            layers += [nn.Linear(in_dim, width), nn.SiLU()]
            in_dim = width
        layers.append(nn.Linear(in_dim, 2 * out_coeff))
        self.net = nn.Sequential(*layers)

    def forward(self, a, logw):
        x = torch.stack([(a - self.a_center) / self.a_scale, (logw - self.logw_center) / self.logw_scale], dim=-1)
        out = self.net(x)
        low = out[:, 0::2].to(CDTYPE) + 1j * out[:, 1::2].to(CDTYPE)
        full = torch.zeros((a.shape[0], self.n_coeff), dtype=CDTYPE, device=a.device)
        full[:, : self.out_coeff] = low
        return full


class ParamToScaledCoeff(nn.Module):
    def __init__(self, coeff_net: ParamToCoeff):
        super().__init__()
        self.coeff_net = coeff_net
        self.scale_head = nn.Sequential(
            nn.Linear(2, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )

    def forward(self, a, logw):
        coeff = self.coeff_net(a, logw)
        x = torch.stack([
            (a - self.coeff_net.a_center) / self.coeff_net.a_scale,
            (logw - self.coeff_net.logw_center) / self.coeff_net.logw_scale,
        ], dim=-1)
        log_scale = self.scale_head(x).squeeze(-1).clamp(-30.0, 30.0)
        return coeff * torch.exp(log_scale).to(CDTYPE).unsqueeze(-1)


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def domain_mats(n: int, y_left: float, y_right: float):
    Dxi, xi = cheb(n)
    half = 0.5 * (y_right - y_left)
    mid = 0.5 * (y_right + y_left)
    y = mid + half * xi
    D1 = Dxi / half
    D2 = D1 @ D1
    return y, D1, D2


def z_from_y(y: np.ndarray) -> np.ndarray:
    return 0.5 * (y + 1.0)


def solve_outer_local(mode: KerrMode, basis: str, *, n: int, y_right: float):
    y, Dy, Dyy = domain_mats(n, -1.0, y_right)
    z = z_from_y(y)
    Dz = 2.0 * Dy
    Dzz = 4.0 * Dyy
    A = np.zeros((n + 1, n + 1), dtype=np.complex128)
    b = np.zeros(n + 1, dtype=np.complex128)
    # cheb ordering: index n -> y=-1 / z=0, index 0 -> y=y_right.
    du_dz = boundary_du_exact(mode, basis, "left")
    du_dy = 0.5 * du_dz
    A[-1, -1] = 1.0
    b[-1] = 1.0
    A[-2, :] = Dy[-1, :]
    b[-2] = du_dy
    idx = np.arange(0, n - 1)
    B2, B1, B0 = coeffs_numeric(z[idx], mode, basis)
    # coeffs_numeric is in z; convert u_z=2u_y, u_zz=4u_yy via Dz/Dzz matrices.
    A[idx, :] = B2[:, None] * Dzz[idx, :] + B1[:, None] * Dz[idx, :]
    A[idx, idx] += B0
    row = np.linalg.norm(A, axis=1)
    row = np.where(row > 1e-300, row, 1.0)
    Ar = A / row[:, None]
    br = b / row
    col = np.linalg.norm(Ar, axis=0)
    col = np.where(col > 1e-300, col, 1.0)
    As = Ar / col[None, :]
    u = np.linalg.solve(As, br) / col
    uy = Dy @ u
    uyy = Dyy @ u
    B2all, B1all, B0all = coeffs_numeric(z[:-1], mode, basis)
    res = B2all * (Dzz @ u)[:-1] + B1all * (Dz @ u)[:-1] + B0all * u[:-1]
    den = np.maximum.reduce([
        np.abs(B2all * (Dzz @ u)[:-1]),
        np.abs(B1all * (Dz @ u)[:-1]),
        np.abs(B0all * u[:-1]),
    ])
    rel = np.abs(res) / np.maximum(den, 1e-300)
    return {
        "y": y,
        "z": z,
        "u": u,
        "uy": uy,
        "uyy": uyy,
        "cond": float(np.linalg.cond(As)),
        "rel_res_median": float(np.median(rel)),
        "rel_res_mean": float(np.mean(rel)),
        "rel_res_max": float(np.max(rel)),
    }


def cheb_eval_matrix(n: int, device: str):
    _, xi = cheb(n)
    k = np.arange(n + 1)
    V = np.cos(np.outer(np.arccos(xi), k))
    return torch.tensor(V, dtype=RDTYPE, device=device)


def build_dataset(args, run_dir: Path):
    aa = np.linspace(args.a_center - args.a_half_width, args.a_center + args.a_half_width, args.n_param_side)
    ll = np.linspace(args.logw_center - args.logw_half_width, args.logw_center + args.logw_half_width, args.n_param_side)
    rows, coeffs = [], []
    for basis in args.bases.split(","):
        basis = basis.strip()
        for a in aa:
            for logw in ll:
                mode = KerrMode(M=1.0, a=float(a), omega=10.0 ** float(logw), ell=args.ell, m=args.m, s=args.s)
                sol = solve_outer_local(mode, basis, n=args.n, y_right=args.y_right)
                c = real_to_cheb(sol["u"])
                coeffs.append(c)
                rows.append({
                    "basis": basis,
                    "a": float(a),
                    "logw": float(logw),
                    "omega": mode.omega,
                    "teacher_res_med": sol["rel_res_median"],
                    "teacher_res_max": sol["rel_res_max"],
                    "cond": sol["cond"],
                })
    coeffs = np.stack(coeffs).astype(np.complex128)
    np.savez(run_dir / "coeff_dataset.npz", coeffs=coeffs, records=np.array(rows, dtype=object))
    write_csv(run_dir / "coeff_dataset.csv", rows)
    return rows, coeffs


def residual_metrics(mode: KerrMode, basis: str, coeff: np.ndarray, args):
    y, Dy, Dyy = domain_mats(args.n, -1.0, args.y_right)
    z = z_from_y(y)
    _, xi = cheb(args.n)
    V = np.cos(np.outer(np.arccos(xi), np.arange(args.n + 1)))
    u = V @ coeff
    Dz = 2.0 * Dy
    Dzz = 4.0 * Dyy
    uz = Dz @ u
    uzz = Dzz @ u
    B2, B1, B0 = coeffs_numeric(z[:-1], mode, basis)
    res = B2 * uzz[:-1] + B1 * uz[:-1] + B0 * u[:-1]
    den = np.maximum.reduce([np.abs(B2 * uzz[:-1]), np.abs(B1 * uz[:-1]), np.abs(B0 * u[:-1])])
    rel = np.abs(res) / np.maximum(den, 1e-300)
    return float(np.median(rel)), float(np.max(rel)), y, u


def train_one_basis(args, all_records, all_coeffs, basis: str, run_dir: Path):
    records = [r for r in all_records if r["basis"] == basis]
    indices = [i for i, r in enumerate(all_records) if r["basis"] == basis]
    coeff_np = all_coeffs[indices]
    a_arr = torch.tensor([r["a"] for r in records], dtype=RDTYPE, device=args.device)
    logw_arr = torch.tensor([r["logw"] for r in records], dtype=RDTYPE, device=args.device)
    target = torch.tensor(coeff_np, dtype=CDTYPE, device=args.device)
    out_coeff = min(args.out_coeff, args.n + 1)
    base_model = ParamToCoeff(args.n + 1, out_coeff, args.width, args.depth, args.a_center, args.logw_center, args.a_half_width, args.logw_half_width)
    model = base_model.to(args.device, dtype=RDTYPE)
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
    V = cheb_eval_matrix(args.n, args.device)
    target_values = target @ V.to(CDTYPE).T
    target_scale = torch.max(torch.abs(target[:, :out_coeff]), dim=1, keepdim=True).values.clamp_min(1e-30)
    target_scaled = target / target_scale
    target_values_scaled = target_scaled @ V.to(CDTYPE).T
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-8)
    rng = np.random.default_rng(123)
    rows = []
    coeff_slice = slice(0, out_coeff)
    for step in range(1, args.steps + 1):
        idx = torch.tensor(rng.choice(target.shape[0], min(args.batch, target.shape[0]), replace=False), dtype=torch.long, device=args.device)
        pred = model(a_arr[idx], logw_arr[idx])
        if args.normalize_coeff:
            pred_train = pred
            target_train = target_scaled[idx]
            target_value_train = target_values_scaled[idx]
        else:
            pred_train = pred
            target_train = target[idx]
            target_value_train = target_values[idx]
        coeff_scale = torch.abs(target_train[:, coeff_slice]).detach().clamp_min(args.coeff_rel_floor)
        coeff_loss = torch.mean((torch.abs(pred_train[:, coeff_slice] - target_train[:, coeff_slice]) / coeff_scale) ** 2)
        pred_values = pred @ V.to(CDTYPE).T
        value_pred_train = pred_values if not args.normalize_coeff else pred_values
        value_loss = torch.mean((torch.abs(value_pred_train - target_value_train) / torch.abs(target_value_train).detach().clamp_min(1.0)) ** 2)
        loss = coeff_loss + args.value_weight * value_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step == 1 or step % args.log_every == 0:
            pred_eval = pred * target_scale[idx] if args.normalize_coeff else pred
            pred_eval_values = pred_eval @ V.to(CDTYPE).T
            relv = torch.abs(pred_eval_values.detach() - target_values[idx]) / torch.abs(target_values[idx]).detach().clamp_min(1e-14)
            row = {"basis": basis, "step": step, "loss": float(loss.detach().cpu()), "value_rel_median": float(relv.median().cpu()), "value_rel_max": float(relv.max().cpu())}
            rows.append(row)
            print(f"{basis} step={step} loss={row['loss']:.3e} val_med={row['value_rel_median']:.3e} val_max={row['value_rel_max']:.3e}", flush=True)
    torch.save({"model": model.state_dict(), "args": vars(args), "basis": basis, "out_coeff": out_coeff, "normalize_coeff": args.normalize_coeff}, run_dir / f"model_{basis}.pt")
    write_csv(run_dir / f"train_log_{basis}.csv", rows)
    return model


def evaluate(args, models: dict[str, nn.Module], all_records, all_coeffs, run_dir: Path):
    rows = []
    fig_dir = run_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    for basis, model in models.items():
        indices = [i for i, r in enumerate(all_records) if r["basis"] == basis]
        records = [all_records[i] for i in indices]
        coeff_np = all_coeffs[indices]
        model.eval()
        for i, rec in enumerate(records):
            if rec["a"] not in np.linspace(args.a_center - args.a_half_width, args.a_center + args.a_half_width, args.eval_param_side):
                continue
            if rec["logw"] not in np.linspace(args.logw_center - args.logw_half_width, args.logw_center + args.logw_half_width, args.eval_param_side):
                continue
            with torch.no_grad():
                pred = model(torch.tensor([rec["a"]], dtype=RDTYPE, device=args.device), torch.tensor([rec["logw"]], dtype=RDTYPE, device=args.device)).squeeze(0).cpu().numpy()
            teacher = coeff_np[i]
            mode = KerrMode(M=1.0, a=rec["a"], omega=rec["omega"], ell=args.ell, m=args.m, s=args.s)
            res_med, res_max, y, u = residual_metrics(mode, basis, pred, args)
            _, _, _, u_teacher = residual_metrics(mode, basis, teacher, args)
            rel = np.abs(u - u_teacher) / np.maximum(np.abs(u_teacher), 1e-14)
            row = {"basis": basis, "a": rec["a"], "logw": rec["logw"], "omega": rec["omega"], "value_rel_median": float(np.median(rel)), "value_rel_max": float(np.max(rel)), "res_rel_median": res_med, "res_rel_max": res_max}
            rows.append(row)
            if abs(rec["a"] - args.a_center) < 1e-12 and abs(rec["logw"] - args.logw_center) < 1e-12:
                fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
                axes[0].plot(y, u.real, label="student")
                axes[0].plot(y, u_teacher.real, "--", label="teacher")
                axes[0].set_ylabel(f"Re(u_{basis})"); axes[0].legend(); axes[0].grid(alpha=.3)
                axes[1].plot(y, u.imag, label="student")
                axes[1].plot(y, u_teacher.imag, "--", label="teacher")
                axes[1].set_ylabel(f"Im(u_{basis})"); axes[1].legend(); axes[1].grid(alpha=.3)
                axes[2].semilogy(y, rel, label="rel")
                axes[2].set_xlabel("physical y"); axes[2].set_ylabel("rel err"); axes[2].legend(); axes[2].grid(alpha=.3)
                fig.suptitle(f"{basis}: val_med={row['value_rel_median']:.2e}, res_med={res_med:.2e}")
                fig.tight_layout()
                fig.savefig(fig_dir / f"center_{basis}.png", dpi=160, bbox_inches="tight")
                plt.close(fig)
    write_csv(run_dir / "eval.csv", rows)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--a-center", type=float, default=0.5)
    parser.add_argument("--logw-center", type=float, default=-1.5)
    parser.add_argument("--a-half-width", type=float, default=0.02)
    parser.add_argument("--logw-half-width", type=float, default=0.08)
    parser.add_argument("--n-param-side", type=int, default=7)
    parser.add_argument("--eval-param-side", type=int, default=5)
    parser.add_argument("--ell", type=int, default=2)
    parser.add_argument("--m", type=int, default=2)
    parser.add_argument("--s", type=int, default=-2)
    parser.add_argument("--y-right", type=float, default=0.0)
    parser.add_argument("--n", type=int, default=80)
    parser.add_argument("--out-coeff", type=int, default=16)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--steps", type=int, default=2500)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--coeff-rel-floor", type=float, default=1e-8)
    parser.add_argument("--value-weight", type=float, default=1.0)
    parser.add_argument("--normalize-coeff", action="store_true")
    parser.add_argument("--bases", default="down,up")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", default="outputs/infinity_spectral_coeff")
    args = parser.parse_args()

    torch.set_default_dtype(RDTYPE)
    run_dir = Path(args.output_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "figures").mkdir(exist_ok=True)
    with open(run_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    print("Building infinity coefficient dataset...", flush=True)
    records, coeffs = build_dataset(args, run_dir)
    models = {}
    for basis in [b.strip() for b in args.bases.split(",") if b.strip()]:
        models[basis] = train_one_basis(args, records, coeffs, basis, run_dir)
    eval_rows = evaluate(args, models, records, coeffs, run_dir)
    summary = {"args": vars(args)}
    for basis in models:
        br = [r for r in eval_rows if r["basis"] == basis]
        summary[basis] = {
            "value_rel_median_median": float(np.median([r["value_rel_median"] for r in br])),
            "value_rel_max_max": float(np.max([r["value_rel_max"] for r in br])),
            "res_rel_median_median": float(np.median([r["res_rel_median"] for r in br])),
            "res_rel_max_max": float(np.max([r["res_rel_max"] for r in br])),
        }
    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved to {run_dir}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
