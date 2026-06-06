#!/usr/bin/env python3
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

from teukspec.atlas.patch_expert import PatchExpert  # noqa: E402
from teukspec.core.chebyshev import cheb_vandermonde  # noqa: E402
from teukspec.teachers.spectral_teacher import branch_specs, solve_teacher  # noqa: E402
from utils.mode import KerrMode  # noqa: E402


RDTYPE = torch.float64
CDTYPE = torch.complex128


class CoeffFiLMResidual(nn.Module):
    def __init__(self, n_coeff: int, hidden_dim: int = 128, depth: int = 4, fourier_bands: int = 4, alpha_init: float = 1.0e-3):
        super().__init__()
        self.n_coeff = int(n_coeff)
        self.fourier_bands = int(fourier_bands)
        in_dim = 2 + 4 * fourier_bands
        self.input = nn.Linear(in_dim, hidden_dim)
        self.film = nn.ModuleList([nn.Linear(2, 2 * hidden_dim) for _ in range(depth)])
        self.blocks = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(depth)])
        self.alpha = nn.Parameter(torch.full((depth,), float(alpha_init)))
        self.output = nn.Linear(hidden_dim, 2 * n_coeff)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        feats = [x]
        for band in range(self.fourier_bands):
            freq = np.pi * float(2**band)
            feats.append(torch.sin(freq * x))
            feats.append(torch.cos(freq * x))
        return torch.cat(feats, dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.nn.functional.silu(self.input(self.features(x)))
        for i, layer in enumerate(self.blocks):
            gamma, beta = self.film[i](x).chunk(2, dim=-1)
            upd = torch.nn.functional.silu((1.0 + gamma) * layer(h) + beta)
            alpha = torch.clamp(self.alpha[i], 0.0, 1.0)
            h = (1.0 - alpha) * h + alpha * upd
        out = self.output(h)
        return out[:, 0::2].to(CDTYPE) + 1j * out[:, 1::2].to(CDTYPE)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def sample_params(cfg: dict, n: int, rng: np.random.Generator, margin: float = 0.03):
    a = rng.uniform(
        cfg["a_center"] - cfg["a_half_width"] * (1.0 - margin),
        cfg["a_center"] + cfg["a_half_width"] * (1.0 - margin),
        n,
    )
    logw = rng.uniform(
        cfg["logw_center"] - cfg["logw_half_width"] * (1.0 - margin),
        cfg["logw_center"] + cfg["logw_half_width"] * (1.0 - margin),
        n,
    )
    return a, logw


def build_dataset(expert: PatchExpert, n_samples: int, seed: int):
    rng = np.random.default_rng(seed)
    cfg = expert.config
    spec = branch_specs(cfg["y_match"], cfg["match_width"])["up"]
    dec = expert.decoders["up"]
    a, logw = sample_params(cfg, n_samples, rng)
    base = np.zeros((n_samples, dec.n_z + 1), dtype=np.complex128)
    teacher = np.zeros_like(base)
    rows = []
    for i, (ai, lwi) in enumerate(zip(a, logw)):
        mode = KerrMode(M=1.0, a=float(ai), omega=10.0 ** float(lwi), ell=cfg.get("ell", 2), m=cfg.get("m", 2), s=cfg.get("s", -2))
        base[i] = dec.eval_coeff(float(ai), float(lwi))
        teacher[i], metrics = solve_teacher(mode, spec, cfg["n"], cfg["grid_kind"])
        rows.append({"a": float(ai), "logw": float(lwi), "omega": float(mode.omega), **metrics})
    x = np.stack([(a - cfg["a_center"]) / cfg["a_half_width"], (logw - cfg["logw_center"]) / cfg["logw_half_width"]], axis=1)
    return x, base, teacher, rows


def coeff_to_values(coeff: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    return coeff @ V.T


def metrics(base: np.ndarray, pred: np.ndarray, teacher: np.ndarray, V_np: np.ndarray) -> dict:
    base_v = base @ V_np.T
    pred_v = pred @ V_np.T
    teacher_v = teacher @ V_np.T
    def rel_l2(a, b):
        return np.linalg.norm(a - b, axis=1) / np.maximum(np.linalg.norm(b, axis=1), 1.0e-300)
    return {
        "base_value_rel_median": float(np.median(rel_l2(base_v, teacher_v))),
        "base_value_rel_max": float(np.max(rel_l2(base_v, teacher_v))),
        "pred_value_rel_median": float(np.median(rel_l2(pred_v, teacher_v))),
        "pred_value_rel_max": float(np.max(rel_l2(pred_v, teacher_v))),
        "base_coeff_rel_median": float(np.median(rel_l2(base, teacher))),
        "base_coeff_rel_max": float(np.max(rel_l2(base, teacher))),
        "pred_coeff_rel_median": float(np.median(rel_l2(pred, teacher))),
        "pred_coeff_rel_max": float(np.max(rel_l2(pred, teacher))),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch-dir", required=True)
    parser.add_argument("--loss-kind", choices=["mse", "log-value", "log1p-value", "mixed-log"], default="mse")
    parser.add_argument("--n-train", type=int, default=96)
    parser.add_argument("--n-val", type=int, default=49)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--fourier-bands", type=int, default=4)
    parser.add_argument("--lambda-coeff", type=float, default=1.0e-2)
    parser.add_argument("--lambda-tail", type=float, default=1.0e-6)
    parser.add_argument("--log-tau", type=float, default=1.0e-6)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--seed", type=int, default=20260606)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    torch.set_default_dtype(RDTYPE)
    expert = PatchExpert(args.patch_dir)
    dec = expert.decoders["up"]
    run_dir = Path(args.patch_dir) / "coeff_film_correctors" / f"up_{args.loss_kind}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    print(json.dumps({"status": "building_train_dataset", "n_train": args.n_train}, indent=2), flush=True)
    x_train, base_train, teacher_train, train_rows = build_dataset(expert, args.n_train, args.seed)
    print(json.dumps({"status": "building_val_dataset", "n_val": args.n_val}, indent=2), flush=True)
    x_val, base_val, teacher_val, val_rows = build_dataset(expert, args.n_val, args.seed + 1)
    write_csv(run_dir / "train_teacher_metrics.csv", train_rows)
    write_csv(run_dir / "val_teacher_metrics.csv", val_rows)

    rng = np.random.default_rng(args.seed + 2)
    V_np = cheb_vandermonde(np.cos(np.pi * np.arange(dec.n_z + 1) / dec.n_z), dec.n_z).astype(np.complex128)
    V = torch.tensor(V_np, dtype=CDTYPE, device=args.device)
    x_t = torch.tensor(x_train, dtype=RDTYPE, device=args.device)
    base_t = torch.tensor(base_train, dtype=CDTYPE, device=args.device)
    teacher_t = torch.tensor(teacher_train, dtype=CDTYPE, device=args.device)
    model = CoeffFiLMResidual(dec.n_z + 1, args.hidden_dim, args.depth, args.fourier_bands).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1.0e-6)

    rows = []
    for step in range(1, args.steps + 1):
        idx = torch.tensor(rng.integers(0, args.n_train, size=args.batch_size), dtype=torch.long, device=args.device)
        x = x_t[idx]
        base = base_t[idx]
        teacher = teacher_t[idx]
        delta = model(x)
        pred = base + delta
        pred_v = coeff_to_values(pred, V)
        teacher_v = coeff_to_values(teacher, V)
        rel_v = torch.linalg.norm(pred_v - teacher_v, dim=1) / torch.clamp(torch.linalg.norm(teacher_v, dim=1), min=1.0e-300)
        rel_c = torch.linalg.norm(pred - teacher, dim=1) / torch.clamp(torch.linalg.norm(teacher, dim=1), min=1.0e-300)
        if args.loss_kind == "mse":
            loss_value = torch.mean(rel_v**2)
        elif args.loss_kind == "log-value":
            loss_value = torch.mean(torch.log10(rel_v + 1.0e-14) ** 2)
        elif args.loss_kind == "log1p-value":
            loss_value = torch.mean(torch.log1p(rel_v / args.log_tau) ** 2)
        else:
            loss_value = torch.mean(rel_v**2) + 1.0e-3 * torch.mean(torch.log1p(rel_v / args.log_tau) ** 2)
        tail = torch.mean(torch.abs(delta[:, -12:]) ** 2)
        loss = loss_value + args.lambda_coeff * torch.mean(rel_c**2) + args.lambda_tail * tail
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step()
        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            with torch.no_grad():
                pred_val = base_val + model(torch.tensor(x_val, dtype=RDTYPE, device=args.device)).detach().cpu().numpy()
                m = metrics(base_val, pred_val, teacher_val, V_np)
            row = {"step": step, "loss": float(loss.detach().cpu()), **m}
            rows.append(row)
            print(json.dumps(row, indent=2), flush=True)

    with torch.no_grad():
        pred_val = base_val + model(torch.tensor(x_val, dtype=RDTYPE, device=args.device)).detach().cpu().numpy()
        final = metrics(base_val, pred_val, teacher_val, V_np)
    torch.save({"state_dict": model.state_dict(), "config": vars(args), "final": final}, run_dir / "up_coeff_film_corrector.pt")
    with open(run_dir / "summary.json", "w") as f:
        json.dump({"run_dir": str(run_dir), **final}, f, indent=2)
    write_csv(run_dir / "training_log.csv", rows)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy([r["step"] for r in rows], [r["base_value_rel_median"] for r in rows], "o-", label="base median")
    ax.semilogy([r["step"] for r in rows], [r["pred_value_rel_median"] for r in rows], "o-", label="corrected median")
    ax.semilogy([r["step"] for r in rows], [r["base_value_rel_max"] for r in rows], "s-", label="base max")
    ax.semilogy([r["step"] for r in rows], [r["pred_value_rel_max"] for r in rows], "s-", label="corrected max")
    ax.set_xlabel("step")
    ax.set_ylabel("up branch value relative error")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "up_coeff_film_training.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps({"run_dir": str(run_dir), **final}, indent=2), flush=True)


if __name__ == "__main__":
    main()
