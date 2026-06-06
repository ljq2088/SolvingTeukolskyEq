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
from utils.amplitude import A_down, A_in, A_up, Delta, Delta_p, K_of_r, d2z_dr2, dz_dr, q_and_qp, r_of_z  # noqa: E402


RDTYPE = torch.float64
CDTYPE = torch.complex128


class DeltaRFiLM(nn.Module):
    def __init__(self, hidden_dim=96, depth=4, fourier_bands=4, alpha_init=1.0e-3):
        super().__init__()
        self.fourier_bands = int(fourier_bands)
        in_dim = 3 + 6 * self.fourier_bands
        self.input = nn.Linear(in_dim, hidden_dim)
        self.blocks = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(depth)])
        self.film = nn.ModuleList([nn.Linear(2, 2 * hidden_dim) for _ in range(depth)])
        self.alpha = nn.Parameter(torch.full((depth,), float(alpha_init)))
        self.output = nn.Linear(hidden_dim, 2)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def features(self, z: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        z_scaled = 2.0 * z[:, None] - 1.0
        base = torch.cat([z_scaled, p], dim=-1)
        feats = [base]
        for band in range(self.fourier_bands):
            freq = np.pi * float(2**band)
            feats.append(torch.sin(freq * base))
            feats.append(torch.cos(freq * base))
        return torch.cat(feats, dim=-1)

    def forward(self, z: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        h = torch.nn.functional.silu(self.input(self.features(z, p)))
        for i, layer in enumerate(self.blocks):
            gamma, beta = self.film[i](p).chunk(2, dim=-1)
            upd = torch.nn.functional.silu((1.0 + gamma) * layer(h) + beta)
            alpha = torch.clamp(self.alpha[i], 0.0, 1.0)
            h = (1.0 - alpha) * h + alpha * upd
        out = self.output(h)
        return out[:, 0].to(CDTYPE) + 1j * out[:, 1].to(CDTYPE)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def complex_grad(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    real = torch.autograd.grad(y.real.sum(), x, create_graph=True, retain_graph=True)[0]
    imag = torch.autograd.grad(y.imag.sum(), x, create_graph=True, retain_graph=True)[0]
    return real.to(CDTYPE) + 1j * imag.to(CDTYPE)


def raw_R_coeffs_np(z: np.ndarray, mode):
    r = r_of_z(z, mode)
    zr = dz_dr(z, mode)
    zrr = d2z_dr2(z, mode)
    D = Delta(r, mode)
    Dp = Delta_p(r, mode)
    K = K_of_r(r, mode)
    V = (K * K + 4j * (r - mode.M) * K) / D - 8j * mode.omega * r - mode.lambda_value
    C2 = D * zr * zr
    C1 = D * zrr - Dp * zr
    C0 = V
    return C2.astype(np.complex128), C1.astype(np.complex128), C0.astype(np.complex128)


def sample_batch(expert: PatchExpert, basis: str, batch_size: int, rng: np.random.Generator):
    cfg = expert.config
    dec = expert.decoders[basis]
    a = rng.uniform(cfg["a_center"] - cfg["a_half_width"], cfg["a_center"] + cfg["a_half_width"], batch_size)
    logw = rng.uniform(cfg["logw_center"] - cfg["logw_half_width"], cfg["logw_center"] + cfg["logw_half_width"], batch_size)
    zl, zr = dec.z_domain
    z = rng.uniform(zl + 0.03 * (zr - zl), zr - 0.03 * (zr - zl), batch_size)
    return a, logw, z


def branch_factor(basis: str, r: np.ndarray, mode):
    if basis == "in":
        return A_in(r, mode)
    if basis == "down":
        return A_down(r, mode)
    if basis == "up":
        return A_up(r, mode)
    raise ValueError(basis)


def physical_R_from_spectral_u(basis: str, z: float, mode, u: complex, uz: complex, uzz: complex):
    z_arr = np.array([z], dtype=float)
    r = r_of_z(z_arr, mode)
    A = branch_factor(basis, r, mode)[0]
    q, qp = q_and_qp(r, mode, basis)
    zr = dz_dr(z_arr, mode)
    zrr = d2z_dr2(z_arr, mode)
    rz = 1.0 / zr
    rzz = -zrr / (zr**3)
    s = q * rz
    sz = qp * rz * rz + q * rzz
    R = A * u
    Rz = A * (uz + s[0] * u)
    Rzz = A * (uzz + 2.0 * s[0] * uz + (sz[0] + s[0] * s[0]) * u)
    return complex(R), complex(Rz), complex(Rzz)


def precompute_cache(expert: PatchExpert, basis: str, n_points: int, rng: np.random.Generator, device: str):
    cfg = expert.config
    a, logw, z = sample_batch(expert, basis, n_points, rng)
    R = np.zeros(n_points, dtype=np.complex128)
    Rz = np.zeros(n_points, dtype=np.complex128)
    Rzz = np.zeros(n_points, dtype=np.complex128)
    C2 = np.zeros(n_points, dtype=np.complex128)
    C1 = np.zeros(n_points, dtype=np.complex128)
    C0 = np.zeros(n_points, dtype=np.complex128)
    for i, (ai, lwi, zi) in enumerate(zip(a, logw, z)):
        omega = 10.0 ** float(lwi)
        mode = expert.mode(float(ai), omega)
        u, uz, uzz = expert.eval_branch_derivatives(float(ai), omega, basis, np.array([zi]))
        R[i], Rz[i], Rzz[i] = physical_R_from_spectral_u(basis, float(zi), mode, u[0], uz[0], uzz[0])
        c2, c1, c0 = raw_R_coeffs_np(np.array([zi]), mode)
        C2[i], C1[i], C0[i] = c2[0], c1[0], c0[0]
    p = np.stack([(a - cfg["a_center"]) / cfg["a_half_width"], (logw - cfg["logw_center"]) / cfg["logw_half_width"]], axis=1)
    scale = np.maximum(np.abs(R), 1.0)
    return {
        "z": torch.tensor(z, dtype=RDTYPE, device=device),
        "p": torch.tensor(p, dtype=RDTYPE, device=device),
        "R": torch.tensor(R, dtype=CDTYPE, device=device),
        "Rz": torch.tensor(Rz, dtype=CDTYPE, device=device),
        "Rzz": torch.tensor(Rzz, dtype=CDTYPE, device=device),
        "C2": torch.tensor(C2, dtype=CDTYPE, device=device),
        "C1": torch.tensor(C1, dtype=CDTYPE, device=device),
        "C0": torch.tensor(C0, dtype=CDTYPE, device=device),
        "scale": torch.tensor(scale, dtype=RDTYPE, device=device),
    }


def residual_rel(R, Rz, Rzz, C2, C1, C0):
    res = C2 * Rzz + C1 * Rz + C0 * R
    den = torch.maximum(torch.maximum(torch.abs(C2 * Rzz), torch.abs(C1 * Rz)), torch.abs(C0 * R)).clamp_min(1.0e-300)
    return torch.abs(res) / den


def loss_cached(model, cache, idx, lambda_amp, lambda_sob, loss_kind: str, log_tau: float):
    z0 = cache["z"][idx].detach().clone().requires_grad_(True)
    p = cache["p"][idx]
    scale = cache["scale"][idx].to(CDTYPE)
    raw = model(z0, p)
    delta = scale * raw
    dz = complex_grad(delta, z0)
    dzz = complex_grad(dz, z0)
    R = cache["R"][idx] + delta
    Rz = cache["Rz"][idx] + dz
    Rzz = cache["Rzz"][idx] + dzz
    rel = residual_rel(R, Rz, Rzz, cache["C2"][idx], cache["C1"][idx], cache["C0"][idx])
    amp = torch.mean(torch.abs(delta) ** 2 / torch.clamp(torch.abs(cache["R"][idx]) ** 2, min=1.0e-300))
    sob = torch.mean(torch.abs(dz / scale) ** 2 + 1.0e-3 * torch.abs(dzz / scale) ** 2)
    if loss_kind == "mse":
        loss_res = torch.mean(rel**2)
    elif loss_kind == "log1p":
        loss_res = torch.mean(torch.log1p(rel / log_tau) ** 2)
    elif loss_kind == "mixed":
        loss_res = torch.mean(rel**2) + 1.0e-6 * torch.mean(torch.log1p(rel / log_tau) ** 2)
    else:
        raise ValueError(loss_kind)
    return loss_res + lambda_amp * amp + lambda_sob * sob, {
        "rel_median": float(torch.median(rel.detach()).cpu()),
        "rel_max": float(torch.max(rel.detach()).cpu()),
        "delta_rel_median": float(torch.median(torch.abs(delta.detach()) / torch.clamp(torch.abs(cache["R"][idx]), min=1.0e-300)).cpu()),
        "loss_res": float(loss_res.detach().cpu()),
        "amp": float(amp.detach().cpu()),
        "sob": float(sob.detach().cpu()),
    }


def evaluate(model, cache):
    idx = torch.arange(cache["z"].shape[0], device=cache["z"].device)
    before = residual_rel(cache["R"], cache["Rz"], cache["Rzz"], cache["C2"], cache["C1"], cache["C0"]).detach().cpu().numpy()
    loss, metrics = loss_cached(model, cache, idx, 0.0, 0.0, "mse", 1.0e-7)
    return {
        "before_median": float(np.median(before)),
        "before_max": float(np.max(before)),
        "after_median": metrics["rel_median"],
        "after_max": metrics["rel_max"],
        "delta_rel_median": metrics["delta_rel_median"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch-dir", required=True)
    parser.add_argument("--basis", choices=["in", "down", "up"], default="up")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--cache-size", type=int, default=512)
    parser.add_argument("--val-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2.0e-5)
    parser.add_argument("--lambda-amp", type=float, default=1.0e-4)
    parser.add_argument("--lambda-sob", type=float, default=1.0e-8)
    parser.add_argument("--loss-kind", choices=["mse", "log1p", "mixed"], default="mse")
    parser.add_argument("--log-tau", type=float, default=1.0e-7)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--fourier-bands", type=int, default=3)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260606)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    torch.set_default_dtype(RDTYPE)
    rng = np.random.default_rng(args.seed)
    expert = PatchExpert(args.patch_dir)
    run_dir = Path(args.patch_dir) / "deltaR_correctors" / f"{args.basis}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    print(json.dumps({"status": "building_cache", "cache_size": args.cache_size, "val_size": args.val_size}, indent=2), flush=True)
    train_cache = precompute_cache(expert, args.basis, args.cache_size, rng, args.device)
    val_cache = precompute_cache(expert, args.basis, args.val_size, rng, args.device)
    model = DeltaRFiLM(args.hidden_dim, args.depth, args.fourier_bands).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1.0e-6)
    rows = []
    print(json.dumps({"run_dir": str(run_dir), "initial": evaluate(model, val_cache)}, indent=2), flush=True)
    for step in range(1, args.steps + 1):
        idx = torch.tensor(rng.integers(0, args.cache_size, size=args.batch_size), dtype=torch.long, device=args.device)
        loss, m = loss_cached(model, train_cache, idx, args.lambda_amp, args.lambda_sob, args.loss_kind, args.log_tau)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            ev = evaluate(model, val_cache)
            row = {"step": step, "loss": float(loss.detach().cpu()), **m, **{f"eval_{k}": v for k, v in ev.items()}}
            rows.append(row)
            print(json.dumps(row, indent=2), flush=True)
    write_csv(run_dir / "training_log.csv", rows)
    final = evaluate(model, val_cache)
    torch.save({"state_dict": model.state_dict(), "config": vars(args), "final": final}, run_dir / f"deltaR_{args.basis}.pt")
    with open(run_dir / "summary.json", "w") as f:
        json.dump({"run_dir": str(run_dir), "final": final}, f, indent=2)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy([r["step"] for r in rows], [r["eval_before_median"] for r in rows], "o-", label="before median")
    ax.semilogy([r["step"] for r in rows], [r["eval_after_median"] for r in rows], "o-", label="after median")
    ax.semilogy([r["step"] for r in rows], [r["eval_before_max"] for r in rows], "s-", label="before max")
    ax.semilogy([r["step"] for r in rows], [r["eval_after_max"] for r in rows], "s-", label="after max")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    ax.set_xlabel("step")
    ax.set_ylabel("full R reduced residual")
    fig.tight_layout()
    fig.savefig(run_dir / "deltaR_training.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps({"run_dir": str(run_dir), "final": final}, indent=2), flush=True)


if __name__ == "__main__":
    main()
