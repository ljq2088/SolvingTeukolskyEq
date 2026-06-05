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

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from teukspec.atlas.patch_expert import PatchExpert  # noqa: E402
from teukspec.correctors.film_pirate_corrector import FiLMPirateCorrector  # noqa: E402
from utils.amplitude import coeffs_numeric  # noqa: E402


RDTYPE = torch.float64
CDTYPE = torch.complex128


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


def sample_batch(expert: PatchExpert, branch: str, batch_size: int, rng: np.random.Generator):
    cfg = expert.config
    dec = expert.decoders[branch]
    a = rng.uniform(cfg["a_center"] - cfg["a_half_width"], cfg["a_center"] + cfg["a_half_width"], batch_size)
    logw = rng.uniform(cfg["logw_center"] - cfg["logw_half_width"], cfg["logw_center"] + cfg["logw_half_width"], batch_size)
    z_left, z_right = dec.z_domain
    z = rng.uniform(z_left + 0.03 * (z_right - z_left), z_right - 0.03 * (z_right - z_left), batch_size)
    return a, logw, z


def build_features(expert: PatchExpert, a: np.ndarray, logw: np.ndarray, device: str) -> torch.Tensor:
    cfg = expert.config
    xi_a = (a - cfg["a_center"]) / cfg["a_half_width"]
    xi_w = (logw - cfg["logw_center"]) / cfg["logw_half_width"]
    feats = np.stack([xi_a, xi_w], axis=1)
    return torch.tensor(feats, dtype=RDTYPE, device=device)


def spec_quantities(expert: PatchExpert, branch: str, a: np.ndarray, logw: np.ndarray, z: np.ndarray):
    u = np.zeros_like(z, dtype=np.complex128)
    uz = np.zeros_like(z, dtype=np.complex128)
    uzz = np.zeros_like(z, dtype=np.complex128)
    B2 = np.zeros_like(z, dtype=np.complex128)
    B1 = np.zeros_like(z, dtype=np.complex128)
    B0 = np.zeros_like(z, dtype=np.complex128)
    for i, (ai, lwi, zi) in enumerate(zip(a, logw, z)):
        omega = 10.0 ** float(lwi)
        mode = expert.mode(float(ai), omega)
        ui, uzi, uzzi = expert.eval_branch_derivatives(float(ai), omega, branch, np.array([zi]))
        b2, b1, b0 = coeffs_numeric(np.array([zi], dtype=float), mode, branch)
        u[i], uz[i], uzz[i] = ui[0], uzi[0], uzzi[0]
        B2[i], B1[i], B0[i] = b2[0], b1[0], b0[0]
    return u, uz, uzz, B2, B1, B0


def to_complex_tensor(x: np.ndarray, device: str) -> torch.Tensor:
    return torch.tensor(x, dtype=CDTYPE, device=device)


def precompute_cache(expert: PatchExpert, branch: str, n_points: int, rng: np.random.Generator, device: str) -> dict[str, torch.Tensor]:
    a, logw, z_np = sample_batch(expert, branch, n_points, rng)
    features = build_features(expert, a, logw, device)
    u_np, uz_np, uzz_np, B2_np, B1_np, B0_np = spec_quantities(expert, branch, a, logw, z_np)
    return {
        "z": torch.tensor(z_np, dtype=RDTYPE, device=device),
        "features": features,
        "u": to_complex_tensor(u_np, device),
        "uz": to_complex_tensor(uz_np, device),
        "uzz": to_complex_tensor(uzz_np, device),
        "B2": to_complex_tensor(B2_np, device),
        "B1": to_complex_tensor(B1_np, device),
        "B0": to_complex_tensor(B0_np, device),
    }


def residual_terms(u: torch.Tensor, uz: torch.Tensor, uzz: torch.Tensor, B2: torch.Tensor, B1: torch.Tensor, B0: torch.Tensor):
    res = B2 * uzz + B1 * uz + B0 * u
    den = torch.maximum(
        torch.maximum(torch.abs(B2 * uzz), torch.abs(B1 * uz)),
        torch.abs(B0 * u),
    ).clamp_min(1.0e-300)
    return torch.abs(res) / den


def loss_from_arrays(model, z_base, features, u_spec, uz_spec, uzz_spec, B2, B1, B0, lambda_amp: float, lambda_sob: float):
    z = z_base.detach().clone().requires_grad_(True)
    delta = model(z, features)
    delta_z = complex_grad(delta, z)
    delta_zz = complex_grad(delta_z, z)

    u = u_spec + delta
    uz = uz_spec + delta_z
    uzz = uzz_spec + delta_zz
    rel = residual_terms(u, uz, uzz, B2, B1, B0)
    amp = torch.mean(torch.abs(delta) ** 2 / torch.clamp(torch.abs(u_spec) ** 2, min=1.0e-300))
    sob = torch.mean(torch.abs(delta_z) ** 2 + 1.0e-3 * torch.abs(delta_zz) ** 2)
    loss_res = torch.mean(rel**2)
    loss = loss_res + lambda_amp * amp + lambda_sob * sob
    return loss, {
        "loss_res": float(loss_res.detach().cpu()),
        "rel_median": float(torch.median(rel.detach()).cpu()),
        "rel_max": float(torch.max(rel.detach()).cpu()),
        "amp": float(amp.detach().cpu()),
        "sob": float(sob.detach().cpu()),
        "delta_rel_median": float(torch.median(torch.abs(delta.detach()) / torch.clamp(torch.abs(u_spec), min=1.0e-300)).cpu()),
    }


def loss_batch(model, expert: PatchExpert, branch: str, batch_size: int, rng: np.random.Generator, device: str, lambda_amp: float, lambda_sob: float):
    cache = precompute_cache(expert, branch, batch_size, rng, device)
    return loss_from_arrays(
        model,
        cache["z"],
        cache["features"],
        cache["u"],
        cache["uz"],
        cache["uzz"],
        cache["B2"],
        cache["B1"],
        cache["B0"],
        lambda_amp,
        lambda_sob,
    )


def loss_cached(model, cache: dict[str, torch.Tensor], batch_size: int, rng: np.random.Generator, lambda_amp: float, lambda_sob: float):
    n = cache["z"].shape[0]
    idx = torch.tensor(rng.integers(0, n, size=batch_size), dtype=torch.long, device=cache["z"].device)
    return loss_from_arrays(
        model,
        cache["z"][idx],
        cache["features"][idx],
        cache["u"][idx],
        cache["uz"][idx],
        cache["uzz"][idx],
        cache["B2"][idx],
        cache["B1"][idx],
        cache["B0"][idx],
        lambda_amp,
        lambda_sob,
    )


def evaluate(model, expert: PatchExpert, branch: str, rng: np.random.Generator, device: str, n_points: int):
    model.eval()
    a, logw, z_np = sample_batch(expert, branch, n_points, rng)
    features = build_features(expert, a, logw, device)
    u_np, uz_np, uzz_np, B2_np, B1_np, B0_np = spec_quantities(expert, branch, a, logw, z_np)
    z = torch.tensor(z_np, dtype=RDTYPE, device=device, requires_grad=True)
    with torch.enable_grad():
        delta = model(z, features)
        delta_z = complex_grad(delta, z)
        delta_zz = complex_grad(delta_z, z)
        B2 = to_complex_tensor(B2_np, device)
        B1 = to_complex_tensor(B1_np, device)
        B0 = to_complex_tensor(B0_np, device)
        def rel_of(u, uz, uzz):
            return residual_terms(u, uz, uzz, B2, B1, B0)
        before = rel_of(to_complex_tensor(u_np, device), to_complex_tensor(uz_np, device), to_complex_tensor(uzz_np, device)).detach().cpu().numpy()
        after = rel_of(
            to_complex_tensor(u_np, device) + delta,
            to_complex_tensor(uz_np, device) + delta_z,
            to_complex_tensor(uzz_np, device) + delta_zz,
        ).detach().cpu().numpy()
        delta_rel = (torch.abs(delta.detach()) / torch.clamp(torch.abs(to_complex_tensor(u_np, device)), min=1.0e-300)).cpu().numpy()
    return {
        "before_median": float(np.median(before)),
        "before_max": float(np.max(before)),
        "after_median": float(np.median(after)),
        "after_max": float(np.max(after)),
        "delta_rel_median": float(np.median(delta_rel)),
        "delta_rel_max": float(np.max(delta_rel)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch-dir", required=True)
    parser.add_argument("--basis", choices=["in", "down", "up"], default="up")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--lambda-amp", type=float, default=1.0e-2)
    parser.add_argument("--lambda-sob", type=float, default=1.0e-5)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--fourier-bands", type=int, default=5)
    parser.add_argument("--alpha-init", type=float, default=1.0e-3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=20260605)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--cache-size", type=int, default=8192)
    args = parser.parse_args()

    torch.set_default_dtype(RDTYPE)
    rng = np.random.default_rng(args.seed)
    expert = PatchExpert(args.patch_dir)
    dec = expert.decoders[args.basis]
    run_dir = Path(args.patch_dir) / "correctors" / f"{args.basis}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    model = FiLMPirateCorrector(
        feature_dim=2,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        fourier_bands=args.fourier_bands,
        basis=args.basis,
        z_left=dec.z_domain[0],
        z_right=dec.z_domain[1],
        alpha_init=args.alpha_init,
    ).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1.0e-6)
    print(json.dumps({"status": "building_cache", "cache_size": args.cache_size}, indent=2), flush=True)
    cache = precompute_cache(expert, args.basis, args.cache_size, rng, args.device)
    rows = []
    initial = evaluate(model, expert, args.basis, rng, args.device, max(256, args.batch_size))
    print(json.dumps({"run_dir": str(run_dir), "initial": initial}, indent=2), flush=True)
    for step in range(1, args.steps + 1):
        model.train()
        opt.zero_grad(set_to_none=True)
        loss, metrics = loss_cached(model, cache, args.batch_size, rng, args.lambda_amp, args.lambda_sob)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            val = evaluate(model, expert, args.basis, rng, args.device, max(256, args.batch_size))
            row = {"step": step, "loss": float(loss.detach().cpu()), **metrics, **{f"eval_{k}": v for k, v in val.items()}}
            rows.append(row)
            print(json.dumps(row, indent=2), flush=True)
    write_csv(run_dir / "training_log.csv", rows)
    final = evaluate(model, expert, args.basis, rng, args.device, 1024)
    torch.save({"state_dict": model.state_dict(), "config": vars(args), "final": final}, run_dir / f"corrector_{args.basis}.pt")
    summary = {"run_dir": str(run_dir), "initial": initial, "final": final}
    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy([r["step"] for r in rows], [r["eval_before_median"] for r in rows], "o-", label="before")
    ax.semilogy([r["step"] for r in rows], [r["eval_after_median"] for r in rows], "o-", label="after")
    ax.set_xlabel("step")
    ax.set_ylabel("median reduced residual")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "residual_training.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
