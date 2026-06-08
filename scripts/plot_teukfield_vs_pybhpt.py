#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pybhpt_usage.compute_solution import compute_pybhpt_solution  # noqa: E402
from scripts.train_teukfield_ampnet import build_model  # noqa: E402
from teukfield.physics.angular import lambda_spheroidal  # noqa: E402
from teukfield.physics.coordinates import r_plus, y_from_z  # noqa: E402
from teukfield.physics.factors import leaver_P_h2  # noqa: E402


def complex_relerr(pred: np.ndarray, ref: np.ndarray, floor: float = 1.0e-300) -> np.ndarray:
    return np.abs(pred - ref) / np.maximum(np.abs(ref), floor)


def ph2_numpy(y_values: np.ndarray, a: float, logw: float, device: str) -> np.ndarray:
    with torch.no_grad():
        y = torch.tensor(y_values[None, :], dtype=torch.float64, device=device)
        a_t = torch.tensor([a], dtype=torch.float64, device=device)
        logw_t = torch.tensor([logw], dtype=torch.float64, device=device)
        omega_t = torch.pow(torch.tensor(10.0, dtype=torch.float64, device=device), logw_t)
        _, _, ph2 = leaver_P_h2(y, a_t, omega_t)
    return ph2.detach().cpu().numpy()[0].astype(np.complex128)


def pybhpt_sorted(a: float, omega: float, r_values: np.ndarray, timeout: float) -> np.ndarray:
    order = np.argsort(r_values)
    inv = np.empty_like(order)
    inv[order] = np.arange(order.size)
    _, ref_sorted = compute_pybhpt_solution(a, omega, ell=2, m=2, r_grid=np.asarray(r_values)[order], timeout=timeout)
    return np.asarray(ref_sorted, dtype=np.complex128)[inv]


def evaluate_one(
    model,
    cfg: dict,
    a: float,
    logw: float,
    z_values_r: np.ndarray,
    y_values_s: np.ndarray,
    device: str,
    timeout: float,
) -> dict:
    omega = 10.0 ** logw
    with torch.no_grad():
        z_r = torch.tensor(z_values_r[None, :], dtype=torch.float64, device=device)
        y_r = y_from_z(z_r)
        y_s = torch.tensor(y_values_s[None, :], dtype=torch.float64, device=device)
        a_t = torch.tensor([a], dtype=torch.float64, device=device)
        logw_t = torch.tensor([logw], dtype=torch.float64, device=device)
        omega_t = torch.pow(torch.tensor(10.0, dtype=torch.float64, device=device), logw_t)
        lambda_t = lambda_spheroidal(a_t, omega_t, ell=cfg["physics"]["l"], m=cfg["physics"]["m"], s=cfg["physics"]["s"])
        out_r = model(y_r, a_t, logw_t, lambda_=lambda_t)
        out_s = model(y_s, a_t, logw_t, lambda_=lambda_t)
        R_pred = out_r["R"].detach().cpu().numpy()[0].astype(np.complex128)
        S_pred = out_s["S"].detach().cpu().numpy()[0].astype(np.complex128)
        r_values = (float((1.0 + np.sqrt(1.0 - a * a))) / z_values_r).astype(np.float64)

    R_ref = pybhpt_sorted(a, omega, r_values, timeout)
    z_values_s = 0.5 * (y_values_s + 1.0)
    r_values_s = (float((1.0 + np.sqrt(1.0 - a * a))) / z_values_s).astype(np.float64)
    R_ref_s = pybhpt_sorted(a, omega, r_values_s, timeout)
    ph2_s = ph2_numpy(y_values_s, a, logw, device)
    S_ref = R_ref_s / ph2_s
    R_pred_s = out_s["R"].detach().cpu().numpy()[0].astype(np.complex128)
    rel_R = complex_relerr(R_pred, R_ref)
    rel_S = complex_relerr(S_pred, S_ref)
    return {
        "a": a,
        "logw": logw,
        "omega": omega,
        "z_R": z_values_r,
        "y_R": 2.0 * z_values_r - 1.0,
        "z_S": z_values_s,
        "y_S": y_values_s,
        "r": r_values,
        "R_pred": R_pred,
        "R_ref": R_ref,
        "r_S": r_values_s,
        "R_pred_Sgrid": R_pred_s,
        "R_ref_Sgrid": R_ref_s,
        "S_pred": S_pred,
        "S_ref": S_ref,
        "rel_R": rel_R,
        "rel_S": rel_S,
    }


def plot_case(case: dict, fig_path: Path) -> None:
    y = case["y_S"]
    r = case["r"]
    fig, axes = plt.subplots(3, 2, figsize=(15, 14))
    fig.suptitle(
        f"Teukfield-AmpNet vs pybhpt, a={case['a']:.6f}, omega={case['omega']:.6e}, "
        f"R med={np.median(case['rel_R']):.2e}, S med={np.median(case['rel_S']):.2e}"
    )

    axes[0, 0].plot(r, case["R_pred"].real, label="Pred Re(R)")
    axes[0, 0].plot(r, case["R_ref"].real, "--", label="pybhpt Re(R)")
    axes[0, 0].set_ylabel("Re(R)")
    axes[0, 0].set_title("R_in over r")
    axes[0, 0].legend()
    axes[1, 0].plot(r, case["R_pred"].imag, label="Pred Im(R)")
    axes[1, 0].plot(r, case["R_ref"].imag, "--", label="pybhpt Im(R)")
    axes[1, 0].set_ylabel("Im(R)")
    axes[1, 0].legend()
    axes[2, 0].plot(r, np.abs(case["R_pred"]), label="Pred |R|")
    axes[2, 0].plot(r, np.abs(case["R_ref"]), "--", label="pybhpt |R|")
    axes[2, 0].set_ylabel("|R|")
    axes[2, 0].set_xlabel("r")
    axes[2, 0].legend()

    axes[0, 1].plot(y, case["S_pred"].real, label="Pred Re(S)")
    axes[0, 1].plot(y, case["S_ref"].real, "--", label="pybhpt Re(S)")
    axes[0, 1].set_ylabel("Re(S)")
    axes[0, 1].set_title("S=R/(P h2) over y")
    axes[0, 1].legend()
    axes[1, 1].plot(y, case["S_pred"].imag, label="Pred Im(S)")
    axes[1, 1].plot(y, case["S_ref"].imag, "--", label="pybhpt Im(S)")
    axes[1, 1].set_ylabel("Im(S)")
    axes[1, 1].legend()
    axes[2, 1].plot(y, np.abs(case["S_pred"]), label="Pred |S|")
    axes[2, 1].plot(y, np.abs(case["S_ref"]), "--", label="pybhpt |S|")
    axes[2, 1].set_ylabel("|S|")
    axes[2, 1].set_xlabel("y")
    axes[2, 1].legend()
    for ax in axes.ravel():
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/teukfield_ampnet_moderate_patch.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--n-param", type=int, default=5)
    parser.add_argument("--n-z", type=int, default=160)
    parser.add_argument("--r-max", type=float, default=200.0)
    parser.add_argument("--r-near-points", type=int, default=120)
    parser.add_argument("--r-far-points", type=int, default=160)
    parser.add_argument("--r-near-max", type=float, default=20.0)
    parser.add_argument("--n-y-s", type=int, default=500)
    parser.add_argument("--y-s-min", type=float, default=-0.98)
    parser.add_argument("--y-s-max", type=float, default=0.98)
    parser.add_argument("--z-min", type=float, default=None)
    parser.add_argument("--z-max", type=float, default=0.98)
    parser.add_argument("--timeout", type=float, default=45.0)
    parser.add_argument("--seed", type=int, default=20260608)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    model = build_model(cfg).to(args.device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=args.device))
    model.eval()

    out_dir = Path(args.output_dir)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    patch = cfg["patch"]
    params = [(patch["a_center"], patch["logw_center"])]
    for _ in range(max(0, args.n_param - 1)):
        params.append(
            (
                float(rng.uniform(patch["a_center"] - patch["a_half_width"], patch["a_center"] + patch["a_half_width"])),
                float(rng.uniform(patch["logw_center"] - patch["logw_half_width"], patch["logw_center"] + patch["logw_half_width"])),
            )
        )
    summaries = []
    failures = []
    for idx, (a, logw) in enumerate(params):
        try:
            rp = 1.0 + np.sqrt(1.0 - a * a)
            r_min = rp / float(args.z_max)
            r_near_max = min(float(args.r_near_max), float(args.r_max))
            if r_near_max <= r_min:
                r_values = np.linspace(r_min, float(args.r_max), args.n_z, dtype=np.float64)
            else:
                r_near = np.linspace(r_min, r_near_max, int(args.r_near_points), endpoint=False, dtype=np.float64)
                r_far = np.linspace(r_near_max, float(args.r_max), int(args.r_far_points), dtype=np.float64)
                r_values = np.unique(np.concatenate([r_near, r_far]))
            if args.z_min is not None:
                r_values = r_values[r_values <= rp / float(args.z_min)]
            z_values = rp / r_values
            y_values_s = np.linspace(float(args.y_s_min), float(args.y_s_max), int(args.n_y_s), dtype=np.float64)
            case = evaluate_one(model, cfg, a, logw, z_values, y_values_s, args.device, args.timeout)
            plot_case(case, fig_dir / f"pybhpt_R_S_case_{idx:02d}_a_{a:.4f}_logw_{logw:.4f}.png")
            summaries.append(
                {
                    "case": idx,
                    "a": a,
                    "logw": logw,
                    "omega": 10.0**logw,
                    "R_rel_median": float(np.median(case["rel_R"])),
                    "R_rel_p90": float(np.quantile(case["rel_R"], 0.9)),
                    "R_rel_max": float(np.max(case["rel_R"])),
                    "S_rel_median": float(np.median(case["rel_S"])),
                    "S_rel_p90": float(np.quantile(case["rel_S"], 0.9)),
                    "S_rel_max": float(np.max(case["rel_S"])),
                }
            )
        except Exception as exc:
            failures.append({"case": idx, "a": a, "logw": logw, "error": str(exc)})

    result = {"checkpoint": args.checkpoint, "summaries": summaries, "failures": failures}
    (out_dir / "pybhpt_R_S_summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
