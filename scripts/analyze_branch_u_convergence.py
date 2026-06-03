#!/usr/bin/env python3
"""Plot branch regular factors and scan spectral convergence.

Physical y = 2 r_+ / r - 1.  Horizon is y=1, infinity is y=-1.

For the horizon branch the current local solver variable is
    u_in := S = R_in / (P_Leaver h)
not R_in/A_in.  For the outer branches:
    u_down = R_down/A_down,  u_up = R_up/A_up.
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

from scripts.test_horizon_local_spectral import solve_horizon_local  # noqa: E402
from scripts.train_infinity_spectral_coeff import solve_outer_local  # noqa: E402
from utils.amplitude import solve_basis_domain  # noqa: E402
from utils.matlcheb import real_to_cheb  # noqa: E402
from utils.mode import KerrMode  # noqa: E402


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
    xi = np.clip((y - mid) / half, -1.0, 1.0)
    theta = np.arccos(xi)
    k = np.arange(coeff.shape[0])
    return np.cos(np.outer(theta, k)) @ coeff


def coeff_tail_metrics(coeff: np.ndarray, tail: int = 12) -> dict[str, float]:
    mag = np.abs(coeff)
    return {
        "coeff_max": float(np.max(mag)),
        "coeff_last": float(mag[-1]),
        "coeff_tail_max": float(np.max(mag[-tail:])),
        "coeff_tail_rel": float(np.max(mag[-tail:]) / max(np.max(mag), 1e-300)),
    }


def solve_branch(mode: KerrMode, branch: str, n: int, y_left: float, y_right: float):
    if branch == "in":
        sol = solve_horizon_local(
            a=mode.a,
            omega=mode.omega,
            ell=mode.ell,
            m=mode.m,
            s=mode.s,
            y_match=y_left,
            n=n,
        )
        y = sol["y"]
        u = sol["S"]
        coeff = real_to_cheb(u)
        res_med = sol["rel_res_median"]
        res_max = sol["rel_res_max"]
        cond = sol["cond_scaled"]
    elif branch == "in_A":
        z_left = 0.5 * (y_left + 1.0)
        sol = solve_basis_domain(mode, "in", n, z_left, 1.0, "right")
        z = sol["z"]
        y = 2.0 * z - 1.0
        u = sol["u"]
        coeff = real_to_cheb(u)
        # Reuse the spectral collocation residual scanner from the z-basis is
        # not duplicated here; coefficient convergence is the relevant metric.
        res_med = np.nan
        res_max = np.nan
        cond = np.nan
    else:
        sol = solve_outer_local(mode, branch, n=n, y_right=y_right)
        y = sol["y"]
        u = sol["u"]
        coeff = real_to_cheb(u)
        res_med = sol["rel_res_median"]
        res_max = sol["rel_res_max"]
        cond = sol["cond"]
    return {
        "branch": branch,
        "n": n,
        "y": y,
        "u": u,
        "coeff": coeff,
        "res_med": float(res_med),
        "res_max": float(res_max),
        "cond": float(cond),
        **coeff_tail_metrics(coeff),
    }


def add_profile_plot(axes, y, u, title):
    axes[0].plot(y, u.real, label=title)
    axes[1].plot(y, u.imag, label=title)
    axes[2].semilogy(y, np.maximum(np.abs(u), 1e-300), label=title)


def plot_profiles(run_dir: Path, sols: dict[str, dict]):
    fig, axes = plt.subplots(3, 1, figsize=(9, 11), sharex=False)
    add_profile_plot(axes, sols["in"]["y"], sols["in"]["u"], "u_in=S")
    if "in_A" in sols:
        add_profile_plot(axes, sols["in_A"]["y"], sols["in_A"]["u"], "u_in=R/A_in")
    add_profile_plot(axes, sols["down"]["y"], sols["down"]["u"], "u_down")
    add_profile_plot(axes, sols["up"]["y"], sols["up"]["u"], "u_up")
    axes[0].set_ylabel("Re(u)")
    axes[1].set_ylabel("Im(u)")
    axes[2].set_ylabel("|u|")
    axes[2].set_xlabel("physical y")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle("Regular branch factors over physical y")
    fig.tight_layout()
    fig.savefig(run_dir / "figures" / "branch_u_profiles.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_coeffs(run_dir: Path, sols: dict[str, dict]):
    fig, ax = plt.subplots(figsize=(9, 6))
    for label, sol in sols.items():
        ax.semilogy(np.abs(sol["coeff"]), marker=".", linewidth=1.0, label=label)
    ax.set_xlabel("Chebyshev mode k")
    ax.set_ylabel("|c_k|")
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_title("Chebyshev coefficient decay")
    fig.tight_layout()
    fig.savefig(run_dir / "figures" / "branch_coeff_decay.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def convergence_scan(mode: KerrMode, branch: str, n_values: list[int], y_left: float, y_right: float):
    ref = solve_branch(mode, branch, max(n_values), y_left, y_right)
    y_eval = np.linspace(y_left, y_right, 401)
    ref_u = cheb_eval(ref["coeff"], y_eval, y_left, y_right)
    rows = []
    for n in n_values:
        sol = solve_branch(mode, branch, n, y_left, y_right)
        u_eval = cheb_eval(sol["coeff"], y_eval, y_left, y_right)
        diff = np.abs(u_eval - ref_u) / np.maximum(np.abs(ref_u), 1e-300)
        row = {
            "branch": branch,
            "N": n,
            "y_left": y_left,
            "y_right": y_right,
            "res_med": sol["res_med"],
            "res_max": sol["res_max"],
            "cond_scaled_or_equilibrated": sol["cond"],
            "vs_ref_rel_median": float(np.median(diff)),
            "vs_ref_rel_max": float(np.max(diff)),
            "coeff_tail_rel": sol["coeff_tail_rel"],
            "coeff_last": sol["coeff_last"],
        }
        rows.append(row)
    return rows


def plot_convergence(run_dir: Path, rows: list[dict]):
    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
    branches = list(dict.fromkeys(row["branch"] for row in rows))
    for branch in branches:
        br = [row for row in rows if row["branch"] == branch]
        n = [row["N"] for row in br]
        axes[0].semilogy(n, [row["res_med"] for row in br], "o-", label=branch)
        axes[1].semilogy(n, [row["vs_ref_rel_median"] for row in br], "o-", label=branch)
        axes[2].semilogy(n, [row["coeff_tail_rel"] for row in br], "o-", label=branch)
    axes[0].set_ylabel("median relative residual")
    axes[1].set_ylabel("median rel diff vs highest N")
    axes[2].set_ylabel("tail coeff / max coeff")
    axes[2].set_xlabel("N")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle("Spectral convergence diagnostics")
    fig.tight_layout()
    fig.savefig(run_dir / "figures" / "branch_convergence.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def parse_ints(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", type=float, default=0.5)
    parser.add_argument("--omega", type=float, default=10.0)
    parser.add_argument("--ell", type=int, default=2)
    parser.add_argument("--m", type=int, default=2)
    parser.add_argument("--s", type=int, default=-2)
    parser.add_argument("--in-y-left", type=float, default=-0.2)
    parser.add_argument("--outer-y-right", type=float, default=0.41)
    parser.add_argument("--profile-n", type=int, default=160)
    parser.add_argument("--n-values", default="40,60,80,120,160")
    parser.add_argument("--output-dir", default="outputs/branch_u_convergence")
    args = parser.parse_args()

    mode = KerrMode(M=1.0, a=args.a, omega=args.omega, ell=args.ell, m=args.m, s=args.s)
    run_dir = Path(args.output_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    (run_dir / "figures").mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    sols = {
        "in": solve_branch(mode, "in", args.profile_n, args.in_y_left, 1.0),
        "in_A": solve_branch(mode, "in_A", args.profile_n, args.in_y_left, 1.0),
        "down": solve_branch(mode, "down", args.profile_n, -1.0, args.outer_y_right),
        "up": solve_branch(mode, "up", args.profile_n, -1.0, args.outer_y_right),
    }
    plot_profiles(run_dir, sols)
    plot_coeffs(run_dir, sols)

    profile_rows = []
    for label, sol in sols.items():
        for y, u in zip(sol["y"], sol["u"]):
            profile_rows.append({
                "branch": label,
                "y": float(y),
                "u_re": float(u.real),
                "u_im": float(u.imag),
                "u_abs": float(abs(u)),
            })
    write_csv(run_dir / "profiles.csv", profile_rows)

    n_values = parse_ints(args.n_values)
    rows = []
    rows += convergence_scan(mode, "in", n_values, args.in_y_left, 1.0)
    rows += convergence_scan(mode, "in_A", n_values, args.in_y_left, 1.0)
    rows += convergence_scan(mode, "down", n_values, -1.0, args.outer_y_right)
    rows += convergence_scan(mode, "up", n_values, -1.0, args.outer_y_right)
    write_csv(run_dir / "convergence.csv", rows)
    plot_convergence(run_dir, rows)

    summary = {
        "args": vars(args),
        "run_dir": str(run_dir),
        "profile": {
            key: {
                "res_med": sol["res_med"],
                "res_max": sol["res_max"],
                "cond": sol["cond"],
                "coeff_tail_rel": sol["coeff_tail_rel"],
                "u_abs_min": float(np.min(np.abs(sol["u"]))),
                "u_abs_max": float(np.max(np.abs(sol["u"]))),
            }
            for key, sol in sols.items()
        },
        "convergence_last": {
            branch: [row for row in rows if row["branch"] == branch][-1]
            for branch in ["in", "in_A", "down", "up"]
            if any(row["branch"] == branch for row in rows)
        },
    }
    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
