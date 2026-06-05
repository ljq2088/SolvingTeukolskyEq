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

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pybhpt_usage.compute_solution import compute_pybhpt_solution  # noqa: E402
from teukspec.atlas.patch_expert import PatchExpert  # noqa: E402
from utils.amplitude import A_in, r_of_z  # noqa: E402


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys = sorted({key for row in rows for key in row})
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def relative_error(pred: np.ndarray, ref: np.ndarray, floor: float = 1.0e-300) -> np.ndarray:
    return np.abs(pred - ref) / np.maximum(np.abs(ref), floor)


def pybhpt_rin_sorted(a: float, omega: float, r_values: np.ndarray, timeout: float) -> np.ndarray:
    r_values = np.asarray(r_values, dtype=float)
    order = np.argsort(r_values)
    inv = np.empty_like(order)
    inv[order] = np.arange(order.size)
    _, ref_sorted = compute_pybhpt_solution(a, omega, ell=2, m=2, r_grid=r_values[order], timeout=timeout)
    return np.asarray(ref_sorted, dtype=np.complex128)[inv]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch-dir", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--n-param-side", type=int, default=3)
    parser.add_argument("--n-z", type=int, default=80)
    parser.add_argument("--z-min-frac", type=float, default=0.08)
    parser.add_argument("--z-max-frac", type=float, default=0.08)
    parser.add_argument("--timeout", type=float, default=45.0)
    args = parser.parse_args()

    expert = PatchExpert(args.patch_dir)
    cfg = expert.config
    run_dir = Path(args.output_dir) if args.output_dir else Path(args.patch_dir) / "pybhpt_eval" / datetime.now().strftime("%Y%m%d_%H%M%S")
    fig_dir = run_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    a_vals = np.linspace(cfg["a_center"] - cfg["a_half_width"], cfg["a_center"] + cfg["a_half_width"], args.n_param_side)
    logw_vals = np.linspace(cfg["logw_center"] - cfg["logw_half_width"], cfg["logw_center"] + cfg["logw_half_width"], args.n_param_side)
    dec = expert.decoders["in"]
    z_left, z_right = dec.z_domain
    z_values = np.linspace(
        z_left + args.z_min_frac * (z_right - z_left),
        z_right - args.z_max_frac * (z_right - z_left),
        args.n_z,
    )

    rows: list[dict] = []
    point_rows: list[dict] = []
    failures: list[dict] = []
    for a in a_vals:
        for logw in logw_vals:
            omega = 10.0 ** float(logw)
            mode = expert.mode(float(a), omega)
            r_values = r_of_z(z_values, mode)
            try:
                R_ref = pybhpt_rin_sorted(float(a), omega, r_values, args.timeout)
                u_pred = dec.eval_u(z_values, float(a), float(logw))
                R_pred = A_in(r_values, mode) * u_pred
                rel = relative_error(R_pred, R_ref, floor=1.0e-300)
                row = {
                    "a": float(a),
                    "logw": float(logw),
                    "omega": float(omega),
                    "n": int(len(z_values)),
                    "rel_median": float(np.median(rel)),
                    "rel_mean": float(np.mean(rel)),
                    "rel_p90": float(np.quantile(rel, 0.9)),
                    "rel_max": float(np.max(rel)),
                    "abs_ref_min": float(np.min(np.abs(R_ref))),
                    "abs_ref_max": float(np.max(np.abs(R_ref))),
                }
                rows.append(row)
                for zi, ri, pred, ref, err in zip(z_values, r_values, R_pred, R_ref, rel):
                    point_rows.append(
                        {
                            "a": float(a),
                            "logw": float(logw),
                            "omega": float(omega),
                            "z": float(zi),
                            "r": float(ri),
                            "R_pred_re": float(pred.real),
                            "R_pred_im": float(pred.imag),
                            "R_ref_re": float(ref.real),
                            "R_ref_im": float(ref.imag),
                            "rel": float(err),
                        }
                    )
                print(json.dumps(row), flush=True)
            except Exception as exc:
                failure = {"a": float(a), "logw": float(logw), "omega": float(omega), "error": repr(exc)}
                failures.append(failure)
                print(json.dumps(failure), flush=True)

    write_csv(run_dir / "pybhpt_point_errors.csv", point_rows)
    write_csv(run_dir / "pybhpt_param_summary.csv", rows)
    write_csv(run_dir / "pybhpt_failures.csv", failures)
    summary = {
        "run_dir": str(run_dir),
        "patch_dir": str(args.patch_dir),
        "n_success": len(rows),
        "n_failure": len(failures),
        "rel_median_over_params": float(np.median([row["rel_median"] for row in rows])) if rows else None,
        "rel_max_over_params": float(np.max([row["rel_max"] for row in rows])) if rows else None,
        "rel_p90_over_params": float(np.median([row["rel_p90"] for row in rows])) if rows else None,
        "failures": failures,
    }
    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    if rows:
        fig, ax = plt.subplots(figsize=(8, 5))
        labels = [f"a={row['a']:.3f}\nω={row['omega']:.3g}" for row in rows]
        ax.semilogy(range(len(rows)), [row["rel_median"] for row in rows], "o-", label="median")
        ax.semilogy(range(len(rows)), [row["rel_p90"] for row in rows], "s-", label="p90")
        ax.semilogy(range(len(rows)), [row["rel_max"] for row in rows], "^-", label="max")
        ax.set_xticks(range(len(rows)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("|R_pred - R_pybhpt| / |R_pybhpt|")
        ax.set_title("Patch decoder R_in vs pybhpt on in-domain")
        ax.grid(alpha=0.3, which="both")
        ax.legend()
        fig.tight_layout()
        fig.savefig(fig_dir / "pybhpt_param_error_summary.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

        center = rows[len(rows) // 2]
        pts = [p for p in point_rows if abs(p["a"] - center["a"]) < 1e-14 and abs(p["logw"] - center["logw"]) < 1e-14]
        if pts:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.semilogy([p["r"] for p in pts], [p["rel"] for p in pts], "-")
            ax.set_xlabel("r")
            ax.set_ylabel("relative error")
            ax.set_title(f"Center-like point error, a={center['a']:.3f}, omega={center['omega']:.6g}")
            ax.grid(alpha=0.3, which="both")
            fig.tight_layout()
            fig.savefig(fig_dir / "pybhpt_center_pointwise_error.png", dpi=180, bbox_inches="tight")
            plt.close(fig)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
