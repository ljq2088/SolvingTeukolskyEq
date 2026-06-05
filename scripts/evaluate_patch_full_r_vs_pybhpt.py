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
from utils.amplitude import A_down, A_in, A_up, r_of_z  # noqa: E402


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys = sorted({key for row in rows for key in row})
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def pybhpt_rin_sorted(a: float, omega: float, r_values: np.ndarray, timeout: float) -> np.ndarray:
    order = np.argsort(r_values)
    inv = np.empty_like(order)
    inv[order] = np.arange(order.size)
    _, ref_sorted = compute_pybhpt_solution(a, omega, ell=2, m=2, r_grid=np.asarray(r_values)[order], timeout=timeout)
    return np.asarray(ref_sorted, dtype=np.complex128)[inv]


def rel(pred: np.ndarray, ref: np.ndarray, floor: float = 1.0e-300) -> np.ndarray:
    return np.abs(pred - ref) / np.maximum(np.abs(ref), floor)


def predict_piecewise(expert: PatchExpert, a: float, omega: float, z_values: np.ndarray, z_switch: float):
    mode = expert.mode(a, omega)
    logw = float(np.log10(omega))
    r = r_of_z(z_values, mode)
    u_in = expert.decoders["in"].eval_u(z_values, a, logw)
    u_down = expert.decoders["down"].eval_u(z_values, a, logw)
    u_up = expert.decoders["up"].eval_u(z_values, a, logw)
    amp = expert.solve_amplitudes(a, omega)
    R_inner = A_in(r, mode) * u_in
    R_outer = amp["B_inc"] * A_down(r, mode) * u_down + amp["B_ref"] * A_up(r, mode) * u_up
    R = np.where(z_values >= z_switch, R_inner, R_outer)
    return R, amp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch-dir", required=True)
    parser.add_argument("--n-param-side", type=int, default=5)
    parser.add_argument("--n-z", type=int, default=160)
    parser.add_argument("--z-min", type=float, default=1.0e-3)
    parser.add_argument("--z-max", type=float, default=1.0 - 1.0e-4)
    parser.add_argument("--timeout", type=float, default=45.0)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--sample-kind", choices=["grid", "random"], default="grid")
    parser.add_argument("--n-param-random", type=int, default=25)
    parser.add_argument("--seed", type=int, default=20260605)
    args = parser.parse_args()

    expert = PatchExpert(args.patch_dir)
    cfg = expert.config
    run_dir = Path(args.output_dir) if args.output_dir else Path(args.patch_dir) / "full_r_pybhpt_eval" / datetime.now().strftime("%Y%m%d_%H%M%S")
    fig_dir = run_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    z_switch = 0.5 * (expert.decoders["in"].z_domain[0] + expert.decoders["down"].z_domain[1])
    rng = np.random.default_rng(args.seed)
    if args.sample_kind == "grid":
        z_values_base = np.geomspace(args.z_min, args.z_max, args.n_z)
        param_pairs = [
            (float(a), float(logw))
            for a in np.linspace(cfg["a_center"] - cfg["a_half_width"], cfg["a_center"] + cfg["a_half_width"], args.n_param_side)
            for logw in np.linspace(cfg["logw_center"] - cfg["logw_half_width"], cfg["logw_center"] + cfg["logw_half_width"], args.n_param_side)
        ]
    else:
        z_values_base = None
        margin_a = 0.03 * cfg["a_half_width"]
        margin_w = 0.03 * cfg["logw_half_width"]
        param_pairs = [
            (
                float(rng.uniform(cfg["a_center"] - cfg["a_half_width"] + margin_a, cfg["a_center"] + cfg["a_half_width"] - margin_a)),
                float(rng.uniform(cfg["logw_center"] - cfg["logw_half_width"] + margin_w, cfg["logw_center"] + cfg["logw_half_width"] - margin_w)),
            )
            for _ in range(args.n_param_random)
        ]
    rows: list[dict] = []
    point_rows: list[dict] = []
    failures: list[dict] = []
    for a, logw in param_pairs:
            if z_values_base is None:
                u = rng.uniform(np.log(args.z_min), np.log(args.z_max), args.n_z)
                z_values = np.sort(np.exp(u))
            else:
                z_values = z_values_base
            omega = 10.0 ** float(logw)
            mode = expert.mode(float(a), omega)
            r_values = r_of_z(z_values, mode)
            try:
                R_pred, amp = predict_piecewise(expert, float(a), omega, z_values, z_switch)
                R_ref = pybhpt_rin_sorted(float(a), omega, r_values, args.timeout)
                err = rel(R_pred, R_ref)
                row = {
                    "a": float(a),
                    "logw": float(logw),
                    "omega": float(omega),
                    "z_switch": float(z_switch),
                    "amp_ls_residual": float(amp["residual"]),
                    "amp_cond": float(amp["cond"]),
                    "rel_median": float(np.median(err)),
                    "rel_p90": float(np.quantile(err, 0.9)),
                    "rel_max": float(np.max(err)),
                    "rel_outer_median": float(np.median(err[z_values < z_switch])),
                    "rel_inner_median": float(np.median(err[z_values >= z_switch])),
                    "B_inc_abs": float(abs(amp["B_inc"])),
                    "B_ref_abs": float(abs(amp["B_ref"])),
                }
                rows.append(row)
                print(json.dumps(row), flush=True)
                for zi, ri, pred, ref, ei in zip(z_values, r_values, R_pred, R_ref, err):
                    point_rows.append(
                        {
                            "a": float(a),
                            "logw": float(logw),
                            "omega": float(omega),
                            "z": float(zi),
                            "r": float(ri),
                            "rel": float(ei),
                            "region": "inner" if zi >= z_switch else "outer",
                            "R_pred_re": float(pred.real),
                            "R_pred_im": float(pred.imag),
                            "R_ref_re": float(ref.real),
                            "R_ref_im": float(ref.imag),
                        }
                    )
            except Exception as exc:
                failure = {"a": float(a), "logw": float(logw), "omega": float(omega), "error": repr(exc)}
                failures.append(failure)
                print(json.dumps(failure), flush=True)
    write_csv(run_dir / "full_r_point_errors.csv", point_rows)
    write_csv(run_dir / "full_r_param_summary.csv", rows)
    write_csv(run_dir / "failures.csv", failures)
    summary = {
        "run_dir": str(run_dir),
        "patch_dir": str(args.patch_dir),
        "z_switch": float(z_switch),
        "n_success": len(rows),
        "n_failure": len(failures),
        "rel_median_over_params": float(np.median([r["rel_median"] for r in rows])) if rows else None,
        "rel_p90_over_params": float(np.median([r["rel_p90"] for r in rows])) if rows else None,
        "rel_max_over_params": float(np.max([r["rel_max"] for r in rows])) if rows else None,
        "outer_median_over_params": float(np.median([r["rel_outer_median"] for r in rows])) if rows else None,
        "inner_median_over_params": float(np.median([r["rel_inner_median"] for r in rows])) if rows else None,
        "amp_ls_residual_median": float(np.median([r["amp_ls_residual"] for r in rows])) if rows else None,
        "amp_ls_residual_max": float(np.max([r["amp_ls_residual"] for r in rows])) if rows else None,
    }
    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    if rows:
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.semilogy([r["rel_median"] for r in rows], "o-", label="full median")
        ax.semilogy([r["rel_outer_median"] for r in rows], "s-", label="outer median")
        ax.semilogy([r["rel_inner_median"] for r in rows], "^-", label="inner median")
        ax.semilogy([r["amp_ls_residual"] for r in rows], "x-", label="amp LS residual")
        ax.set_xlabel("parameter sample")
        ax.set_ylabel("relative error / residual")
        ax.grid(alpha=0.3, which="both")
        ax.legend()
        fig.tight_layout()
        fig.savefig(fig_dir / "full_r_error_summary.png", dpi=180, bbox_inches="tight")
        plt.close(fig)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
