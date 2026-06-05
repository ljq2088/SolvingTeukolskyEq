#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from teukspec.atlas.patch_expert import PatchExpert  # noqa: E402
from teukspec.core.chebyshev import clenshaw_eval, z_to_comp_xi  # noqa: E402
from teukspec.teachers.spectral_teacher import branch_specs, solve_teacher, write_csv  # noqa: E402
from utils.mode import KerrMode  # noqa: E402


def interior_linspace(left: float, right: float, n: int) -> np.ndarray:
    return np.linspace(left + 0.08 * (right - left), right - 0.08 * (right - left), n)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch-dir", required=True)
    parser.add_argument("--n-offgrid", type=int, default=49)
    parser.add_argument("--n-z", type=int, default=80)
    args = parser.parse_args()
    patch_dir = Path(args.patch_dir)
    expert = PatchExpert(patch_dir)
    cfg = expert.config
    specs = branch_specs(cfg["y_match"], cfg["match_width"])
    rng = np.random.default_rng(20260605)
    n_side = int(np.sqrt(args.n_offgrid))
    n_side = max(n_side, 3)
    a_rand = rng.uniform(cfg["a_center"] - cfg["a_half_width"], cfg["a_center"] + cfg["a_half_width"], n_side * n_side)
    lw_rand = rng.uniform(cfg["logw_center"] - cfg["logw_half_width"], cfg["logw_center"] + cfg["logw_half_width"], n_side * n_side)
    rows = []
    amp_rows = []
    for a, lw in zip(a_rand, lw_rand):
        mode = KerrMode(M=1.0, a=float(a), omega=10.0 ** float(lw), ell=cfg.get("ell", 2), m=cfg.get("m", 2), s=cfg.get("s", -2))
        for basis, decoder in expert.decoders.items():
            spec = specs[basis]
            coeff_teacher, metrics = solve_teacher(mode, spec, cfg["n"], cfg["grid_kind"])
            z_eval = interior_linspace(spec.z_left, spec.z_right, args.n_z)
            xi_eval = z_to_comp_xi(z_eval, spec.z_left, spec.z_right, mode, spec.domain, cfg["grid_kind"])
            u_teacher = clenshaw_eval(coeff_teacher, xi_eval)
            u_pred = decoder.eval_u(z_eval, float(a), float(lw))
            value_rel = np.linalg.norm(u_pred - u_teacher) / max(np.linalg.norm(u_teacher), 1.0e-300)
            rel_res = expert.residual_scan(float(a), mode.omega, basis, z_eval)
            rows.append(
                {
                    "basis": basis,
                    "a": float(a),
                    "logw": float(lw),
                    "omega": float(mode.omega),
                    "value_rel_l2": float(value_rel),
                    "residual_median": float(np.median(rel_res)),
                    "residual_max": float(np.max(rel_res)),
                    "teacher_tail_rel": float(metrics["tail_rel"]),
                    "teacher_res_med": float(metrics["res_med"]),
                    "teacher_res_max": float(metrics["res_max"]),
                }
            )
        z_match = expert.default_match_z_values(36)
        amp = expert.solve_amplitudes(float(a), mode.omega, z_match)
        amp_rows.append(
            {
                "a": float(a),
                "logw": float(lw),
                "omega": float(mode.omega),
                "B_inc_re": amp["B_inc"].real,
                "B_inc_im": amp["B_inc"].imag,
                "B_ref_re": amp["B_ref"].real,
                "B_ref_im": amp["B_ref"].imag,
                "amp_ls_residual": amp["residual"],
                "amp_cond": amp["cond"],
            }
        )
    write_csv(patch_dir / "offgrid_validation.csv", rows)
    write_csv(patch_dir / "amplitude_validation.csv", amp_rows)
    summary = {"patch_dir": str(patch_dir), "basis": {}, "amplitude": {}}
    for basis in sorted({row["basis"] for row in rows}):
        br = [row for row in rows if row["basis"] == basis]
        summary["basis"][basis] = {
            "value_rel_median": float(np.median([r["value_rel_l2"] for r in br])),
            "value_rel_max": float(np.max([r["value_rel_l2"] for r in br])),
            "residual_median": float(np.median([r["residual_median"] for r in br])),
            "residual_max": float(np.max([r["residual_max"] for r in br])),
            "teacher_tail_median": float(np.median([r["teacher_tail_rel"] for r in br])),
        }
    summary["amplitude"] = {
        "ls_residual_median": float(np.median([r["amp_ls_residual"] for r in amp_rows])),
        "ls_residual_max": float(np.max([r["amp_ls_residual"] for r in amp_rows])),
        "cond_median": float(np.median([r["amp_cond"] for r in amp_rows])),
        "cond_max": float(np.max([r["amp_cond"] for r in amp_rows])),
    }
    with open(patch_dir / "validation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    fig_dir = patch_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    for basis in sorted(summary["basis"]):
        br = [row for row in rows if row["basis"] == basis]
        ax.semilogy([r["value_rel_l2"] for r in br], "o", label=basis)
    ax.set_xlabel("off-grid sample")
    ax.set_ylabel("L2 value relative error")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "value_error_offgrid.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 5))
    for basis in sorted(summary["basis"]):
        br = [row for row in rows if row["basis"] == basis]
        ax.semilogy([r["residual_median"] for r in br], "o", label=basis)
    ax.set_xlabel("off-grid sample")
    ax.set_ylabel("median reduced residual")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "residual_offgrid.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
