#!/usr/bin/env python3
from __future__ import annotations

import argparse
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

from teukspec.core.chebyshev import (  # noqa: E402
    cheb_lobatto_physical,
    coeff_l2_value_errors,
    eval_tensor_decoder,
    fit_tensor_decoder,
)
from teukspec.decoders.param_cheb_decoder import ParamChebDecoder  # noqa: E402
from teukspec.teachers.spectral_teacher import branch_specs, build_teacher_grid, write_csv  # noqa: E402


def selected_basis(value: str) -> list[str]:
    if value == "all":
        return ["in", "down", "up"]
    return [value]


def plot_basis(run_dir: Path, basis: str, decoder: np.ndarray, teacher: np.ndarray, pred: np.ndarray, a_vals: np.ndarray, logw_vals: np.ndarray):
    fig_dir = run_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    ia = int(np.argmin(np.abs(a_vals - np.mean(a_vals))))
    iw = int(np.argmin(np.abs(logw_vals - np.mean(logw_vals))))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].semilogy(np.abs(teacher[ia, iw]), "o-", ms=3, lw=1, label="teacher")
    axes[0].semilogy(np.abs(pred[ia, iw]), "s--", ms=3, lw=1, label="param Cheb")
    axes[0].set_title(f"{basis} radial Cheb coefficients at patch center")
    axes[0].set_xlabel("radial Cheb mode")
    axes[0].set_ylabel("|c_k|")
    axes[0].grid(alpha=0.3, which="both")
    axes[0].legend()
    modal = np.max(np.abs(decoder), axis=2)
    im = axes[1].imshow(np.log10(np.maximum(modal, 1.0e-300)), origin="lower", aspect="auto")
    axes[1].set_title(f"{basis} parameter Cheb tensor max over radial modes")
    axes[1].set_xlabel("logw Cheb mode")
    axes[1].set_ylabel("a Cheb mode")
    fig.colorbar(im, ax=axes[1])
    fig.tight_layout()
    fig.savefig(fig_dir / f"{basis}_decoder_coeff_decay.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--basis", choices=["all", "in", "down", "up"], default="all")
    parser.add_argument("--a-center", type=float, default=0.5)
    parser.add_argument("--a-half-width", type=float, default=0.12)
    parser.add_argument("--logw-center", type=float, default=-1.5)
    parser.add_argument("--logw-half-width", type=float, default=0.25)
    parser.add_argument("--n-param-side", type=int, default=9)
    parser.add_argument("--deg-a", type=int, default=8)
    parser.add_argument("--deg-logw", type=int, default=8)
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--grid-kind", choices=["linear", "anmr", "auto"], default="anmr")
    parser.add_argument("--y-match", type=float, default=-0.25)
    parser.add_argument("--match-width", type=float, default=0.12)
    parser.add_argument("--ell", type=int, default=2)
    parser.add_argument("--m", type=int, default=2)
    parser.add_argument("--s", type=int, default=-2)
    parser.add_argument("--output-dir", default="outputs/param_cheb_patch_decoder")
    args = parser.parse_args()

    run_dir = Path(args.output_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    basis_names = selected_basis(args.basis)
    specs = branch_specs(args.y_match, args.match_width)
    a_vals, xi_a = cheb_lobatto_physical(args.a_center, args.a_half_width, args.n_param_side)
    logw_vals, xi_w = cheb_lobatto_physical(args.logw_center, args.logw_half_width, args.n_param_side)

    config = vars(args).copy()
    config["run_dir"] = str(run_dir)
    config["basis_names"] = basis_names
    config["specs"] = {name: spec.__dict__ for name, spec in specs.items()}
    with open(run_dir / "patch_config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(json.dumps({"run_dir": str(run_dir), "basis": basis_names}, indent=2), flush=True)

    coeffs, teacher_rows = build_teacher_grid(
        basis_names=basis_names,
        specs=specs,
        a_vals=a_vals,
        logw_vals=logw_vals,
        n=args.n,
        grid_kind=args.grid_kind,
        ell=args.ell,
        m=args.m,
        s=args.s,
    )
    write_csv(run_dir / "teacher_metrics.csv", teacher_rows)

    eval_rows = []
    summary = {"run_dir": str(run_dir), "basis": {}}
    for basis in basis_names:
        decoder_coeffs = fit_tensor_decoder(coeffs[basis], xi_a, xi_w, args.deg_a, args.deg_logw)
        pred = eval_tensor_decoder(decoder_coeffs, xi_a, xi_w)
        coeff_rel, value_rel = coeff_l2_value_errors(pred, coeffs[basis])
        decoder = ParamChebDecoder(
            decoder_coeffs,
            args.a_center,
            args.a_half_width,
            args.logw_center,
            args.logw_half_width,
            (specs[basis].z_left, specs[basis].z_right),
            args.n,
            args.deg_a,
            args.deg_logw,
            basis,
            args.grid_kind,
            ell=args.ell,
            m=args.m,
            s=args.s,
            domain=specs[basis].domain,
        )
        decoder.save_npz(run_dir / f"decoder_{basis}.npz")
        np.savez(
            run_dir / f"teacher_{basis}.npz",
            coeffs=coeffs[basis],
            pred=pred,
            a_vals=a_vals,
            logw_vals=logw_vals,
            xi_a=xi_a,
            xi_w=xi_w,
        )
        plot_basis(run_dir, basis, decoder_coeffs, coeffs[basis], pred, a_vals, logw_vals)
        flat = 0
        for av in a_vals:
            for lw in logw_vals:
                eval_rows.append(
                    {
                        "basis": basis,
                        "a": float(av),
                        "logw": float(lw),
                        "omega": float(10.0**lw),
                        "coeff_rel_l2": float(coeff_rel[flat]),
                        "value_rel_l2": float(value_rel[flat]),
                    }
                )
                flat += 1
        tails = [row[f"{basis}_tail_rel"] for row in teacher_rows if f"{basis}_tail_rel" in row]
        summary["basis"][basis] = {
            "coeff_rel_median": float(np.median(coeff_rel)),
            "coeff_rel_max": float(np.max(coeff_rel)),
            "value_rel_median": float(np.median(value_rel)),
            "value_rel_max": float(np.max(value_rel)),
            "teacher_tail_median": float(np.median(tails)),
            "teacher_tail_max": float(np.max(tails)),
        }
    write_csv(run_dir / "param_decoder_eval.csv", eval_rows)
    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
