#!/usr/bin/env python3
"""Extract high-frequency Teukolsky amplitudes from finite-radius GSN samples.

The direct two-domain Teukolsky spectral matcher loses the tiny high-frequency
reflection amplitude.  This script tests the more stable variable

    R / A_down = B_inc + B_ref * (A_up / A_down)

on a far-field window, where the outgoing component is algebraically amplified
by |A_up/A_down| ~ r^4.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.amplitude import A_down, A_up  # noqa: E402
from utils.amplitude import compute_smatrix_with_abel  # noqa: E402
from utils.mode import KerrMode  # noqa: E402

DEFAULT_GSN_PROJECT = Path("/home/ljq/code/GSN/GeneralizedSasakiNakamura.jl")


def parse_float_list(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def run_gsn_samples(
    *,
    a: float,
    omega: float,
    r_values: np.ndarray,
    gsn_project: Path,
    timeout: float,
    work_dir: Path,
) -> tuple[dict[str, complex], np.ndarray]:
    julia = shutil.which("julia")
    if julia is None:
        raise RuntimeError("julia not found")
    if not (gsn_project / "Project.toml").exists():
        raise RuntimeError(f"GSN Project.toml not found: {gsn_project}")

    script_path = work_dir / f"gsn_profile_a{a:.6g}_w{omega:.6g}.jl"
    csv_path = work_dir / f"gsn_profile_a{a:.6g}_w{omega:.6g}.csv"
    amp_path = work_dir / f"gsn_amplitudes_a{a:.6g}_w{omega:.6g}.csv"
    r_list = ",".join(f"{float(r):.17g}" for r in r_values)
    script_path.write_text(
        f"""
using GeneralizedSasakiNakamura
s = -2
l = 2
m = 2
a = {a:.17g}
omega = {omega:.17g}
r_values = [{r_list}]

Rin = Teukolsky_radial(s, l, m, a, omega, IN)
open("{amp_path}", "w") do io
    println(io, "B_inc_re,B_inc_im,B_ref_re,B_ref_im,B_trans_re,B_trans_im,lambda_re,lambda_im")
    println(io, join([
        real(Rin.incidence_amplitude),
        imag(Rin.incidence_amplitude),
        real(Rin.reflection_amplitude),
        imag(Rin.reflection_amplitude),
        real(Rin.transmission_amplitude),
        imag(Rin.transmission_amplitude),
        real(Rin.mode.lambda),
        imag(Rin.mode.lambda)
    ], ","))
end

rows = []
for r in r_values
    v = Rin(r)
    push!(rows, [r, real(v), imag(v)])
end
open("{csv_path}", "w") do io
    println(io, "r,R_re,R_im")
    for row in rows
        println(io, join(row, ","))
    end
end
"""
    )
    proc = subprocess.run(
        [julia, f"--project={gsn_project}", str(script_path)],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"GSN failed: {proc.stderr.strip()[:500]}")
    if not csv_path.exists() or not amp_path.exists():
        raise RuntimeError("GSN did not produce expected outputs")

    with open(amp_path, newline="") as f:
        raw = next(csv.DictReader(f))
    amps = {
        "B_inc": complex(float(raw["B_inc_re"]), float(raw["B_inc_im"])),
        "B_ref": complex(float(raw["B_ref_re"]), float(raw["B_ref_im"])),
        "B_trans": complex(float(raw["B_trans_re"]), float(raw["B_trans_im"])),
        "lambda": complex(float(raw["lambda_re"]), float(raw["lambda_im"])),
    }

    rows = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append((float(row["r"]), complex(float(row["R_re"]), float(row["R_im"]))))
    return amps, np.array(rows, dtype=object)


def fit_amplitudes(
    mode: KerrMode,
    r: np.ndarray,
    R: np.ndarray,
    *,
    weight: str,
    order: int,
    fixed_binc: complex | None,
) -> tuple[complex, complex, float, float]:
    Ad = A_down(r, mode)
    Au = A_up(r, mode)
    y = R / Ad
    q = Au / Ad
    powers = [r ** (-k) for k in range(order + 1)]
    if fixed_binc is None:
        X_down = np.column_stack(powers)
        y_fit = y
        binc_index = 0
        bref_index = order + 1
    else:
        down_corrections = powers[1:] if order >= 1 else []
        X_down = np.column_stack(down_corrections) if down_corrections else np.empty((len(r), 0))
        y_fit = y - fixed_binc
        binc_index = None
        bref_index = X_down.shape[1]
    X_up = np.column_stack([q * p for p in powers])
    X = np.column_stack([X_down, X_up])

    if weight == "none":
        weights = np.ones_like(r, dtype=np.float64)
    elif weight == "unit-q":
        weights = 1.0 / np.maximum(np.linalg.norm(X, axis=1), 1.0e-300)
    elif weight == "r-minus4":
        weights = r ** (-4)
    else:
        raise ValueError(f"unknown weight={weight}")

    Xw = X * weights[:, None]
    yw = y_fit * weights
    coef, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
    residual = np.linalg.norm(X @ coef - y_fit) / max(np.linalg.norm(y_fit), 1.0e-300)
    cond = np.linalg.cond(Xw)
    B_inc = complex(coef[binc_index]) if fixed_binc is None else complex(fixed_binc)
    B_ref = complex(coef[bref_index])
    return B_inc, B_ref, float(residual), float(cond)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--a-list", default="0.5")
    parser.add_argument("--omega-list", default="1.0,10.0")
    parser.add_argument("--r-min-list", default="20,50,100,200")
    parser.add_argument("--r-max-list", default="100,200,500,1000")
    parser.add_argument("--n-r", type=int, default=128)
    parser.add_argument("--weight-list", default="none,unit-q,r-minus4")
    parser.add_argument("--order-list", default="0,1,2,3")
    parser.add_argument("--fit-mode-list", default="free,fixed-binc")
    parser.add_argument("--fixed-binc-source", choices=["gsn", "spectral"], default="gsn")
    parser.add_argument("--gsn-project", default=str(DEFAULT_GSN_PROJECT))
    parser.add_argument("--timeout", type=float, default=240.0)
    parser.add_argument("--output-dir", default="outputs/highw_bref_extraction")
    args = parser.parse_args()

    a_values = parse_float_list(args.a_list)
    omega_values = parse_float_list(args.omega_list)
    r_min_values = parse_float_list(args.r_min_list)
    r_max_values = parse_float_list(args.r_max_list)
    weight_values = [item.strip() for item in args.weight_list.split(",") if item.strip()]
    order_values = [int(item.strip()) for item in args.order_list.split(",") if item.strip()]
    fit_modes = [item.strip() for item in args.fit_mode_list.split(",") if item.strip()]

    out_dir = Path(args.output_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    gsn_project = Path(args.gsn_project).expanduser()

    rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    for a in a_values:
        for omega in omega_values:
            r_global_min = min(r_min_values)
            r_global_max = max(r_max_values)
            r_global = np.geomspace(r_global_min, r_global_max, args.n_r)
            print(f"GSN sample a={a:g} omega={omega:g} r=[{r_global_min:g},{r_global_max:g}]", flush=True)
            try:
                amps, samples = run_gsn_samples(
                    a=a,
                    omega=omega,
                    r_values=r_global,
                    gsn_project=gsn_project,
                    timeout=args.timeout,
                    work_dir=out_dir,
                )
            except Exception as exc:
                failures.append({"a": a, "omega": omega, "error": str(exc)})
                print(f"  FAILED: {exc}", flush=True)
                continue

            r_all = np.array([float(item[0]) for item in samples], dtype=np.float64)
            R_all = np.array([complex(item[1]) for item in samples], dtype=np.complex128)
            mode = KerrMode(M=1.0, a=a, omega=omega, ell=2, m=2, s=-2)
            spectral_binc = None
            if args.fixed_binc_source == "spectral":
                spectral_binc = complex(
                    compute_smatrix_with_abel(mode, N_in=80, N_out=80, z_m=0.3)["B_inc"]
                )

            for r_min in r_min_values:
                for r_max in r_max_values:
                    if r_max <= r_min:
                        continue
                    mask = (r_all >= r_min) & (r_all <= r_max)
                    if np.count_nonzero(mask) < 8:
                        continue
                    for weight in weight_values:
                        for order in order_values:
                            for fit_mode in fit_modes:
                                if fit_mode == "fixed-binc":
                                    fixed_binc = spectral_binc if args.fixed_binc_source == "spectral" else amps["B_inc"]
                                else:
                                    fixed_binc = None
                                if fit_mode not in {"free", "fixed-binc"}:
                                    raise ValueError(f"unknown fit_mode={fit_mode}")
                                B_inc_fit, B_ref_fit, fit_residual, fit_cond = fit_amplitudes(
                                    mode,
                                    r_all[mask],
                                    R_all[mask],
                                    weight=weight,
                                    order=order,
                                    fixed_binc=fixed_binc,
                                )
                                B_inc_ref = amps["B_inc"]
                                B_ref_ref = amps["B_ref"]
                                rows.append({
                                "a": a,
                                "omega": omega,
                                "r_min": r_min,
                                "r_max": r_max,
                                "n_fit": int(np.count_nonzero(mask)),
                                "weight": weight,
                                "order": int(order),
                                "fit_mode": fit_mode,
                                "fit_residual": fit_residual,
                                "fit_cond": fit_cond,
                                "B_inc_fit_re": B_inc_fit.real,
                                "B_inc_fit_im": B_inc_fit.imag,
                                "B_inc_fit_abs": abs(B_inc_fit),
                                "B_ref_fit_re": B_ref_fit.real,
                                "B_ref_fit_im": B_ref_fit.imag,
                                "B_ref_fit_abs": abs(B_ref_fit),
                                "B_inc_gsn_re": B_inc_ref.real,
                                "B_inc_gsn_im": B_inc_ref.imag,
                                "B_inc_gsn_abs": abs(B_inc_ref),
                                "B_ref_gsn_re": B_ref_ref.real,
                                "B_ref_gsn_im": B_ref_ref.imag,
                                "B_ref_gsn_abs": abs(B_ref_ref),
                                "B_inc_relerr": abs(B_inc_fit - B_inc_ref) / max(abs(B_inc_ref), 1.0e-300),
                                "B_ref_relerr": abs(B_ref_fit - B_ref_ref) / max(abs(B_ref_ref), 1.0e-300),
                                })

    if rows:
        fieldnames = list(rows[0].keys())
        with open(out_dir / "highw_bref_fit_cases.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    selected_rows = []
    if rows:
        grouped: dict[tuple[float, float], list[dict[str, object]]] = {}
        for row in rows:
            grouped.setdefault((float(row["a"]), float(row["omega"])), []).append(row)
        for (a, omega), group in sorted(grouped.items()):
            selected = min(
                group,
                key=lambda row: float(row["fit_residual"])
                * (1.0 + math.log10(max(float(row["fit_cond"]), 1.0))),
            )
            selected = dict(selected)
            selected["selection_score"] = float(selected["fit_residual"]) * (
                1.0 + math.log10(max(float(selected["fit_cond"]), 1.0))
            )
            selected_rows.append(selected)
        with open(out_dir / "selected_highw_bref_fits.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(selected_rows[0].keys()))
            writer.writeheader()
            writer.writerows(selected_rows)

    summary = {"args": vars(args), "n_cases": len(rows), "failures": failures}
    if rows:
        finite = rows
        best_bref = min(finite, key=lambda row: float(row["B_ref_relerr"]))
        best_joint = min(
            finite,
            key=lambda row: max(float(row["B_ref_relerr"]), float(row["B_inc_relerr"])),
        )
        summary["best_bref"] = best_bref
        summary["best_joint"] = best_joint
        summary["B_ref_relerr_median"] = float(np.median([row["B_ref_relerr"] for row in finite]))
        summary["B_ref_relerr_min"] = float(min(row["B_ref_relerr"] for row in finite))
        summary["selected"] = selected_rows
        summary["selected_B_ref_relerr_median"] = float(
            np.median([float(row["B_ref_relerr"]) for row in selected_rows])
        )
        summary["selected_B_ref_relerr_max"] = float(
            max(float(row["B_ref_relerr"]) for row in selected_rows)
        )
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved high-w extraction diagnostics to {out_dir}")
    print(json.dumps({k: summary[k] for k in summary if k != "best_bref" and k != "best_joint"}, indent=2))
    if "best_bref" in summary:
        print("Best B_ref fit:")
        print(json.dumps(summary["best_bref"], indent=2))


if __name__ == "__main__":
    main()
