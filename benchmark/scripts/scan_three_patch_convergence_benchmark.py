from __future__ import annotations

import os

# ---- must be set before numpy/matplotlib ----
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
import csv
import math
import sys
from itertools import product
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from utils.mode import KerrMode
from utils.amplitude_three_patch import TeukRadAmplitudeIn3PatchWithAbelChecks


DEFAULT_A_VALUES = [0.1, 0.3, 0.5, 0.7, 0.9, 0.99]
DEFAULT_OMEGA_VALUES = [1.0e-3, 1.0e-2, 1.0e-1, 1.0, 10.0]
DEFAULT_MATCH_OMEGA_VALUES = [1.0e-3, 1.0e-2]
DEFAULT_MATCH_PAIRS = [
    (0.10, 0.90),
    (0.10, 0.80),
    (0.05, 0.90),
    (0.10, 0.95),
    (0.20, 0.90),
]
DEFAULT_N_VALUES = [32, 48, 64, 80, 96, 128]


def parse_float_list(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def parse_n_list(text: str) -> list[int]:
    vals = [int(x.strip()) for x in text.split(",") if x.strip()]
    vals = sorted(set(vals))
    if not vals:
        raise ValueError("N list must not be empty")
    return vals


def zpair_label(z1: float, z2: float) -> str:
    return f"z1={z1:.2f}, z2={z2:.2f}"


def slug_float(x: float) -> str:
    s = f"{x:.6g}"
    s = s.replace("+", "")
    s = s.replace("-", "m")
    s = s.replace(".", "p")
    return s


def safe_ratio(numer: complex, denom: complex, floor: float = 1.0e-30):
    if abs(denom) <= floor:
        return None
    return numer / denom


def rel_complex_err(x: complex, xref: complex, floor: float = 1.0e-30) -> float:
    return abs(x - xref) / max(abs(xref), floor)


def finite_xy(rows: list[dict], x_key: str, y_key: str):
    xs = []
    ys = []
    for row in rows:
        x = row.get(x_key, np.nan)
        y = row.get(y_key, np.nan)
        if np.isfinite(x) and np.isfinite(y):
            xs.append(x)
            ys.append(y)
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)


def write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = sorted({k for row in rows for k in row.keys()}) if rows else []
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_row(mode: KerrMode, N_edge: int, z1: float, z2: float, scenario: str) -> dict:
    row = {
        "scenario": scenario,
        "a": float(mode.a),
        "omega": float(mode.omega),
        "N_edge": int(N_edge),
        "N_mid": int(2 * N_edge),
        "z1": float(z1),
        "z2": float(z2),
        "zpair": zpair_label(z1, z2),
        "ok": 0,
        "error": "",
        "B_inc_re": np.nan,
        "B_inc_im": np.nan,
        "B_ref_re": np.nan,
        "B_ref_im": np.nan,
        "B_trans_re": np.nan,
        "B_trans_im": np.nan,
        "ratio_ref_over_inc_re": np.nan,
        "ratio_ref_over_inc_im": np.nan,
        "outer_abel_residual": np.nan,
        "inner_abel_residual": np.nan,
        "detS_residual": np.nan,
        "cond_M_left": np.nan,
        "cond_T_mid": np.nan,
        "cond_M_outer_at_z2": np.nan,
        "solve_in_relres": np.nan,
        "solve_out_relres": np.nan,
        "solve_in_cond_raw": np.nan,
        "solve_out_cond_raw": np.nan,
        "solve_in_cond_scaled": np.nan,
        "solve_out_cond_scaled": np.nan,
        "state_in_z1_relerr": np.nan,
        "state_out_z1_relerr": np.nan,
    }

    try:
        amp = TeukRadAmplitudeIn3PatchWithAbelChecks(
            mode,
            N_left=N_edge,
            N_mid=2 * N_edge,
            N_right=N_edge,
            z1=z1,
            z2=z2,
        )
        ratio = safe_ratio(amp.B_ref, amp.B_inc)
        row.update(
            {
                "ok": 1,
                "B_inc_re": amp.B_inc.real,
                "B_inc_im": amp.B_inc.imag,
                "B_ref_re": amp.B_ref.real,
                "B_ref_im": amp.B_ref.imag,
                "B_trans_re": amp.B_trans.real,
                "B_trans_im": amp.B_trans.imag,
                "ratio_ref_over_inc_re": ratio.real if ratio is not None else np.nan,
                "ratio_ref_over_inc_im": ratio.imag if ratio is not None else np.nan,
                "outer_abel_residual": float(amp.outer_abel_residual),
                "inner_abel_residual": float(amp.inner_abel_residual),
                "detS_residual": float(amp.detS_residual),
                "cond_M_left": float(amp.smatrix.get("cond_M_left", np.nan)),
                "cond_T_mid": float(amp.smatrix.get("cond_T_mid", np.nan)),
                "cond_M_outer_at_z2": float(amp.smatrix.get("cond_M_outer_at_z2", np.nan)),
                "solve_in_relres": float(amp.smatrix.get("solve_in_relres", np.nan)),
                "solve_out_relres": float(amp.smatrix.get("solve_out_relres", np.nan)),
                "solve_in_cond_raw": float(amp.smatrix.get("solve_in_cond_raw", np.nan)),
                "solve_out_cond_raw": float(amp.smatrix.get("solve_out_cond_raw", np.nan)),
                "solve_in_cond_scaled": float(amp.smatrix.get("solve_in_cond_scaled", np.nan)),
                "solve_out_cond_scaled": float(amp.smatrix.get("solve_out_cond_scaled", np.nan)),
                "state_in_z1_relerr": float(amp.smatrix.get("state_in_z1_relerr", np.nan)),
                "state_out_z1_relerr": float(amp.smatrix.get("state_out_z1_relerr", np.nan)),
            }
        )
    except Exception as e:
        row["error"] = str(e)

    return row


def build_convergence_rows(detail_rows: list[dict], N_ref: int) -> list[dict]:
    ref_map: dict[tuple[str, float, float, float, float], dict] = {}
    for row in detail_rows:
        if row["ok"] != 1 or row["N_edge"] != N_ref:
            continue
        ref_map[(row["scenario"], row["a"], row["omega"], row["z1"], row["z2"])] = row

    conv_rows = []
    for row in detail_rows:
        if row["ok"] != 1 or row["N_edge"] == N_ref:
            continue

        key = (row["scenario"], row["a"], row["omega"], row["z1"], row["z2"])
        ref = ref_map.get(key)
        if ref is None:
            continue

        B_inc = complex(row["B_inc_re"], row["B_inc_im"])
        B_ref = complex(row["B_ref_re"], row["B_ref_im"])
        ratio = safe_ratio(B_ref, B_inc)

        B_inc_ref = complex(ref["B_inc_re"], ref["B_inc_im"])
        B_ref_ref = complex(ref["B_ref_re"], ref["B_ref_im"])
        ratio_ref = safe_ratio(B_ref_ref, B_inc_ref)

        conv_rows.append(
            {
                "scenario": row["scenario"],
                "a": row["a"],
                "omega": row["omega"],
                "N_edge": row["N_edge"],
                "N_mid": row["N_mid"],
                "z1": row["z1"],
                "z2": row["z2"],
                "zpair": row["zpair"],
                "err_B_inc": rel_complex_err(B_inc, B_inc_ref),
                "err_B_ref": rel_complex_err(B_ref, B_ref_ref),
                "err_ratio_ref_over_inc": (
                    rel_complex_err(ratio, ratio_ref)
                    if (ratio is not None and ratio_ref is not None)
                    else np.nan
                ),
            }
        )

    return conv_rows


def plot_metric_curves(
    rows: list[dict],
    x_key: str,
    y_key: str,
    curve_key: str,
    title: str,
    ylabel: str,
    out_path: Path,
    yscale: str = "log",
) -> None:
    fig = plt.figure(figsize=(7.2, 5.0))
    ax = fig.add_subplot(111)

    curve_values = []
    for row in rows:
        val = row[curve_key]
        if val not in curve_values:
            curve_values.append(val)

    plotted = False
    for cval in curve_values:
        subset = [r for r in rows if r[curve_key] == cval]
        subset = sorted(subset, key=lambda r: (r[x_key], r.get("N_mid", 0)))
        xs, ys = finite_xy(subset, x_key, y_key)
        if xs.size == 0:
            continue
        ax.plot(xs, ys, marker="o", label=str(cval))
        plotted = True

    ax.set_xlabel("N (outer patches) / 2N (middle patch)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    if yscale:
        ax.set_yscale(yscale)
    if plotted:
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_baseline_plots(
    detail_rows: list[dict],
    conv_rows: list[dict],
    a_values: list[float],
    out_dir: Path,
) -> None:
    baseline_detail = [r for r in detail_rows if r["scenario"] == "baseline" and r["ok"] == 1]
    baseline_conv = [r for r in conv_rows if r["scenario"] == "baseline"]

    for a in a_values:
        detail_a = [r for r in baseline_detail if math.isclose(r["a"], a, rel_tol=0.0, abs_tol=1.0e-15)]
        conv_a = [r for r in baseline_conv if math.isclose(r["a"], a, rel_tol=0.0, abs_tol=1.0e-15)]
        if not detail_a:
            continue

        stem = f"baseline_a_{slug_float(a)}"
        plot_metric_curves(
            conv_a,
            x_key="N_edge",
            y_key="err_B_inc",
            curve_key="omega",
            title=f"Three-patch convergence of B_inc (a={a})\n(z1=0.10, z2=0.90)",
            ylabel="relative error vs highest N",
            out_path=out_dir / f"{stem}_conv_B_inc.png",
        )
        plot_metric_curves(
            conv_a,
            x_key="N_edge",
            y_key="err_B_ref",
            curve_key="omega",
            title=f"Three-patch convergence of B_ref (a={a})\n(z1=0.10, z2=0.90)",
            ylabel="relative error vs highest N",
            out_path=out_dir / f"{stem}_conv_B_ref.png",
        )
        plot_metric_curves(
            conv_a,
            x_key="N_edge",
            y_key="err_ratio_ref_over_inc",
            curve_key="omega",
            title=f"Three-patch convergence of B_ref/B_inc (a={a})\n(z1=0.10, z2=0.90)",
            ylabel="relative error vs highest N",
            out_path=out_dir / f"{stem}_conv_ratio.png",
        )
        plot_metric_curves(
            detail_a,
            x_key="N_edge",
            y_key="outer_abel_residual",
            curve_key="omega",
            title=f"Outer Abel residual (a={a})\n(z1=0.10, z2=0.90)",
            ylabel="outer Abel residual",
            out_path=out_dir / f"{stem}_outer_abel.png",
        )
        plot_metric_curves(
            detail_a,
            x_key="N_edge",
            y_key="inner_abel_residual",
            curve_key="omega",
            title=f"Inner Abel residual (a={a})\n(z1=0.10, z2=0.90)",
            ylabel="inner Abel residual",
            out_path=out_dir / f"{stem}_inner_abel.png",
        )
        plot_metric_curves(
            detail_a,
            x_key="N_edge",
            y_key="detS_residual",
            curve_key="omega",
            title=f"det(S) residual (a={a})\n(z1=0.10, z2=0.90)",
            ylabel="det(S) residual",
            out_path=out_dir / f"{stem}_detS.png",
        )


def save_match_plots(
    detail_rows: list[dict],
    conv_rows: list[dict],
    a_values: list[float],
    match_omegas: list[float],
    out_dir: Path,
) -> None:
    match_detail = [r for r in detail_rows if r["scenario"] == "match_sweep" and r["ok"] == 1]
    match_conv = [r for r in conv_rows if r["scenario"] == "match_sweep"]

    for a in a_values:
        for omega in match_omegas:
            detail_sel = [
                r for r in match_detail
                if math.isclose(r["a"], a, rel_tol=0.0, abs_tol=1.0e-15)
                and math.isclose(r["omega"], omega, rel_tol=0.0, abs_tol=1.0e-15)
            ]
            conv_sel = [
                r for r in match_conv
                if math.isclose(r["a"], a, rel_tol=0.0, abs_tol=1.0e-15)
                and math.isclose(r["omega"], omega, rel_tol=0.0, abs_tol=1.0e-15)
            ]
            if not detail_sel:
                continue

            stem = f"match_a_{slug_float(a)}_omega_{slug_float(omega)}"
            plot_metric_curves(
                conv_sel,
                x_key="N_edge",
                y_key="err_B_inc",
                curve_key="zpair",
                title=f"Match-point convergence of B_inc (a={a}, omega={omega})",
                ylabel="relative error vs highest N",
                out_path=out_dir / f"{stem}_conv_B_inc.png",
            )
            plot_metric_curves(
                conv_sel,
                x_key="N_edge",
                y_key="err_B_ref",
                curve_key="zpair",
                title=f"Match-point convergence of B_ref (a={a}, omega={omega})",
                ylabel="relative error vs highest N",
                out_path=out_dir / f"{stem}_conv_B_ref.png",
            )
            plot_metric_curves(
                conv_sel,
                x_key="N_edge",
                y_key="err_ratio_ref_over_inc",
                curve_key="zpair",
                title=f"Match-point convergence of B_ref/B_inc (a={a}, omega={omega})",
                ylabel="relative error vs highest N",
                out_path=out_dir / f"{stem}_conv_ratio.png",
            )
            plot_metric_curves(
                detail_sel,
                x_key="N_edge",
                y_key="outer_abel_residual",
                curve_key="zpair",
                title=f"Outer Abel residual (a={a}, omega={omega})",
                ylabel="outer Abel residual",
                out_path=out_dir / f"{stem}_outer_abel.png",
            )
            plot_metric_curves(
                detail_sel,
                x_key="N_edge",
                y_key="inner_abel_residual",
                curve_key="zpair",
                title=f"Inner Abel residual (a={a}, omega={omega})",
                ylabel="inner Abel residual",
                out_path=out_dir / f"{stem}_inner_abel.png",
            )
            plot_metric_curves(
                detail_sel,
                x_key="N_edge",
                y_key="detS_residual",
                curve_key="zpair",
                title=f"det(S) residual (a={a}, omega={omega})",
                ylabel="det(S) residual",
                out_path=out_dir / f"{stem}_detS.png",
            )


def write_summary(
    out_dir: Path,
    a_values: list[float],
    omega_values: list[float],
    match_omegas: list[float],
    match_pairs: list[tuple[float, float]],
    N_values: list[int],
) -> None:
    summary_path = out_dir / "README.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Three-patch spectral convergence benchmark\n")
        f.write("========================================\n\n")
        f.write(f"a values: {a_values}\n")
        f.write(f"omega values (baseline): {omega_values}\n")
        f.write(f"omega values (match sweep): {match_omegas}\n")
        f.write(f"N_edge values: {N_values}\n")
        f.write("middle patch order is fixed to 2*N_edge\n")
        f.write("baseline match pair: (z1, z2) = (0.10, 0.90)\n")
        f.write(f"match sweep pairs: {match_pairs}\n\n")
        f.write("CSV files:\n")
        f.write("  - baseline_detail.csv\n")
        f.write("  - baseline_convergence.csv\n")
        f.write("  - match_sweep_detail.csv\n")
        f.write("  - match_sweep_convergence.csv\n\n")
        f.write("Plots:\n")
        f.write("  * baseline: for each a, curves over omega at fixed (z1, z2) = (0.10, 0.90)\n")
        f.write("  * match_sweep: for each (a, omega), curves over the five requested match-point pairs\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark convergence and conservation diagnostics of the three-patch Teukolsky amplitude solver."
        )
    )
    parser.add_argument("--a-values", type=str, default=",".join(str(x) for x in DEFAULT_A_VALUES))
    parser.add_argument("--omega-values", type=str, default=",".join(str(x) for x in DEFAULT_OMEGA_VALUES))
    parser.add_argument(
        "--match-omega-values",
        type=str,
        default=",".join(str(x) for x in DEFAULT_MATCH_OMEGA_VALUES),
    )
    parser.add_argument("--N-list", type=str, default=",".join(str(x) for x in DEFAULT_N_VALUES))
    parser.add_argument(
        "--out-dir-name",
        type=str,
        default="three_patch_convergence_benchmark",
        help="Directory name under benchmark/outputs/.",
    )
    parser.add_argument("--ell", type=int, default=2)
    parser.add_argument("--m", type=int, default=2)
    parser.add_argument("--s", type=int, default=-2)
    parser.add_argument("--skip-plots", action="store_true")
    args = parser.parse_args()

    a_values = parse_float_list(args.a_values)
    omega_values = parse_float_list(args.omega_values)
    match_omegas = parse_float_list(args.match_omega_values)
    N_values = parse_n_list(args.N_list)
    N_ref = max(N_values)

    match_pairs = list(DEFAULT_MATCH_PAIRS)
    baseline_pair = DEFAULT_MATCH_PAIRS[0]

    out_dir = ROOT / "benchmark" / "outputs" / args.out_dir_name
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline_rows: list[dict] = []
    match_rows: list[dict] = []

    baseline_jobs = len(a_values) * len(omega_values) * len(N_values)
    match_jobs = len(a_values) * len(match_omegas) * len(match_pairs) * len(N_values)
    total_jobs = baseline_jobs + match_jobs

    pbar = tqdm(total=total_jobs, desc="Three-patch benchmark")

    # ---------------------------------------------------------
    # baseline scan: fixed (z1, z2) = (0.10, 0.90), full omega list
    # ---------------------------------------------------------
    for a, omega, N_edge in product(a_values, omega_values, N_values):
        mode = KerrMode(
            M=1.0,
            a=float(a),
            omega=float(omega),
            ell=args.ell,
            m=args.m,
            lam=None,
            s=args.s,
        )
        row = make_row(mode, N_edge=N_edge, z1=baseline_pair[0], z2=baseline_pair[1], scenario="baseline")
        baseline_rows.append(row)
        postfix = {
            "part": "baseline",
            "a": f"{a:.2f}",
            "omega": f"{omega:.1e}",
            "N": N_edge,
            "ok": row["ok"],
        }
        if row["ok"] == 1:
            postfix["detS"] = f"{row['detS_residual']:.2e}"
        else:
            postfix["err"] = row["error"][:18]
        pbar.update(1)
        pbar.set_postfix(postfix)

    # ---------------------------------------------------------
    # match-point scan: five requested pairs, only omega = 1e-3, 1e-2 by default
    # ---------------------------------------------------------
    for a, omega, (z1, z2), N_edge in product(a_values, match_omegas, match_pairs, N_values):
        mode = KerrMode(
            M=1.0,
            a=float(a),
            omega=float(omega),
            ell=args.ell,
            m=args.m,
            lam=None,
            s=args.s,
        )
        row = make_row(mode, N_edge=N_edge, z1=z1, z2=z2, scenario="match_sweep")
        match_rows.append(row)
        postfix = {
            "part": "match",
            "a": f"{a:.2f}",
            "omega": f"{omega:.1e}",
            "z": f"({z1:.2f},{z2:.2f})",
            "N": N_edge,
            "ok": row["ok"],
        }
        if row["ok"] == 1:
            postfix["detS"] = f"{row['detS_residual']:.2e}"
        else:
            postfix["err"] = row["error"][:18]
        pbar.update(1)
        pbar.set_postfix(postfix)

    pbar.close()

    baseline_conv = build_convergence_rows(baseline_rows, N_ref=N_ref)
    match_conv = build_convergence_rows(match_rows, N_ref=N_ref)

    baseline_detail_csv = out_dir / "baseline_detail.csv"
    baseline_conv_csv = out_dir / "baseline_convergence.csv"
    match_detail_csv = out_dir / "match_sweep_detail.csv"
    match_conv_csv = out_dir / "match_sweep_convergence.csv"

    write_csv(baseline_detail_csv, baseline_rows)
    write_csv(baseline_conv_csv, baseline_conv)
    write_csv(match_detail_csv, match_rows)
    write_csv(match_conv_csv, match_conv)

    print(f"[saved] {baseline_detail_csv}")
    print(f"[saved] {baseline_conv_csv}")
    print(f"[saved] {match_detail_csv}")
    print(f"[saved] {match_conv_csv}")

    write_summary(
        out_dir=out_dir,
        a_values=a_values,
        omega_values=omega_values,
        match_omegas=match_omegas,
        match_pairs=match_pairs,
        N_values=N_values,
    )

    if args.skip_plots:
        print("[done] skip-plots enabled")
        return

    print("[stage] saving baseline plots")
    save_baseline_plots(baseline_rows, baseline_conv, a_values=a_values, out_dir=out_dir)

    print("[stage] saving match-point sweep plots")
    save_match_plots(
        match_rows,
        match_conv,
        a_values=a_values,
        match_omegas=match_omegas,
        out_dir=out_dir,
    )

    print(f"[done] plots saved under {out_dir}")


if __name__ == "__main__":
    main()