from __future__ import annotations

from __future__ import annotations

import os

# ---- 必须在 numpy / matplotlib 之前 ----
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
import csv
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from utils.mode import KerrMode
from utils.amplitude import TeukRadAmplitudeInWithAbelChecks


def parse_n_list(text: str):
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def rel_complex_err(x: complex, xref: complex, floor: float = 1.0e-30) -> float:
    return abs(x - xref) / max(abs(xref), floor)


def safe_abs2(x):
    if x is None:
        return np.nan
    return abs(x) ** 2


def main():
    parser = argparse.ArgumentParser(
        description="Scan (a, omega) domain, check flux conservation and spectral-order convergence."
    )
    parser.add_argument("--a-min", type=float, default=0.001)
    parser.add_argument("--a-max", type=float, default=0.999)
    parser.add_argument("--n-a", type=int, default=50)

    parser.add_argument("--omega-min", type=float, default=1.0e-4)
    parser.add_argument("--omega-max", type=float, default=10.0)
    parser.add_argument("--n-omega", type=int, default=50)

    parser.add_argument("--ell", type=int, default=2)
    parser.add_argument("--m", type=int, default=2)
    parser.add_argument("--s", type=int, default=-2)

    parser.add_argument("--N-list", type=str, default="24,32,48,64,80,128")
    parser.add_argument("--z-m", type=float, default=0.3)

    parser.add_argument("--out-prefix", type=str, default="amp_scan")
    parser.add_argument("--skip-plots", action="store_true", help="Only save CSVs, skip all matplotlib figures.")
    parser.add_argument("--skip-summary", action="store_true", help="Skip writing the text summary file.")
    args = parser.parse_args()

    a_values = np.linspace(args.a_min, args.a_max, args.n_a)
    omega_values = np.geomspace(args.omega_min, args.omega_max, args.n_omega)
    N_values = parse_n_list(args.N_list)
    N_ref = max(N_values)

    out_dir = ROOT / "benchmark" / "outputs"/"conservation_scan"
    out_dir.mkdir(parents=True, exist_ok=True)

    detail_rows = []
    ref_map = {}

    for ia, a in enumerate(a_values):
        for iw, omega in enumerate(omega_values):
            mode = KerrMode(
                M=1.0,
                a=float(a),
                omega=float(omega),
                ell=args.ell,
                m=args.m,
                lam=None,
                s=args.s,
            )

            for N in N_values:
                row = {
                    "a": float(a),
                    "omega": float(omega),
                    "N": int(N),
                    "ok": 0,
                    "k_hor": float(mode.k_hor),
                    "B_inc_re": np.nan,
                    "B_inc_im": np.nan,
                    "B_ref_re": np.nan,
                    "B_ref_im": np.nan,
                    "B_trans_re": np.nan,
                    "B_trans_im": np.nan,
                    "ratio_ref_over_inc_abs2": np.nan,
                    "B_trans_over_B_inc_abs2": np.nan,
                    "cons_residual": np.nan,
                    "flux_residual_scaled": np.nan,
                    "outer_abel_residual": np.nan,
                    "inner_abel_residual": np.nan,
                    "detS_residual": np.nan,
                }

                try:
                    amp = TeukRadAmplitudeInWithAbelChecks(mode, N_in=N, N_out=N, z_m=args.z_m)

                    B_inc = amp.B_inc
                    B_ref = amp.B_ref
                    B_trans = amp.B_trans

                    row.update(
                        {
                            "ok": 1,
                            "B_inc_re": B_inc.real,
                            "B_inc_im": B_inc.imag,
                            "B_ref_re": B_ref.real,
                            "B_ref_im": B_ref.imag,
                            "B_trans_re": B_trans.real,
                            "B_trans_im": B_trans.imag,

                            # Abel diagnostics
                            "outer_abel_num_re": amp.smatrix["outer_abel_num"].real,
                            "outer_abel_num_im": amp.smatrix["outer_abel_num"].imag,
                            "outer_abel_th_re": amp.smatrix["outer_abel_th"].real,
                            "outer_abel_th_im": amp.smatrix["outer_abel_th"].imag,
                            "outer_abel_residual": amp.outer_abel_residual,

                            "inner_abel_num_re": amp.smatrix["inner_abel_num"].real,
                            "inner_abel_num_im": amp.smatrix["inner_abel_num"].imag,
                            "inner_abel_th_re": amp.smatrix["inner_abel_th"].real,
                            "inner_abel_th_im": amp.smatrix["inner_abel_th"].imag,
                            "inner_abel_residual": amp.inner_abel_residual,

                            "detS_num_re": amp.smatrix["detS_num"].real,
                            "detS_num_im": amp.smatrix["detS_num"].imag,
                            "detS_th_re": amp.smatrix["detS_th"].real,
                            "detS_th_im": amp.smatrix["detS_th"].imag,
                            "detS_residual": amp.detS_residual,

                            # full transfer matrix columns
                            "Cin_down_re": amp.smatrix["Cin_down"].real,
                            "Cin_down_im": amp.smatrix["Cin_down"].imag,
                            "Cin_up_re": amp.smatrix["Cin_up"].real,
                            "Cin_up_im": amp.smatrix["Cin_up"].imag,
                            "Cout_down_re": amp.smatrix["Cout_down"].real,
                            "Cout_down_im": amp.smatrix["Cout_down"].imag,
                            "Cout_up_re": amp.smatrix["Cout_up"].real,
                            "Cout_up_im": amp.smatrix["Cout_up"].imag,
                        }
                    )
                    



                    if N == N_ref:
                        ref_map[(float(a), float(omega))] = {
                            "B_inc": B_inc,
                            "B_ref": B_ref,
                            "outer_abel_residual": amp.outer_abel_residual,
                            "inner_abel_residual": amp.inner_abel_residual,
                            "detS_residual": amp.detS_residual,
                        }

                except Exception as e:
                    row["error"] = str(e)

                detail_rows.append(row)
                # print(
                #     f"[scan] a={a:.6f}, omega={omega:.6e}, N={N:3d}, ok={row['ok']}, "
                #     f"cons={row['cons_residual']}"
                # )
                print(
                    f"[scan] a={a:.6f}, omega={omega:.6e}, N={N:3d}, ok={row['ok']}, "
                    f"outer={row['outer_abel_residual']:.3e}, "
                    f"inner={row['inner_abel_residual']:.3e}, "
                    f"detS={row['detS_residual']:.3e}"
                )
    # convergence against reference N_ref
    conv_rows = []
    for row in detail_rows:
        if row["ok"] != 1:
            continue
        if row["N"] == N_ref:
            continue

        key = (row["a"], row["omega"])
        if key not in ref_map:
            continue

        ref = ref_map[key]
        B_inc = complex(row["B_inc_re"], row["B_inc_im"])
        B_ref = complex(row["B_ref_re"], row["B_ref_im"])
        ratio = None
        if np.isfinite(row["ratio_ref_over_inc_abs2"]):
            # reconstruct directly from row B's to avoid phase loss
            ratio = B_ref / B_inc if abs(B_inc) > 1.0e-30 else np.nan
        else:
            ratio = np.nan

        conv_rows.append(
            {
                "a": row["a"],
                "omega": row["omega"],
                "N": row["N"],
                "err_B_inc": rel_complex_err(B_inc, ref["B_inc"]),
                "err_B_ref": rel_complex_err(B_ref, ref["B_ref"]),
                "err_ratio_ref_over_inc": (
                    rel_complex_err(ratio, ref["ratio_ref_over_inc"])
                    if ratio is not np.nan and ref["ratio_ref_over_inc"] is not None
                    else np.nan
                ),
            }
        )

    # write CSVs
    detail_csv = out_dir / f"{args.out_prefix}_detail.csv"
    conv_csv = out_dir / f"{args.out_prefix}_convergence.csv"

    detail_fieldnames = sorted({k for row in detail_rows for k in row.keys()})
    with open(detail_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=detail_fieldnames)
        writer.writeheader()
        writer.writerows(detail_rows)

    conv_fieldnames = sorted({k for row in conv_rows for k in row.keys()})
    with open(conv_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=conv_fieldnames)
        writer.writeheader()
        writer.writerows(conv_rows)

    print(f"[saved] {detail_csv}", flush=True)
    print(f"[saved] {conv_csv}", flush=True)
    if args.skip_plots:
        print("[done] skip-plots enabled, stop after CSV export", flush=True)
        return
    # heatmaps for highest-N conservation
        # heatmaps for highest-N conservation
    print("[stage] build conservation heatmaps", flush=True)

    outer_map = np.full((args.n_a, args.n_omega), np.nan)
    inner_map = np.full((args.n_a, args.n_omega), np.nan)
    detS_map  = np.full((args.n_a, args.n_omega), np.nan)

    for ia, a in enumerate(a_values):
        for iw, omega in enumerate(omega_values):
            key = (float(a), float(omega))
            if key in ref_map:
                outer_map[ia, iw] = ref_map[key]["outer_abel_residual"]
                inner_map[ia, iw] = ref_map[key]["inner_abel_residual"]
                detS_map[ia, iw]  = ref_map[key]["detS_residual"]

    try:
        fig1, axes = plt.subplots(1, 3, figsize=(13, 5))

        im0 = axes[0].imshow(
            np.log10(np.clip(outer_map, 1e-30, None)),
            origin="lower",
            aspect="auto",
            extent=[np.log10(omega_values[0]), np.log10(omega_values[-1]), a_values[0], a_values[-1]],
        )
        axes[0].set_title(f"log10 outer map (N={N_ref})")
        axes[0].set_xlabel("log10 omega")
        axes[0].set_ylabel("a")
        fig1.colorbar(im0, ax=axes[0])

        im1 = axes[1].imshow(
            np.log10(np.clip(inner_map, 1e-30, None)),
            origin="lower",
            aspect="auto",
            extent=[np.log10(omega_values[0]), np.log10(omega_values[-1]), a_values[0], a_values[-1]],
        )
        axes[1].set_title(f"log10 inner map (N={N_ref})")
        axes[1].set_xlabel("log10 omega")
        axes[1].set_ylabel("a")
        fig1.colorbar(im1, ax=axes[1])

        im2 = axes[2].imshow(
            np.log10(np.clip(detS_map, 1e-30, None)),
            origin="lower",
            aspect="auto",
            extent=[np.log10(omega_values[0]), np.log10(omega_values[-1]), a_values[0], a_values[-1]],
        )
        axes[2].set_title(f"log10 detS map (N={N_ref})")
        axes[2].set_xlabel("log10 omega")
        axes[2].set_ylabel("a")
        fig1.colorbar(im2, ax=axes[2])
        fig1.tight_layout()
        fig1_path = out_dir / f"{args.out_prefix}_conservation_heatmaps.png"
        print("[stage] save conservation heatmaps", flush=True)
        fig1.savefig(fig1_path, dpi=180, bbox_inches="tight")
        plt.close(fig1)
        print(f"[saved] {fig1_path}", flush=True)

    except Exception as e:
        print(f"[warn] failed to build/save conservation heatmaps: {e}", flush=True)

        # convergence summary lines
    print("[stage] build convergence summary plots", flush=True)

    Ns_sorted = sorted([N for N in N_values if N != N_ref])

    med_Binc, max_Binc = [], []
    med_Bref, max_Bref = [], []
    med_ratio, max_ratio = [], []
    med_cons, max_cons = [], []

    for N in Ns_sorted:
        rowsN = [r for r in conv_rows if r["N"] == N]
        detN = [r for r in detail_rows if r["N"] == N and r["ok"] == 1]

        err_Binc = np.array([r["err_B_inc"] for r in rowsN], dtype=float)
        err_Bref = np.array([r["err_B_ref"] for r in rowsN], dtype=float)
        err_ratio = np.array(
            [r["err_ratio_ref_over_inc"] for r in rowsN if np.isfinite(r["err_ratio_ref_over_inc"])],
            dtype=float,
        )
        consN = np.array([r["cons_residual"] for r in detN], dtype=float)

        med_Binc.append(np.nanmedian(err_Binc) if err_Binc.size else np.nan)
        max_Binc.append(np.nanmax(err_Binc) if err_Binc.size else np.nan)

        med_Bref.append(np.nanmedian(err_Bref) if err_Bref.size else np.nan)
        max_Bref.append(np.nanmax(err_Bref) if err_Bref.size else np.nan)

        med_ratio.append(np.nanmedian(err_ratio) if err_ratio.size else np.nan)
        max_ratio.append(np.nanmax(err_ratio) if err_ratio.size else np.nan)

        med_cons.append(np.nanmedian(consN) if consN.size else np.nan)
        max_cons.append(np.nanmax(consN) if consN.size else np.nan)

    try:
        fig2, axes = plt.subplots(1, 2, figsize=(13, 5))

        axes[0].plot(Ns_sorted, med_Binc, marker="o", label="median rel err B_inc")
        axes[0].plot(Ns_sorted, max_Binc, marker="o", linestyle="--", label="max rel err B_inc")
        axes[0].plot(Ns_sorted, med_Bref, marker="s", label="median rel err B_ref")
        axes[0].plot(Ns_sorted, max_Bref, marker="s", linestyle="--", label="max rel err B_ref")
        axes[0].plot(Ns_sorted, med_ratio, marker="^", label="median rel err B_ref/B_inc")
        axes[0].plot(Ns_sorted, max_ratio, marker="^", linestyle="--", label="max rel err B_ref/B_inc")
        axes[0].set_yscale("log")
        axes[0].set_xlabel("spectral order N")
        axes[0].set_ylabel("relative error vs highest N")
        axes[0].set_title(f"convergence to N_ref={N_ref}")
        axes[0].grid(alpha=0.3)
        axes[0].legend()

        axes[1].plot(Ns_sorted, med_cons, marker="o", label="median conservation residual")
        axes[1].plot(Ns_sorted, max_cons, marker="o", linestyle="--", label="max conservation residual")
        axes[1].set_yscale("log")
        axes[1].set_xlabel("spectral order N")
        axes[1].set_ylabel("residual")
        axes[1].set_title("conservation check vs spectral order")
        axes[1].grid(alpha=0.3)
        axes[1].legend()

        fig2.tight_layout()
        fig2_path = out_dir / f"{args.out_prefix}_convergence_summary.png"
        print("[stage] save convergence summary plots", flush=True)
        fig2.savefig(fig2_path, dpi=180, bbox_inches="tight")
        plt.close(fig2)
        print(f"[saved] {fig2_path}", flush=True)

    except Exception as e:
        print(f"[warn] failed to build/save convergence summary plots: {e}", flush=True)
    if args.skip_summary:
        print("[done] skip-summary enabled", flush=True)
        return

    print("[stage] write text summary", flush=True)
    try:
        summary_txt = out_dir / f"{args.out_prefix}_summary.txt"
        with open(summary_txt, "w", encoding="utf-8") as f:
            f.write(f"a in [{args.a_min}, {args.a_max}], n_a={args.n_a}\n")
            f.write(f"omega in [{args.omega_min}, {args.omega_max}], n_omega={args.n_omega}\n")
            f.write(f"N_list = {N_values}\n")
            f.write(f"z_m = {args.z_m}\n")
            f.write(f"N_ref = {N_ref}\n\n")

            ref_cons_all = np.array(list(ref_map[k]["cons_residual"] for k in ref_map.keys()), dtype=float)
            ref_flux_all = np.array(list(ref_map[k]["flux_residual_scaled"] for k in ref_map.keys()), dtype=float)

            f.write("Highest-order conservation residual summary:\n")
            f.write(f"  median cons residual = {np.nanmedian(ref_cons_all):.6e}\n")
            f.write(f"  max    cons residual = {np.nanmax(ref_cons_all):.6e}\n")
            f.write(f"  median flux residual = {np.nanmedian(ref_flux_all):.6e}\n")
            f.write(f"  max    flux residual = {np.nanmax(ref_flux_all):.6e}\n\n")

            for N, mbi, xbi, mbr, xbr, mrr, xrr in zip(
                Ns_sorted, med_Binc, max_Binc, med_Bref, max_Bref, med_ratio, max_ratio
            ):
                f.write(
                    f"N={N:4d}: "
                    f"med/max err B_inc = {mbi:.6e}/{xbi:.6e}, "
                    f"med/max err B_ref = {mbr:.6e}/{xbr:.6e}, "
                    f"med/max err ratio = {mrr:.6e}/{xrr:.6e}\n"
                )

        print(f"[saved] {summary_txt}", flush=True)

    except Exception as e:
        print(f"[warn] failed to write summary: {e}", flush=True)


if __name__ == "__main__":
    main()
