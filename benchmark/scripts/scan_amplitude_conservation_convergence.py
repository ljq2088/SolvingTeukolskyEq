from __future__ import annotations

import os

# ---- 必须在 numpy / matplotlib 之前 ----
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("MPLBACKEND", "Agg")
from tqdm import tqdm
import argparse
import csv
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


def parse_n_list(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def rel_complex_err(x: complex, xref: complex, floor: float = 1.0e-30) -> float:
    return abs(x - xref) / max(abs(xref), floor)


def safe_ratio(numer: complex, denom: complex, floor: float = 1.0e-30):
    if abs(denom) <= floor:
        return None
    return numer / denom


def row_complex(row: dict, prefix: str) -> complex:
    return complex(row[f"{prefix}_re"], row[f"{prefix}_im"])


def nanmedian_safe(arr) -> float:
    arr = np.asarray(arr, dtype=float)
    return float(np.nanmedian(arr)) if arr.size else np.nan


def nanmax_safe(arr) -> float:
    arr = np.asarray(arr, dtype=float)
    return float(np.nanmax(arr)) if arr.size else np.nan


def print_scan_status(done: int, total: int, row: dict) -> None:
    head = (
        f"[scan {done:6d}/{total:6d}] "
        f"a={row['a']:.6f}, omega={row['omega']:.6e}, N={row['N']:3d}"
    )
    if row["ok"] == 1:
        msg = (
            f"{head}, ok=1, "
            f"outer={row['outer_abel_residual']:.3e}, "
            f"inner={row['inner_abel_residual']:.3e}, "
            f"detS={row['detS_residual']:.3e}"
        )
    else:
        msg = f"{head}, ok=0, error={row.get('error', 'unknown')}"
    print(msg, flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Scan (a, omega) domain, check Abel-type invariant diagnostics and spectral-order convergence."
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
    #对于ω小于1e-2的时候,需要使用更大的N
    
    parser.add_argument("--N-list", type=str, default="24,32,48,64,80,128,160,180,256")
    parser.add_argument("--z-m", type=float, default=0.3)

    parser.add_argument("--out-prefix", type=str, default="amp_scan")
    parser.add_argument("--skip-plots", action="store_true", help="Only save CSVs, skip all matplotlib figures.")
    parser.add_argument("--skip-summary", action="store_true", help="Skip writing the text summary file.")
    args = parser.parse_args()

    a_values = np.linspace(args.a_min, args.a_max, args.n_a)
    omega_values = np.geomspace(args.omega_min, args.omega_max, args.n_omega)
    N_values = parse_n_list(args.N_list)
    N_ref = max(N_values)

    out_dir = ROOT / "benchmark" / "outputs" / "abel_scan"
    out_dir.mkdir(parents=True, exist_ok=True)

    detail_rows: list[dict] = []
    ref_map: dict[tuple[float, float], dict] = {}

    total_jobs = len(a_values) * len(omega_values) * len(N_values)
    done_jobs = 0
    pbar = tqdm(total=total_jobs, desc="Scanning")
    for a in a_values:
        for omega in omega_values:
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
                    "error": "",
                    "k_hor": float(mode.k_hor),

                    "B_inc_re": np.nan,
                    "B_inc_im": np.nan,
                    "B_ref_re": np.nan,
                    "B_ref_im": np.nan,
                    "B_trans_re": np.nan,
                    "B_trans_im": np.nan,

                    "ratio_ref_over_inc_re": np.nan,
                    "ratio_ref_over_inc_im": np.nan,

                    "outer_abel_num_re": np.nan,
                    "outer_abel_num_im": np.nan,
                    "outer_abel_th_re": np.nan,
                    "outer_abel_th_im": np.nan,
                    "outer_abel_residual": np.nan,

                    "inner_abel_num_re": np.nan,
                    "inner_abel_num_im": np.nan,
                    "inner_abel_th_re": np.nan,
                    "inner_abel_th_im": np.nan,
                    "inner_abel_residual": np.nan,

                    "detS_num_re": np.nan,
                    "detS_num_im": np.nan,
                    "detS_th_re": np.nan,
                    "detS_th_im": np.nan,
                    "detS_residual": np.nan,

                    "Cin_down_re": np.nan,
                    "Cin_down_im": np.nan,
                    "Cin_up_re": np.nan,
                    "Cin_up_im": np.nan,
                    "Cout_down_re": np.nan,
                    "Cout_down_im": np.nan,
                    "Cout_up_re": np.nan,
                    "Cout_up_im": np.nan,
                }

                try:
                    amp = TeukRadAmplitudeInWithAbelChecks(mode, N_in=N, N_out=N, z_m=args.z_m)

                    B_inc = amp.B_inc
                    B_ref = amp.B_ref
                    B_trans = amp.B_trans
                    ratio_ref_over_inc = safe_ratio(B_ref, B_inc)

                    row.update(
                        {
                            "ok": 1,
                            "B_inc_re": B_inc.real,
                            "B_inc_im": B_inc.imag,
                            "B_ref_re": B_ref.real,
                            "B_ref_im": B_ref.imag,
                            "B_trans_re": B_trans.real,
                            "B_trans_im": B_trans.imag,

                            "ratio_ref_over_inc_re": (
                                ratio_ref_over_inc.real if ratio_ref_over_inc is not None else np.nan
                            ),
                            "ratio_ref_over_inc_im": (
                                ratio_ref_over_inc.imag if ratio_ref_over_inc is not None else np.nan
                            ),

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
                            "ratio_ref_over_inc": ratio_ref_over_inc,
                            "outer_abel_residual": amp.outer_abel_residual,
                            "inner_abel_residual": amp.inner_abel_residual,
                            "detS_residual": amp.detS_residual,
                        }

                except Exception as e:
                    row["error"] = str(e)

                detail_rows.append(row)
                done_jobs += 1
                # print_scan_status(done_jobs, total_jobs, row)
                #把打印的逻辑改成进度条的后缀显示
                postfix = {
                'a': f"{row['a']:.3f}",
                'ω': f"{row['omega']:.3e}",
                'N': row['N'],
                'ok': row['ok']
            }
                if row['ok'] == 1:
                    postfix['outer'] = f"{row['outer_abel_residual']:.2e}"
                    postfix['inner'] = f"{row['inner_abel_residual']:.2e}"
                    postfix['detS'] = f"{row['detS_residual']:.2e}"
                else:
                    postfix['err'] = row.get('error', 'unknown')[:20]  # 截断过长的错误信息
                pbar.update(1) 
                
                pbar.set_postfix(postfix)

    pbar.close()
    # ---------------------------------------------------------
    # convergence against highest N_ref
    # ---------------------------------------------------------
    conv_rows: list[dict] = []
    for row in detail_rows:
        if row["ok"] != 1 or row["N"] == N_ref:
            continue

        key = (row["a"], row["omega"])
        if key not in ref_map:
            continue

        ref = ref_map[key]
        B_inc = row_complex(row, "B_inc")
        B_ref = row_complex(row, "B_ref")
        ratio = safe_ratio(B_ref, B_inc)

        err_ratio = np.nan
        if ratio is not None and ref["ratio_ref_over_inc"] is not None:
            err_ratio = rel_complex_err(ratio, ref["ratio_ref_over_inc"])

        conv_rows.append(
            {
                "a": row["a"],
                "omega": row["omega"],
                "N": row["N"],
                "err_B_inc": rel_complex_err(B_inc, ref["B_inc"]),
                "err_B_ref": rel_complex_err(B_ref, ref["B_ref"]),
                "err_ratio_ref_over_inc": err_ratio,
            }
        )

    # ---------------------------------------------------------
    # write CSVs
    # ---------------------------------------------------------
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

    # ---------------------------------------------------------
    # heatmaps at highest N_ref
    # ---------------------------------------------------------
    print("[stage] build Abel heatmaps", flush=True)

    outer_map = np.full((args.n_a, args.n_omega), np.nan)
    inner_map = np.full((args.n_a, args.n_omega), np.nan)
    detS_map = np.full((args.n_a, args.n_omega), np.nan)

    for ia, a in enumerate(a_values):
        for iw, omega in enumerate(omega_values):
            key = (float(a), float(omega))
            if key in ref_map:
                outer_map[ia, iw] = ref_map[key]["outer_abel_residual"]
                inner_map[ia, iw] = ref_map[key]["inner_abel_residual"]
                detS_map[ia, iw] = ref_map[key]["detS_residual"]

    try:
        fig1, axes = plt.subplots(1, 3, figsize=(15, 5))
        extent = [np.log10(omega_values[0]), np.log10(omega_values[-1]), a_values[0], a_values[-1]]

        maps = [
            (outer_map, r"$\log_{10}\,\epsilon_{\mathrm{outer}}$"),
            (inner_map, r"$\log_{10}\,\epsilon_{\mathrm{inner}}$"),
            (detS_map,  r"$\log_{10}\,\epsilon_{\det S}$"),
        ]

        for ax, (data, title) in zip(axes, maps):
            im = ax.imshow(
                np.log10(np.clip(data, 1e-30, None)),
                origin="lower",
                aspect="auto",
                extent=extent,
            )
            ax.set_title(title + f"\n(highest N = {N_ref})")
            ax.set_xlabel(r"$\log_{10}\omega$")
            ax.set_ylabel(r"$a$")
            cbar = fig1.colorbar(im, ax=ax)
            cbar.set_label(r"$\log_{10}$ residual")

        fig1.suptitle(
            f"Abel-invariant residual heatmaps (ell={args.ell}, m={args.m}, s={args.s}, z_m={args.z_m})"
        )
        fig1.tight_layout()

        fig1_path = out_dir / f"{args.out_prefix}_abel_heatmaps.png"
        print("[stage] save Abel heatmaps", flush=True)
        fig1.savefig(fig1_path, dpi=180, bbox_inches="tight")
        plt.close(fig1)
        print(f"[saved] {fig1_path}", flush=True)

    except Exception as e:
        print(f"[warn] failed to build/save Abel heatmaps: {e}", flush=True)

    # ---------------------------------------------------------
    # convergence summary
    # ---------------------------------------------------------
    print("[stage] build convergence summary plots", flush=True)

    Ns_no_ref = sorted([N for N in N_values if N != N_ref])
    Ns_all = sorted(N_values)

    med_Binc, max_Binc = [], []
    med_Bref, max_Bref = [], []
    med_ratio, max_ratio = [], []

    med_outer, max_outer = [], []
    med_inner, max_inner = [], []
    med_detS, max_detS = [], []

    for N in Ns_no_ref:
        rowsN = [r for r in conv_rows if r["N"] == N]
        err_Binc = np.array([r["err_B_inc"] for r in rowsN], dtype=float)
        err_Bref = np.array([r["err_B_ref"] for r in rowsN], dtype=float)
        err_ratio = np.array(
            [r["err_ratio_ref_over_inc"] for r in rowsN if np.isfinite(r["err_ratio_ref_over_inc"])],
            dtype=float,
        )

        med_Binc.append(nanmedian_safe(err_Binc))
        max_Binc.append(nanmax_safe(err_Binc))
        med_Bref.append(nanmedian_safe(err_Bref))
        max_Bref.append(nanmax_safe(err_Bref))
        med_ratio.append(nanmedian_safe(err_ratio))
        max_ratio.append(nanmax_safe(err_ratio))

    for N in Ns_all:
        detN = [r for r in detail_rows if r["N"] == N and r["ok"] == 1]
        outerN = np.array([r["outer_abel_residual"] for r in detN], dtype=float)
        innerN = np.array([r["inner_abel_residual"] for r in detN], dtype=float)
        detSN = np.array([r["detS_residual"] for r in detN], dtype=float)

        med_outer.append(nanmedian_safe(outerN))
        max_outer.append(nanmax_safe(outerN))
        med_inner.append(nanmedian_safe(innerN))
        max_inner.append(nanmax_safe(innerN))
        med_detS.append(nanmedian_safe(detSN))
        max_detS.append(nanmax_safe(detSN))

    try:
        fig2, axes = plt.subplots(1, 2, figsize=(14, 5))

        # amplitude convergence against highest N_ref
        axes[0].plot(Ns_no_ref, med_Binc, marker="o", label=r"median rel err $B_{\rm inc}$")
        axes[0].plot(Ns_no_ref, max_Binc, marker="o", linestyle="--", label=r"max rel err $B_{\rm inc}$")
        axes[0].plot(Ns_no_ref, med_Bref, marker="s", label=r"median rel err $B_{\rm ref}$")
        axes[0].plot(Ns_no_ref, max_Bref, marker="s", linestyle="--", label=r"max rel err $B_{\rm ref}$")
        axes[0].plot(Ns_no_ref, med_ratio, marker="^", label=r"median rel err $B_{\rm ref}/B_{\rm inc}$")
        axes[0].plot(Ns_no_ref, max_ratio, marker="^", linestyle="--", label=r"max rel err $B_{\rm ref}/B_{\rm inc}$")
        axes[0].set_yscale("log")
        axes[0].set_xlabel("spectral order N")
        axes[0].set_ylabel(f"relative error vs N_ref={N_ref}")
        axes[0].set_title("Amplitude convergence")
        axes[0].grid(alpha=0.3)
        axes[0].legend()

        # Abel residuals vs N
        axes[1].plot(Ns_all, med_outer, marker="o", label=r"median $\epsilon_{\mathrm{outer}}$")
        axes[1].plot(Ns_all, max_outer, marker="o", linestyle="--", label=r"max $\epsilon_{\mathrm{outer}}$")
        axes[1].plot(Ns_all, med_inner, marker="s", label=r"median $\epsilon_{\mathrm{inner}}$")
        axes[1].plot(Ns_all, max_inner, marker="s", linestyle="--", label=r"max $\epsilon_{\mathrm{inner}}$")
        axes[1].plot(Ns_all, med_detS, marker="^", label=r"median $\epsilon_{\det S}$")
        axes[1].plot(Ns_all, max_detS, marker="^", linestyle="--", label=r"max $\epsilon_{\det S}$")
        axes[1].set_yscale("log")
        axes[1].set_xlabel("spectral order N")
        axes[1].set_ylabel("Abel residual")
        axes[1].set_title("Abel diagnostics vs spectral order")
        axes[1].grid(alpha=0.3)
        axes[1].legend()

        fig2.suptitle(
            f"Amplitude and Abel convergence summary (ell={args.ell}, m={args.m}, s={args.s}, z_m={args.z_m})"
        )
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

    # ---------------------------------------------------------
    # text summary
    # ---------------------------------------------------------
    print("[stage] write text summary", flush=True)
    try:
        summary_txt = out_dir / f"{args.out_prefix}_summary.txt"
        with open(summary_txt, "w", encoding="utf-8") as f:
            f.write(f"a in [{args.a_min}, {args.a_max}], n_a={args.n_a}\n")
            f.write(f"omega in [{args.omega_min}, {args.omega_max}], n_omega={args.n_omega}\n")
            f.write(f"N_list = {N_values}\n")
            f.write(f"z_m = {args.z_m}\n")
            f.write(f"N_ref = {N_ref}\n")
            f.write(f"(ell, m, s) = ({args.ell}, {args.m}, {args.s})\n\n")

            f.write("Highest-order Abel residual summary:\n")
            ref_outer_all = np.array([ref_map[k]["outer_abel_residual"] for k in ref_map.keys()], dtype=float)
            ref_inner_all = np.array([ref_map[k]["inner_abel_residual"] for k in ref_map.keys()], dtype=float)
            ref_detS_all = np.array([ref_map[k]["detS_residual"] for k in ref_map.keys()], dtype=float)

            f.write(f"  median outer residual = {np.nanmedian(ref_outer_all):.6e}\n")
            f.write(f"  max    outer residual = {np.nanmax(ref_outer_all):.6e}\n")
            f.write(f"  median inner residual = {np.nanmedian(ref_inner_all):.6e}\n")
            f.write(f"  max    inner residual = {np.nanmax(ref_inner_all):.6e}\n")
            f.write(f"  median detS  residual = {np.nanmedian(ref_detS_all):.6e}\n")
            f.write(f"  max    detS  residual = {np.nanmax(ref_detS_all):.6e}\n\n")

            f.write("Amplitude convergence against highest N_ref:\n")
            for N, mbi, xbi, mbr, xbr, mrr, xrr in zip(
                Ns_no_ref, med_Binc, max_Binc, med_Bref, max_Bref, med_ratio, max_ratio
            ):
                f.write(
                    f"N={N:4d}: "
                    f"med/max err B_inc = {mbi:.6e}/{xbi:.6e}, "
                    f"med/max err B_ref = {mbr:.6e}/{xbr:.6e}, "
                    f"med/max err ratio = {mrr:.6e}/{xrr:.6e}\n"
                )

            f.write("\nAbel residual summary by spectral order:\n")
            for N, mo, xo, mi, xi, md, xd in zip(
                Ns_all, med_outer, max_outer, med_inner, max_inner, med_detS, max_detS
            ):
                f.write(
                    f"N={N:4d}: "
                    f"med/max outer = {mo:.6e}/{xo:.6e}, "
                    f"med/max inner = {mi:.6e}/{xi:.6e}, "
                    f"med/max detS  = {md:.6e}/{xd:.6e}\n"
                )

        print(f"[saved] {summary_txt}", flush=True)

    except Exception as e:
        print(f"[warn] failed to write summary: {e}", flush=True)


if __name__ == "__main__":
    main()