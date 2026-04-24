from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("MPLBACKEND", "Agg")

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
from utils.amplitude_three_patch import TeukRadAmplitudeIn3PatchWithAbelChecks


def parse_n_list(text: str):
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def rel_complex_err(x: complex, xref: complex, floor: float = 1.0e-30) -> float:
    return abs(x - xref) / max(abs(xref), floor)


def safe_ratio(numer: complex, denom: complex, floor: float = 1.0e-30):
    if abs(denom) <= floor:
        return None
    return numer / denom


def nanmedian_safe(arr):
    arr = np.asarray(arr, dtype=float)
    return float(np.nanmedian(arr)) if arr.size else np.nan


def nanmax_safe(arr):
    arr = np.asarray(arr, dtype=float)
    return float(np.nanmax(arr)) if arr.size else np.nan


def main():
    parser = argparse.ArgumentParser(
        description="Full-band benchmark comparing two-patch and three-patch spectral amplitudes."
    )
    parser.add_argument("--a-min", type=float, default=0.001)
    parser.add_argument("--a-max", type=float, default=0.999)
    parser.add_argument("--n-a", type=int, default=24)

    parser.add_argument("--omega-min", type=float, default=1.0e-4)
    parser.add_argument("--omega-max", type=float, default=10.0)
    parser.add_argument("--n-omega", type=int, default=28)

    parser.add_argument("--ell", type=int, default=2)
    parser.add_argument("--m", type=int, default=2)
    parser.add_argument("--s", type=int, default=-2)

    parser.add_argument("--N-edge-list", type=str, default="24,32,48,64,80,96")
    parser.add_argument("--N-mid-factor", type=float, default=2.0)

    parser.add_argument("--z-m", type=float, default=0.3)
    parser.add_argument("--z1", type=float, default=0.1)
    parser.add_argument("--z2", type=float, default=0.9)

    parser.add_argument("--omega-mp-cut", type=float, default=1.0e-2)
    parser.add_argument("--mp-dps-loww", type=int, default=80)

    parser.add_argument("--out-prefix", type=str, default="compare_2patch_3patch")
    parser.add_argument("--skip-plots", action="store_true")
    args = parser.parse_args()

    a_values = np.linspace(args.a_min, args.a_max, args.n_a)
    omega_values = np.geomspace(args.omega_min, args.omega_max, args.n_omega)
    N_edge_list = parse_n_list(args.N_edge_list)
    N_ref = max(N_edge_list)

    out_dir = ROOT / "benchmark" / "outputs" / "compare_2patch_3patch"
    out_dir.mkdir(parents=True, exist_ok=True)

    detail_rows = []
    ref_map = {}
    cross_rows = []

    # ---------------------------------------------------------
    # scan all points, both methods, all N
    # ---------------------------------------------------------
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

            for N_edge in N_edge_list:
                N_mid = int(round(args.N_mid_factor * N_edge))

                # ---------------- 2-patch ----------------
                row2 = {
                    "method": "two_patch",
                    "a": float(a),
                    "omega": float(omega),
                    "N_edge": int(N_edge),
                    "N_mid": np.nan,
                    "ok": 0,
                    "error": "",
                    "use_mp_backend": 0,
                    "B_inc_re": np.nan,
                    "B_inc_im": np.nan,
                    "B_ref_re": np.nan,
                    "B_ref_im": np.nan,
                    "ratio_ref_over_inc_re": np.nan,
                    "ratio_ref_over_inc_im": np.nan,
                    "outer_abel_residual": np.nan,
                    "inner_abel_residual": np.nan,
                    "detS_residual": np.nan,
                    "solve_in_cond_raw": np.nan,
                    "solve_in_cond_scaled": np.nan,
                }

                try:
                    amp2 = TeukRadAmplitudeInWithAbelChecks(
                        mode,
                        N_in=N_edge,
                        N_out=N_edge,
                        z_m=args.z_m,
                        omega_mp_cut=args.omega_mp_cut,
                        mp_dps_loww=args.mp_dps_loww,
                    )
                    ratio2 = safe_ratio(amp2.B_ref, amp2.B_inc)

                    row2.update(
                        {
                            "ok": 1,
                            "use_mp_backend": int(bool(amp2.smatrix.get("use_mp_backend", False))),
                            "B_inc_re": amp2.B_inc.real,
                            "B_inc_im": amp2.B_inc.imag,
                            "B_ref_re": amp2.B_ref.real,
                            "B_ref_im": amp2.B_ref.imag,
                            "ratio_ref_over_inc_re": ratio2.real if ratio2 is not None else np.nan,
                            "ratio_ref_over_inc_im": ratio2.imag if ratio2 is not None else np.nan,
                            "outer_abel_residual": amp2.outer_abel_residual,
                            "inner_abel_residual": amp2.inner_abel_residual,
                            "detS_residual": amp2.detS_residual,
                            "solve_in_cond_raw": float(amp2.smatrix.get("solve_in_cond_raw", np.nan)),
                            "solve_in_cond_scaled": float(amp2.smatrix.get("solve_in_cond_scaled", np.nan)),
                        }
                    )
                except Exception as e:
                    row2["error"] = str(e)

                detail_rows.append(row2)

                # ---------------- 3-patch ----------------
                row3 = {
                    "method": "three_patch",
                    "a": float(a),
                    "omega": float(omega),
                    "N_edge": int(N_edge),
                    "N_mid": int(N_mid),
                    "ok": 0,
                    "error": "",
                    "use_mp_backend": 0,
                    "B_inc_re": np.nan,
                    "B_inc_im": np.nan,
                    "B_ref_re": np.nan,
                    "B_ref_im": np.nan,
                    "ratio_ref_over_inc_re": np.nan,
                    "ratio_ref_over_inc_im": np.nan,
                    "outer_abel_residual": np.nan,
                    "inner_abel_residual": np.nan,
                    "detS_residual": np.nan,
                    "cond_M_left": np.nan,
                    "cond_T_mid": np.nan,
                    "cond_M_outer_at_z2": np.nan,
                    "solve_in_cond_raw": np.nan,
                    "solve_in_cond_scaled": np.nan,
                    "state_in_z1_relerr": np.nan,
                }

                try:
                    amp3 = TeukRadAmplitudeIn3PatchWithAbelChecks(
                        mode,
                        N_left=N_edge,
                        N_mid=N_mid,
                        N_right=N_edge,
                        z1=args.z1,
                        z2=args.z2,
                        omega_mp_cut=args.omega_mp_cut,
                        mp_dps_loww=args.mp_dps_loww,
                    )
                    ratio3 = safe_ratio(amp3.B_ref, amp3.B_inc)

                    row3.update(
                        {
                            "ok": 1,
                            "use_mp_backend": int(bool(amp3.smatrix.get("use_mp_backend", False))),
                            "B_inc_re": amp3.B_inc.real,
                            "B_inc_im": amp3.B_inc.imag,
                            "B_ref_re": amp3.B_ref.real,
                            "B_ref_im": amp3.B_ref.imag,
                            "ratio_ref_over_inc_re": ratio3.real if ratio3 is not None else np.nan,
                            "ratio_ref_over_inc_im": ratio3.imag if ratio3 is not None else np.nan,
                            "outer_abel_residual": amp3.outer_abel_residual,
                            "inner_abel_residual": amp3.inner_abel_residual,
                            "detS_residual": amp3.detS_residual,
                            "cond_M_left": float(amp3.smatrix.get("cond_M_left", np.nan)),
                            "cond_T_mid": float(amp3.smatrix.get("cond_T_mid", np.nan)),
                            "cond_M_outer_at_z2": float(amp3.smatrix.get("cond_M_outer_at_z2", np.nan)),
                            "solve_in_cond_raw": float(amp3.smatrix.get("solve_in_cond_raw", np.nan)),
                            "solve_in_cond_scaled": float(amp3.smatrix.get("solve_in_cond_scaled", np.nan)),
                            "state_in_z1_relerr": float(amp3.smatrix.get("state_in_z1_relerr", np.nan)),
                        }
                    )
                except Exception as e:
                    row3["error"] = str(e)

                detail_rows.append(row3)

                # highest-order reference map for self-convergence
                if N_edge == N_ref and row2["ok"] == 1:
                    ref_map[("two_patch", float(a), float(omega))] = {
                        "B_inc": complex(row2["B_inc_re"], row2["B_inc_im"]),
                        "B_ref": complex(row2["B_ref_re"], row2["B_ref_im"]),
                        "ratio": complex(row2["ratio_ref_over_inc_re"], row2["ratio_ref_over_inc_im"])
                        if np.isfinite(row2["ratio_ref_over_inc_re"]) else None,
                    }

                if N_edge == N_ref and row3["ok"] == 1:
                    ref_map[("three_patch", float(a), float(omega))] = {
                        "B_inc": complex(row3["B_inc_re"], row3["B_inc_im"]),
                        "B_ref": complex(row3["B_ref_re"], row3["B_ref_im"]),
                        "ratio": complex(row3["ratio_ref_over_inc_re"], row3["ratio_ref_over_inc_im"])
                        if np.isfinite(row3["ratio_ref_over_inc_re"]) else None,
                    }

            # direct cross-method comparison at highest N_edge
            row2_ref = next(
                (r for r in detail_rows[::-1]
                 if r["method"] == "two_patch" and r["a"] == float(a) and r["omega"] == float(omega) and r["N_edge"] == N_ref),
                None
            )
            row3_ref = next(
                (r for r in detail_rows[::-1]
                 if r["method"] == "three_patch" and r["a"] == float(a) and r["omega"] == float(omega) and r["N_edge"] == N_ref),
                None
            )

            if row2_ref is not None and row3_ref is not None and row2_ref["ok"] == 1 and row3_ref["ok"] == 1:
                Binc2 = complex(row2_ref["B_inc_re"], row2_ref["B_inc_im"])
                Bref2 = complex(row2_ref["B_ref_re"], row2_ref["B_ref_im"])
                Binc3 = complex(row3_ref["B_inc_re"], row3_ref["B_inc_im"])
                Bref3 = complex(row3_ref["B_ref_re"], row3_ref["B_ref_im"])

                ratio2 = safe_ratio(Bref2, Binc2)
                ratio3 = safe_ratio(Bref3, Binc3)

                cross_rows.append(
                    {
                        "a": float(a),
                        "omega": float(omega),
                        "err_B_inc_3_vs_2": rel_complex_err(Binc3, Binc2),
                        "err_B_ref_3_vs_2": rel_complex_err(Bref3, Bref2),
                        "err_ratio_3_vs_2": rel_complex_err(ratio3, ratio2) if (ratio2 is not None and ratio3 is not None) else np.nan,
                        "outer_two": row2_ref["outer_abel_residual"],
                        "outer_three": row3_ref["outer_abel_residual"],
                        "detS_two": row2_ref["detS_residual"],
                        "detS_three": row3_ref["detS_residual"],
                        "cond_M_left_three": row3_ref["cond_M_left"],
                        "cond_M_outer_at_z2_three": row3_ref["cond_M_outer_at_z2"],
                        "use_mp_three": row3_ref["use_mp_backend"],
                    }
                )

    # ---------------------------------------------------------
    # self-convergence table
    # ---------------------------------------------------------
    conv_rows = []
    for row in detail_rows:
        if row["ok"] != 1 or row["N_edge"] == N_ref:
            continue

        key = (row["method"], row["a"], row["omega"])
        if key not in ref_map:
            continue

        ref = ref_map[key]
        Binc = complex(row["B_inc_re"], row["B_inc_im"])
        Bref = complex(row["B_ref_re"], row["B_ref_im"])
        ratio = (
            complex(row["ratio_ref_over_inc_re"], row["ratio_ref_over_inc_im"])
            if np.isfinite(row["ratio_ref_over_inc_re"]) else None
        )

        conv_rows.append(
            {
                "method": row["method"],
                "a": row["a"],
                "omega": row["omega"],
                "N_edge": row["N_edge"],
                "err_B_inc": rel_complex_err(Binc, ref["B_inc"]),
                "err_B_ref": rel_complex_err(Bref, ref["B_ref"]),
                "err_ratio": rel_complex_err(ratio, ref["ratio"]) if (ratio is not None and ref["ratio"] is not None) else np.nan,
            }
        )

    # ---------------------------------------------------------
    # save csv
    # ---------------------------------------------------------
    detail_csv = out_dir / f"{args.out_prefix}_detail.csv"
    conv_csv = out_dir / f"{args.out_prefix}_convergence.csv"
    cross_csv = out_dir / f"{args.out_prefix}_cross_method.csv"

    with open(detail_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sorted({k for r in detail_rows for k in r.keys()}))
        writer.writeheader()
        writer.writerows(detail_rows)

    with open(conv_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sorted({k for r in conv_rows for k in r.keys()}))
        writer.writeheader()
        writer.writerows(conv_rows)

    with open(cross_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sorted({k for r in cross_rows for k in r.keys()}))
        writer.writeheader()
        writer.writerows(cross_rows)

    print(f"[saved] {detail_csv}")
    print(f"[saved] {conv_csv}")
    print(f"[saved] {cross_csv}")

    if args.skip_plots:
        return

    # ---------------------------------------------------------
    # highest-order heatmaps
    # ---------------------------------------------------------
    def build_map(rows, value_key):
        arr = np.full((args.n_a, args.n_omega), np.nan)
        amap = {float(a): ia for ia, a in enumerate(a_values)}
        omap = {float(w): iw for iw, w in enumerate(omega_values)}
        for r in rows:
            ia = amap.get(float(r["a"]))
            iw = omap.get(float(r["omega"]))
            if ia is not None and iw is not None:
                arr[ia, iw] = r[value_key]
        return arr

    cross_map_ratio = build_map(cross_rows, "err_ratio_3_vs_2")
    cross_map_cond = build_map(cross_rows, "cond_M_left_three")
    cross_map_det3 = build_map(cross_rows, "detS_three")
    cross_map_outer3 = build_map(cross_rows, "outer_three")

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    extent = [np.log10(omega_values[0]), np.log10(omega_values[-1]), a_values[0], a_values[-1]]

    panels = [
        (np.log10(np.clip(cross_map_ratio, 1e-30, None)), r"$\log_{10}\,\mathrm{err}(B_{\rm ref}/B_{\rm inc})_{3{\rm p}\,vs\,2{\rm p}}$"),
        (np.log10(np.clip(cross_map_outer3, 1e-30, None)), r"$\log_{10}\,\epsilon_{\mathrm{outer}}^{(3{\rm p})}$"),
        (np.log10(np.clip(cross_map_det3, 1e-30, None)), r"$\log_{10}\,\epsilon_{\det S}^{(3{\rm p})}$"),
        (np.log10(np.clip(cross_map_cond, 1e-30, None)), r"$\log_{10}\,\kappa(M_{\rm left})^{(3{\rm p})}$"),
    ]

    for ax, (data, title) in zip(axes.ravel(), panels):
        im = ax.imshow(data, origin="lower", aspect="auto", extent=extent)
        ax.set_title(title + f"\n(highest N_edge={N_ref})")
        ax.set_xlabel(r"$\log_{10}\omega$")
        ax.set_ylabel(r"$a$")
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(r"$\log_{10}$ value")

    fig.tight_layout()
    fig_path = out_dir / f"{args.out_prefix}_heatmaps.png"
    fig.savefig(fig_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {fig_path}")


if __name__ == "__main__":
    main()