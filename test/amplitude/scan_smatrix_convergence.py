from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.amplitude import TeukRadAmplitudeIn
from utils.mode import KerrMode


def parse_orders(text: str) -> list[int]:
    vals = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        vals.append(int(item))
    vals = sorted(set(vals))
    if len(vals) < 2:
        raise ValueError("Need at least two spectral orders")
    return vals


def safe_rel_err(val: complex | None, ref: complex | None, eps: float = 1.0e-30) -> float:
    if val is None or ref is None:
        return np.nan
    denom = abs(ref)
    if not np.isfinite(denom) or denom <= eps:
        return np.nan
    num = abs(val - ref)
    if not np.isfinite(num):
        return np.nan
    return float(num / denom)


def solve_at_order(
    *,
    l: int,
    m: int,
    s: int,
    a: float,
    omega: float,
    lam: float | None,
    order: int,
    z_m: float,
):
    mode = KerrMode(M=1.0, a=a, omega=omega, ell=l, m=m, lam=lam, s=s)
    solver = TeukRadAmplitudeIn(mode=mode, N_in=order, N_out=order, z_m=z_m)
    result = solver.result
    smat = solver.smatrix["S"]
    return {
        "order": int(order),
        "lam": float(result.lam),
        "B_inc": complex(result.B_inc),
        "B_ref": complex(result.B_ref),
        "B_trans": complex(result.B_trans),
        "ratio_ref_over_inc": result.ratio_ref_over_inc,
        "ratio_inc_over_ref": result.ratio_inc_over_ref,
        "S": np.asarray(smat, dtype=np.complex128),
    }


def run_fixed_case(
    *,
    l: int,
    m: int,
    s: int,
    a: float,
    omega: float,
    orders: list[int],
    z_m: float,
):
    records = []
    lam_ref = None
    for order in orders:
        rec = solve_at_order(
            l=l,
            m=m,
            s=s,
            a=a,
            omega=omega,
            lam=lam_ref,
            order=order,
            z_m=z_m,
        )
        lam_ref = rec["lam"]
        records.append(rec)

    ref = records[-1]
    for rec in records:
        rec["err_ratio_ref_over_inc"] = safe_rel_err(rec["ratio_ref_over_inc"], ref["ratio_ref_over_inc"])
        rec["err_ratio_inc_over_ref"] = safe_rel_err(rec["ratio_inc_over_ref"], ref["ratio_inc_over_ref"])
        rec["err_B_inc"] = safe_rel_err(rec["B_inc"], ref["B_inc"])
        rec["err_B_ref"] = safe_rel_err(rec["B_ref"], ref["B_ref"])
        rec["err_S"] = float(
            np.linalg.norm(rec["S"] - ref["S"]) / max(np.linalg.norm(ref["S"]), 1.0e-30)
        )
    return records


def make_fixed_case_figure(records: list[dict], out_path: Path, *, a: float, omega: float):
    orders = np.array([rec["order"] for rec in records], dtype=int)
    err_ratio = np.array([rec["err_ratio_ref_over_inc"] for rec in records], dtype=float)
    err_binc = np.array([rec["err_B_inc"] for rec in records], dtype=float)
    err_bref = np.array([rec["err_B_ref"] for rec in records], dtype=float)
    err_s = np.array([rec["err_S"] for rec in records], dtype=float)
    ratio_abs = np.array(
        [
            np.nan if rec["ratio_ref_over_inc"] is None else abs(rec["ratio_ref_over_inc"])
            for rec in records
        ],
        dtype=float,
    )
    ratio_arg = np.array(
        [
            np.nan if rec["ratio_ref_over_inc"] is None else np.angle(rec["ratio_ref_over_inc"])
            for rec in records
        ],
        dtype=float,
    )

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    axes[0, 0].semilogy(orders, err_ratio, "o-", label="rel err ratio_ref/inc")
    axes[0, 0].semilogy(orders, err_binc, "o-", label="rel err B_inc")
    axes[0, 0].semilogy(orders, err_bref, "o-", label="rel err B_ref")
    axes[0, 0].semilogy(orders, err_s, "o-", label="rel err S")
    axes[0, 0].set_xlabel("spectral order N_in=N_out")
    axes[0, 0].set_ylabel("relative error vs highest order")
    axes[0, 0].grid(alpha=0.3)
    axes[0, 0].legend()

    axes[0, 1].plot(orders, ratio_abs, "o-")
    axes[0, 1].set_xlabel("spectral order N_in=N_out")
    axes[0, 1].set_ylabel(r"$|B_{ref}/B_{inc}|$")
    axes[0, 1].grid(alpha=0.3)

    axes[1, 0].plot(orders, ratio_arg, "o-")
    axes[1, 0].set_xlabel("spectral order N_in=N_out")
    axes[1, 0].set_ylabel(r"$\arg(B_{ref}/B_{inc})$")
    axes[1, 0].grid(alpha=0.3)

    ax = axes[1, 1]
    ax.axis("off")
    ref = records[-1]
    text = (
        f"fixed case\n"
        f"a={a:.6f}\n"
        f"omega={omega:.6f}\n"
        f"reference order={ref['order']}\n"
        f"lambda={ref['lam']:.12e}\n"
        f"|B_inc|={abs(ref['B_inc']):.6e}\n"
        f"|B_ref|={abs(ref['B_ref']):.6e}\n"
        f"|B_ref/B_inc|="
        f"{np.nan if ref['ratio_ref_over_inc'] is None else abs(ref['ratio_ref_over_inc']):.6e}"
    )
    ax.text(0.02, 0.98, text, va="top", ha="left", family="monospace")

    fig.suptitle("Two-domain spectral convergence at fixed (a, omega)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_scan_grid(
    *,
    n_a: int,
    n_omega: int,
    a_min: float,
    a_max: float,
    omega_min: float,
    omega_max: float,
):
    a_vals = np.linspace(a_min, a_max, n_a)
    omega_vals = np.geomspace(omega_min, omega_max, n_omega)
    return a_vals, omega_vals


def run_scan(
    *,
    l: int,
    m: int,
    s: int,
    orders: list[int],
    z_m: float,
    a_vals: np.ndarray,
    omega_vals: np.ndarray,
    tol: float,
):
    n_a = len(a_vals)
    n_w = len(omega_vals)
    final_err = np.full((n_a, n_w), np.nan, dtype=float)
    best_order = np.full((n_a, n_w), np.nan, dtype=float)
    lam_grid = np.full((n_a, n_w), np.nan, dtype=float)
    success = np.zeros((n_a, n_w), dtype=bool)
    failure_messages: list[dict] = []

    for ia, a in enumerate(a_vals):
        for iw, omega in enumerate(omega_vals):
            try:
                records = run_fixed_case(
                    l=l,
                    m=m,
                    s=s,
                    a=float(a),
                    omega=float(omega),
                    orders=orders,
                    z_m=z_m,
                )
                err = records[-2]["err_ratio_ref_over_inc"]
                final_err[ia, iw] = err
                lam_grid[ia, iw] = records[-1]["lam"]
                success[ia, iw] = True

                found = np.nan
                for rec in records[1:]:
                    if np.isfinite(rec["err_ratio_ref_over_inc"]) and rec["err_ratio_ref_over_inc"] < tol:
                        found = rec["order"]
                        break
                best_order[ia, iw] = found
            except Exception as e:
                failure_messages.append(
                    {
                        "a": float(a),
                        "omega": float(omega),
                        "err": str(e),
                    }
                )

    return {
        "a_vals": a_vals,
        "omega_vals": omega_vals,
        "final_err": final_err,
        "best_order": best_order,
        "lam_grid": lam_grid,
        "success": success,
        "failures": failure_messages,
        "tol": float(tol),
    }


def make_scan_figure(scan: dict, out_path: Path):
    a_vals = scan["a_vals"]
    omega_vals = scan["omega_vals"]
    extent = [math.log10(omega_vals.min()), math.log10(omega_vals.max()), a_vals.min(), a_vals.max()]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)

    im0 = axes[0].imshow(
        np.log10(scan["final_err"]),
        origin="lower",
        aspect="auto",
        extent=extent,
        interpolation="nearest",
    )
    axes[0].set_title(r"$\log_{10}$ final relative error")
    axes[0].set_xlabel(r"$\log_{10}\omega$")
    axes[0].set_ylabel("a")
    fig.colorbar(im0, ax=axes[0], shrink=0.9)

    im1 = axes[1].imshow(
        scan["best_order"],
        origin="lower",
        aspect="auto",
        extent=extent,
        interpolation="nearest",
    )
    axes[1].set_title("first order meeting tol")
    axes[1].set_xlabel(r"$\log_{10}\omega$")
    fig.colorbar(im1, ax=axes[1], shrink=0.9)

    im2 = axes[2].imshow(
        scan["success"].astype(float),
        origin="lower",
        aspect="auto",
        extent=extent,
        interpolation="nearest",
        vmin=0.0,
        vmax=1.0,
    )
    axes[2].set_title("success mask")
    axes[2].set_xlabel(r"$\log_{10}\omega$")
    fig.colorbar(im2, ax=axes[2], shrink=0.9)

    fig.suptitle("Two-domain spectral convergence scan on (a, omega)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def to_jsonable_records(records: list[dict]) -> list[dict]:
    out = []
    for rec in records:
        out.append(
            {
                "order": int(rec["order"]),
                "lam": float(rec["lam"]),
                "B_inc_re": float(np.real(rec["B_inc"])),
                "B_inc_im": float(np.imag(rec["B_inc"])),
                "B_ref_re": float(np.real(rec["B_ref"])),
                "B_ref_im": float(np.imag(rec["B_ref"])),
                "ratio_ref_over_inc_re": None if rec["ratio_ref_over_inc"] is None else float(np.real(rec["ratio_ref_over_inc"])),
                "ratio_ref_over_inc_im": None if rec["ratio_ref_over_inc"] is None else float(np.imag(rec["ratio_ref_over_inc"])),
                "err_ratio_ref_over_inc": float(rec["err_ratio_ref_over_inc"]) if np.isfinite(rec["err_ratio_ref_over_inc"]) else None,
                "err_B_inc": float(rec["err_B_inc"]) if np.isfinite(rec["err_B_inc"]) else None,
                "err_B_ref": float(rec["err_B_ref"]) if np.isfinite(rec["err_B_ref"]) else None,
                "err_S": float(rec["err_S"]) if np.isfinite(rec["err_S"]) else None,
            }
        )
    return out


def main():
    parser = argparse.ArgumentParser(description="Convergence test for two-domain spectral amplitudes")
    parser.add_argument("--l", type=int, default=2)
    parser.add_argument("--m", type=int, default=2)
    parser.add_argument("--s", type=int, default=-2)
    parser.add_argument("--orders", type=str, default="16,24,32,48,64,80,96")
    parser.add_argument("--z-m", type=float, default=0.3)
    parser.add_argument("--fixed-a", type=float, default=0.581667)
    parser.add_argument("--fixed-omega", type=float, default=1.0)
    parser.add_argument("--scan-a-min", type=float, default=0.001)
    parser.add_argument("--scan-a-max", type=float, default=0.999)
    parser.add_argument("--scan-omega-min", type=float, default=1.0e-4)
    parser.add_argument("--scan-omega-max", type=float, default=10.0)
    parser.add_argument("--scan-n-a", type=int, default=15)
    parser.add_argument("--scan-n-omega", type=int, default=21)
    parser.add_argument("--tol", type=float, default=1.0e-6)
    parser.add_argument("--out-dir", type=str, default="outputs/amplitude_smatrix_convergence")
    args = parser.parse_args()

    out_dir = (PROJECT_ROOT / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    orders = parse_orders(args.orders)

    fixed_records = run_fixed_case(
        l=args.l,
        m=args.m,
        s=args.s,
        a=args.fixed_a,
        omega=args.fixed_omega,
        orders=orders,
        z_m=args.z_m,
    )
    fixed_fig = out_dir / "fixed_case_convergence.png"
    make_fixed_case_figure(fixed_records, fixed_fig, a=args.fixed_a, omega=args.fixed_omega)

    a_vals, omega_vals = build_scan_grid(
        n_a=args.scan_n_a,
        n_omega=args.scan_n_omega,
        a_min=args.scan_a_min,
        a_max=args.scan_a_max,
        omega_min=args.scan_omega_min,
        omega_max=args.scan_omega_max,
    )
    scan = run_scan(
        l=args.l,
        m=args.m,
        s=args.s,
        orders=orders,
        z_m=args.z_m,
        a_vals=a_vals,
        omega_vals=omega_vals,
        tol=args.tol,
    )

    scan_fig = out_dir / "scan_convergence_maps.png"
    make_scan_figure(scan, scan_fig)

    np.savez_compressed(
        out_dir / "scan_data.npz",
        a_vals=scan["a_vals"],
        omega_vals=scan["omega_vals"],
        final_err=scan["final_err"],
        best_order=scan["best_order"],
        lam_grid=scan["lam_grid"],
        success=scan["success"],
    )

    with open(out_dir / "fixed_case_records.json", "w", encoding="utf-8") as f:
        json.dump(to_jsonable_records(fixed_records), f, ensure_ascii=False, indent=2)

    summary = {
        "l": args.l,
        "m": args.m,
        "s": args.s,
        "orders": orders,
        "z_m": args.z_m,
        "fixed_case": {
            "a": args.fixed_a,
            "omega": args.fixed_omega,
            "reference_order": orders[-1],
            "lambda": fixed_records[-1]["lam"],
        },
        "scan": {
            "a_min": args.scan_a_min,
            "a_max": args.scan_a_max,
            "omega_min": args.scan_omega_min,
            "omega_max": args.scan_omega_max,
            "n_a": args.scan_n_a,
            "n_omega": args.scan_n_omega,
            "tol": args.tol,
            "n_success": int(scan["success"].sum()),
            "n_total": int(scan["success"].size),
            "n_failed": int((~scan["success"]).sum()),
        },
        "outputs": {
            "fixed_fig": str(fixed_fig),
            "scan_fig": str(scan_fig),
            "scan_data": str(out_dir / "scan_data.npz"),
            "fixed_records": str(out_dir / "fixed_case_records.json"),
            "failures": str(out_dir / "scan_failures.json"),
        },
    }

    with open(out_dir / "scan_failures.json", "w", encoding="utf-8") as f:
        json.dump(scan["failures"], f, ensure_ascii=False, indent=2)

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"saved fixed figure -> {fixed_fig}")
    print(f"saved scan figure  -> {scan_fig}")
    print(f"saved scan data    -> {out_dir / 'scan_data.npz'}")
    print(f"saved summary      -> {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
