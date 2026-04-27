from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.mode import KerrMode
from utils.amplitude import (
    solve_basis_domain,
    compute_smatrix_with_abel,
    r_of_z,
    dz_dr,
    q_and_qp,
    A_down,
    A_up,
    A_in,
    A_out,
    basis_values_at_match,
    abel_constant_from_pair,
    outer_abel_theory,
    inner_abel_theory,
    rel_complex_residual,
)


def format_complex(z: complex) -> str:
    z = complex(z)
    return f"{z.real:+.16e}{z.imag:+.16e}j"


def safe_ratio(numer: complex, denom: complex, floor: float = 1.0e-30):
    if abs(denom) <= floor:
        return None
    return numer / denom


def cheb_lobatto_points(n: int, a: float, b: float) -> np.ndarray:
    if n < 2:
        return np.array([a, b], dtype=float)
    k = np.arange(n, dtype=float)
    x = np.cos(np.pi * k / (n - 1))
    z = 0.5 * (a + b) + 0.5 * (b - a) * x
    return np.sort(z.astype(float))


def uniform_points(n: int, a: float, b: float) -> np.ndarray:
    if n < 2:
        return np.array([a, b], dtype=float)
    return np.linspace(a, b, n, dtype=float)


def prefactor_for_basis(r, mode: KerrMode, basis: str):
    if basis == "down":
        return A_down(r, mode)
    if basis == "up":
        return A_up(r, mode)
    if basis == "in":
        return A_in(r, mode)
    if basis == "out":
        return A_out(r, mode)
    raise ValueError(f"Unknown basis={basis}")


def _complex_interp_linear(x_query, x_nodes, y_nodes):
    """
    复数线性插值。这里要求 x_query 落在 [x_nodes[0], x_nodes[-1]] 内。
    """
    xq = np.asarray(x_query, dtype=np.float64)
    yn = np.asarray(y_nodes, dtype=np.complex128)
    xr = np.asarray(x_nodes, dtype=np.float64)

    re = np.interp(xq, xr, yn.real)
    im = np.interp(xq, xr, yn.imag)
    out = re + 1j * im
    if np.ndim(x_query) == 0:
        return complex(out)
    return out


def _strict_interior_arrays(mode: KerrMode, basis: str, sol: dict):
    """
    只用严格内部节点 z[1:-1] 构造 R, R_r。
    不触碰真端点 z=0 或 z=1。
    """
    z_all = np.asarray(sol["z"], dtype=np.float64)
    u_all = np.asarray(sol["u"], dtype=np.complex128)
    uz_all = np.asarray(sol["uz"], dtype=np.complex128)

    if len(z_all) <= 2:
        raise ValueError("Need at least one strict interior node; increase spectral order N.")

    z_mid = z_all[1:-1]
    u_mid = u_all[1:-1]
    uz_mid = uz_all[1:-1]

    r_mid = r_of_z(z_mid, mode)
    zr_mid = dz_dr(z_mid, mode)
    q_mid, _ = q_and_qp(r_mid, mode, basis)
    F_mid = prefactor_for_basis(r_mid, mode, basis)

    R_mid = F_mid * u_mid
    Rr_mid = F_mid * (zr_mid * uz_mid + q_mid * u_mid)

    return (
        np.asarray(z_mid, dtype=np.float64),
        np.asarray(R_mid, dtype=np.complex128),
        np.asarray(Rr_mid, dtype=np.complex128),
    )


def _finite_endpoint_value(mode: KerrMode, basis: str, sol: dict, zq: float) -> tuple[complex, complex]:
    """
    在一个有限点 zq in (0,1) 计算 (R, R_r)。

    规则：
    1. 若 zq 恰好是当前 patch 的有限边界，则用 basis_values_at_match；
    2. 否则只用严格内部节点做线性插值；
    3. 严禁对 z=0 或 z=1 调用 basis_values_at_match。
    """
    z_all = np.asarray(sol["z"], dtype=np.float64)
    z_patch_left = float(z_all[0])
    z_patch_right = float(z_all[-1])
    tol = 1.0e-14

    # 只允许在“有限边界”上人工算端点
    if abs(zq - z_patch_left) < tol and z_patch_left > 0.0:
        return basis_values_at_match(mode, basis, sol, "left")

    if abs(zq - z_patch_right) < tol and z_patch_right < 1.0:
        return basis_values_at_match(mode, basis, sol, "right")

    # 否则只用严格内部节点
    z_mid, R_mid, Rr_mid = _strict_interior_arrays(mode, basis, sol)

    if not (z_mid[0] <= zq <= z_mid[-1]):
        raise ValueError(
            f"Requested zq={zq} is outside strict-interior interpolation range "
            f"[{z_mid[0]}, {z_mid[-1]}]. Increase N or choose a narrower overlap."
        )

    Rq = _complex_interp_linear(zq, z_mid, R_mid)
    Rrq = _complex_interp_linear(zq, z_mid, Rr_mid)
    return complex(Rq), complex(Rrq)


def build_overlap_profile(mode: KerrMode, basis: str, sol: dict, z_left: float, z_right: float) -> dict:
    """
    只在 overlap 区间 [z_left, z_right] 上构造 profile。
    真端点 z=0,1 完全排除。

    端点处理：
    - 若 overlap 端点恰好是 patch 的有限边界，则用 basis_values_at_match 人工算
    - 否则用严格内部节点做线性插值
    """
    z_mid, R_mid, Rr_mid = _strict_interior_arrays(mode, basis, sol)

    R_left, Rr_left = _finite_endpoint_value(mode, basis, sol, z_left)
    R_right, Rr_right = _finite_endpoint_value(mode, basis, sol, z_right)

    mask = (z_mid > z_left) & (z_mid < z_right)

    z_full = np.concatenate(([z_left], z_mid[mask], [z_right]))
    R_full = np.concatenate(([R_left], R_mid[mask], [R_right]))
    Rr_full = np.concatenate(([Rr_left], Rr_mid[mask], [Rr_right]))

    return {
        "basis": basis,
        "z": np.asarray(z_full, dtype=np.float64),
        "R": np.asarray(R_full, dtype=np.complex128),
        "R_r": np.asarray(Rr_full, dtype=np.complex128),
    }


def eval_profile_overlap(profile: dict, zq) -> tuple[np.ndarray, np.ndarray]:
    z_nodes = np.asarray(profile["z"], dtype=np.float64)
    R_nodes = np.asarray(profile["R"], dtype=np.complex128)
    Rr_nodes = np.asarray(profile["R_r"], dtype=np.complex128)

    zq_arr = np.atleast_1d(np.asarray(zq, dtype=np.float64))
    if np.any(zq_arr < z_nodes[0] - 1.0e-14) or np.any(zq_arr > z_nodes[-1] + 1.0e-14):
        raise ValueError(
            f"Query points must lie inside overlap profile range [{z_nodes[0]}, {z_nodes[-1]}]."
        )

    Rq = _complex_interp_linear(zq, z_nodes, R_nodes)
    Rrq = _complex_interp_linear(zq, z_nodes, Rr_nodes)

    return np.asarray(Rq, dtype=np.complex128), np.asarray(Rrq, dtype=np.complex128)


def robust_scale(*arrays, floor: float = 1.0e-30) -> float:
    vals = []
    for arr in arrays:
        arr = np.asarray(arr)
        vals.append(np.abs(arr).ravel())
    allv = np.concatenate(vals)
    allv = allv[np.isfinite(allv)]
    if allv.size == 0:
        return 1.0
    med = float(np.median(allv))
    return max(med, floor)


def fit_column_from_overlap(
    R1: np.ndarray,
    R2: np.ndarray,
    Rt: np.ndarray,
    R1r: np.ndarray,
    R2r: np.ndarray,
    Rtr: np.ndarray,
    weight_value: float = 1.0,
    weight_derivative: float = 1.0,
) -> tuple[np.ndarray, dict]:
    scale_val = robust_scale(R1, R2, Rt)
    scale_der = robust_scale(R1r, R2r, Rtr)

    A_val = np.column_stack([R1 / scale_val, R2 / scale_val])
    y_val = Rt / scale_val

    A_der = np.column_stack([R1r / scale_der, R2r / scale_der])
    y_der = Rtr / scale_der

    A = np.vstack([weight_value * A_val, weight_derivative * A_der])
    y = np.concatenate([weight_value * y_val, weight_derivative * y_der])

    coef, residuals, rank, svals = np.linalg.lstsq(A, y, rcond=None)

    relres_val = np.linalg.norm((np.column_stack([R1, R2]) @ coef) - Rt) / max(np.linalg.norm(Rt), 1.0e-30)
    relres_der = np.linalg.norm((np.column_stack([R1r, R2r]) @ coef) - Rtr) / max(np.linalg.norm(Rtr), 1.0e-30)
    relres_all = np.linalg.norm(A @ coef - y) / max(np.linalg.norm(y), 1.0e-30)

    diag = {
        "rank": int(rank),
        "smax": float(np.abs(svals[0])) if len(svals) else np.nan,
        "smin": float(np.abs(svals[-1])) if len(svals) else np.nan,
        "cond": float(np.linalg.cond(A)),
        "relres_value": float(relres_val),
        "relres_derivative": float(relres_der),
        "relres_all": float(relres_all),
        "scale_value": float(scale_val),
        "scale_derivative": float(scale_der),
        "residuals_raw": residuals,
    }
    return np.asarray(coef, dtype=np.complex128), diag


def pointwise_single_fit(
    R1: np.ndarray,
    R2: np.ndarray,
    Rt: np.ndarray,
    R1r: np.ndarray,
    R2r: np.ndarray,
    Rtr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(R1)
    coefs = np.full((n, 2), np.nan + 1j * np.nan, dtype=np.complex128)
    conds = np.full(n, np.nan, dtype=float)

    for i in range(n):
        M = np.array([[R1[i], R2[i]], [R1r[i], R2r[i]]], dtype=np.complex128)
        y = np.array([Rt[i], Rtr[i]], dtype=np.complex128)
        try:
            conds[i] = np.linalg.cond(M)
            coefs[i, :] = np.linalg.solve(M, y)
        except np.linalg.LinAlgError:
            pass

    return coefs, conds


def abel_residuals_over_points(
    mode: KerrMode,
    z_pts: np.ndarray,
    prof_a: dict,
    prof_b: dict,
    theory: complex,
) -> np.ndarray:
    Ra, Rra = eval_profile_overlap(prof_a, z_pts)
    Rb, Rrb = eval_profile_overlap(prof_b, z_pts)

    res = np.empty_like(z_pts, dtype=float)
    for i, z in enumerate(z_pts):
        r = float(r_of_z(z, mode))
        cnum = abel_constant_from_pair(Ra[i], Rra[i], Rb[i], Rrb[i], r, mode)
        res[i] = rel_complex_residual(cnum, theory)
    return res


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Fit two-sided spectral S-matrix coefficients on an overlap region. "
            "This final version never touches the true endpoints z=0 or z=1."
        )
    )
    parser.add_argument("--M", type=float, default=1.0)
    parser.add_argument("--a", type=float, default=0.1)
    parser.add_argument("--omega", type=float, default=0.1)
    parser.add_argument("--ell", type=int, default=2)
    parser.add_argument("--m", type=int, default=2)
    parser.add_argument("--s", type=int, default=-2)
    parser.add_argument("--lam", type=float, default=None)

    parser.add_argument("--N-out", type=int, default=80)
    parser.add_argument("--N-in", type=int, default=80)

    parser.add_argument("--z-left", type=float, default=0.24)
    parser.add_argument("--z-right", type=float, default=0.36)
    parser.add_argument("--n-fit-points", type=int, default=21)
    parser.add_argument("--sample-kind", choices=["cheb_lobatto", "uniform"], default="cheb_lobatto")

    parser.add_argument("--weight-value", type=float, default=1.0)
    parser.add_argument("--weight-derivative", type=float, default=1.0)

    parser.add_argument("--save-csv", action="store_true")
    parser.add_argument("--out-prefix", type=str, default="overlap_fit")

    args = parser.parse_args()

    if not (0.0 < args.z_left < args.z_right < 1.0):
        raise ValueError("Require 0 < z_left < z_right < 1.")

    mode = KerrMode(
        M=args.M,
        a=args.a,
        omega=args.omega,
        ell=args.ell,
        m=args.m,
        lam=args.lam,
        s=args.s,
    )

    z_center = 0.5 * (args.z_left + args.z_right)

    if args.sample_kind == "cheb_lobatto":
        z_fit = cheb_lobatto_points(args.n_fit_points, args.z_left, args.z_right)
    else:
        z_fit = uniform_points(args.n_fit_points, args.z_left, args.z_right)

    # ---------------------------------------------------------
    # 两边谱解
    # outer basis on [0, z_right]
    # inner basis on [z_left, 1]
    # ---------------------------------------------------------
    sol_down = solve_basis_domain(mode, "down", args.N_out, 0.0, args.z_right, "left")
    sol_up = solve_basis_domain(mode, "up", args.N_out, 0.0, args.z_right, "left")
    sol_in = solve_basis_domain(mode, "in", args.N_in, args.z_left, 1.0, "right")
    sol_out = solve_basis_domain(mode, "out", args.N_in, args.z_left, 1.0, "right")

    # ---------------------------------------------------------
    # 只在 overlap 上构造 profile，不触碰 z=0,1
    # ---------------------------------------------------------
    prof_down = build_overlap_profile(mode, "down", sol_down, args.z_left, args.z_right)
    prof_up = build_overlap_profile(mode, "up", sol_up, args.z_left, args.z_right)
    prof_in = build_overlap_profile(mode, "in", sol_in, args.z_left, args.z_right)
    prof_out = build_overlap_profile(mode, "out", sol_out, args.z_left, args.z_right)

    R_down, Rr_down = eval_profile_overlap(prof_down, z_fit)
    R_up, Rr_up = eval_profile_overlap(prof_up, z_fit)
    R_in, Rr_in = eval_profile_overlap(prof_in, z_fit)
    R_out, Rr_out = eval_profile_overlap(prof_out, z_fit)

    # ---------------------------------------------------------
    # overlap 上拟合两列
    # ---------------------------------------------------------
    coef_in, diag_in = fit_column_from_overlap(
        R_down, R_up, R_in, Rr_down, Rr_up, Rr_in,
        weight_value=args.weight_value,
        weight_derivative=args.weight_derivative,
    )
    coef_out, diag_out = fit_column_from_overlap(
        R_down, R_up, R_out, Rr_down, Rr_up, Rr_out,
        weight_value=args.weight_value,
        weight_derivative=args.weight_derivative,
    )

    Cin_down, Cin_up = coef_in
    Cout_down, Cout_up = coef_out

    S_fit = np.array(
        [[Cin_down, Cin_up], [Cout_down, Cout_up]],
        dtype=np.complex128,
    )

    B_inc = complex(Cin_down)
    B_ref = complex(Cin_up)
    B_trans = 1.0 + 0.0j
    ratio_ref_over_inc = safe_ratio(B_ref, B_inc)

    # ---------------------------------------------------------
    # overlap 中点单点解
    # ---------------------------------------------------------
    Rc_down, Rrc_down = eval_profile_overlap(prof_down, z_center)
    Rc_up, Rrc_up = eval_profile_overlap(prof_up, z_center)
    Rc_in, Rrc_in = eval_profile_overlap(prof_in, z_center)
    Rc_out, Rrc_out = eval_profile_overlap(prof_out, z_center)

    M_center = np.array(
        [[Rc_down, Rc_up], [Rrc_down, Rrc_up]],
        dtype=np.complex128,
    )
    y_center_in = np.array([Rc_in, Rrc_in], dtype=np.complex128)
    y_center_out = np.array([Rc_out, Rrc_out], dtype=np.complex128)

    coef_in_center = np.linalg.solve(M_center, y_center_in)
    coef_out_center = np.linalg.solve(M_center, y_center_out)

    S_center = np.array(
        [[coef_in_center[0], coef_in_center[1]], [coef_out_center[0], coef_out_center[1]]],
        dtype=np.complex128,
    )

    # ---------------------------------------------------------
    # 当前仓库原始单点方法（有限 z_center，安全）
    # ---------------------------------------------------------
    sm_orig = compute_smatrix_with_abel(mode, N_in=args.N_in, N_out=args.N_out, z_m=z_center)
    S_orig = sm_orig["S"]

    # ---------------------------------------------------------
    # overlap 上每个点各自单点解，观察漂移
    # ---------------------------------------------------------
    coef_in_pts, conds_in_pts = pointwise_single_fit(R_down, R_up, R_in, Rr_down, Rr_up, Rr_in)
    coef_out_pts, conds_out_pts = pointwise_single_fit(R_down, R_up, R_out, Rr_down, Rr_up, Rr_out)

    # ---------------------------------------------------------
    # Abel 诊断
    # ---------------------------------------------------------
    outer_res = abel_residuals_over_points(mode, z_fit, prof_down, prof_up, outer_abel_theory(mode))
    inner_res = abel_residuals_over_points(mode, z_fit, prof_in, prof_out, inner_abel_theory(mode))

    detS_fit = np.linalg.det(S_fit)
    detS_th = inner_abel_theory(mode) / outer_abel_theory(mode)
    detS_fit_res = rel_complex_residual(detS_fit, detS_th)

    detS_center = np.linalg.det(S_center)
    detS_center_res = rel_complex_residual(detS_center, detS_th)

    detS_orig = sm_orig["detS_num"]
    detS_orig_res = sm_orig["detS_residual"]

    # ---------------------------------------------------------
    # 打印
    # ---------------------------------------------------------
    print("=== Overlap-fit two-sided matching (final safe version) ===")
    print(f"mode = (M={args.M}, a={args.a}, s={args.s}, ell={args.ell}, m={args.m}, omega={args.omega}, lambda={mode.lambda_value})")
    print(f"outer domain = [0, {args.z_right}]")
    print(f"inner domain = [{args.z_left}, 1]")
    print(f"overlap      = [{args.z_left}, {args.z_right}]")
    print(f"z_center     = {z_center}")
    print(f"n_fit_points = {args.n_fit_points} ({args.sample_kind})")
    print(f"weight_value = {args.weight_value}")
    print(f"weight_derivative = {args.weight_derivative}")

    print("\n=== Fitted S-matrix (overlap least squares) ===")
    print(f"Cin_down  = {format_complex(Cin_down)}")
    print(f"Cin_up    = {format_complex(Cin_up)}")
    print(f"Cout_down = {format_complex(Cout_down)}")
    print(f"Cout_up   = {format_complex(Cout_up)}")

    print("\n=== Physical amplitudes from fitted first column ===")
    print(f"B_inc   = {format_complex(B_inc)}")
    print(f"B_ref   = {format_complex(B_ref)}")
    print(f"B_trans = {format_complex(B_trans)}")
    if ratio_ref_over_inc is None:
        print("B_ref/B_inc = None")
    else:
        print(f"B_ref/B_inc = {format_complex(ratio_ref_over_inc)}")

    print("\n=== Fit diagnostics ===")
    print(f"[in ] cond(A)           = {diag_in['cond']:.6e}")
    print(f"[in ] relres(value)     = {diag_in['relres_value']:.6e}")
    print(f"[in ] relres(deriv)     = {diag_in['relres_derivative']:.6e}")
    print(f"[in ] relres(all)       = {diag_in['relres_all']:.6e}")
    print(f"[out] cond(A)           = {diag_out['cond']:.6e}")
    print(f"[out] relres(value)     = {diag_out['relres_value']:.6e}")
    print(f"[out] relres(deriv)     = {diag_out['relres_derivative']:.6e}")
    print(f"[out] relres(all)       = {diag_out['relres_all']:.6e}")

    print("\n=== Abel diagnostics over overlap points ===")
    print(f"outer_abel_residual median = {np.median(outer_res):.6e}")
    print(f"outer_abel_residual max    = {np.max(outer_res):.6e}")
    print(f"inner_abel_residual median = {np.median(inner_res):.6e}")
    print(f"inner_abel_residual max    = {np.max(inner_res):.6e}")

    print("\n=== det(S) diagnostics ===")
    print(f"detS_theory                = {format_complex(detS_th)}")
    print(f"detS_fit                   = {format_complex(detS_fit)}")
    print(f"detS_fit_residual          = {detS_fit_res:.6e}")
    print(f"detS_center(single-point)  = {format_complex(detS_center)}")
    print(f"detS_center_residual       = {detS_center_res:.6e}")
    print(f"detS_orig(current method)  = {format_complex(detS_orig)}")
    print(f"detS_orig_residual         = {detS_orig_res:.6e}")

    print("\n=== Comparison: overlap-fit vs single-point at overlap center ===")
    print(f"|Cin_down_fit - Cin_down_center| / max(1,|center|) = "
          f"{abs(Cin_down - coef_in_center[0]) / max(1.0, abs(coef_in_center[0])):.6e}")
    print(f"|Cin_up_fit   - Cin_up_center|   / max(1,|center|) = "
          f"{abs(Cin_up - coef_in_center[1]) / max(1.0, abs(coef_in_center[1])):.6e}")
    print(f"|Cout_down_fit- Cout_down_center|/ max(1,|center|) = "
          f"{abs(Cout_down - coef_out_center[0]) / max(1.0, abs(coef_out_center[0])):.6e}")
    print(f"|Cout_up_fit  - Cout_up_center|  / max(1,|center|) = "
          f"{abs(Cout_up - coef_out_center[1]) / max(1.0, abs(coef_out_center[1])):.6e}")

    print("\n=== Comparison: overlap-fit vs repo current single-point method ===")
    print(f"|Cin_down_fit - Cin_down_orig| / max(1,|orig|) = "
          f"{abs(Cin_down - sm_orig['Cin_down']) / max(1.0, abs(sm_orig['Cin_down'])):.6e}")
    print(f"|Cin_up_fit   - Cin_up_orig|   / max(1,|orig|) = "
          f"{abs(Cin_up - sm_orig['Cin_up']) / max(1.0, abs(sm_orig['Cin_up'])):.6e}")
    print(f"|Cout_down_fit- Cout_down_orig|/ max(1,|orig|) = "
          f"{abs(Cout_down - sm_orig['Cout_down']) / max(1.0, abs(sm_orig['Cout_down'])):.6e}")
    print(f"|Cout_up_fit  - Cout_up_orig|  / max(1,|orig|) = "
          f"{abs(Cout_up - sm_orig['Cout_up']) / max(1.0, abs(sm_orig['Cout_up'])):.6e}")

    print("\n=== Pointwise single-point solve spread across overlap ===")
    valid_in = np.isfinite(conds_in_pts)
    valid_out = np.isfinite(conds_out_pts)
    if np.any(valid_in):
        print(f"[in ] pointwise cond median = {np.nanmedian(conds_in_pts):.6e}")
        print(f"[in ] pointwise cond max    = {np.nanmax(conds_in_pts):.6e}")
        diff_in = np.abs(coef_in_pts[valid_in] - coef_in[None, :])
        print(f"[in ] median |point-fit - overlap-fit| for Cin_down = {np.nanmedian(diff_in[:,0]):.6e}")
        print(f"[in ] median |point-fit - overlap-fit| for Cin_up   = {np.nanmedian(diff_in[:,1]):.6e}")
    if np.any(valid_out):
        print(f"[out] pointwise cond median = {np.nanmedian(conds_out_pts):.6e}")
        print(f"[out] pointwise cond max    = {np.nanmax(conds_out_pts):.6e}")
        diff_out = np.abs(coef_out_pts[valid_out] - coef_out[None, :])
        print(f"[out] median |point-fit - overlap-fit| for Cout_down = {np.nanmedian(diff_out[:,0]):.6e}")
        print(f"[out] median |point-fit - overlap-fit| for Cout_up   = {np.nanmedian(diff_out[:,1]):.6e}")

    if args.save_csv:
        out_dir = ROOT / "benchmark" / "outputs" / "overlap_fit"
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / f"{args.out_prefix}.csv"

        rows = []
        for i, z in enumerate(z_fit):
            row = {
                "z": float(z),
                "r": float(r_of_z(z, mode)),

                "R_down_re": R_down[i].real,
                "R_down_im": R_down[i].imag,
                "Rr_down_re": Rr_down[i].real,
                "Rr_down_im": Rr_down[i].imag,

                "R_up_re": R_up[i].real,
                "R_up_im": R_up[i].imag,
                "Rr_up_re": Rr_up[i].real,
                "Rr_up_im": Rr_up[i].imag,

                "R_in_re": R_in[i].real,
                "R_in_im": R_in[i].imag,
                "Rr_in_re": Rr_in[i].real,
                "Rr_in_im": Rr_in[i].imag,

                "R_out_re": R_out[i].real,
                "R_out_im": R_out[i].imag,
                "Rr_out_re": Rr_out[i].real,
                "Rr_out_im": Rr_out[i].imag,

                "outer_abel_residual": float(outer_res[i]),
                "inner_abel_residual": float(inner_res[i]),

                "coef_in_point_down_re": coef_in_pts[i, 0].real if np.isfinite(conds_in_pts[i]) else np.nan,
                "coef_in_point_down_im": coef_in_pts[i, 0].imag if np.isfinite(conds_in_pts[i]) else np.nan,
                "coef_in_point_up_re": coef_in_pts[i, 1].real if np.isfinite(conds_in_pts[i]) else np.nan,
                "coef_in_point_up_im": coef_in_pts[i, 1].imag if np.isfinite(conds_in_pts[i]) else np.nan,
                "coef_in_point_cond": float(conds_in_pts[i]) if np.isfinite(conds_in_pts[i]) else np.nan,

                "coef_out_point_down_re": coef_out_pts[i, 0].real if np.isfinite(conds_out_pts[i]) else np.nan,
                "coef_out_point_down_im": coef_out_pts[i, 0].imag if np.isfinite(conds_out_pts[i]) else np.nan,
                "coef_out_point_up_re": coef_out_pts[i, 1].real if np.isfinite(conds_out_pts[i]) else np.nan,
                "coef_out_point_up_im": coef_out_pts[i, 1].imag if np.isfinite(conds_out_pts[i]) else np.nan,
                "coef_out_point_cond": float(conds_out_pts[i]) if np.isfinite(conds_out_pts[i]) else np.nan,
            }
            rows.append(row)

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        print(f"\n[saved] {csv_path}")


if __name__ == "__main__":
    main()