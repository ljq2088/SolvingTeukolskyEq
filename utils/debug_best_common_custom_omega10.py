#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
debug_best_common_custom_omega10.py

用途
----
这是一个“最优情况 / 当前最好方案”的独立调试脚本，用于复现目前在线测试中
omega=10 时表现最好的 common-custom middle-patch 方案：

    R(r) = F_mid(r) U(r),
    F_mid(r) = r^p exp(+ i omega r_*),
    p = 3.4.

核心思想
--------
原始三段谱方法的 middle patch 传播 full Teukolsky R。高频时，B_ref 是
B_inc 的 ~1e-9 小分量，直接传播 full state 容易把反射通道压没。

本脚本将 middle patch 改成统一的 custom 变量 U，并且在同一个 U-state
空间里完成：

    1. right in-mode state at z2 -> U-space;
    2. 通过 custom middle transport 反传到 z1;
    3. left down/up states at z1 -> U-space;
    4. 解 2x2 得到物理外区振幅 B_inc, B_ref。

注意
----
这不是最终根治方案，而是目前在线测试中 omega=10 最好的可调试版本。
典型结果应接近：

    B_inc ~ 31.20436 + 3.94034 i
    B_ref ~ 3.65e-8 - 7.08e-9 i
    |B_ref/B_inc| ~ 1.18e-9

GSN 参考：
    B_inc = 31.2044 + 3.94032 i
    B_ref = 3.399e-8 - 7.713e-9 i
    |B_ref/B_inc| = 1.108e-9

运行方式
--------
在仓库根目录运行：

    python benchmark/scripts/debug_best_common_custom_omega10.py

或者手动传参：

    python benchmark/scripts/debug_best_common_custom_omega10.py \
        --omega 10 --lam -2.45630962430791 --N-mid 28 --z1 0.625 --z2 0.705 --p 3.4
"""

from __future__ import annotations
import sys
sys.path.append("/home/ljq/code/PINN/SolvingTeukolsky")
import argparse
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import mpmath as mp
import matplotlib.pyplot as plt
import os
# ----------------------------------------------------------------------
# 0. 路径设置：确保脚本可以从 benchmark/scripts 下导入仓库 utils
# ----------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ----------------------------------------------------------------------
# 1. 复用仓库现有基础函数
# ----------------------------------------------------------------------
from utils.mode import KerrMode
from utils.amplitude import (
    Delta,
    Delta_p,
    K_of_r,
    r_of_z,
    dz_dr,
    d2z_dr2,
    A_down,
    A_up,
    A_in,
    A_out,
    q_and_qp as q_and_qp_two_patch,
    cheb_D,
    solve_basis_domain,
    basis_values_at_match,
    abel_constant_from_pair,
    outer_abel_theory,
    inner_abel_theory,
)

# ----------------------------------------------------------------------
# 2. 数值安全工具
# ----------------------------------------------------------------------
def safe_ratio(num: complex, den: complex, floor: float = 1.0e-300) -> complex:
    """安全复数比值，避免分母为 0。"""
    if abs(den) < floor:
        return complex(np.nan, np.nan)
    return num / den


def solve_scaled_lstsq(A: np.ndarray, b: np.ndarray, floor: float = 1.0e-300):
    """
    对线性系统 A x = b 做列归一化后最小二乘求解。

    这里即使用于 2x2，也使用 lstsq 而不是 solve，方便输出 relres 和条件数。
    这个步骤不是为了“修复”主要误差，而是为了避免纯代数列尺度差异污染调试。
    """
    A = np.asarray(A, dtype=complex)
    b = np.asarray(b, dtype=complex)

    col_norm = np.linalg.norm(A, axis=0)
    col_norm = np.where(col_norm > floor, col_norm, 1.0)

    A_scaled = A / col_norm[None, :]
    x_scaled, *_ = np.linalg.lstsq(A_scaled, b, rcond=None)
    x = x_scaled / col_norm

    relres = np.linalg.norm(A @ x - b) / max(np.linalg.norm(b), floor)

    diag = {
        "relres": float(relres),
        "cond_raw": float(np.linalg.cond(A)),
        "cond_scaled": float(np.linalg.cond(A_scaled)),
        "col_norm_0": float(col_norm[0]),
        "col_norm_1": float(col_norm[1]),
    }
    return x, diag

def solve_rowcol_scaled_square(A: np.ndarray, b: np.ndarray, floor: float = 1.0e-300):
    """
    Solve A x = b with row and column scaling.

    This is used for the middle Chebyshev collocation matrix.
    It does not change the exact discrete solution, but reduces numerical
    damage from badly scaled ODE rows.
    """
    A = np.asarray(A, dtype=complex)
    b = np.asarray(b, dtype=complex)

    # Row scaling
    row_norm = np.linalg.norm(A, axis=1)
    row_norm = np.where(row_norm > floor, row_norm, 1.0)
    A1 = A / row_norm[:, None]
    b1 = b / row_norm

    # Column scaling
    col_norm = np.linalg.norm(A1, axis=0)
    col_norm = np.where(col_norm > floor, col_norm, 1.0)
    A2 = A1 / col_norm[None, :]

    y = np.linalg.solve(A2, b1)
    x = y / col_norm

    relres = np.linalg.norm(A @ x - b) / max(np.linalg.norm(b), floor)

    diag = {
        "relres": float(relres),
        "cond_raw": float(np.linalg.cond(A)),
        "cond_rowcol": float(np.linalg.cond(A2)),
        "row_norm_min": float(np.min(row_norm)),
        "row_norm_max": float(np.max(row_norm)),
        "col_norm_min": float(np.min(col_norm)),
        "col_norm_max": float(np.max(col_norm)),
    }
    return x, diag

# ----------------------------------------------------------------------
# 3. middle physical-up prefactor: F_mid = A_up = r^3 exp(+i omega r_*)
# ----------------------------------------------------------------------
def middle_factor(r: np.ndarray | float, mode: KerrMode):
    """
    Middle prefactor fixed to the physical outgoing infinity factor:

        F_mid(r) = A_up(r) = r^3 exp(+i omega r_*).

    This removes the phenomenological exponent p.
    """
    return A_up(r, mode)


def middle_q_and_qp(r: np.ndarray | float, mode: KerrMode):
    """
    q = F'/F and q' for F = A_up.

    Since A_up is already implemented in utils.amplitude, use the existing
    q_and_qp for basis='up'. This guarantees phase convention consistency.
    """
    return q_and_qp_two_patch(r, mode, "up")
# ----------------------------------------------------------------------
# 4. custom 中段 ODE 系数
# ----------------------------------------------------------------------
def coeffs_middle_up(z: np.ndarray, mode: KerrMode):
    """
    Coefficients for the middle equation after

        R = A_up U.

    Original radial equation:

        Delta R_rr - Delta' R_r + V R = 0.

    In compact coordinate z, the U equation is:

        B2 U_zz + B1 U_z + B0 U = 0.
    """
    z = np.asarray(z)
    r = r_of_z(z, mode)

    zr = dz_dr(z, mode)
    zrr = d2z_dr2(z, mode)

    D = Delta(r, mode)
    Dp = Delta_p(r, mode)
    K = K_of_r(r, mode)

    V = (K * K + 4j * (r - mode.M) * K) / D - 8j * mode.omega * r - mode.lambda_value

    q, qp = middle_q_and_qp(r, mode)

    B2 = D * zr * zr
    B1 = D * (zrr + 2.0 * q * zr) - Dp * zr
    B0 = D * (qp + q * q) - Dp * q + V

    return B2.astype(complex), B1.astype(complex), B0.astype(complex)


# ----------------------------------------------------------------------
# 5. 在 middle patch 上构造 custom 基解
# ----------------------------------------------------------------------
# def solve_custom_basis_domain(
#     mode: KerrMode,
#     N: int,
#     z_a: float,
#     z_b: float,
#     p: float,
#     sigma: float,
#     *,
#     bc_side: str,
#     U_bc: complex,
#     Uz_bc: complex,
# ):
#     """
#     在 [z_a, z_b] 上解 custom 变量 U 的二阶 ODE。

#     边界条件：
#         U = U_bc
#         U_z = Uz_bc

#     bc_side:
#         "left"  : 在 z_a 处给定 U, U_z；
#         "right" : 在 z_b 处给定 U, U_z。

#     注意：
#         这里 U_z 是对 z 的导数，不是对 r 的导数。
#     """
#     Dmat, z = cheb_D(N, z_a, z_b)
#     D2 = Dmat @ Dmat

#     A = np.zeros((N + 1, N + 1), dtype=complex)
#     b = np.zeros(N + 1, dtype=complex)

#     if bc_side == "left":
#         # 左端函数值
#         A[0, :] = 0.0
#         A[0, 0] = 1.0
#         b[0] = U_bc

#         # 左端 z 导数
#         A[1, :] = Dmat[0, :]
#         b[1] = Uz_bc

#         # 内点 ODE
#         idx = np.arange(2, N + 1)
#         B2, B1, B0 = coeffs_custom_mid(z[idx], mode, p, sigma)
#         A[idx, :] = B2[:, None] * D2[idx, :] + B1[:, None] * Dmat[idx, :]
#         A[idx, idx] += B0

#     elif bc_side == "right":
#         # 内点 ODE
#         idx = np.arange(0, N - 1)
#         B2, B1, B0 = coeffs_custom_mid(z[idx], mode, p, sigma)
#         A[idx, :] = B2[:, None] * D2[idx, :] + B1[:, None] * Dmat[idx, :]
#         A[idx, idx] += B0

#         # 右端 z 导数
#         A[-2, :] = Dmat[-1, :]
#         b[-2] = Uz_bc

#         # 右端函数值
#         A[-1, :] = 0.0
#         A[-1, -1] = 1.0
#         b[-1] = U_bc

#     else:
#         raise ValueError("bc_side must be 'left' or 'right'.")

#     U = np.linalg.solve(A, b)
#     Uz = Dmat @ U

#     return {
#         "z": z,
#         "U": U,
#         "Uz": Uz,
#         "D": Dmat,
#         "matrix_cond": float(np.linalg.cond(A)),
#     }

def solve_middle_up_basis_domain(
    mode: KerrMode,
    N: int,
    z_a: float,
    z_b: float,
    *,
    bc_side: str,
    U_bc: complex,
    Uz_bc: complex,
):
    """
    Solve the middle equation with the physical-up prefactor:

        R = A_up U.

    Boundary data are given for U and U_z.
    """
    Dmat, z = cheb_D(N, z_a, z_b)
    D2 = Dmat @ Dmat

    A = np.zeros((N + 1, N + 1), dtype=complex)
    b = np.zeros(N + 1, dtype=complex)

    if bc_side == "left":
        # U(z_a) = U_bc
        A[0, :] = 0.0
        A[0, 0] = 1.0
        b[0] = U_bc

        # U_z(z_a) = Uz_bc
        A[1, :] = Dmat[0, :]
        b[1] = Uz_bc

        # Interior ODE rows
        idx = np.arange(2, N + 1)
        B2, B1, B0 = coeffs_middle_up(z[idx], mode)
        A[idx, :] = B2[:, None] * D2[idx, :] + B1[:, None] * Dmat[idx, :]
        A[idx, idx] += B0

    elif bc_side == "right":
        # Interior ODE rows
        idx = np.arange(0, N - 1)
        B2, B1, B0 = coeffs_middle_up(z[idx], mode)
        A[idx, :] = B2[:, None] * D2[idx, :] + B1[:, None] * Dmat[idx, :]
        A[idx, idx] += B0

        # U_z(z_b) = Uz_bc
        A[-2, :] = Dmat[-1, :]
        b[-2] = Uz_bc

        # U(z_b) = U_bc
        A[-1, :] = 0.0
        A[-1, -1] = 1.0
        b[-1] = U_bc

    else:
        raise ValueError("bc_side must be 'left' or 'right'.")

    # Row/column scaled solve
    U, diag = solve_rowcol_scaled_square(A, b)
    Uz = Dmat @ U

    return {
        "z": z,
        "U": U,
        "Uz": Uz,
        "D": Dmat,
        "matrix_cond_raw": diag["cond_raw"],
        "matrix_cond_scaled": diag["cond_rowcol"],
        "matrix_relres": diag["relres"],
    }
def R_state_to_U_state(
    R: complex,
    Rr: complex,
    z: float,
    mode: KerrMode,
) -> np.ndarray:
    """
    Convert physical state [R, R_r] to middle-up state [U, U_r]:

        R = A_up U.
    """
    r = r_of_z(z, mode)
    F = middle_factor(r, mode)
    q, _ = middle_q_and_qp(r, mode)

    U = R / F
    Ur = Rr / F - q * U
    return np.array([U, Ur], dtype=complex)


def U_state_to_R_state(
    U: complex,
    Ur: complex,
    z: float,
    mode: KerrMode,
) -> np.ndarray:
    """
    Convert middle-up state [U, U_r] back to physical state [R, R_r].
    """
    r = r_of_z(z, mode)
    F = middle_factor(r, mode)
    q, _ = middle_q_and_qp(r, mode)

    R = F * U
    Rr = F * (q * U + Ur)
    return np.array([R, Rr], dtype=complex)

def U_z_to_U_r(Uz: complex, z: float, mode: KerrMode) -> complex:
    """
    谱方法给出的是 U_z，匹配 state 使用 U_r。

        U_r = z_r U_z
    """
    return dz_dr(z, mode) * Uz


def U_r_to_U_z(Ur: complex, z: float, mode: KerrMode) -> complex:
    """
    如果已知 U_r，要给谱 solver 的边界 U_z：

        U_z = U_r / z_r
    """
    return Ur / dz_dr(z, mode)


def custom_values_at_edge(sol: Dict, side: str, mode: KerrMode) -> np.ndarray:
    """
    从 custom 谱解中取边界 state [U, U_r]。

    side="left"  : z=z_a；
    side="right" : z=z_b。
    """
    if side == "left":
        j = 0
    elif side == "right":
        j = -1
    else:
        raise ValueError("side must be 'left' or 'right'.")

    z = float(sol["z"][j])
    U = complex(sol["U"][j])
    Uz = complex(sol["Uz"][j])
    Ur = U_z_to_U_r(Uz, z, mode)

    return np.array([U, Ur], dtype=complex)


def solve_middle_up_transport(
    mode: KerrMode,
    N_mid: int,
    z1: float,
    z2: float,
):
    """
    Construct middle transport matrix in U-space with R=A_up U.

    T_U maps:

        [U, U_r](z1) -> [U, U_r](z2).
    """
    # Basis 0: [U, U_r](z1) = [1, 0]
    sol0 = solve_middle_up_basis_domain(
        mode,
        N_mid,
        z1,
        z2,
        bc_side="left",
        U_bc=1.0 + 0.0j,
        Uz_bc=U_r_to_U_z(0.0 + 0.0j, z1, mode),
    )

    # Basis 1: [U, U_r](z1) = [0, 1]
    sol1 = solve_middle_up_basis_domain(
        mode,
        N_mid,
        z1,
        z2,
        bc_side="left",
        U_bc=0.0 + 0.0j,
        Uz_bc=U_r_to_U_z(1.0 + 0.0j, z1, mode),
    )

    y0_z2 = custom_values_at_edge(sol0, "right", mode)
    y1_z2 = custom_values_at_edge(sol1, "right", mode)

    T_U = np.column_stack([y0_z2, y1_z2])

    return {
        "T_U": T_U,
        "sol_U_val": sol0,
        "sol_U_der": sol1,
        "cond_T_U": float(np.linalg.cond(T_U)),
        "cond_A_val_raw": float(sol0["matrix_cond_raw"]),
        "cond_A_der_raw": float(sol1["matrix_cond_raw"]),
        "cond_A_val_scaled": float(sol0["matrix_cond_scaled"]),
        "cond_A_der_scaled": float(sol1["matrix_cond_scaled"]),
        "A_val_relres": float(sol0["matrix_relres"]),
        "A_der_relres": float(sol1["matrix_relres"]),
    }


def combine_middle_U_values(mid: Dict, y_z1: np.ndarray):
    """
    Given initial U-state y_z1=[U,U_r] at z1, reconstruct U(z)
    on the middle Chebyshev nodes.

    The middle basis sol_U_val corresponds to [1,0] at z1.
    The middle basis sol_U_der corresponds to [0,1] at z1.
    """
    U0 = np.asarray(mid["sol_U_val"]["U"], dtype=complex)
    U1 = np.asarray(mid["sol_U_der"]["U"], dtype=complex)
    return y_z1[0] * U0 + y_z1[1] * U1


def solve_value_only_fit(
    mid: Dict,
    U_down_z1: np.ndarray,
    U_up_z1: np.ndarray,
    U_in_z1_back: np.ndarray,
    *,
    use_interior_only: bool = True,
):
    """
    Multi-point value-only fit:

        U_in(z_j) = B_inc U_down(z_j) + B_ref U_up(z_j).

    This avoids using U_r rows, because the derivative carrier ratio for
    B_ref is typically ~1e-9 and contaminates B_ref extraction.
    """
    z = np.asarray(mid["sol_U_val"]["z"], dtype=float)

    U_down_vals = combine_middle_U_values(mid, U_down_z1)
    U_up_vals = combine_middle_U_values(mid, U_up_z1)
    U_in_vals = combine_middle_U_values(mid, U_in_z1_back)

    if use_interior_only and len(z) > 4:
        idx = np.arange(1, len(z) - 1)
    else:
        idx = np.arange(len(z))

    A = np.column_stack([U_down_vals[idx], U_up_vals[idx]])
    b = U_in_vals[idx]

    # Row balancing: each z_j row contributes comparably.
    row_norm = np.abs(A[:, 0]) + np.abs(A[:, 1])
    row_norm = np.where(row_norm > 1.0e-300, row_norm, 1.0)
    A_bal = A / row_norm[:, None]
    b_bal = b / row_norm

    coef, diag = solve_scaled_lstsq(A_bal, b_bal)

    residual = A @ coef - b
    relres = np.linalg.norm(residual) / max(np.linalg.norm(b), 1.0e-300)

    return coef, {
        "value_fit_relres": float(relres),
        "value_fit_cond": float(np.linalg.cond(A)),
        "value_fit_cond_bal": float(np.linalg.cond(A_bal)),
        "value_fit_num_points": int(len(idx)),
    }

def solve_ratio_value_fit(
    mid: Dict,
    U_down_z1: np.ndarray,
    U_up_z1: np.ndarray,
    U_in_z1_back: np.ndarray,
    *,
    use_interior_only: bool = True,
    h_min: float = 0.0,
):
    """
    Ratio-form value fit:

        U_in/U_down = B_inc + B_ref * (U_up/U_down).

    This exposes the lever arm H=U_up/U_down and is useful for judging
    whether the interval has enough sensitivity to B_ref.
    """
    z = np.asarray(mid["sol_U_val"]["z"], dtype=float)

    U_down_vals = combine_middle_U_values(mid, U_down_z1)
    U_up_vals = combine_middle_U_values(mid, U_up_z1)
    U_in_vals = combine_middle_U_values(mid, U_in_z1_back)

    if use_interior_only and len(z) > 4:
        idx = np.arange(1, len(z) - 1)
    else:
        idx = np.arange(len(z))

    Ud = U_down_vals[idx]
    Uu = U_up_vals[idx]
    Ui = U_in_vals[idx]

    mask = np.abs(Ud) > 1.0e-300
    H = Uu[mask] / Ud[mask]
    Y = Ui[mask] / Ud[mask]

    if h_min > 0:
        keep = np.abs(H) >= h_min
        H = H[keep]
        Y = Y[keep]

    A = np.column_stack([np.ones_like(H), H])

    coef, diag = solve_scaled_lstsq(A, Y)
    residual = A @ coef - Y
    relres = np.linalg.norm(residual) / max(np.linalg.norm(Y), 1.0e-300)

    H_abs = np.abs(H)
    H_phase = np.unwrap(np.angle(H))

    return coef, {
        "ratio_fit_relres": float(relres),
        "ratio_fit_cond": float(np.linalg.cond(A)),
        "ratio_fit_num_points": int(len(H)),
        "H_abs_min": float(np.min(H_abs)),
        "H_abs_max": float(np.max(H_abs)),
        "H_abs_med": float(np.median(H_abs)),
        "H_phase_span": float(np.max(H_phase) - np.min(H_phase)),
    }

# ----------------------------------------------------------------------
# 6. 主计算：当前最好 common-custom 方案
# ----------------------------------------------------------------------
def compute_best_common_custom(
    mode: KerrMode,
    *,
    N_left: int,
    N_mid: int,
    N_right: int,
    z1: float,
    z2: float,
):
    """
    当前最好方案：

    left patch:
        使用外区物理 down/up basis，仍然由原 solve_basis_domain 计算。

    middle patch:
        使用 common custom 变量：
            R = r^p exp(+i omega r_*) U
        在 U-state 空间传播。

    right patch:
        使用 horizon in/out basis，仍然由原 solve_basis_domain 计算。

    matching:
        1. right in-state at z2 转成 U-state；
        2. 通过 T_U^{-1} 反传到 z1；
        3. left down/up states at z1 转成 U-state；
        4. 在同一个 U-state 空间中解 B_inc, B_ref。
    """
    # --------------------------------------------------------------
    # left patch: [0,z1], physical down/up
    # --------------------------------------------------------------
    sol_down = solve_basis_domain(mode, "down", N_left, 0.0, z1, "left")
    sol_up = solve_basis_domain(mode, "up", N_left, 0.0, z1, "left")

    R_down_z1, Rr_down_z1 = basis_values_at_match(mode, "down", sol_down, "right")
    R_up_z1, Rr_up_z1 = basis_values_at_match(mode, "up", sol_up, "right")

    # 把 left physical states 转成 custom U-state。
    U_down_z1 = R_state_to_U_state(R_down_z1, Rr_down_z1, z1, mode)
    U_up_z1 = R_state_to_U_state(R_up_z1, Rr_up_z1, z1, mode)

    M_left_U = np.column_stack([U_down_z1, U_up_z1])

    # --------------------------------------------------------------
    # middle patch: [z1,z2], custom transport in U-space
    # --------------------------------------------------------------
    mid = solve_middle_up_transport(mode, N_mid, z1, z2)
    T_U = mid["T_U"]

    # --------------------------------------------------------------
    # right patch: [z2,1], physical horizon in/out
    # --------------------------------------------------------------
    sol_in = solve_basis_domain(mode, "in", N_right, z2, 1.0, "right")
    sol_out = solve_basis_domain(mode, "out", N_right, z2, 1.0, "right")

    R_in_z2, Rr_in_z2 = basis_values_at_match(mode, "in", sol_in, "left")
    R_out_z2, Rr_out_z2 = basis_values_at_match(mode, "out", sol_out, "left")

    # 右端 in-state 转成 custom U-state。
    U_in_z2 = R_state_to_U_state(R_in_z2, Rr_in_z2, z2, mode)
    U_out_z2 = R_state_to_U_state(R_out_z2, Rr_out_z2, z2, mode)

    # --------------------------------------------------------------
    # 核心步骤：不要在 physical R-space 里回传；
    # 而是在 custom U-space 中回传。
    # --------------------------------------------------------------
    U_in_z1_back, diag_back_in = solve_scaled_lstsq(T_U, U_in_z2)
    U_out_z1_back, diag_back_out = solve_scaled_lstsq(T_U, U_out_z2)

    # --------------------------------------------------------------
    # 在同一个 U-state 空间里分解 left down/up 系数。
    # 这两个系数就是物理外区的 B_inc, B_ref。
    # --------------------------------------------------------------
    # coef_in, diag_coef_in = solve_scaled_lstsq(M_left_U, U_in_z1_back)
    # coef_out, diag_coef_out = solve_scaled_lstsq(M_left_U, U_out_z1_back)

    # B_inc = complex(coef_in[0])
    # B_ref = complex(coef_in[1])

    coef_in_state, diag_coef_in = solve_scaled_lstsq(M_left_U, U_in_z1_back)
    coef_out, diag_coef_out = solve_scaled_lstsq(M_left_U, U_out_z1_back)

    B_inc_state = complex(coef_in_state[0])
    B_ref_state = complex(coef_in_state[1])

    # New: multi-point value-only fit across middle patch.
    coef_in_value, diag_value_fit = solve_value_only_fit(
        mid,
        U_down_z1,
        U_up_z1,
        U_in_z1_back,
        use_interior_only=True,
    )

    coef_in_ratio, diag_ratio_fit = solve_ratio_value_fit(
    mid,
    U_down_z1,
    U_up_z1,
    U_in_z1_back,
    use_interior_only=True,
)

    B_inc_ratio = complex(coef_in_ratio[0])
    B_ref_ratio = complex(coef_in_ratio[1])

    B_inc_value = complex(coef_in_value[0])
    B_ref_value = complex(coef_in_value[1])

    # Default output: use value-only extraction for B_ref diagnostics.
    # Keep state result as comparison.
    B_inc = B_inc_value
    B_ref = B_ref_value
    # --------------------------------------------------------------
    # 中间携带量诊断：
    # 在 z1 和 z2，计算 B_inc*U_down 与 B_ref*U_up 的量级。
    # 如果这两个量差很多，说明反射通道仍然没有真正被抬到同阶。
    # --------------------------------------------------------------
    carrier = {}
    for label, z in [("z1", z1), ("z2", z2)]:
        if label == "z1":
            U_down = U_down_z1
            U_up = U_up_z1
        else:
            # 把 left down/up 经 middle transport 到 z2
            U_down = T_U @ U_down_z1
            U_up = T_U @ U_up_z1

        inc_vec = B_inc * U_down
        ref_vec = B_ref * U_up

        carrier[label] = {
            "z": z,
            "r": float(r_of_z(z, mode)),
            "abs_inc_U": [float(abs(inc_vec[0])), float(abs(inc_vec[1]))],
            "abs_ref_U": [float(abs(ref_vec[0])), float(abs(ref_vec[1]))],
            "ratio_value_component": float(abs(ref_vec[0]) / max(abs(inc_vec[0]), 1e-300)),
            "ratio_derivative_component": float(abs(ref_vec[1]) / max(abs(inc_vec[1]), 1e-300)),
        }
        # ----- 新增：Abel 常数与 S 矩阵行列式 -----
    r_match_outer = r_of_z(z1, mode)
    r_match_inner = r_of_z(z2, mode)

    outer_abel_num = abel_constant_from_pair(
        R_down_z1, Rr_down_z1, R_up_z1, Rr_up_z1, r_match_outer, mode
    )
    outer_abel_th = outer_abel_theory(mode)

    inner_abel_num = abel_constant_from_pair(
        R_in_z2, Rr_in_z2, R_out_z2, Rr_out_z2, r_match_inner, mode
    )
    inner_abel_th = inner_abel_theory(mode)

    # S 矩阵 (2x2) 的数值行列式
    # 这里统一采用当前默认输出所使用的 value-only 第一列，
    # 以及 state-fit 第二列，保持与上面的 B_inc/B_ref 选择一致。
    detS_num = coef_in_value[0] * coef_out[1] - coef_in_value[1] * coef_out[0]
    detS_th = inner_abel_th / outer_abel_th

    abel_diag = {
        "outer_abel_num": complex(outer_abel_num),
        "outer_abel_th": complex(outer_abel_th),
        "inner_abel_num": complex(inner_abel_num),
        "inner_abel_th": complex(inner_abel_th),
        "detS_num": complex(detS_num),
        "detS_th": complex(detS_th),
    }
    # ----- 新增：基解纯度诊断 -----
    # 右侧 in 解在 custom U 空间中的纯度（应在 in/out 基上分解）
    # M_right_U = [U_in_z2, U_out_z2]
    M_right_U = np.column_stack([U_in_z2, U_out_z2])
    # 解 U_in_z2 = M_right_U * coef_right, coef_right 应为 [1,0]
    coef_right, diag_right = solve_scaled_lstsq(M_right_U, U_in_z2)
    parasitic_out_in_z2 = coef_right[1]  # 寄生 out 振幅

    # 左侧 down 解在 custom U 空间中的纯度（应在 down/up 基上分解）
    # 注意：M_left_U 已经是 [U_down_z1, U_up_z1]
    # 解 U_down_z1 = M_left_U * coef_left_down, 理论应为 [1,0]
    coef_left_down, diag_left_down = solve_scaled_lstsq(M_left_U, U_down_z1)
    parasitic_up_in_down = coef_left_down[1]  # down 解中寄生的 up 振幅

    purity_diag = {
        "parasitic_out_in_in": complex(parasitic_out_in_z2),
        "parasitic_up_in_down": complex(parasitic_up_in_down),
        "rel_out_in_in": float(abs(parasitic_out_in_z2)),
        "rel_up_in_down": float(abs(parasitic_up_in_down)),
    }
    # ------------------------------------
    # -----------------------------------------------
    # 2x2 重构误差：只检验代数一致性，不代表 ODE 物理误差。
    U_in_z1_recon = M_left_U @ coef_in_state

    diagnostics = {
        "cond_M_left_U": float(np.linalg.cond(M_left_U)),
        "cond_T_U": float(np.linalg.cond(T_U)),
        "back_in_relres": diag_back_in["relres"],
        "coef_in_relres": diag_coef_in["relres"],
        "state_in_z1_U_relerr": float(
            np.linalg.norm(U_in_z1_recon - U_in_z1_back)
            / max(np.linalg.norm(U_in_z1_back), 1e-300)
        ),
        # "mid_cond_A_val": float(mid["cond_A_val"]),
        # "mid_cond_A_der": float(mid["cond_A_der"]),
        "value_fit_relres": diag_value_fit["value_fit_relres"],
        "value_fit_cond": diag_value_fit["value_fit_cond"],
        "value_fit_cond_bal": diag_value_fit["value_fit_cond_bal"],
        "value_fit_num_points": diag_value_fit["value_fit_num_points"],
        "ratio_fit_relres": diag_ratio_fit["ratio_fit_relres"],
        "ratio_fit_cond": diag_ratio_fit["ratio_fit_cond"],
        "ratio_fit_num_points": diag_ratio_fit["ratio_fit_num_points"],
        "H_abs_min": diag_ratio_fit["H_abs_min"],
        "H_abs_max": diag_ratio_fit["H_abs_max"],
        "H_abs_med": diag_ratio_fit["H_abs_med"],
        "H_phase_span": diag_ratio_fit["H_phase_span"],
            }

    return {
        "B_inc": B_inc,
        "B_ref": B_ref,
        "B_ref_over_B_inc": safe_ratio(B_ref, B_inc),
        "B_inc_out": complex(coef_out[0]),
        "B_ref_out": complex(coef_out[1]),
        "carrier": carrier,
        "diagnostics": diagnostics,
        "abel_diag": abel_diag, 
        "purity_diag": purity_diag,
        "sol_down": sol_down,
        "sol_up": sol_up,
        "sol_in": sol_in,
        "sol_out": sol_out,
        "mid": mid,
        "B_inc_state": B_inc_state,
        "B_ref_state": B_ref_state,
        "B_ref_over_B_inc_state": safe_ratio(B_ref_state, B_inc_state),
        "B_inc_value": B_inc_value,
        "B_ref_value": B_ref_value,
        "B_ref_over_B_inc_value": safe_ratio(B_ref_value, B_inc_value),
        "B_inc_ratio": B_inc_ratio,
        "B_ref_ratio": B_ref_ratio,
        "B_ref_over_B_inc_ratio": safe_ratio(B_ref_ratio, B_inc_ratio),
    }


# ----------------------------------------------------------------------
# 7. 打印结果
# ----------------------------------------------------------------------
def print_complex(name: str, z: complex):
    print(f"{name:28s} = {z.real:+.16e} {z.imag:+.16e}j   |.|={abs(z):.16e}")


def main():
    parser = argparse.ArgumentParser()

    # mode 参数
    parser.add_argument("--M", type=float, default=1.0)
    parser.add_argument("--a", type=float, default=0.1)
    parser.add_argument("--s", type=int, default=-2)
    parser.add_argument("--ell", type=int, default=2)
    parser.add_argument("--m", type=int, default=2)
    parser.add_argument("--omega", type=float, default=10.0)
    parser.add_argument("--lam", type=float, default=-2.45630962430791)

    # 当前最好参数
    parser.add_argument("--N-left", type=int, default=28)
    parser.add_argument("--N-mid", type=int, default=28)
    parser.add_argument("--N-right", type=int, default=28)
    parser.add_argument("--z1", type=float, default=0.625)
    parser.add_argument("--z2", type=float, default=0.705)
    parser.add_argument(
        "--z1-scan",
        type=str,
        default="",
        help="Comma-separated z1 values for moderate-z1 scan, e.g. '0.625,0.5,0.4,0.3,0.2'.",
    )

    # GSN 参考值，仅用于打印误差
    parser.add_argument("--ref-Binc-re", type=float, default=31.2044)
    parser.add_argument("--ref-Binc-im", type=float, default=3.94032)
    parser.add_argument("--ref-Bref-re", type=float, default=3.399e-8)
    parser.add_argument("--ref-Bref-im", type=float, default=-7.713e-9)
    parser.add_argument("--mp-dps", type=int, default=0, help="Use mpmath high precision for middle patch (0=off)")
    args = parser.parse_args()

    mode = KerrMode(
        M=args.M,
        a=args.a,
        omega=args.omega,
        ell=args.ell,
        m=args.m,
        lam=args.lam,
        s=args.s,
    )
    if args.z1_scan.strip():
        z1_values = [float(x) for x in args.z1_scan.split(",") if x.strip()]

        print("\n=== Moderate-z1 scan with physical-up middle ansatz and value-only fit ===")
        print(f"fixed z2={args.z2}, N_left={args.N_left}, N_mid={args.N_mid}, N_right={args.N_right}")
        print("columns: z1, B_inc_value, B_ref_value, |B_ref/B_inc|, value_fit_relres, cond_T_U\n")
        Bref_ref = complex(args.ref_Bref_re, args.ref_Bref_im)

        print(
            "z1        |Bref_val/Binc|  err_val   "
            "|Bref_rat/Binc|  err_rat   "
            "cond_val  cond_rat  H_med  H_phase  carrier_v"
        )
        for z1_val in z1_values:
            if not (0.0 < z1_val < args.z2 < 1.0):
                print(f"[skip] invalid z1={z1_val}, z2={args.z2}")
                continue

            res = compute_best_common_custom(
                mode,
                N_left=args.N_left,
                N_mid=args.N_mid,
                N_right=args.N_right,
                z1=z1_val,
                z2=args.z2,
            )

            ratio = res["B_ref_over_B_inc"]
            print(
                f"z1={z1_val:.6f}  "
                f"Binc={res['B_inc']:+.6e}  "
                f"Bref={res['B_ref']:+.6e}  "
                f"|Bref/Binc|={abs(ratio):.6e}  "
                f"value_relres={res['diagnostics']['value_fit_relres']:.3e}  "
                f"cond_T={res['diagnostics']['cond_T_U']:.3e}"
            )
            rv = res["B_ref_over_B_inc_value"]
            rr = res["B_ref_over_B_inc_ratio"]

            err_val = abs(res["B_ref_value"] - Bref_ref) / max(abs(Bref_ref), 1e-300)
            err_rat = abs(res["B_ref_ratio"] - Bref_ref) / max(abs(Bref_ref), 1e-300)

            print(
                f"{z1_val:.6f}  "
                f"{abs(rv):.6e}  {err_val:.3e}  "
                f"{abs(rr):.6e}  {err_rat:.3e}  "
                f"{res['diagnostics']['value_fit_cond']:.3e}  "
                f"{res['diagnostics']['ratio_fit_cond']:.3e}  "
                f"{res['diagnostics']['H_abs_med']:.3e}  "
                f"{res['diagnostics']['H_phase_span']:.3e}  "
                f"{res['carrier']['z1']['ratio_value_component']:.3e}"
            )
        return

    result = compute_best_common_custom(
    mode,
    N_left=args.N_left,
    N_mid=args.N_mid,
    N_right=args.N_right,
    z1=args.z1,
    z2=args.z2,
)

    Binc_ref = complex(args.ref_Binc_re, args.ref_Binc_im)
    Bref_ref = complex(args.ref_Bref_re, args.ref_Bref_im)

    print("\n=== Best common-custom middle-patch debug run ===")
    print(f"mode: M={args.M}, a={args.a}, s={args.s}, ell={args.ell}, m={args.m}, omega={args.omega}, lambda={args.lam}")
    print(f"N_left={args.N_left}, N_mid={args.N_mid}, N_right={args.N_right}")
    print(f"z1={args.z1}, z2={args.z2}")
    print("middle ansatz: R = A_up * U = r^3 * exp(+i omega r_*) * U\n")

    print("=== Amplitudes ===")
    print_complex("B_inc", result["B_inc"])
    print_complex("B_ref", result["B_ref"])
    print_complex("B_ref/B_inc", result["B_ref_over_B_inc"])

    print("\n=== GSN reference comparison ===")
    print_complex("B_inc_ref", Binc_ref)
    print_complex("B_ref_ref", Bref_ref)
    print_complex("Bref/Binc ref", Bref_ref / Binc_ref)

    err_Binc = abs(result["B_inc"] - Binc_ref) / max(abs(Binc_ref), 1e-300)
    err_Bref = abs(result["B_ref"] - Bref_ref) / max(abs(Bref_ref), 1e-300)
    err_ratio = abs(result["B_ref_over_B_inc"] - Bref_ref / Binc_ref) / max(abs(Bref_ref / Binc_ref), 1e-300)

    print(f"\nrelerr(B_inc)        = {err_Binc:.6e}")
    print(f"relerr(B_ref)        = {err_Bref:.6e}")
    print(f"relerr(B_ref/B_inc)  = {err_ratio:.6e}")

    print("\n=== Linear algebra diagnostics ===")
    for k, v in result["diagnostics"].items():
        print(f"{k:28s} = {v:.6e}")

    print("\n=== Carrier-size diagnostics in U-space ===")
    print("These compare |B_inc * U_down| and |B_ref * U_up| in the propagated variable.")
    print("If the ratios are still tiny, the reflected channel is still not truly scale-balanced.\n")
    # ----- 新增：Abel & det(S) 诊断 -----
    print("\n=== Abel & S-matrix determinant diagnostics ===")
    ad = result["abel_diag"]
    print_complex("outer_abel_num", ad["outer_abel_num"])
    print_complex("outer_abel_th", ad["outer_abel_th"])
    print(f"{'outer_abel_relerr':28s} = {abs(ad['outer_abel_num']-ad['outer_abel_th'])/max(abs(ad['outer_abel_th']),1e-300):.6e}")
    print_complex("inner_abel_num", ad["inner_abel_num"])
    print_complex("inner_abel_th", ad["inner_abel_th"])
    print(f"{'inner_abel_relerr':28s} = {abs(ad['inner_abel_num']-ad['inner_abel_th'])/max(abs(ad['inner_abel_th']),1e-300):.6e}")
    print_complex("detS_num", ad["detS_num"])
    print_complex("detS_th", ad["detS_th"])
    print(f"{'detS_relerr':28s} = {abs(ad['detS_num']-ad['detS_th'])/max(abs(ad['detS_th']),1e-300):.6e}")
    # ----- 新增：基解纯度打印 -----
    print("\n=== Purity diagnostics (parasitic amplitudes) ===")
    pd = result["purity_diag"]
    print_complex("out in in-solution", pd["parasitic_out_in_in"])
    print(f"{'|out| in in-sol':28s} = {pd['rel_out_in_in']:.6e}")
    print_complex("up in down-solution", pd["parasitic_up_in_down"])
    print(f"{'|up| in down-sol':28s} = {pd['rel_up_in_down']:.6e}")

    # ------------------------------------
    # 使用行列式关系估算 B_ref
    Cin_down = result["B_inc"]
    Cin_up = result["B_ref"]
    Cout_down = result["B_inc_out"]
    Cout_up = result["B_ref_out"]
    det_th = ad["detS_th"]
    print("\n=== State matching result: single-point (U, U_r) ===")
    print_complex("B_inc_state", result["B_inc_state"])
    print_complex("B_ref_state", result["B_ref_state"])
    print_complex("B_ref/B_inc state", result["B_ref_over_B_inc_state"])

    print("\n=== Value-only multi-point fit result ===")
    print_complex("B_inc_value", result["B_inc_value"])
    print_complex("B_ref_value", result["B_ref_value"])
    print_complex("B_ref/B_inc value", result["B_ref_over_B_inc_value"])
    # 公式: det = Cin_down * Cout_up - Cin_up * Cout_down
    # 若 Cout_down 不那么微小，可直接解出 Cin_up ≈ (Cin_down * Cout_up - det_th) / Cout_down
    if abs(Cout_down) > 1e-14:
        B_ref_alt = (Cin_down * Cout_up - det_th) / Cout_down
        print_complex("B_ref_alt (from det)", B_ref_alt)
        ref_ref = complex(args.ref_Bref_re, args.ref_Bref_im)
        err_alt = abs(B_ref_alt - ref_ref) / max(abs(ref_ref), 1e-300)
        print(f"{'relerr(B_ref_alt)':28s} = {err_alt:.6e}")
    else:
        print("Cout_down too small for alternative B_ref extraction.")
    # -----------------------------------------
    for label, info in result["carrier"].items():
        print(f"[{label}] z={info['z']:.8f}, r={info['r']:.8f}")
        print(f"  |B_inc * U_down|  value,derivative = {info['abs_inc_U'][0]:.6e}, {info['abs_inc_U'][1]:.6e}")
        print(f"  |B_ref * U_up|    value,derivative = {info['abs_ref_U'][0]:.6e}, {info['abs_ref_U'][1]:.6e}")
        print(f"  ref/inc ratio     value,derivative = {info['ratio_value_component']:.6e}, {info['ratio_derivative_component']:.6e}\n")
    # ----- 绘制基底函数 (已提取渐近因子的部分) -----
    outdir = "high_omega_debug"
    os.makedirs(outdir, exist_ok=True)

    sol_down = result["sol_down"]
    sol_up = result["sol_up"]
    sol_in = result["sol_in"]
    z1 = args.z1
    z2 = args.z2

    fig, (ax_left1,ax_left2, ax_mid1, ax_mid2, ax_right1, ax_right2) = plt.subplots(1, 6, figsize=(20, 5))

    # 左图：down 基底在 [0, z1] 的 u(z)
    ax_left1.semilogy(sol_down["z"], np.real(sol_down["u"]), label="down (real u)")

    ax_left1.set_xlabel("z")
    ax_left1.set_ylabel("real u (log scale)")
    ax_left1.set_title(f"Left patch [0, {z1}]")
    ax_left1.legend()
    ax_left1.grid(True, which="both", ls="--", alpha=0.5)

    ax_left2.plot(sol_down["z"], np.imag(sol_down["u"]), label="down (imag u)")

    ax_left2.set_xlabel("z")
    ax_left2.set_ylabel("imag u ")
    ax_left2.set_title(f"Left patch [0, {z1}]")
    ax_left2.legend()
    ax_left2.grid(True, which="both", ls="--", alpha=0.5)
    # 中图：up 基底在 [0, z1] 的 u(z)
    ax_mid1.semilogy(sol_up["z"], np.real(sol_up["u"]), label="up (real u)")

    ax_mid1.set_xlabel("z")
    ax_mid1.set_ylabel("real u (log scale)")
    ax_mid1.set_title(f"Left patch [0, {z1}]")
    ax_mid1.legend()
    ax_mid1.grid(True, which="both", ls="--", alpha=0.5)

    ax_mid2.plot(sol_up["z"], np.imag(sol_up["u"]), label="up (imag u)")

    ax_mid2.set_xlabel("z")
    ax_mid2.set_ylabel("imag u ")
    ax_mid2.set_title(f"Left patch [0, {z1}]")
    ax_mid2.legend()
    ax_mid2.grid(True, which="both", ls="--", alpha=0.5)
    # 右图：in 基底在 [z2, 1] 的 u(z)
    ax_right1.semilogy(sol_in["z"], np.real(sol_in["u"]), label="in (real u)")

    ax_right1.set_xlabel("z")   
    ax_right1.set_ylabel("real u (log scale)")
    ax_right1.set_title(f"Right patch [{z2}, 1]")
    ax_right1.legend()
    ax_right1.grid(True, which="both", ls="--", alpha=0.5)

    ax_right2.plot(sol_in["z"], np.imag(sol_in["u"]), label="in (imag u)")

    ax_right2.set_xlabel("z")   
    ax_right2.set_ylabel("imag u")
    ax_right2.set_title(f"Right patch [{z2}, 1]")
    ax_right2.legend()
    ax_right2.grid(True, which="both", ls="--", alpha=0.5)


    plt.tight_layout()
    outpath = os.path.join(outdir, "basis_functions.png")
    plt.savefig(outpath, dpi=150)
    plt.close()
    print(f"\nPlots saved to {outpath}")

if __name__ == "__main__":
    main()
