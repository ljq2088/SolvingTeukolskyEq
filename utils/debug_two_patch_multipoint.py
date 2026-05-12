#!/usr/bin/env python3
"""
debug_two_patch_multipoint.py
-----------------------------
最简单的两段匹配：在重叠区间内取多个 z 点，只匹配函数值（不匹配导数）。
利用左基底 (down/up) 和右基底 (in) 的渐近展开项，直接最小二乘求解 B_inc, B_ref。
对 B_ref 的基函数除以 r_pivot^4，将微小系数放大，最后再乘回去，以改善数值条件。
"""

import numpy as np
import argparse
import sys
sys.path.append("/home/ljq/code/PINN/SolvingTeukolsky")
from pathlib import Path

# 加入仓库路径
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.mode import KerrMode
from utils.amplitude import (
    r_of_z, Delta, A_down, A_up, A_in,
    r_star, solve_basis_domain, basis_values_at_match
)
from scipy.interpolate import BarycentricInterpolator

# ---------------------- 工具函数 ----------------------
def setup_bases(mode, N_left, N_right, z_left_max, z_right_min):
    """生成左右基解，返回 sol_down, sol_up, sol_in 以及插值器"""
    # 左区 [0, z_left_max]
    sol_down = solve_basis_domain(mode, "down", N_left, 0.0, z_left_max, "left")
    sol_up   = solve_basis_domain(mode, "up",   N_left, 0.0, z_left_max, "left")
    # 右区 [z_right_min, 1]
    sol_in   = solve_basis_domain(mode, "in",   N_right, z_right_min, 1.0, "right")

    # 构建插值器（使用 scipy BarycentricInterpolator）
    interp_down = BarycentricInterpolator(sol_down["z"], sol_down["u"])
    interp_up   = BarycentricInterpolator(sol_up["z"],   sol_up["u"])
    interp_in   = BarycentricInterpolator(sol_in["z"],   sol_in["u"])

    return sol_down, sol_up, sol_in, interp_down, interp_up, interp_in


def compute_coefficients(mode, interp_down, interp_up, interp_in,
                         z_left_max, z_right_min, N_points=30, r_pivot=None):
    """在重叠区间内多点采值，构建设计矩阵，返回 B_inc, B_ref"""
    # 重叠区间
    z_vals = np.linspace(z_right_min, z_left_max, N_points)
    r_vals = r_of_z(z_vals, mode)

    # 如果未指定 pivot，取区间中点
    if r_pivot is None:
        z_mid = (z_left_max + z_right_min) / 2
        r_pivot = r_of_z(z_mid, mode)
    factor = r_pivot ** 4

    # 计算各基函数在采样点的物理值（不含未知振幅）
    A_d = A_down(r_vals, mode)
    A_u = A_up(r_vals, mode)
    A_i = A_in(r_vals, mode)

    u_d = interp_down(z_vals)   # 复数插值，scipy 支持
    u_u = interp_up(z_vals)
    u_i = interp_in(z_vals)

    # 基函数列（右端项为纯 in 波）
    col1 = A_d * u_d                         # down 贡献
    col2 = (A_u * u_u) / factor              # up 贡献，缩放
    rhs  = A_i * u_i                         # in 波

    # 构造矩阵 (N_points, 2)
    M = np.column_stack([col1, col2])
    # 最小二乘求解
    coeffs, _, _, _ = np.linalg.lstsq(M, rhs, rcond=None)
    B_inc = coeffs[0]
    B_ref_scaled = coeffs[1]
    B_ref = B_ref_scaled / factor
    return B_inc, B_ref, factor


# ---------------------- 主程序 ----------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--M", type=float, default=1.0)
    parser.add_argument("--a", type=float, default=0.1)
    parser.add_argument("--s", type=int, default=-2)
    parser.add_argument("--ell", type=int, default=2)
    parser.add_argument("--m", type=int, default=2)
    parser.add_argument("--omega", type=float, default=10.0)
    parser.add_argument("--lam", type=float, default=-2.45630962430791)
    parser.add_argument("--N-left", type=int, default=80, help="左区 Chebyshev 阶数")
    parser.add_argument("--N-right", type=int, default=80, help="右区 Chebyshev 阶数")
    parser.add_argument("--z-left-max", type=float, default=0.7,
                        help="左区右端点（必须大于 z-right-min 以形成重叠）")
    parser.add_argument("--z-right-min", type=float, default=0.6,
                        help="右区左端点（必须小于 z-left-max）")
    parser.add_argument("--N-points", type=int, default=30,
                        help="重叠区内采样点数")
    parser.add_argument("--pivot-r", type=float, default=None,
                        help="缩放因子 r_pivot，若不提供则自动使用区间中点")

    args = parser.parse_args()

    mode = KerrMode(M=args.M, a=args.a, omega=args.omega,
                    ell=args.ell, m=args.m, lam=args.lam, s=args.s)

    # 检查重叠
    assert args.z_left_max > args.z_right_min, "重叠区间无效：z_left_max 必须大于 z_right_min"

    # 解基函数
    sol_down, sol_up, sol_in, interp_down, interp_up, interp_in = \
        setup_bases(mode, args.N_left, args.N_right, args.z_left_max, args.z_right_min)

    # 多点拟合
    B_inc, B_ref, factor = compute_coefficients(
        mode, interp_down, interp_up, interp_in,
        args.z_left_max, args.z_right_min, N_points=args.N_points,
        r_pivot=args.pivot_r
    )

    print("=== Two-patch multipoint fit ===")
    print(f"Overlap interval: z ∈ [{args.z_right_min}, {args.z_left_max}]")
    print(f"Number of sample points: {args.N_points}")
    print(f"Scaling pivot r = {r_of_z((args.z_left_max+args.z_right_min)/2, mode):.8f}, factor = r^4 = {factor:.6e}")
    print(f"B_inc = {B_inc.real:+16.8e} {B_inc.imag:+16.8e} j   |B_inc| = {abs(B_inc):.8e}")
    print(f"B_ref = {B_ref.real:+16.8e} {B_ref.imag:+16.8e} j   |B_ref| = {abs(B_ref):.8e}")
    if abs(B_inc) > 1e-15:
        print(f"B_ref/B_inc = {B_ref/B_inc:.8e}")

    # 可选的与参考值对比（若需要）
    # ...

if __name__ == "__main__":
    main()