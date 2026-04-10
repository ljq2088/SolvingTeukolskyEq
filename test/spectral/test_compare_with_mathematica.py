from __future__ import annotations

import os
import sys
import math
import yaml
import numpy as np
import torch
import matplotlib.pyplot as plt

from wolframclient.evaluation import WolframLanguageSession
from wolframclient.language import wlexpr

# 让脚本能从项目根目录导入
sys.path.insert(0, os.getcwd())

from physical_ansatz.residual import AuxCache, get_ramp_and_p_from_cfg
from physical_ansatz.prefactor import U_factor
from physical_ansatz.mapping import r_plus
from physical_ansatz.transform_y import *

def load_cfg(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

from model.chebyshev_trunk import clenshaw_evaluate
# def clenshaw_evaluate(coeff: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
#     """
#     计算
#         f(y) = sum_k coeff_k T_k(y)
#     coeff: (Nc,) complex
#     y:     (Ny,) real
#     返回:   (Ny,) complex
#     """
#     if coeff.ndim != 1:
#         raise ValueError(f"coeff must be 1D, got shape {tuple(coeff.shape)}")
#     if y.ndim != 1:
#         raise ValueError(f"y must be 1D, got shape {tuple(y.shape)}")

#     dtype = coeff.dtype
#     device = coeff.device

#     x = y.to(dtype=dtype, device=device)   # (Ny,)
#     Nc = coeff.shape[0]

#     b_kplus1 = torch.zeros_like(x)
#     b_kplus2 = torch.zeros_like(x)

#     # Clenshaw recurrence
#     for k in range(Nc - 1, 0, -1):
#         b_k = 2.0 * x * b_kplus1 - b_kplus2 + coeff[k]
#         b_kplus2 = b_kplus1
#         b_kplus1 = b_k

#     out = x * b_kplus1 - b_kplus2 + coeff[0]
#     return out


def sample_mathematica_rin(
    s: int,
    l: int,
    m: int,
    a: float,
    omega: float,
    rmin: float = 2.0,
    rmax: float = 10.0,
    npts: int = 100,
    kernel_path: str = r"/mnt/f/mma/WolframKernel.exe",
    wl_path_win: str = r"F:/EMRI/Radial_flow/Radial_Function.wl",
):
    """
    调 Mathematica 采样 Rin
    返回:
        r_mma:  (N,)
        R_mma:  (N,) complex128
    """
    session = WolframLanguageSession(kernel=kernel_path)
    try:
        session.evaluate(wlexpr(rf'Get["{wl_path_win}"]'))

        expr = (
            f"SampleRinOnGrid[{s}, {l}, {m}, "
            f"{a:.16g}, {omega:.16g}, "
            f"{rmin:.16g}, {rmax:.16g}, {npts}]"
        )
        result = session.evaluate(wlexpr(expr))

        arr = np.array(result, dtype=float)
        if arr.ndim != 2 or arr.shape[1] < 3:
            raise ValueError(f"Unexpected Mathematica output shape: {arr.shape}")

        r_mma = arr[:, 0]
        R_mma = arr[:, 1] + 1j * arr[:, 2]
        return r_mma, R_mma

    finally:
        session.terminate()


def best_fit_complex_scale(R_py: np.ndarray, R_ref: np.ndarray) -> complex:
    """
    求复常数 C，使得
        C * R_py ≈ R_ref
    在最小二乘意义下最优。

    C = argmin_C || C R_py - R_ref ||_2^2
      = <R_py, R_ref> / <R_py, R_py>
    """
    denom = np.vdot(R_py, R_py)
    if abs(denom) == 0:
        raise ZeroDivisionError("Denominator is zero when fitting complex scale.")
    C = np.vdot(R_py, R_ref) / denom
    return C


def main():
    # -------------------------------------------------
    # 0. 路径
    # -------------------------------------------------
    cfg_path = "config/teukolsky_radial.yaml"
    ckpt_path = "outputs/single_case_coeffs.pt"

    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # -------------------------------------------------
    # 1. 读配置与 checkpoint
    # -------------------------------------------------
    cfg = load_cfg(cfg_path)
    ckpt = torch.load(ckpt_path, map_location="cpu")

    s = int(cfg["problem"].get("s", -2))
    l = int(cfg["problem"]["l"])
    m = int(cfg["problem"]["m"])
    M = float(cfg["problem"].get("M", 1.0))

    a0 = float(ckpt["a"])
    omega0 = float(ckpt["omega"])
    order = int(ckpt["order"])

    coeff_re = ckpt["coeff_re"].to(torch.float64)
    coeff_im = ckpt["coeff_im"].to(torch.float64)

    if coeff_re.ndim == 2:
        coeff_re = coeff_re[0]
    if coeff_im.ndim == 2:
        coeff_im = coeff_im[0]

    coeff = torch.complex(coeff_re, coeff_im)  # (Nc,)

    print("===== checkpoint info =====")
    print(f"s = {s}, l = {l}, m = {m}")
    print(f"a = {a0}, omega = {omega0}, order = {order}")
    print(f"coeff shape = {tuple(coeff.shape)}")
    print("===========================\n")

    # -------------------------------------------------
    # 2. 用 Mathematica 取同一组参数下的 Rin 样本
    #    这里直接用 Mathematica 返回的 r 网格做比较
    # -------------------------------------------------
    r_mma, R_mma = sample_mathematica_rin(
        s=s,
        l=l,
        m=m,
        a=a0,
        omega=omega0,
        rmin=2.0,
        rmax=10.0,
        npts=100,
        kernel_path=r"/mnt/f/mma/WolframKernel.exe",
        wl_path_win=r"F:/EMRI/Radial_flow/Radial_Function.wl",
    )

    print("Mathematica sample shape =", R_mma.shape)
    print("first 5 r from Mathematica =", r_mma[:5])
    print()

    # -------------------------------------------------
    # 3. 在同一组 r 点上重建 Python 的 R
    #
    #    先算:
    #      x = r_plus / r
    #      y = 2x - 1
    #      f(y) = sum b_k T_k(y)
    #      R'(x(y)) = (exp(x-1)-1) f(y) + 1
    #      R(r) = U(r) R'(r)
    # -------------------------------------------------
    device = torch.device("cpu")
    dtype_real = torch.float64
    dtype_cplx = torch.complex128

    r_t = torch.tensor(r_mma, dtype=dtype_real, device=device)

    a_t = torch.tensor(a0, dtype=dtype_real, device=device)
    omega_t = torch.tensor(omega0, dtype=dtype_real, device=device)

    rp = r_plus(a_t, M)
    x_t = rp/r_t
    y_t = 2.0 * x_t - 1.0

    # 3.1 计算 f(y)
    coeff = coeff.to(dtype=dtype_cplx, device=device)
    f_t = clenshaw_evaluate(coeff, y_t)   # (N,) complex

    # 3.2 还原 R'(x(y))
    g_t = torch.exp(x_t - 1.0).to(dtype=dtype_cplx) - 1.0
    Rp_t = g_t * f_t + 1.0

    # 3.3 从 cfg 里取 p 和 R_amp，构造 U(r)
    cache = AuxCache()
    p, ramp_t = get_ramp_and_p_from_cfg(cfg, cache, a_t, omega_t)

    # U_factor 返回 complex
    U_t = U_factor(
        r_t,                  # (N,)
        a_t,                  # scalar
        omega_t,              # scalar
        p,
        ramp_t,               # scalar complex
        m,
        s,
        M,
    )

    R_py_t = U_t * Rp_t
    R_py = R_py_t.detach().cpu().numpy()

    # -------------------------------------------------
    # 4. 考虑归一化差一个复常数倍
    #
    #    找 C 使得:
    #      C * R_py ≈ R_mma
    # -------------------------------------------------
    C = best_fit_complex_scale(R_py, R_mma)
    R_py_scaled = C * R_py

    rel_err_raw = np.linalg.norm(R_py - R_mma) / np.linalg.norm(R_mma)
    rel_err_scaled = np.linalg.norm(R_py_scaled - R_mma) / np.linalg.norm(R_mma)

    print("===== comparison summary =====")
    print(f"p = {p}")
    print(f"R_amp = {complex(ramp_t.detach().cpu().item())}")
    print(f"best-fit complex scale C = {C}")
    print(f"|C| = {abs(C)}")
    print(f"arg(C) = {np.angle(C)}  [rad]")
    print(f"relative error before scaling  = {rel_err_raw:.6e}")
    print(f"relative error after scaling   = {rel_err_scaled:.6e}")
    print("==============================\n")

    # -------------------------------------------------
    # 5. 画图
    # -------------------------------------------------
    os.makedirs("outputs", exist_ok=True)

    # 图 1：Re/Im 对比（缩放后）
    fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    axes[0].plot(r_mma, R_mma.real, label="Mathematica Re[R]")
    axes[0].plot(r_mma, R_py_scaled.real, "--", label="Python scaled Re[R]")
    axes[0].set_ylabel("Re[R]")
    axes[0].legend()
    axes[0].set_title("Comparison of R(r): Mathematica vs Python (scaled)")

    axes[1].plot(r_mma, R_mma.imag, label="Mathematica Im[R]")
    axes[1].plot(r_mma, R_py_scaled.imag, "--", label="Python scaled Im[R]")
    axes[1].set_xlabel("r / M")
    axes[1].set_ylabel("Im[R]")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("outputs/compare_R_re_im_scaled.png", dpi=200)
    plt.close()

    # 图 2：模长对比（缩放后）
    plt.figure(figsize=(8, 5))
    plt.plot(r_mma, np.abs(R_mma), label="|R_mma|")
    plt.plot(r_mma, np.abs(R_py_scaled), "--", label="|C R_py|")
    plt.xlabel("r / M")
    plt.ylabel("|R|")
    plt.title("Magnitude comparison of R(r)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/compare_R_abs_scaled.png", dpi=200)
    plt.close()

    # 图 3：raw vs scaled（看常数倍差异有多大）
    fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    axes[0].plot(r_mma, R_mma.real, label="Mathematica Re[R]")
    axes[0].plot(r_mma, R_py.real, "--", label="Python raw Re[R]")
    axes[0].set_ylabel("Re[R]")
    axes[0].legend()
    axes[0].set_title("Raw comparison before fitting a constant factor")

    axes[1].plot(r_mma, R_mma.imag, label="Mathematica Im[R]")
    axes[1].plot(r_mma, R_py.imag, "--", label="Python raw Im[R]")
    axes[1].set_xlabel("r / M")
    axes[1].set_ylabel("Im[R]")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("outputs/compare_R_re_im_raw.png", dpi=200)
    plt.close()

    print("Saved figures:")
    print("  outputs/compare_R_re_im_scaled.png")
    print("  outputs/compare_R_abs_scaled.png")
    print("  outputs/compare_R_re_im_raw.png")


if __name__ == "__main__":
    main()