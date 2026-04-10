from __future__ import annotations

import os
import sys
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


def load_cfg(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def clenshaw_evaluate(coeff: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    计算
        f(y) = sum_k coeff_k T_k(y)
    coeff: (Nc,) complex
    y:     (Ny,) real
    返回:   (Ny,) complex
    """
    if coeff.ndim != 1:
        raise ValueError(f"coeff must be 1D, got shape {tuple(coeff.shape)}")
    if y.ndim != 1:
        raise ValueError(f"y must be 1D, got shape {tuple(y.shape)}")

    dtype = coeff.dtype
    device = coeff.device

    x = y.to(dtype=dtype, device=device)
    Nc = coeff.shape[0]

    b_kplus1 = torch.zeros_like(x)
    b_kplus2 = torch.zeros_like(x)

    for k in range(Nc - 1, 0, -1):
        b_k = 2.0 * x * b_kplus1 - b_kplus2 + coeff[k]
        b_kplus2 = b_kplus1
        b_kplus1 = b_k

    out = x * b_kplus1 - b_kplus2 + coeff[0]
    return out


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
        r_mma: (N,)
        R_mma: (N,) complex128
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
    """
    denom = np.vdot(R_py, R_py)
    if abs(denom) == 0:
        raise ZeroDivisionError("Denominator is zero when fitting complex scale.")
    return np.vdot(R_py, R_ref) / denom


def reconstruct_full_R_from_ckpt(
    cfg: dict,
    ckpt_path: str,
    r_query: np.ndarray,
):
    """
    从 direct spectral checkpoint 重建 full R(r)

    返回:
        R_py:  (N,) complex
        Rp_py: (N,) complex
        U_py:  (N,) complex
        meta:  dict
    """
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")

    s = int(cfg["problem"].get("s", -2))
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

    device = torch.device("cpu")
    dtype_real = torch.float64
    dtype_cplx = torch.complex128

    r_t = torch.tensor(r_query, dtype=dtype_real, device=device)
    a_t = torch.tensor(a0, dtype=dtype_real, device=device)
    omega_t = torch.tensor(omega0, dtype=dtype_real, device=device)

    # r -> x -> y
    rp = r_plus(a_t, M)
    x_t = rp / r_t
    y_t = 2.0 * x_t - 1.0

    # reconstruct f and R'
    coeff = coeff.to(dtype=dtype_cplx, device=device)
    f_t = clenshaw_evaluate(coeff, y_t)
    g_t = torch.exp(x_t - 1.0).to(dtype=dtype_cplx) - 1.0
    Rp_t = g_t * f_t + 1.0

    # prefactor U
    cache = AuxCache()
    p, ramp_t = get_ramp_and_p_from_cfg(cfg, cache, a_t, omega_t)

    U_t = U_factor(
        r_t,
        a_t,
        omega_t,
        p,
        ramp_t,
        m,
        s,
        M,
    )

    R_t = U_t * Rp_t

    meta = {
        "a": a0,
        "omega": omega0,
        "order": order,
        "p": p,
        "R_amp": complex(ramp_t.detach().cpu().item()),
    }

    return (
        R_t.detach().cpu().numpy(),
        Rp_t.detach().cpu().numpy(),
        U_t.detach().cpu().numpy(),
        meta,
    )


def main():
    # -------------------------------------------------
    # 0. 路径设置
    # -------------------------------------------------
    cfg_path = "config/teukolsky_radial.yaml"

    # 改这里：你刚才 direct spectral 保存出来的文件
    ckpt_path = "outputs/direct_spectral_a0.1_w0.1_o24.pt"

    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    cfg = load_cfg(cfg_path)

    s = int(cfg["problem"].get("s", -2))
    l = int(cfg["problem"]["l"])
    m = int(cfg["problem"]["m"])

    # -------------------------------------------------
    # 1. 先从 checkpoint 读出参数，确保 Mathematica 用同一组
    # -------------------------------------------------
    ckpt = torch.load(ckpt_path, map_location="cpu")
    a0 = float(ckpt["a"])
    omega0 = float(ckpt["omega"])

    print("===== direct spectral checkpoint info =====")
    print(f"s = {s}, l = {l}, m = {m}")
    print(f"a = {a0}, omega = {omega0}")
    print(f"order = {int(ckpt['order'])}")
    print("==========================================\n")

    # -------------------------------------------------
    # 2. Mathematica full R on the same grid
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

    # -------------------------------------------------
    # 3. Reconstruct Python direct spectral full R
    # -------------------------------------------------
    R_py, Rp_py, U_py, meta = reconstruct_full_R_from_ckpt(
        cfg=cfg,
        ckpt_path=ckpt_path,
        r_query=r_mma,
    )

    print("===== reconstruction meta =====")
    print(f"p = {meta['p']}")
    print(f"R_amp = {meta['R_amp']}")
    print("================================\n")

    # -------------------------------------------------
    # 4. Fit one complex constant factor
    # -------------------------------------------------
    C = best_fit_complex_scale(R_py, R_mma)
    R_py_scaled = C * R_py

    rel_err_raw = np.linalg.norm(R_py - R_mma) / np.linalg.norm(R_mma)
    rel_err_scaled = np.linalg.norm(R_py_scaled - R_mma) / np.linalg.norm(R_mma)

    print("===== full-R comparison summary =====")
    print(f"best-fit complex scale C = {C}")
    print(f"|C| = {abs(C)}")
    print(f"arg(C) = {np.angle(C)} [rad]")
    print(f"relative error before scaling = {rel_err_raw:.6e}")
    print(f"relative error after scaling  = {rel_err_scaled:.6e}")
    print("=====================================\n")

    # -------------------------------------------------
    # 5. Save plots
    # -------------------------------------------------
    os.makedirs("outputs", exist_ok=True)

    # 图1：Re/Im 对比（scaled）
    fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    axes[0].plot(r_mma, R_mma.real, label="Mathematica Re[R]")
    axes[0].plot(r_mma, R_py_scaled.real, "--", label="Direct spectral scaled Re[R]")
    axes[0].set_ylabel("Re[R]")
    axes[0].legend()
    axes[0].set_title("Full R(r): Mathematica vs direct spectral (scaled)")

    axes[1].plot(r_mma, R_mma.imag, label="Mathematica Im[R]")
    axes[1].plot(r_mma, R_py_scaled.imag, "--", label="Direct spectral scaled Im[R]")
    axes[1].set_xlabel("r / M")
    axes[1].set_ylabel("Im[R]")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("outputs/compare_direct_fullR_re_im_scaled.png", dpi=200)
    plt.close()

    # 图2：模长对比（scaled）
    plt.figure(figsize=(8, 5))
    plt.plot(r_mma, np.abs(R_mma), label="|R_mma|")
    plt.plot(r_mma, np.abs(R_py_scaled), "--", label="|C R_direct|")
    plt.xlabel("r / M")
    plt.ylabel("|R|")
    plt.title("Full R(r) magnitude: Mathematica vs direct spectral")
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/compare_direct_fullR_abs_scaled.png", dpi=200)
    plt.close()

    # 图3：raw 对比
    fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    axes[0].plot(r_mma, R_mma.real, label="Mathematica Re[R]")
    axes[0].plot(r_mma, R_py.real, "--", label="Direct spectral raw Re[R]")
    axes[0].set_ylabel("Re[R]")
    axes[0].legend()
    axes[0].set_title("Raw full R(r) before fitting constant factor")

    axes[1].plot(r_mma, R_mma.imag, label="Mathematica Im[R]")
    axes[1].plot(r_mma, R_py.imag, "--", label="Direct spectral raw Im[R]")
    axes[1].set_xlabel("r / M")
    axes[1].set_ylabel("Im[R]")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("outputs/compare_direct_fullR_re_im_raw.png", dpi=200)
    plt.close()

    print("Saved figures:")
    print("  outputs/compare_direct_fullR_re_im_scaled.png")
    print("  outputs/compare_direct_fullR_abs_scaled.png")
    print("  outputs/compare_direct_fullR_re_im_raw.png")


if __name__ == "__main__":
    main()