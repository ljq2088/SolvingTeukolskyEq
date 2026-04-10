"""
用 Mathematica 计算 QNM 径向函数，然后除以 r³ exp(iω r_*) 得到剩余部分
其中 r_* 是乌龟坐标（tortoise coordinate）
参数: s=-2, l=2, m=2, a=0.1, n=0
r 范围: 2.1M → 5000M
输出: mma/divide_by_r3_exp/ 下按参数分类的图
"""
from __future__ import annotations

import os
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from wolframclient.evaluation import WolframLanguageSession
from wolframclient.language import wlexpr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from physical_ansatz.prefactor import r_star

# ── 参数 ──────────────────────────────────────────────
S      = -2
L      = 2
M_MODE = 2
A      = 0.1
N_OVT  = 0
RMIN   = 1000.0
RMAX   = 5000.0
NPTS   = 5000
M      = 1.0
scale_factor = 1e10

KERNEL_PATH  = r"/mnt/f/mma/WolframKernel.exe"
WL_PATH_WIN  = r"F:/EMRI/Radial_flow/QNM.wl"

_BASE_DIR = os.path.join(os.path.dirname(__file__), "divide_by_r3_exp")

def sample_qnm(rmin, rmax, npts):
    session = WolframLanguageSession(kernel=KERNEL_PATH)
    try:
        session.evaluate(wlexpr(rf'Get["{WL_PATH_WIN}"]'))
        freq = session.evaluate(
            wlexpr(f'GetQNMFrequency[{S}, {L}, {M_MODE}, {N_OVT}, {A:.16g}]')
        )
        omega = complex(freq["ReFrequency"], freq["ImFrequency"])
        result = session.evaluate(
            wlexpr(
                f'SampleQNMRadialOnGrid[{S}, {L}, {M_MODE}, {N_OVT}, '
                f'{A:.16g}, {rmin:.16g}, {rmax:.16g}, {npts}]'
            )
        )
        arr = np.array(result, dtype=float)
        r = arr[:, 0]
        R = arr[:, 1] + 1j * arr[:, 2]
        return omega, r, R
    finally:
        session.terminate()

def compute_r3_exp_factor(r_np, omega_cplx):
    """计算 r³ exp(iω r_*)"""
    device = torch.device("cpu")
    r_t = torch.tensor(r_np, dtype=torch.float64, device=device)
    a_t = torch.tensor(A, dtype=torch.float64, device=device)
    omega_t = torch.tensor(omega_cplx, dtype=torch.complex128, device=device)
    
    rstar_t = r_star(r_t, a_t)
    factor = (r_t ** 3) * torch.exp(1j * omega_t * rstar_t)
    return factor.detach().cpu().numpy()

def make_plots(r, R_divided, omega, out_dir):
    wr, wi = omega.real, omega.imag
    title = (
        rf"$R_{{QNM}} / (r^3 e^{{i\omega r_*}})$: $l={L},\ m={M_MODE},\ a={A},\ s={S}$"
        "\n"
        rf"$\omega = {wr:.5f}{wi:+.5f}i$  (n={N_OVT})"
    )

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(r, R_divided.real, lw=1.0)
    axes[0].set_ylabel(r"Re")
    axes[0].set_title(title)
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(r, R_divided.imag, lw=1.0, color="C1")
    axes[1].set_xlabel(r"$r / M$")
    axes[1].set_ylabel(r"Im")
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "R_over_r3exp_linear.png"), dpi=200)
    plt.close()

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].semilogx(r, R_divided.real, lw=1.0)
    axes[0].set_ylabel(r"Re")
    axes[0].set_title(title)
    axes[0].grid(True, which="both", alpha=0.3)
    axes[1].semilogx(r, R_divided.imag, lw=1.0, color="C1")
    axes[1].set_xlabel(r"$r / M$")
    axes[1].set_ylabel(r"Im")
    axes[1].grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "R_over_r3exp_logr.png"), dpi=200)
    plt.close()
    print(f"Saved to: {out_dir}")

if __name__ == "__main__":
    print(f"Parameters: s={S}, l={L}, m={M_MODE}, n={N_OVT}, a={A}")
    print(f"r range: [{RMIN}, {RMAX}], npts={NPTS}\n")

    omega, r, R = sample_qnm(RMIN, RMAX, NPTS)
    print(f"QNM ω = {omega.real:.8f} {omega.imag:+.8f}i")

    out_dir = os.path.join(
        _BASE_DIR,
        f"s{S}",
        f"l{L}_m{M_MODE}_n{N_OVT}",
        f"a{A:.4g}",
        f"omega_{omega.real:.5f}{omega.imag:+.5f}i",
        f"r{RMIN:.4g}_to_{RMAX:.4g}",
    )
    os.makedirs(out_dir, exist_ok=True)

    factor = compute_r3_exp_factor(r, omega)
    R_divided = R / factor*scale_factor
    print(f"|R/factor| range: [{np.abs(R_divided).min():.4e}, {np.abs(R_divided).max():.4e}]")

    make_plots(r, R_divided, omega, out_dir)
    print(R_divided[:50])
    print("Done.")
