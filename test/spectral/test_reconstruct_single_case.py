from __future__ import annotations

import os
import sys
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, os.getcwd())

from model.chebyshev_trunk import ChebyshevTrunk, coeffs_from_re_im
from dataset.grids import map_y_to_x


def main():
    ckpt_path = "outputs/single_case_coeffs.pt"
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")

    a0 = ckpt["a"]
    omega0 = ckpt["omega"]
    order = ckpt["order"]

    coeff_re = ckpt["coeff_re"]   # shape (1, Nc)
    coeff_im = ckpt["coeff_im"]   # shape (1, Nc)

    print(f"a = {a0}")
    print(f"omega = {omega0}")
    print(f"order = {order}")
    print(f"coeff_re shape = {tuple(coeff_re.shape)}")
    print(f"coeff_im shape = {tuple(coeff_im.shape)}")

    # -------------------------------------------------
    # 1. 构造 trunk
    # -------------------------------------------------
    trunk = ChebyshevTrunk(order=order)

    # -------------------------------------------------
    # 2. 合成 complex 系数
    # -------------------------------------------------
    coeff = coeffs_from_re_im(coeff_re, coeff_im)   # (1, Nc)

    # -------------------------------------------------
    # 3. 在稠密 y 网格上重建 f(y)
    #    这里先直接取 [-1,1]，画图没问题
    # -------------------------------------------------
    Ny_plot = 1000
    y_plot = torch.linspace(-1.0, 1.0, Ny_plot, dtype=torch.float64)

    f_plot = trunk.evaluate(coeff, y_plot, method="clenshaw")   # (1, Ny) or (Ny,)
    if f_plot.ndim == 2:
        f_plot = f_plot[0]

    # -------------------------------------------------
    # 4. 由 f(y) 还原 R'(x(y))
    #    R'(x(y)) = (exp(x-1)-1) * f(y) + 1
    # -------------------------------------------------
    x_plot = map_y_to_x(y_plot)
    g_plot = torch.exp(x_plot - 1.0) - 1.0
    Rp_plot = g_plot.to(dtype=f_plot.dtype) * f_plot + 1.0

    # -------------------------------------------------
    # 5. 系数模长
    # -------------------------------------------------
    coeff_abs = torch.abs(coeff[0]).detach().cpu()
    k = torch.arange(coeff_abs.numel())

    # -------------------------------------------------
    # 6. 打印一些数值范围
    # -------------------------------------------------
    print("\n=== Reconstruction summary ===")
    print("max |f|   =", torch.max(torch.abs(f_plot)).item())
    print("max |R'|  =", torch.max(torch.abs(Rp_plot)).item())
    print("min |R'|  =", torch.min(torch.abs(Rp_plot)).item())
    print("R'(y=1)   =", Rp_plot[-1].item())
    print("R'(y=-1)  =", Rp_plot[0].item())
    print("max |coeff| =", torch.max(coeff_abs).item())
    print("==============================\n")

    # -------------------------------------------------
    # 7. 画图
    # -------------------------------------------------
    os.makedirs("outputs", exist_ok=True)

    y_np = y_plot.detach().cpu().numpy()
    x_np = x_plot.detach().cpu().numpy()

    f_re = f_plot.real.detach().cpu().numpy()
    f_im = f_plot.imag.detach().cpu().numpy()
    f_abs = torch.abs(f_plot).detach().cpu().numpy()

    Rp_re = Rp_plot.real.detach().cpu().numpy()
    Rp_im = Rp_plot.imag.detach().cpu().numpy()
    Rp_abs = torch.abs(Rp_plot).detach().cpu().numpy()

    # 图 1: f(y)
    plt.figure(figsize=(8, 5))
    plt.plot(y_np, f_re, label="Re f(y)")
    plt.plot(y_np, f_im, label="Im f(y)")
    plt.plot(y_np, f_abs, label="|f(y)|")
    plt.xlabel("y")
    plt.ylabel("f")
    plt.title("Reconstruction of f(y)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/reconstruct_f_vs_y.png", dpi=200)
    plt.close()

    # 图 2: R'(y)
    plt.figure(figsize=(8, 5))
    plt.plot(y_np, Rp_re, label="Re R'(y)")
    plt.plot(y_np, Rp_im, label="Im R'(y)")
    plt.plot(y_np, Rp_abs, label="|R'(y)|")
    plt.xlabel("y")
    plt.ylabel("R'")
    plt.title("Reconstruction of R'(y)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/reconstruct_Rp_vs_y.png", dpi=200)
    plt.close()

    # 图 3: R'(x)
    plt.figure(figsize=(8, 5))
    plt.plot(x_np, Rp_re, label="Re R'(x)")
    plt.plot(x_np, Rp_im, label="Im R'(x)")
    plt.plot(x_np, Rp_abs, label="|R'(x)|")
    plt.xlabel("x")
    plt.ylabel("R'")
    plt.title("Reconstruction of R'(x)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/reconstruct_Rp_vs_x.png", dpi=200)
    plt.close()

    # 图 4: |coeff_k|
    plt.figure(figsize=(8, 5))
    plt.semilogy(k.numpy(), coeff_abs.numpy(), marker="o")
    plt.xlabel("k")
    plt.ylabel("|b_k|")
    plt.title("Magnitude of spectral coefficients")
    plt.tight_layout()
    plt.savefig("outputs/reconstruct_coeff_abs.png", dpi=200)
    plt.close()

    print("Saved:")
    print("  outputs/reconstruct_f_vs_y.png")
    print("  outputs/reconstruct_Rp_vs_y.png")
    print("  outputs/reconstruct_Rp_vs_x.png")
    print("  outputs/reconstruct_coeff_abs.png")


if __name__ == "__main__":
    main()