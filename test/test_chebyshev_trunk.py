import sys
sys.path.append("/home/ljq/code/PINN/SolvingTeukolsky")
import torch

from dataset.grids import chebyshev_grid_bundle
from model.chebyshev_trunk import ChebyshevTrunk, coeffs_from_re_im


def main():
    order = 8
    grid = chebyshev_grid_bundle(order=order, dtype=torch.float64)
    trunk = ChebyshevTrunk(order=order)

    y_nodes = grid.y_nodes
    Tmat = grid.Tmat

    # 一组复系数
    coeff_re = torch.zeros(order + 1, dtype=torch.float64)
    coeff_im = torch.zeros(order + 1, dtype=torch.float64)

    coeff_re[0] = 1.0
    coeff_re[1] = 0.2
    coeff_re[2] = -0.4
    coeff_re[3] = 0.1

    coeff_im[1] = 0.1
    coeff_im[2] = 0.3
    coeff_im[3] = -0.2

    coeff = coeffs_from_re_im(coeff_re, coeff_im)

    # 1) 用训练节点基底矩阵重建
    f_tmat = trunk.reconstruct(coeff, Tmat=Tmat)

    # 2) 用 Clenshaw 在同样节点上评估
    f_clenshaw = trunk.evaluate(coeff, y_nodes, method="clenshaw")

    err = torch.max(torch.abs(f_tmat - f_clenshaw)).item()

    print("max |f_tmat - f_clenshaw| =", err)


if __name__ == "__main__":
    main()