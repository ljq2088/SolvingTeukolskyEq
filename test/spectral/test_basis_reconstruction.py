import torch
import sys
sys.path.append("/home/ljq/code/PINN/SolvingTeukolsky")
from dataset.grids import chebyshev_grid_bundle


def test_real_reconstruction():
    """
    测试 1：实系数重建
    取
        f(y) = 1 + 0.2 T1(y) - 0.4 T2(y) + 0.1 T3(y)
    检查 Tmat @ coeff 是否与显式表达一致
    """
    order = 8
    grid = chebyshev_grid_bundle(order=order, dtype=torch.float64)

    y = grid.y_nodes          # (N+1,)
    Tmat = grid.Tmat          # (N+1, N+1)

    # 系数 b0, b1, b2, b3, ... , bN
    coeff = torch.zeros(order + 1, dtype=torch.float64)
    coeff[0] = 1.0
    coeff[1] = 0.2
    coeff[2] = -0.4
    coeff[3] = 0.1

    # 用基底矩阵重建
    f_num = Tmat @ coeff

    # 显式写出 T0,T1,T2,T3
    T0 = torch.ones_like(y)
    T1 = y
    T2 = 2.0 * y**2 - 1.0
    T3 = 4.0 * y**3 - 3.0 * y

    f_true = 1.0 * T0 + 0.2 * T1 - 0.4 * T2 + 0.1 * T3

    err = torch.max(torch.abs(f_num - f_true)).item()
    print("[real reconstruction]")
    print("max |f_num - f_true| =", err)


def test_complex_reconstruction():
    """
    测试 2：复系数重建
    检查 Tmat 对 complex 系数也能正常工作
    """
    order = 8
    grid = chebyshev_grid_bundle(order=order, dtype=torch.float64)

    y = grid.y_nodes
    Tmat = grid.Tmat.to(torch.complex128)

    coeff = torch.zeros(order + 1, dtype=torch.complex128)
    coeff[0] = 1.0 + 0.0j
    coeff[1] = 0.2 + 0.1j
    coeff[2] = -0.4 + 0.3j
    coeff[3] = 0.1 - 0.2j

    f_num = Tmat @ coeff

    T0 = torch.ones_like(y, dtype=torch.complex128)
    T1 = y.to(torch.complex128)
    T2 = 2.0 * T1**2 - 1.0
    T3 = 4.0 * T1**3 - 3.0 * T1

    f_true = coeff[0] * T0 + coeff[1] * T1 + coeff[2] * T2 + coeff[3] * T3

    err = torch.max(torch.abs(f_num - f_true)).item()
    print("[complex reconstruction]")
    print("max |f_num - f_true| =", err)


def test_batch_right_multiply():
    """
    测试 3：模拟后面训练时的 batch 写法
        f = coeff_batch @ Tmat.T
    """
    order = 8
    grid = chebyshev_grid_bundle(order=order, dtype=torch.float64)

    y = grid.y_nodes
    Tmat = grid.Tmat.to(torch.complex128)

    coeff_batch = torch.zeros((2, order + 1), dtype=torch.complex128)

    # sample 0
    coeff_batch[0, 0] = 1.0 + 0.0j
    coeff_batch[0, 1] = 0.2 + 0.1j
    coeff_batch[0, 2] = -0.4 + 0.3j
    coeff_batch[0, 3] = 0.1 - 0.2j

    # sample 1
    coeff_batch[1, 0] = -0.5 + 0.2j
    coeff_batch[1, 1] = 0.3 - 0.4j
    coeff_batch[1, 2] = 0.7 + 0.0j
    coeff_batch[1, 4] = -0.2 + 0.5j

    # 后面训练里会用这个写法
    f_batch = coeff_batch @ Tmat.T   # (B, Ny)

    # 用逐项显式求和做真值
    y_c = y.to(torch.complex128)
    T0 = torch.ones_like(y_c)
    T1 = y_c
    T2 = 2.0 * y_c**2 - 1.0
    T3 = 4.0 * y_c**3 - 3.0 * y_c
    T4 = 8.0 * y_c**4 - 8.0 * y_c**2 + 1.0

    f0_true = (
        coeff_batch[0, 0] * T0
        + coeff_batch[0, 1] * T1
        + coeff_batch[0, 2] * T2
        + coeff_batch[0, 3] * T3
    )

    f1_true = (
        coeff_batch[1, 0] * T0
        + coeff_batch[1, 1] * T1
        + coeff_batch[1, 2] * T2
        + coeff_batch[1, 4] * T4
    )

    err0 = torch.max(torch.abs(f_batch[0] - f0_true)).item()
    err1 = torch.max(torch.abs(f_batch[1] - f1_true)).item()

    print("[batch reconstruction]")
    print("max |sample0 - true| =", err0)
    print("max |sample1 - true| =", err1)


if __name__ == "__main__":
    test_real_reconstruction()
    test_complex_reconstruction()
    test_batch_right_multiply()