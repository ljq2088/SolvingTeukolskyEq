"""Chebyshev 节点生成与微分矩阵 D,D2 的预计算与缓存"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class ChebyshevGrid:
    """
    Chebyshev--Gauss--Lobatto 网格与谱微分对象
    """
    order: int
    y_nodes: torch.Tensor   # (N+1,)
    D: torch.Tensor         # (N+1, N+1), d/dy
    D2: torch.Tensor        # (N+1, N+1), d²/dy²
    Tmat: torch.Tensor      # (N+1, N+1), T_k(y_j)


def chebyshev_lobatto_nodes(
    order: int,
    *,
    dtype: torch.dtype = torch.float64,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Chebyshev--Gauss--Lobatto 节点
    y_j = cos(pi * j / N),  j=0,...,N

    返回:
        y_nodes: shape (N+1,)
    """
    if order < 1:
        raise ValueError(f"order must be >= 1, got {order}")

    j = torch.arange(order + 1, dtype=dtype, device=device)
    y_nodes = torch.cos(torch.pi * j / order)
    return y_nodes


def chebyshev_diff_matrix_from_nodes(y_nodes: torch.Tensor) -> torch.Tensor:
    """
    由 Chebyshev--Gauss--Lobatto 节点构造一阶谱微分矩阵 D

    对于节点 y_i, y_j，定义:
      c_0 = c_N = 2, 其余 c_j = 1

    非对角元:
      D_ij = (c_i / c_j) * (-1)^(i+j) / (y_i - y_j),   i != j

    对角元:
      D_ii = -y_i / (2 * (1 - y_i^2)),                 1 <= i <= N-1
      D_00 = (2N^2 + 1) / 6
      D_NN = -(2N^2 + 1) / 6
    """
    if y_nodes.ndim != 1:
        raise ValueError(f"y_nodes must be 1D, got shape {tuple(y_nodes.shape)}")

    N = y_nodes.numel() - 1
    if N < 1:
        raise ValueError("Need at least 2 nodes to build differentiation matrix.")

    dtype = y_nodes.dtype
    device = y_nodes.device

    c = torch.ones(N + 1, dtype=dtype, device=device)
    c[0] = 2.0
    c[-1] = 2.0

    ii = torch.arange(N + 1, device=device)
    sign = (-1.0) ** (ii[:, None] + ii[None, :])

    Y = y_nodes[:, None]
    dY = Y - Y.T

    # 先构造非对角元
    D = torch.zeros((N + 1, N + 1), dtype=dtype, device=device)
    mask_offdiag = ~torch.eye(N + 1, dtype=torch.bool, device=device)

    C_ratio = (c[:, None] / c[None, :]) * sign
    D[mask_offdiag] = (C_ratio[mask_offdiag] / dY[mask_offdiag])

    # 再填对角元
    if N >= 2:
        y_interior = y_nodes[1:-1]
        D[1:-1, 1:-1].diagonal().copy_(
            -y_interior / (2.0 * (1.0 - y_interior * y_interior))
        )

    D[0, 0] = (2.0 * N * N + 1.0) / 6.0
    D[-1, -1] = -(2.0 * N * N + 1.0) / 6.0

    return D


def chebyshev_diff2_matrix(D: torch.Tensor) -> torch.Tensor:
    """
    二阶谱微分矩阵 D2 = D @ D
    """
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError(f"D must be square 2D matrix, got shape {tuple(D.shape)}")
    return D @ D


def chebyshev_basis_matrix(
    y: torch.Tensor,
    order: int,
) -> torch.Tensor:
    """
    构造基底矩阵 Tmat，其中
      Tmat[j, k] = T_k(y_j)

    输入:
      y: shape (Ny,)
      order: 最高阶 N

    返回:
      Tmat: shape (Ny, N+1)
    """
    if y.ndim != 1:
        raise ValueError(f"y must be 1D, got shape {tuple(y.shape)}")
    if order < 0:
        raise ValueError(f"order must be >= 0, got {order}")

    Ny = y.numel()
    dtype = y.dtype
    device = y.device

    Tmat = torch.empty((Ny, order + 1), dtype=dtype, device=device)

    # T_0 = 1
    Tmat[:, 0] = 1.0

    if order >= 1:
        # T_1 = y
        Tmat[:, 1] = y

    # 三项递推:
    # T_{k+1}(y) = 2 y T_k(y) - T_{k-1}(y)
    for k in range(1, order):
        Tmat[:, k + 1] = 2.0 * y * Tmat[:, k] - Tmat[:, k - 1]

    return Tmat


def chebyshev_grid_bundle(
    order: int,
    *,
    dtype: torch.dtype = torch.float64,
    device: Optional[torch.device] = None,
) -> ChebyshevGrid:
    """
    一次性返回 Chebyshev 谱方法所需对象:
      y_nodes, D, D2, Tmat
    """
    y_nodes = chebyshev_lobatto_nodes(order, dtype=dtype, device=device)
    D = chebyshev_diff_matrix_from_nodes(y_nodes)
    D2 = chebyshev_diff2_matrix(D)
    Tmat = chebyshev_basis_matrix(y_nodes, order)

    return ChebyshevGrid(
        order=order,
        y_nodes=y_nodes,
        D=D,
        D2=D2,
        Tmat=Tmat,
    )


def map_y_to_x(y: torch.Tensor) -> torch.Tensor:
    """
    y in [-1,1] -> x in [0,1]
    x = (y + 1) / 2
    """
    return 0.5 * (y + 1.0)


def map_x_to_y(x: torch.Tensor) -> torch.Tensor:
    """
    x in [0,1] -> y in [-1,1]
    y = 2x - 1
    """
    return 2.0 * x - 1.0


def map_r_to_x(r: torch.Tensor, r_plus: torch.Tensor | float) -> torch.Tensor:
    """
    x = r_plus / r
    """
    return r_plus / r


def map_r_to_y(r: torch.Tensor, r_plus: torch.Tensor | float) -> torch.Tensor:
    """
    r -> x -> y
    y = 2 * (r_plus / r) - 1
    """
    x = map_r_to_x(r, r_plus)
    return map_x_to_y(x)