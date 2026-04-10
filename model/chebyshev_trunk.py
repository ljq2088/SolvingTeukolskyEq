"""trunk网络：计算 T_n(ξ(x)) 基底函数，并从系数合成 f(x)"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from dataset.grids import chebyshev_basis_matrix, map_r_to_y


def coeffs_from_re_im(
    coeff_re: torch.Tensor,
    coeff_im: torch.Tensor,
) -> torch.Tensor:
    """
    将实部/虚部系数组合成 complex 系数

    输入:
      coeff_re, coeff_im:
        shape 可为 (Nc,) 或 (B,Nc)

    返回:
      coeff: complex tensor, same shape
    """
    if coeff_re.shape != coeff_im.shape:
        raise ValueError(
            f"coeff_re and coeff_im must have same shape, got "
            f"{tuple(coeff_re.shape)} vs {tuple(coeff_im.shape)}"
        )
    return torch.complex(coeff_re, coeff_im)


def reconstruct_from_tmat(
    coeff: torch.Tensor,
    Tmat: torch.Tensor,
) -> torch.Tensor:
    """
    用基底矩阵 Tmat 重建函数值

    数学形式:
      f(y_j) = Σ_k coeff_k * T_k(y_j)

    输入:
      coeff:
        - (Nc,)      单样本
        - (B, Nc)    batch
      Tmat:
        - (Ny, Nc)

    返回:
      f:
        - (Ny,)      若 coeff 是 1D
        - (B, Ny)    若 coeff 是 2D
    """
    if Tmat.ndim != 2:
        raise ValueError(f"Tmat must be 2D, got shape {tuple(Tmat.shape)}")

    Nc = Tmat.shape[1]

    if coeff.ndim == 1:
        if coeff.shape[0] != Nc:
            raise ValueError(
                f"coeff has wrong size: expected {Nc}, got {coeff.shape[0]}"
            )
        return Tmat.to(dtype=coeff.dtype, device=coeff.device) @ coeff

    if coeff.ndim == 2:
        if coeff.shape[1] != Nc:
            raise ValueError(
                f"coeff has wrong size: expected second dim {Nc}, got {coeff.shape[1]}"
            )
        return coeff @ Tmat.to(dtype=coeff.dtype, device=coeff.device).T

    raise ValueError(f"coeff must be 1D or 2D, got shape {tuple(coeff.shape)}")


def clenshaw_evaluate(
    coeff: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    """
    用 Clenshaw recurrence 在任意 y 上计算
      f(y) = Σ_k coeff_k T_k(y)

    输入:
      coeff:
        - (Nc,)
        - (B, Nc)
      y:
        - scalar tensor
        - (Ny,)

    返回:
      若 coeff 是 (Nc,):
        - scalar 或 (Ny,)
      若 coeff 是 (B,Nc):
        - (B,) 或 (B,Ny)
    """
    if coeff.ndim not in (1, 2):
        raise ValueError(f"coeff must be 1D or 2D, got shape {tuple(coeff.shape)}")

    coeff_was_1d = (coeff.ndim == 1)
    if coeff_was_1d:
        coeff_ = coeff.unsqueeze(0)   # (1, Nc)
    else:
        coeff_ = coeff                # (B, Nc)

    y_was_scalar = (y.ndim == 0)
    y_ = y.reshape(-1)

    B, Nc = coeff_.shape
    device = coeff_.device
    dtype = coeff_.dtype

    x = y_.to(device=device, dtype=dtype)              # (Ny,)
    x = x.unsqueeze(0)                                 # (1, Ny), for broadcast

    b_kplus1 = torch.zeros((B, x.shape[1]), dtype=dtype, device=device)
    b_kplus2 = torch.zeros((B, x.shape[1]), dtype=dtype, device=device)

    # Clenshaw: for k = N,...,1
    for k in range(Nc - 1, 0, -1):
        c_k = coeff_[:, k:k+1]                         # (B,1)
        b_k = 2.0 * x * b_kplus1 - b_kplus2 + c_k
        b_kplus2 = b_kplus1
        b_kplus1 = b_k

    out = x * b_kplus1 - b_kplus2 + coeff_[:, 0:1]    # (B, Ny)

    if coeff_was_1d:
        out = out[0]                                  # (Ny,)

    if y_was_scalar:
        out = out[..., 0]                             # scalar or (B,)

    return out


class ChebyshevTrunk(nn.Module):
    """
    解析 trunk（不是学习网络）

    职责：
      1. 在给定 y 节点上生成 T_k(y) 基底矩阵
      2. 用 coeff 重建训练节点上的 f(y_j)
      3. 用 Clenshaw 在任意 y 上评估 f(y)
      4. 通过 r -> y 的映射，在任意 r 上评估 f
    """

    def __init__(self, order: int) -> None:
        super().__init__()
        if order < 0:
            raise ValueError(f"order must be >= 0, got {order}")
        self.order = int(order)

    def basis_matrix(self, y: torch.Tensor) -> torch.Tensor:
        """
        返回 Tmat[j,k] = T_k(y_j)

        输入:
          y: (Ny,)

        返回:
          Tmat: (Ny, order+1)
        """
        return chebyshev_basis_matrix(y, self.order)

    def reconstruct(
        self,
        coeff: torch.Tensor,
        y_nodes: Optional[torch.Tensor] = None,
        Tmat: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        在训练/离散节点上重建函数值

        二选一提供：
          - y_nodes
          - Tmat

        输入:
          coeff:
            (Nc,) 或 (B,Nc)

        返回:
          f_nodes:
            (Ny,) 或 (B,Ny)
        """
        if Tmat is None:
            if y_nodes is None:
                raise ValueError("Need either y_nodes or Tmat.")
            Tmat = self.basis_matrix(y_nodes)

        return reconstruct_from_tmat(coeff, Tmat)

    def evaluate(
        self,
        coeff: torch.Tensor,
        y: torch.Tensor,
        *,
        method: str = "clenshaw",
    ) -> torch.Tensor:
        """
        在任意 y 上评估 f(y)

        method:
          - "clenshaw" : 推荐，适合任意点查询
          - "tmat"     : 用显式基底矩阵（适合少量点或调试）
        """
        if method == "clenshaw":
            return clenshaw_evaluate(coeff, y)

        if method == "tmat":
            if y.ndim == 0:
                y_ = y.reshape(1)
                out = reconstruct_from_tmat(coeff, self.basis_matrix(y_))
                return out[..., 0]
            return reconstruct_from_tmat(coeff, self.basis_matrix(y))

        raise ValueError(f"Unknown method={method}")

    def evaluate_from_re_im(
        self,
        coeff_re: torch.Tensor,
        coeff_im: torch.Tensor,
        y: torch.Tensor,
        *,
        method: str = "clenshaw",
    ) -> torch.Tensor:
        """
        由实部/虚部系数在任意 y 上评估
        """
        coeff = coeffs_from_re_im(coeff_re, coeff_im)
        return self.evaluate(coeff, y, method=method)

    def evaluate_at_r(
        self,
        coeff: torch.Tensor,
        r: torch.Tensor,
        r_plus: torch.Tensor | float,
        *,
        method: str = "clenshaw",
    ) -> torch.Tensor:
        """
        在任意 r 上评估 f

        通过:
          r -> x = r_plus/r -> y = 2x - 1
        """
        y = map_r_to_y(r, r_plus)
        return self.evaluate(coeff, y, method=method)

    def forward(
        self,
        coeff: torch.Tensor,
        y: torch.Tensor,
        *,
        method: str = "clenshaw",
    ) -> torch.Tensor:
        """
        nn.Module 风格接口
        """
        return self.evaluate(coeff, y, method=method)