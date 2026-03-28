"""物理残差计算：r = L[R] 或 L[f]，将 f 的节点值与导数代入系数，返回标量 loss 与诊断量"""
# physical_ansatz/residual.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple, Any, Optional

import torch

from utils.compute_lambda import compute_lambda
from utils.amplitude_ratio import compute_amplitude_ratio

@dataclass
class AuxCache:
    """
    缓存外部求解器结果，避免训练中重复调用 GremLinEqRe / kerr_matcher。
    key 用 (a, omega, l, m, s, extra...) 的 float 量化值。
    """
    lambda_cache: Dict[Tuple, complex] = field(default_factory=dict)
    ramp_cache: Dict[Tuple, complex] = field(default_factory=dict)

def _to_pyfloat(x: torch.Tensor) -> float:
    return float(x.detach().cpu().item())

def _to_pycomplex(x: torch.Tensor) -> complex:
    # 允许 complex tensor 或 real tensor
    if torch.is_complex(x):
        v = x.detach().cpu().item()
        return complex(v)
    return complex(float(x.detach().cpu().item()), 0.0)

def get_lambda_from_cfg(
    cfg: Dict[str, Any],
    cache: AuxCache,
    a: torch.Tensor,
    omega: torch.Tensor,
) -> torch.Tensor:
    prob = cfg["problem"]
    mode = prob["lambda"]["mode"]
    s = int(prob.get("s", -2))
    l = int(prob["l"])
    m = int(prob["m"])

    if mode == "given":
        val = prob["lambda"]["value"]
        if val is None:
            raise ValueError("lambda.mode=given 但 lambda.value=null")
        lam = complex(val)
        return torch.tensor(lam, dtype=torch.complex128, device=a.device)

    if mode == "compute":
        aa = _to_pyfloat(a)
        ww = _to_pycomplex(omega)
        if abs(ww.imag) > 0:
            raise ValueError("当前 compute_lambda 接口只接受实频 omega；请先把 omega.imag 固定为 0 或改造外部接口。")

        key = ("lambda", round(aa, 12), round(ww.real, 12), l, m, s)
        if key not in cache.lambda_cache:
            lam_val = compute_lambda(aa, ww.real, l, m, s)
            cache.lambda_cache[key] = complex(lam_val)
        lam = cache.lambda_cache[key]
        return torch.tensor(lam, dtype=torch.complex128, device=a.device)

    raise ValueError(f"Unknown lambda.mode={mode}")

def get_ramp_and_p_from_cfg(
    cfg: Dict[str, Any],
    cache: AuxCache,
    a: torch.Tensor,
    omega: torch.Tensor,
) -> Tuple[int, torch.Tensor]:
    prob = cfg["problem"]
    s = int(prob.get("s", -2))
    l = int(prob["l"])
    m = int(prob["m"])

    ramp_cfg = prob.get("R_amp", {"mode": "off"})
    mode = ramp_cfg.get("mode", "off")

    if mode == "off":
        # 不启用反射项：R_amp=0，p 仍给个默认（不会被用到）
        return 0, torch.tensor(0.0 + 0.0j, dtype=torch.complex128, device=a.device)

    if mode == "given":
        val = ramp_cfg["given"]["value"]
        if val is None:
            raise ValueError("R_amp.mode=given 但 R_amp.given.value=null")
        p = int(ramp_cfg.get("compute", {}).get("p", 0))
        return p, torch.tensor(complex(val), dtype=torch.complex128, device=a.device)

    if mode == "compute":
        aa = _to_pyfloat(a)
        ww = _to_pycomplex(omega)
        if abs(ww.imag) > 0:
            raise ValueError("当前 compute_amplitude_ratio 接口只接受实频 omega。")

        ccfg = ramp_cfg["compute"]
        p = int(ccfg["p"])
        r_match = float(ccfg.get("r_match", 8.0))
        n_cheb = int(ccfg.get("n_cheb", 32))

        # 尝试从 cache 获取已计算的 lambda，避免重复计算
        lambda_key = ("lambda", round(aa, 12), round(ww.real, 12), l, m, s)
        lambda_sep = cache.lambda_cache.get(lambda_key, None)

        key = ("ramp", round(aa, 12), round(ww.real, 12), l, m, s, p, r_match, n_cheb)
        if key not in cache.ramp_cache:
            out = compute_amplitude_ratio(aa, ww.real, l, m, lambda_sep=lambda_sep, r_match=r_match, n_cheb=n_cheb, s=s)
            cache.ramp_cache[key] = complex(out["ratio"])
            # 同时缓存 lambda（如果之前没有）
            if lambda_key not in cache.lambda_cache:
                cache.lambda_cache[lambda_key] = complex(out["lambda"])
        ramp = cache.ramp_cache[key]
        return p, torch.tensor(ramp, dtype=torch.complex128, device=a.device)

    raise ValueError(f"Unknown R_amp.mode={mode}")



from physical_ansatz.teukolsky_coeffs import coeffs_x
from physical_ansatz.transform_y import transform_coeffs_x_to_y

def residual_from_nodes(
    y_nodes: torch.Tensor,   # (Ny,) or (Ny,1), real in [-1,1]
    f: torch.Tensor,         # (B,Ny) complex - this is f(y)
    fy: torch.Tensor,        # (B,Ny) complex - df/dy
    fyy: torch.Tensor,       # (B,Ny) complex - d²f/dy²
    a_batch: torch.Tensor,   # (B,) real
    omega_batch: torch.Tensor,# (B,) real or complex
    lambda_batch: torch.Tensor,# (B,) complex
    p: int,
    ramp_batch: torch.Tensor,# (B,) complex
    cfg: Dict[str, Any],
) -> torch.Tensor:
    """
    计算 residual: B2*f_yy + B1*f_y + B0*f = rhs

    坐标变换：x = (y+1)/2, y ∈ [-1,1] → x ∈ [0,1]
    函数变换：R'(x) = (exp(x-1)-1)*f(x) + 1

    返回 (B,Ny) complex
    """
    prob = cfg["problem"]
    m = int(prob["m"])
    s = int(prob.get("s", -2))
    M = float(prob.get("M", 1.0))

    # 转换 y → x
    if y_nodes.ndim == 1:
        y = y_nodes[None, :, None]          # (1,Ny,1)
    else:
        y = y_nodes[None, :, :]             # (1,Ny,1)

    x = (y + 1.0) / 2.0                     # x ∈ [0,1]

    B = a_batch.shape[0]
    x = x.expand(B, -1, -1)                 # (B,Ny,1)
    y = y.expand(B, -1, -1)                 # (B,Ny,1)

    a = a_batch.reshape(B, 1, 1)
    omega = omega_batch.reshape(B, 1, 1)
    lam = lambda_batch.reshape(B, 1, 1)
    ramp = ramp_batch.reshape(B, 1, 1)

    # 先计算 x 坐标下的系数 A2, A1, A0
    A2, A1, A0 = coeffs_x(x, a, omega, m, p, ramp, lam, s=s, M=M)

    # 变换到 y 坐标下的系数 B2, B1, B0
    B2, B1, B0, rhs = transform_coeffs_x_to_y(A2, A1, A0, y)

    # f,fy,fyy: (B,Ny) -> (B,Ny,1)
    f_   = f.unsqueeze(-1)
    fy_  = fy.unsqueeze(-1)
    fyy_ = fyy.unsqueeze(-1)

    res = B2*fyy_ + B1*fy_ + B0*f_ - rhs
    return res.squeeze(-1)   # (B,Ny)

def complex_mse(z: torch.Tensor) -> torch.Tensor:
    return torch.mean(z.real*z.real + z.imag*z.imag)

def teukolsky_residual_loss_coeff(
    cfg,
    y_nodes,              # (Ny,)
    D, D2,                # (Ny,Ny), wrt y
    Tmat,                 # (Ny,Nc), T_k(y_j)
    coeff_re, coeff_im,   # (B,Nc)
    a_batch,              # (B,)
    omega_batch,          # (B,)
    lambda_batch,         # (B,) complex
    ramp_batch,           # (B,) complex
    p: int,
    exclude_endpoints: bool = True,
):
    # 1) build complex coeffs
    coeff = torch.complex(coeff_re, coeff_im)   # (B,Nc)

    # 2) reconstruct f(y)
    f = coeff @ Tmat.T                          # (B,Ny)

    # 3) spectral derivatives wrt y
    fy  = f @ D.T
    fyy = f @ D2.T

    # 4) residual in f-equation
    res = residual_from_nodes(
        y_nodes=y_nodes,
        f=f,
        fy=fy,
        fyy=fyy,
        a_batch=a_batch,
        omega_batch=omega_batch,
        lambda_batch=lambda_batch,
        p=p,
        ramp_batch=ramp_batch,
        cfg=cfg,
    )

    # 5) interior only
    if exclude_endpoints and res.shape[1] > 2:
        res_used = res[:, 1:-1]
    else:
        res_used = res

    loss = torch.mean(res_used.real**2 + res_used.imag**2)

    diag = {
        "loss_res": float(loss.detach().cpu()),
        "res_abs_max": float(torch.abs(res_used).max().detach().cpu()),
        "f_abs_max": float(torch.abs(f).max().detach().cpu()),
    }
    return loss, diag