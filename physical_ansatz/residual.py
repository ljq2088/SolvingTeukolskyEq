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

def residual_from_nodes(
    x_nodes: torch.Tensor,   # (Nx,) or (Nx,1), real
    R: torch.Tensor,         # (B,Nx) complex - this is R'(x), the regular part
    Rx: torch.Tensor,        # (B,Nx) complex - dR'/dx
    Rxx: torch.Tensor,       # (B,Nx) complex - d²R'/dx²
    a_batch: torch.Tensor,   # (B,) real
    omega_batch: torch.Tensor,# (B,) real or complex
    lambda_batch: torch.Tensor,# (B,) complex
    p: int,
    ramp_batch: torch.Tensor,# (B,) complex
    cfg: Dict[str, Any],
) -> torch.Tensor:
    """
    计算 residual: A2*R'_xx + A1*R'_x + A0*R' = 0

    注意：这里的 R 实际上是 R'(x)，即 R(r) = R'(x)*U(r) 中的 regular 部分

    返回 (B,Nx) complex
    """
    prob = cfg["problem"]
    m = int(prob["m"])
    s = int(prob.get("s", -2))
    M = float(prob.get("M", 1.0))

    # reshape for broadcast
    if x_nodes.ndim == 1:
        x = x_nodes[None, :, None]          # (1,Nx,1)
    else:
        x = x_nodes[None, :, :]             # (1,Nx,1)

    B = a_batch.shape[0]
    x = x.expand(B, -1, -1)                 # (B,Nx,1)

    a = a_batch.reshape(B, 1, 1)
    omega = omega_batch.reshape(B, 1, 1)
    lam = lambda_batch.reshape(B, 1, 1)
    ramp = ramp_batch.reshape(B, 1, 1)

    # coeffs_x 返回 (B,Nx,1) 形状（内部广播）
    A2, A1, A0 = coeffs_x(x, a, omega, m, p, ramp, lam, s=s, M=M)

    # R,Rx,Rxx: (B,Nx) -> (B,Nx,1)
    R_  = R.unsqueeze(-1)
    Rx_ = Rx.unsqueeze(-1)
    Rxx_= Rxx.unsqueeze(-1)

    res = A2*Rxx_ + A1*Rx_ + A0*R_
    return res.squeeze(-1)   # (B,Nx)

def complex_mse(z: torch.Tensor) -> torch.Tensor:
    return torch.mean(z.real*z.real + z.imag*z.imag)

def teukolsky_residual_loss(
    cfg: Dict[str, Any],
    cache: AuxCache,
    x_nodes: torch.Tensor,     # (Nx,) or (Nx,1)
    D: torch.Tensor,           # (Nx,Nx) real
    D2: torch.Tensor,          # (Nx,Nx) real
    R_nodes: torch.Tensor,     # (B,Nx) complex
    a_batch: torch.Tensor,     # (B,) real
    omega_batch: torch.Tensor, # (B,) real or complex
    exclude_endpoints: bool = True,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    返回:
      loss (scalar)
      diagnostics dict
    """
    # 1) 先算 Rx, Rxx
    Rx  = R_nodes @ D.T
    Rxx = R_nodes @ D2.T

    # 2) 逐样本算 lambda 与 R_amp（不可导，作为常量进入）
    B = a_batch.shape[0]
    lam_list = []
    p_val: Optional[int] = None
    ramp_list = []

    for i in range(B):
        lam_i = get_lambda_from_cfg(cfg, cache, a_batch[i], omega_batch[i])
        p_i, ramp_i = get_ramp_and_p_from_cfg(cfg, cache, a_batch[i], omega_batch[i])
        lam_list.append(lam_i)
        ramp_list.append(ramp_i)
        if p_val is None:
            p_val = p_i
        else:
            # p 必须一致（同一个训练任务）
            if p_i != p_val:
                raise ValueError(f"Inconsistent p across batch: {p_val} vs {p_i}")

    lambda_batch = torch.stack(lam_list).to(dtype=torch.complex128)
    ramp_batch   = torch.stack(ramp_list).to(dtype=torch.complex128)
    p = int(p_val or 0)

    # 3) residual
    res = residual_from_nodes(
        x_nodes=x_nodes,
        R=R_nodes,
        Rx=Rx,
        Rxx=Rxx,
        a_batch=a_batch,
        omega_batch=omega_batch,
        lambda_batch=lambda_batch,
        p=p,
        ramp_batch=ramp_batch,
        cfg=cfg,
    )

    # 4) 端点处理：默认只在内点算 loss（避免 x=1 视界处 Δ=0 引发数值问题）
    if exclude_endpoints and res.shape[1] > 2:
        res_used = res[:, 1:-1]
    else:
        res_used = res

    loss = complex_mse(res_used)

    diag = {
        "loss_res": float(loss.detach().cpu().item()),
        "res_abs_max": float(torch.max(torch.abs(res_used)).detach().cpu().item()),
    }
    return loss, diag