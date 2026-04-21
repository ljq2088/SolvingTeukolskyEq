"""物理残差计算：r = L[R] 或 L[f]，将 f 的节点值与导数代入系数，返回标量 loss 与诊断量"""
# physical_ansatz/residual.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple, Any, Optional

import torch

from utils.amplitude import TeukRadAmplitudeIn
from utils.mode import KerrMode

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

        mode_obj = KerrMode(M=1.0, a=aa, omega=ww.real, ell=l, m=m, lam=None, s=s)
        key = ("lambda", round(mode_obj.a, 12), round(mode_obj.omega, 12), mode_obj.ell, mode_obj.m, mode_obj.s)
        if key not in cache.lambda_cache:
            cache.lambda_cache[key] = complex(mode_obj.lambda_value)
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

        mode_obj = KerrMode(
            M=1.0,
            a=aa,
            omega=ww.real,
            ell=l,
            m=m,
            lam=None,
            s=s,
        )
        lambda_key = ("lambda", round(mode_obj.a, 12), round(mode_obj.omega, 12), mode_obj.ell, mode_obj.m, mode_obj.s)
        lambda_sep = cache.lambda_cache.get(lambda_key, None)
        if lambda_sep is not None:
            mode_obj = KerrMode(
                M=mode_obj.M,
                a=mode_obj.a,
                omega=mode_obj.omega,
                ell=mode_obj.ell,
                m=mode_obj.m,
                lam=lambda_sep,
                s=mode_obj.s,
            )

        key = (
            "ramp",
            round(mode_obj.a, 12),
            round(mode_obj.omega, 12),
            mode_obj.ell,
            mode_obj.m,
            mode_obj.s,
            round(mode_obj.lambda_value.real, 12),
            round(mode_obj.lambda_value.imag, 12),
            p,
            r_match,
            n_cheb,
        )
        if key not in cache.ramp_cache:
            z_m = mode_obj.rp / r_match
            amp = TeukRadAmplitudeIn(mode_obj, N_in=n_cheb, N_out=n_cheb, z_m=z_m)
            result = amp.result
            if result.ratio_inc_over_ref is None:
                raise FloatingPointError(
                    f"TeukRadAmplitudeIn returned singular incidence/reflection ratio "
                    f"for a={aa}, omega={ww.real}, l={l}, m={m}"
                )
            cache.ramp_cache[key] = complex(result.ratio_inc_over_ref)
            cache.lambda_cache[lambda_key] = complex(mode_obj.lambda_value)
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

# def teukolsky_residual_loss_coeff(
#     cfg,
#     y_nodes,              # (Ny,)
#     D, D2,                # (Ny,Ny), wrt y
#     Tmat,                 # (Ny,Nc), T_k(y_j)
#     coeff_re, coeff_im,   # (B,Nc)
#     a_batch,              # (B,)
#     omega_batch,          # (B,)
#     lambda_batch,         # (B,) complex
#     ramp_batch,           # (B,) complex
#     p: int,
#     exclude_endpoints: bool = True,
#     n_boundary_drop: int | None = None,
# ):
#     coeff = torch.complex(coeff_re, coeff_im)   # (B,Nc)

#     # 对齐 dtype / device
#     Tmat_c = Tmat.to(dtype=coeff.dtype, device=coeff.device)
#     D_c    = D.to(dtype=coeff.dtype, device=coeff.device)
#     D2_c   = D2.to(dtype=coeff.dtype, device=coeff.device)

#     # 1) 全节点重建 f, fy, fyy
#     f_full   = coeff @ Tmat_c.T
#     fy_full  = f_full @ D_c.T
#     fyy_full = f_full @ D2_c.T

#     # 2) 决定去掉多少边界点
#     if n_boundary_drop is None:
#         n_drop = 1 if exclude_endpoints else 0
#     else:
#         n_drop = int(n_boundary_drop)

#     if n_drop < 0:
#         raise ValueError(f"n_boundary_drop must be >= 0, got {n_drop}")
#     if 2 * n_drop >= y_nodes.shape[0]:
#         raise ValueError(
#             f"Too many boundary points dropped: n_drop={n_drop}, Ny={y_nodes.shape[0]}"
#         )

#     if n_drop > 0:
#         sl = slice(n_drop, -n_drop)
#     else:
#         sl = slice(None)

#     # 3) 只把内点送进 operator 系数计算
#     y_used   = y_nodes[sl]
#     f_used   = f_full[:, sl]
#     fy_used  = fy_full[:, sl]
#     fyy_used = fyy_full[:, sl]

#     res = residual_from_nodes(
#         y_nodes=y_used,
#         f=f_used,
#         fy=fy_used,
#         fyy=fyy_used,
#         a_batch=a_batch,
#         omega_batch=omega_batch,
#         lambda_batch=lambda_batch,
#         p=p,
#         ramp_batch=ramp_batch,
#         cfg=cfg,
#     )

#     loss = complex_mse(res)

#     diag = {
#         "loss_res": float(loss.detach().cpu().item()),
#         "res_abs_max": float(torch.abs(res).max().detach().cpu().item()),
#         "f_abs_max": float(torch.abs(f_used).max().detach().cpu().item()),
#         "fy_abs_max": float(torch.abs(fy_used).max().detach().cpu().item()),
#         "fyy_abs_max": float(torch.abs(fyy_used).max().detach().cpu().item()),
#     }
#     return loss, diag

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
    n_boundary_drop: int | None = None,
    collocation_idx: torch.Tensor | None = None,   # 新增
):
    coeff = torch.complex(coeff_re, coeff_im)   # (B,Nc)

    # 对齐 dtype / device
    Tmat_c = Tmat.to(dtype=coeff.dtype, device=coeff.device)
    D_c    = D.to(dtype=coeff.dtype, device=coeff.device)
    D2_c   = D2.to(dtype=coeff.dtype, device=coeff.device)

    # 1) 先重建所有节点上的 f
    f_full = coeff @ Tmat_c.T   # (B,Ny)

    # 2) 决定边界剔除范围
    if n_boundary_drop is None:
        n_drop = 1 if exclude_endpoints else 0
    else:
        n_drop = int(n_boundary_drop)

    Ny = y_nodes.shape[0]
    if n_drop < 0:
        raise ValueError(f"n_boundary_drop must be >= 0, got {n_drop}")
    if 2 * n_drop >= Ny:
        raise ValueError(
            f"Too many boundary points dropped: n_drop={n_drop}, Ny={Ny}"
        )

    interior_idx = torch.arange(
        n_drop, Ny - n_drop, device=coeff.device, dtype=torch.long
    )

    # 3) 如果没有指定采样点，就默认用全部内点
    if collocation_idx is None:
        idx = interior_idx
    else:
        idx = collocation_idx.to(device=coeff.device, dtype=torch.long)

        # 基本合法性检查
        if idx.ndim != 1:
            raise ValueError(f"collocation_idx must be 1D, got shape {tuple(idx.shape)}")
        if torch.any(idx < n_drop) or torch.any(idx >= Ny - n_drop):
            raise ValueError(
                f"collocation_idx contains boundary/out-of-range indices. "
                f"Allowed range is [{n_drop}, {Ny - n_drop - 1}]"
            )

    # 4) 只在 idx 这些点上取值/导数
    y_used = y_nodes.index_select(0, idx)

    f_used = f_full.index_select(1, idx)             # (B,Ns)

    # 注意：导数只算抽中的这些点，不再算全体 fy_full/fyy_full
    D_sub  = D_c.index_select(0, idx)                # (Ns,Ny)
    D2_sub = D2_c.index_select(0, idx)               # (Ns,Ny)

    fy_used  = f_full @ D_sub.T                      # (B,Ns)
    fyy_used = f_full @ D2_sub.T                     # (B,Ns)

    res = residual_from_nodes(
        y_nodes=y_used,
        f=f_used,
        fy=fy_used,
        fyy=fyy_used,
        a_batch=a_batch,
        omega_batch=omega_batch,
        lambda_batch=lambda_batch,
        p=p,
        ramp_batch=ramp_batch,
        cfg=cfg,
    )

    loss = complex_mse(res)

    diag = {
        "loss_res": float(loss.detach().cpu().item()),
        "res_abs_max": float(torch.abs(res).max().detach().cpu().item()),
        "f_abs_max": float(torch.abs(f_used).max().detach().cpu().item()),
        "fy_abs_max": float(torch.abs(fy_used).max().detach().cpu().item()),
        "fyy_abs_max": float(torch.abs(fyy_used).max().detach().cpu().item()),
        "n_collocation": int(idx.numel()),
    }
    return loss, diag



def diagnose_operator_scales(
    cfg: Dict[str, Any],
    y_nodes: torch.Tensor,         # (Ny,)
    a_batch: torch.Tensor,         # (B,)
    omega_batch: torch.Tensor,     # (B,)
    lambda_batch: torch.Tensor,    # (B,) complex
    ramp_batch: torch.Tensor,      # (B,) complex
    p: int,
    n_boundary_report: int = 3,
) -> Dict[str, Any]:
    """
    训练前诊断：检查 A2,A1,A0, B2,B1,B0, rhs 的尺度，
    以及零函数初值 f=0 时的 residual = -rhs 的尺度。

    返回一个 dict，便于在训练脚本里打印。
    """
    from physical_ansatz.transform_y import transform_coeffs_x_to_y
    from physical_ansatz.teukolsky_coeffs import coeffs_x

    prob = cfg["problem"]
    m = int(prob["m"])
    s = int(prob.get("s", -2))
    M = float(prob.get("M", 1.0))

    if y_nodes.ndim != 1:
        raise ValueError(f"y_nodes must be 1D, got shape {tuple(y_nodes.shape)}")

    B = a_batch.shape[0]
    Ny = y_nodes.shape[0]

    # shape -> (B,Ny,1)
    y = y_nodes[None, :, None].expand(B, -1, -1)
    x = (y + 1.0) / 2.0

    a = a_batch.reshape(B, 1, 1)
    omega = omega_batch.reshape(B, 1, 1)
    lam = lambda_batch.reshape(B, 1, 1)
    ramp = ramp_batch.reshape(B, 1, 1)

    A2, A1, A0 = coeffs_x(x, a, omega, m, p, ramp, lam, s=s, M=M)
    B2, B1, B0, rhs = transform_coeffs_x_to_y(A2, A1, A0, y)

    # 零函数初值时，f=fy=fyy=0 => residual = -rhs
    res0 = -rhs

    def summarize(name: str, z: torch.Tensor) -> Dict[str, Any]:
        # z: (B,Ny,1)
        az = torch.abs(z.squeeze(-1))  # (B,Ny)

        flat_idx = torch.argmax(az)
        b_idx = int(flat_idx // Ny)
        j_idx = int(flat_idx % Ny)

        out = {
            "max_abs": float(az.max().detach().cpu().item()),
            "argmax_batch": b_idx,
            "argmax_j": j_idx,
            "y_at_argmax": float(y_nodes[j_idx].detach().cpu().item()),
            "x_at_argmax": float(((y_nodes[j_idx] + 1.0) / 2.0).detach().cpu().item()),
        }

        k = min(n_boundary_report, Ny // 2)
        if k > 0:
            out["left_boundary_max"] = float(torch.abs(z[:, :k, :]).max().detach().cpu().item())
            out["right_boundary_max"] = float(torch.abs(z[:, -k:, :]).max().detach().cpu().item())
            if Ny > 2 * k:
                out["interior_max"] = float(torch.abs(z[:, k:-k, :]).max().detach().cpu().item())
            else:
                out["interior_max"] = float(torch.abs(z).max().detach().cpu().item())

        return out

    return {
        "A2": summarize("A2", A2),
        "A1": summarize("A1", A1),
        "A0": summarize("A0", A0),
        "B2": summarize("B2", B2),
        "B1": summarize("B1", B1),
        "B0": summarize("B0", B0),
        "rhs": summarize("rhs", rhs),
        "res0": summarize("res0", res0),
    }
