"""
PINN版本的 residual 计算。

注意：
这里的 cfg 必须是 physics config，
即 config/teukolsky_radial.yaml 对应的结构，
至少应包含 cfg["problem"] 中的 M, s, l, m, lambda, R_amp。
不要把 pinn_config.yaml 直接传进来。
"""
import torch
import numpy as np
from .residual import AuxCache, get_lambda_from_cfg, get_ramp_and_p_from_cfg
from .teukolsky_coeffs import coeffs_x
from .mapping import r_plus, r_from_x
from .prefactor import *
from .transform_y import transform_coeffs_x_to_y, g_factor,h_factor


def compute_Rprime_derivatives_autograd(
        model,
        a_batch,
        omega_batch,
        y_points,
        u_batch=None,
        v_batch=None,
    ):
    """
    使用自动微分计算 R', R'_y, R'_yy

    Args:
        model: PINN_MLP 模型，输出R'
        a_batch: (B,)
        omega_batch: (B,)
        y_points: (N,) 需要 requires_grad=True

    Returns:
        Rprime: (B, N) 复数
        Rprime_y: (B, N) 复数
        Rprime_yy: (B, N) 复数
    """
    if not y_points.requires_grad:
        y_points = y_points.clone().requires_grad_(True)

    # 前向传播得到 R'
    if u_batch is None and v_batch is None:
        Rprime = model(a_batch, omega_batch, y_points)  # (B, N)
    else:
        Rprime = model(a_batch, omega_batch, y_points, u=u_batch, v=v_batch)  # (B, N)

    # 分离实部和虚部
    Rprime_re = Rprime.real
    Rprime_im = Rprime.imag

    # 计算一阶导数
    Rprime_y_re_list = []
    Rprime_y_im_list = []

    for i in range(Rprime.shape[0]):
        grad_re = torch.autograd.grad(
            outputs=Rprime_re[i].sum(),
            inputs=y_points,
            create_graph=True,
            retain_graph=True,
        )[0]
        Rprime_y_re_list.append(grad_re)

        grad_im = torch.autograd.grad(
            outputs=Rprime_im[i].sum(),
            inputs=y_points,
            create_graph=True,
            retain_graph=True,
        )[0]
        Rprime_y_im_list.append(grad_im)

    Rprime_y_re = torch.stack(Rprime_y_re_list, dim=0)
    Rprime_y_im = torch.stack(Rprime_y_im_list, dim=0)
    Rprime_y = torch.complex(Rprime_y_re, Rprime_y_im)

    # 计算二阶导数
    Rprime_yy_re_list = []
    Rprime_yy_im_list = []

    for i in range(Rprime.shape[0]):
        grad2_re = torch.autograd.grad(
            outputs=Rprime_y_re[i].sum(),
            inputs=y_points,
            create_graph=True,
            retain_graph=True,
        )[0]
        Rprime_yy_re_list.append(grad2_re)

        grad2_im = torch.autograd.grad(
            outputs=Rprime_y_im[i].sum(),
            inputs=y_points,
            create_graph=True,
            retain_graph=True,
        )[0]
        Rprime_yy_im_list.append(grad2_im)

    Rprime_yy_re = torch.stack(Rprime_yy_re_list, dim=0)
    Rprime_yy_im = torch.stack(Rprime_yy_im_list, dim=0)
    Rprime_yy = torch.complex(Rprime_yy_re, Rprime_yy_im)

    return Rprime, Rprime_y, Rprime_yy


def compute_pointwise_pde_residual(
    model,
    cfg,
    a_batch,
    omega_batch,
    lambda_batch,
    ramp_batch,
    p,
    y_interior,
    normalize=False,
    eps=1e-12,
    u_batch=None,
    v_batch=None,
):
    device = a_batch.device
    M = float(cfg["problem"].get("M", 1.0))
    s = int(cfg["problem"].get("s", -2))
    m = int(cfg["problem"].get("m", 2))

    Rprime_int, Rprime_y_int, Rprime_yy_int = compute_Rprime_derivatives_autograd(
        model,
        a_batch,
        omega_batch,
        y_interior,
        u_batch=u_batch,
        v_batch=v_batch,
    )

    x_int = (y_interior + 1.0) / 2.0

    A2_list, A1_list, A0_list = [], [], []
    for i in range(a_batch.shape[0]):
        A2, A1, A0 = coeffs_x(
            x=x_int,
            a=a_batch[i],
            omega=omega_batch[i],
            m=m,
            p=p,
            R_amp=ramp_batch[i],
            lambda_=lambda_batch[i],
            s=s,
            M=M,
        )
        A2_list.append(A2)
        A1_list.append(A1)
        A0_list.append(A0)

    A2_int = torch.stack(A2_list, dim=0)
    A1_int = torch.stack(A1_list, dim=0)
    A0_int = torch.stack(A0_list, dim=0)

    
    h=h_factor(a_batch,omega_batch,m,M,s)
    B2_int, B1_int, B0_int,rhs = transform_coeffs_x_to_y(
        A2_int, A1_int, A0_int, y_interior,h=h
    )

    residual_int = B2_int * Rprime_yy_int + B1_int * Rprime_y_int + B0_int * Rprime_int-rhs
    term2 = B2_int * Rprime_yy_int
    term1 = B1_int * Rprime_y_int
    term0 = B0_int * Rprime_int

    residual_int = term2 + term1 + term0 - rhs
    pointwise = torch.abs(residual_int) ** 2

    if normalize:
        scale = (
            1.0
            + torch.abs(term2.detach()) ** 2
            + torch.abs(term1.detach()) ** 2
            + torch.abs(term0.detach()) ** 2
            + torch.abs(rhs.detach()) ** 2
        )
        pointwise = pointwise / scale.clamp_min(eps)

    return residual_int, pointwise


# def pinn_residual_loss(
#     model,
#     cfg,
#     a_batch,
#     omega_batch,
#     lambda_batch,
#     ramp_batch,
#     p,
#     y_interior,
#     y_boundary,
#     weight_interior=1.0,
#     weight_boundary=10.0,
# ):
#     """
#     PINN residual loss（直接训练R'）

#     方程：A2(x) R'_xx + A1(x) R'_x + A0(x) R' = 0
#     在y坐标下：B2(y) R'_yy + B1(y) R'_y + B0(y) R' = 0
#     """
#     device = a_batch.device
#     dtype = a_batch.dtype
#     M = float(cfg["problem"].get("M", 1.0))
#     s = int(cfg["problem"].get("s", -2))
#     m = int(cfg["problem"].get("m", 2))

#     # ========== 内点 residual ==========
#     Rprime_int, Rprime_y_int, Rprime_yy_int = compute_Rprime_derivatives_autograd(
#         model, a_batch, omega_batch, y_interior
#     )

#     # y -> x
#     x_int = (y_interior + 1.0) / 2.0

#     # 计算方程系数 A2, A1, A0（在x坐标）
#     A2_list, A1_list, A0_list = [], [], []
#     for i in range(a_batch.shape[0]):
#         A2, A1, A0 = coeffs_x(
#             x=x_int, a=a_batch[i], omega=omega_batch[i],
#             m=m, p=p, R_amp=ramp_batch[i],
#             lambda_=lambda_batch[i], s=s, M=M,
#         )
#         A2_list.append(A2)
#         A1_list.append(A1)
#         A0_list.append(A0)

#     A2_int = torch.stack(A2_list, dim=0)
#     A1_int = torch.stack(A1_list, dim=0)
#     A0_int = torch.stack(A0_list, dim=0)

#     # 转换到y坐标：dx/dy = 1/2
#     # R'_x = R'_y * dy/dx = R'_y * 2
#     # R'_xx = R'_yy * (dy/dx)^2 = R'_yy * 4
#     B2_int = A2_int * 4.0
#     B1_int = A1_int * 2.0
#     B0_int = A0_int

#     # Residual
#     residual_int = B2_int * Rprime_yy_int + B1_int * Rprime_y_int + B0_int * Rprime_int

#     loss_interior = torch.mean(torch.abs(residual_int) ** 2)

#     # ========== 边界条件 ==========

#         # validate() 可能会传入空的 y_boundary，此时需要显式跳过，
#     # 否则 torch.mean(empty_tensor) 会返回 NaN。
#     if y_boundary.numel() == 0:
#         loss_boundary = torch.zeros((), device=device, dtype=loss_interior.dtype)
#     else:
#         # 边界上R'应该较小
#         Rprime_bd, _, _ = compute_Rprime_derivatives_autograd(
#             model, a_batch, omega_batch, y_boundary
#         )
#         loss_boundary = torch.mean(torch.abs(Rprime_bd) ** 2)



#     # 总loss
#     total_loss = weight_interior * loss_interior + weight_boundary * loss_boundary

#     info = {
#         'loss_interior': loss_interior.item(),
#         'loss_boundary': loss_boundary.item(),
#         'total_loss': total_loss.item(),
#     }

#     return total_loss, info


def pinn_residual_loss(
    model,
    cfg,
    a_batch,
    omega_batch,
    lambda_batch,
    ramp_batch,
    p,
    y_interior,
    y_boundary,
    weight_interior=1.0,
    weight_boundary=10.0,
    normalize_residual=False,
    residual_scale_eps=1e-12,
    return_pointwise=False,
    u_batch=None,
    v_batch=None,
):
    residual_int, pointwise_interior = compute_pointwise_pde_residual(
        model=model,
        cfg=cfg,
        a_batch=a_batch,
        omega_batch=omega_batch,
        lambda_batch=lambda_batch,
        ramp_batch=ramp_batch,
        p=p,
        y_interior=y_interior,
        normalize=normalize_residual,
        eps=residual_scale_eps,
        u_batch=u_batch,
        v_batch=v_batch,
    )

    loss_interior = torch.mean(pointwise_interior)

    if y_boundary.numel() == 0:
        loss_boundary = torch.zeros(
            (), device=a_batch.device, dtype=loss_interior.dtype
        )
    else:
        Rprime_bd, _, _ = compute_Rprime_derivatives_autograd(
            model,
            a_batch,
            omega_batch,
            y_boundary,
            u_batch=u_batch,
            v_batch=v_batch,
        )
        loss_boundary = torch.mean(torch.abs(Rprime_bd) ** 2)

    total_loss = weight_interior * loss_interior + weight_boundary * loss_boundary

    info = {
        "loss_interior": float(loss_interior.detach().cpu().item()),
        "loss_boundary": float(loss_boundary.detach().cpu().item()),
        "total_loss": float(total_loss.detach().cpu().item()),
    }

    if return_pointwise:
        info["pointwise_interior"] = pointwise_interior.detach()

    return total_loss, info



# def compute_data_anchor_loss(
#     model,
#     cfg,
#     a_batch,
#     omega_batch,
#     y_anchors,
#     R_mma_anchors,
# ):
#     """
#     数据锚点loss：使用Mathematica结果作为监督，约束R'而非R

#     Args:
#         model: PINN模型
#         cfg: 配置
#         a_batch: (B,)
#         omega_batch: (B,)
#         y_anchors: (N_anchor,) y坐标锚点
#         R_mma_anchors: (B, N_anchor) Mathematica的R(r)值

#     Returns:
#         loss_anchor: 锚点loss
#     """
#     device = a_batch.device
#     dtype = a_batch.dtype
#     M = float(cfg["problem"].get("M", 1.0))
#     s = int(cfg["problem"].get("s", -2))
#     m = int(cfg["problem"].get("m", 2))

#     # PINN预测R'
#     Rprime_pred, _, _ = compute_Rprime_derivatives_autograd(
#         model, a_batch, omega_batch, y_anchors
#     )

#     # y -> x -> r，计算U(r)
#     x_anchors = (y_anchors + 1.0) / 2.0
#     U_list = []

#     cache = AuxCache()
#     for i in range(a_batch.shape[0]):
#         rp = r_plus(a_batch[i], M)
#         r_i = r_from_x(x_anchors, rp)

#         # 计算U(r)
#         p_i, ramp_i = get_ramp_and_p_from_cfg(cfg, cache, a_batch[i], omega_batch[i])
#         U_i = U_factor(r_i, a_batch[i], omega_batch[i], p_i, ramp_i, m, s, M)
#         U_list.append(U_i)

#     U_batch = torch.stack(U_list, dim=0)  # (B, N_anchor)

#     # 从Mathematica的R计算真实的R' = R / U
#     Rprime_mma = R_mma_anchors / U_batch

#     # 锚点loss：约束R'
#     loss_anchor = torch.mean(torch.abs(Rprime_pred - Rprime_mma) ** 2)

#     return loss_anchor

def compute_data_anchor_loss(
    model,
    cfg,
    a_batch,
    omega_batch,
    y_anchors,
    R_mma_anchors,
    relative=False,
    eps=1e-12,
    u_batch=None,
    v_batch=None,
):
    M = float(cfg["problem"].get("M", 1.0))
    s = int(cfg["problem"].get("s", -2))
    m = int(cfg["problem"].get("m", 2))

    Rprime_pred, _, _ = compute_Rprime_derivatives_autograd(
        model,
        a_batch,
        omega_batch,
        y_anchors,
        u_batch=u_batch,
        v_batch=v_batch,
    )

    x_anchors = (y_anchors + 1.0) / 2.0
    U_list = []
    cache = AuxCache()

    for i in range(a_batch.shape[0]):
        rp = r_plus(a_batch[i], M)
        r_i = r_from_x(x_anchors, rp)
        p_i, ramp_i = get_ramp_and_p_from_cfg(cfg, cache, a_batch[i], omega_batch[i])
        P, P_r, P_rr = Leaver_prefactors(r_i, a_batch[i], omega_batch[i],m,M,s)
        Q, Q_r, Q_rr = prefactor_Q(r_i, a_batch[i], omega_batch[i],p_i,ramp_i,M,s)
        U_i,_,_ =U_prefactor(P,P_r,P_rr,Q,Q_r,Q_rr)
        U_list.append(U_i)

    U_batch = torch.stack(U_list, dim=0)
    Rprime_mma = R_mma_anchors / U_batch
    
    g,_,_ = g_factor(x_anchors)
    h=h_factor(a_batch,omega_batch,m,M,s)
    Rprime_pred = Rprime_pred*g+h.view(-1, 1)
    err2 = torch.abs(Rprime_pred - Rprime_mma) ** 2
    if relative:
        scale = torch.mean(torch.abs(Rprime_mma.detach()) ** 2, dim=1, keepdim=True)
        loss_anchor = torch.mean(err2 / scale.clamp_min(eps))
    else:
        loss_anchor = torch.mean(err2)

    return loss_anchor

def compute_variance_regularizer(
    model,
    cfg,
    a_batch,
    omega_batch,
    y_points,
    target="Rprime",
    kappa=20.0,
    eps=1.0e-12,
    m=2.0,
    u_batch=None,
    v_batch=None,
):
    """
    用于惩罚“输出几乎为常数”的伪平凡解。

    参数
    ----
    model : PINN_MLP
    cfg : dict
    a_batch, omega_batch : [B]
    y_points : [N]
    target : "f" or "Rprime"
    kappa : float
    eps : float
    """

    # 模型输出: [B, N] complex
    if u_batch is None and v_batch is None:
        f_pred = model(a_batch, omega_batch, y_points)
    else:
        f_pred = model(a_batch, omega_batch, y_points, u=u_batch, v=v_batch)

    h=h_factor(a_batch,omega_batch,m)

    if target == "f":
        z = f_pred

    elif target == "Rprime":
        x_points = 0.5 * (y_points + 1.0)
        g_val, _, _ = g_factor(x_points)   # [N]
        z = g_val.unsqueeze(0) * f_pred + h.view(-1, 1)

    else:
        raise ValueError(f"Unknown target for variance regularizer: {target}")

    # 复数标准差:
    # sigma_b = sqrt( mean( |z - mean(z)|^2 ) + eps )
    z_mean = z.mean(dim=1, keepdim=True)
    sigma_b = torch.sqrt(torch.mean(torch.abs(z - z_mean) ** 2, dim=1) + eps)

    # batch 平均
    sigma = sigma_b.mean()

    loss_var = 1.0 / (torch.expm1(kappa * sigma) + eps)

    info = {
        "sigma_var": float(sigma.detach().cpu().item()),
        "loss_var": float(loss_var.detach().cpu().item()),
    }
    return loss_var, info