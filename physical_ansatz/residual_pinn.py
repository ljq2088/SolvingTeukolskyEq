"""
PINN版本的 residual 计算。

注意：
这里的 cfg 必须是 physics config，
即 config/teukolsky_radial.yaml 对应的结构，
至少应包含 cfg["problem"] 中的 M, s, l, m, lambda。
不要把 pinn_config.yaml 直接传进来。
"""
import torch
import numpy as np
from .residual import AuxCache, get_lambda_from_cfg, get_ramp_and_p_from_cfg
from .teukolsky_coeffs import coeffs_x
from .mapping import r_plus, r_from_x
from .prefactor import build_prefactor_primitives, Leaver_prefactors, prefactor_Q, U_prefactor
from .transform_y import (
    transform_coeffs_x_to_y,
    transform_coeffs_x_to_y_S,
    h_factor,
    horizon_regularity_slope,
    compose_reduced_shape_from_f,
    g_factor,
    h1_factor,
)

def compute_f_derivatives_autograd(
    model,
    a_batch,
    omega_batch,
    y_points,
    u_batch=None,
    v_batch=None,
):
    """
    Compute f(y), f_y, f_yy.

    If the model has forward_with_derivatives (e.g. ChebCoeffNet), use exact
    Chebyshev derivatives. Otherwise fall back to autograd.
    """
    # ChebCoeffNet: exact derivatives, no autograd overhead
    if hasattr(model, "forward_with_derivatives"):
        return model.forward_with_derivatives(
            a_batch, omega_batch, y_points, u=u_batch, v=v_batch
        )

    # Fallback: per-sample autograd to avoid O(B²) memory from retain_graph
    # on full batch. Each sample retains its own (1,N) graph, B× smaller.

    # Ensure y_points is 2D (B,N) for per-sample slicing
    if y_points.ndim == 1:
        y_2d = y_points.unsqueeze(0).expand(a_batch.shape[0], -1)
    else:
        y_2d = y_points

    f_list, fy_list, fyy_list = [], [], []

    B = a_batch.shape[0]
    for i in range(B):
        a_i = a_batch[i:i+1]
        omega_i = omega_batch[i:i+1]
        y_i = y_2d[i:i+1].detach().clone().requires_grad_(True)
        u_i = u_batch[i:i+1] if u_batch is not None else None
        v_i = v_batch[i:i+1] if v_batch is not None else None

        f_i = model(a_i, omega_i, y_i, u=u_i, v=v_i)
        f_list.append(f_i)

        fy_re_i = torch.autograd.grad(
            f_i.real.sum(), y_i, create_graph=True, retain_graph=True,
        )[0]
        fy_im_i = torch.autograd.grad(
            f_i.imag.sum(), y_i, create_graph=True, retain_graph=True,
        )[0]
        fy_i = torch.complex(fy_re_i, fy_im_i)
        fy_list.append(fy_i)

        # fyy from fy — retain_graph=True on all; graph consumed by caller's backward
        fyy_re_i = torch.autograd.grad(
            fy_re_i.sum(), y_i, create_graph=False, retain_graph=True,
        )[0]
        fyy_im_i = torch.autograd.grad(
            fy_im_i.sum(), y_i, create_graph=False, retain_graph=True,
        )[0]
        fyy_i = torch.complex(fyy_re_i, fyy_im_i)
        fyy_list.append(fyy_i)

    f = torch.cat(f_list, dim=0)
    fy = torch.cat(fy_list, dim=0)
    fyy = torch.cat(fyy_list, dim=0)

    return f, fy, fyy

def compute_pointwise_pde_residual(
    model,
    cfg,
    a_batch,
    omega_batch,
    lambda_batch,
    y_interior,
    normalize=False,
    eps=1e-12,
    u_batch=None,
    v_batch=None,
    normalize_mode="term",
):
    M = float(cfg["problem"].get("M", 1.0))
    s = int(cfg["problem"].get("s", -2))
    m = int(cfg["problem"].get("m", 2))

    model_out, fy_out, fyy_out = compute_f_derivatives_autograd(
        model, a_batch, omega_batch, y_interior,
        u_batch=u_batch, v_batch=v_batch,
    )

    output_type = getattr(model, "output_type", "f")

    x_int = (y_interior + 1.0) / 2.0

    A2_list, A1_list, A0_list = [], [], []
    for i in range(a_batch.shape[0]):
        A2, A1, A0 = coeffs_x(
            x=x_int,
            a=a_batch[i],
            omega=omega_batch[i],
            m=m,
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

    if output_type == "S":
        # S-PDE: D2*S_yy + D1*S_y + D0*S = 0
        rp_vals = torch.stack([r_plus(a_batch[i], M) for i in range(a_batch.shape[0])])
        r_int = rp_vals.unsqueeze(-1) / x_int.unsqueeze(0)
        D2_int, D1_int, D0_int = transform_coeffs_x_to_y_S(
            A2_int, A1_int, A0_int, r_int, a_batch, omega_batch,
            m=m, M=M, s=s,
        )
        residual_int = D2_int * fyy_out + D1_int * fy_out + D0_int * model_out
        term2 = D2_int * fyy_out
        term1 = D1_int * fy_out
        term0 = D0_int * model_out
    else:
        slope = horizon_regularity_slope(
            a=a_batch,
            omega=omega_batch,
            lambda_=lambda_batch,
            m=m,
            M=M,
            s=s,
        )

        B2_int, B1_int, B0_int, rhs = transform_coeffs_x_to_y(
            A2_int, A1_int, A0_int, y_interior, slope=slope
        )

        residual_int = B2_int * fyy_out + B1_int * fy_out + B0_int * model_out - rhs
        term2 = B2_int * fyy_out
        term1 = B1_int * fy_out
        term0 = B0_int * model_out

    pointwise = torch.abs(residual_int) ** 2

    if normalize:
        if normalize_mode == "term":
            scale = (
                1.0
                + torch.abs(term2.detach()) ** 2
                + torch.abs(term1.detach()) ** 2
                + torch.abs(term0.detach()) ** 2
                + torch.abs(rhs.detach()) ** 2
            )
        elif normalize_mode == "coeff":
            # Normalize by coefficient magnitudes — prevents vanishing
            # coefficients near y=1 from hiding PDE violations.
            if output_type == "S":
                scale = (
                    torch.abs(D2_int.detach()) ** 2
                    + torch.abs(D1_int.detach()) ** 2
                    + torch.abs(D0_int.detach()) ** 2
                    + eps
                )
            else:
                scale = (
                    torch.abs(B2_int.detach()) ** 2
                    + torch.abs(B1_int.detach()) ** 2
                    + torch.abs(B0_int.detach()) ** 2
                    + torch.abs(rhs.detach()) ** 2
                    + eps
                )
        else:
            raise ValueError(f"Unknown normalize_mode: {normalize_mode}")
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

    y_interior,
    y_boundary,
    weight_interior=1.0,
    weight_boundary=10.0,
    normalize_residual=False,
    residual_scale_eps=1e-12,
    return_pointwise=False,
    u_batch=None,
    v_batch=None,
    normalize_mode="term",
):
    residual_int, pointwise_interior = compute_pointwise_pde_residual(
        model=model,
        cfg=cfg,
        a_batch=a_batch,
        omega_batch=omega_batch,
        lambda_batch=lambda_batch,


        y_interior=y_interior,
        normalize=normalize_residual,
        eps=residual_scale_eps,
        u_batch=u_batch,
        v_batch=v_batch,
        normalize_mode=normalize_mode,
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
    lambda_batch=None,
):
    M = float(cfg["problem"].get("M", 1.0))
    s = int(cfg["problem"].get("s", -2))
    m = int(cfg["problem"].get("m", 2))

    Rprime_pred, _, _ = compute_f_derivatives_autograd(
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
        rp_pref, rm_pref, rs_pref, rs_r_pref, rs_rr_pref = build_prefactor_primitives(r_i, a_batch[i], M=M)
        P, P_r, P_rr = Leaver_prefactors(r_i, a_batch[i], omega_batch[i],m,M,s, rp=rp_pref, rm=rm_pref)
        Q, Q_r, Q_rr = prefactor_Q(r_i, a_batch[i], omega_batch[i],p_i,ramp_i,M,s, rp=rp_pref, rs=rs_pref, rs_r=rs_r_pref, rs_rr=rs_rr_pref)
        U_i,_,_ =U_prefactor(P,P_r,P_rr,Q,Q_r,Q_rr)
        U_list.append(U_i)

    U_batch = torch.stack(U_list, dim=0)
    Rprime_mma = R_mma_anchors / U_batch

    slope = horizon_regularity_slope(
        a=a_batch, omega=omega_batch, lambda_=lambda_batch, m=m, M=M, s=s,
    )
    g,_,_ = g_factor(x_anchors, slope)
    h=h_factor(a_batch,omega_batch,m,M,s)
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
    lambda_batch,
    y_points,
    target="shape",
    kappa=20.0,
    eps=1.0e-12,
    m=2.0,
    u_batch=None,
    v_batch=None,
):
    if u_batch is None and v_batch is None:
        f_pred = model(a_batch, omega_batch, y_points)
    else:
        f_pred = model(a_batch, omega_batch, y_points, u=u_batch, v=v_batch)

    if target == "f":
        z = f_pred

    elif target in ("shape", "Rprime"):
        slope = horizon_regularity_slope(
            a=a_batch,
            omega=omega_batch,
            lambda_=lambda_batch,
            m=int(m),
            M=float(cfg["problem"].get("M", 1.0)),
            s=int(cfg["problem"].get("s", -2)),
        )
        z = compose_reduced_shape_from_f(
            f=f_pred,
            y=y_points,
            slope=slope,
        )

    else:
        raise ValueError(f"Unknown target for variance regularizer: {target}")

    z_mean = z.mean(dim=1, keepdim=True)
    sigma_b = torch.sqrt(torch.mean(torch.abs(z - z_mean) ** 2, dim=1) + eps)
    sigma = sigma_b.mean()

    loss_var = 1.0 / (torch.expm1(kappa * sigma) + eps)
    info = {
        "sigma_var": float(sigma.detach().cpu().item()),
        "loss_var": float(loss_var.detach().cpu().item()),
    }
    return loss_var, info


def compute_integral_consistency_loss(
    model,
    cfg,
    a_batch,
    omega_batch,
    lambda_batch,
    y_interior,
    u_batch=None,
    v_batch=None,
    n_anchors=32,
    # Pre-computed intermediates (speed: avoid double autograd + double coeffs)
    _f=None,
    _f_y=None,
    _S=None,
    _S_y=None,
    _A2=None,
    _A1=None,
    _A0=None,
    _slope=None,
):
    """
    Integral consistency loss: penalizes mismatch between S_nn(y_j) and
    S_int(y_j) obtained by integrating the ODE from the horizon.

    For a linear ODE S_xx = -(A1*S_x + A0*S)/A2, the IVP from x=1 gives:
        S(x_j) = 1 + c_H*(x_j-1) + ∫_{x_j}^1 (t-x_j) * S_xx_ode(t) dt

    This couples ALL collocation points between x_j and the horizon,
    preventing the optimizer from converging to wrong fundamental solutions.
    """
    M = float(cfg["problem"].get("M", 1.0))
    s = int(cfg["problem"].get("s", -2))
    m = int(cfg["problem"].get("m", 2))

    x = (y_interior + 1.0) / 2.0
    x_b = x.unsqueeze(0) if x.ndim == 1 else x

    if _A2 is not None and _A1 is not None and _A0 is not None:
        A2, A1, A0 = _A2, _A1, _A0
    else:
        A2_list, A1_list, A0_list = [], [], []
        for i in range(a_batch.shape[0]):
            A2_i, A1_i, A0_i = coeffs_x(
                x=x, a=a_batch[i], omega=omega_batch[i],
                m=m, lambda_=lambda_batch[i], s=s, M=M,
            )
            A2_list.append(A2_i)
            A1_list.append(A1_i)
            A0_list.append(A0_i)
        A2 = torch.stack(A2_list, dim=0)
        A1 = torch.stack(A1_list, dim=0)
        A0 = torch.stack(A0_list, dim=0)

    if _slope is not None:
        slope = _slope
    else:
        slope = horizon_regularity_slope(
            a=a_batch, omega=omega_batch, lambda_=lambda_batch,
            m=m, M=M, s=s,
        )

    # Compute S and S_x: either directly from model or via ansatz S = W*f + G
    if _S is not None and _S_y is not None:
        S = _S                  # (B, N) complex, from model directly
        S_x = 2.0 * _S_y       # (B, N) dy/dx = 2
    else:
        if _f is not None and _f_y is not None:
            f, f_y = _f, _f_y
        else:
            f, f_y, _f_yy = compute_f_derivatives_autograd(
                model, a_batch, omega_batch, y_interior,
                u_batch=u_batch, v_batch=v_batch,
            )

        h1, h1_x, _h1_xx = h1_factor(x_b)
        g, g_x, _g_xx = g_factor(x_b, slope)

        W = g * h1                      # (B, N)
        W_x = g_x * h1 + g * h1_x       # (B, N)
        G = g + 1.0                     # (B, N)
        G_x = g_x                       # (B, N)

        S = W * f + G                   # (B, N)  complex
        f_x = 2.0 * f_y                 # (B, N)
        S_x = W_x * f + W * f_x + G_x   # (B, N)

    # S_xx from ODE: A2*S_xx + A1*S_x + A0*S = 0 => S_xx = -(A1*S_x + A0*S)/A2
    S_xx_ode = -(A1 * S_x + A0 * S) / A2.clamp_min(1e-12)

    # Sort by x ascending (infinity→horizon)
    x_1d = x_b[0]
    sort_idx = torch.argsort(x_1d)
    x_sorted = x_1d[sort_idx]  # (N,)

    # Select far-field anchors: distribute in x ∈ [x_min, x_mid]
    n_avail = len(x_sorted)
    n_anchors_use = min(n_anchors, n_avail // 2)
    far_end = n_avail // 2
    anchor_pos = torch.linspace(0, far_end - 1, n_anchors_use,
                                 dtype=torch.long, device=x_1d.device)

    B = a_batch.shape[0]
    S_sorted = S[:, sort_idx]            # (B, N)
    S_xx_sorted = S_xx_ode[:, sort_idx]  # (B, N)

    # Trapezoidal cumulative integration from x_min to each x_k
    dx = x_sorted[1:] - x_sorted[:-1]  # (N-1,)

    # integrand1: t * S_xx(t), integrand2: S_xx(t)
    t_avg = 0.5 * (x_sorted[:-1] + x_sorted[1:])  # (N-1,)
    S_xx_avg = 0.5 * (S_xx_sorted[:, :-1] + S_xx_sorted[:, 1:])  # (B, N-1)

    dS1 = t_avg.unsqueeze(0) * S_xx_avg * dx.unsqueeze(0)  # (B, N-1)
    dS2 = S_xx_avg * dx.unsqueeze(0)                        # (B, N-1)

    zero_col = torch.zeros(B, 1, device=S.device, dtype=S.dtype)
    CumSum1 = torch.cat([zero_col, torch.cumsum(dS1, dim=-1)], dim=-1)  # (B, N)
    CumSum2 = torch.cat([zero_col, torch.cumsum(dS2, dim=-1)], dim=-1)  # (B, N)

    # CumSum[:, k] = ∫_{x_0}^{x_k} integrand dt

    # For anchor at sorted index j:
    # ∫_{x_j}^{x_max} (t-x_j)*S_xx dt = (CS1_max - CS1_j) - x_j*(CS2_max - CS2_j)
    CS1_max = CumSum1[:, -1:]   # (B, 1)
    CS2_max = CumSum2[:, -1:]   # (B, 1)
    CS1_j = CumSum1[:, anchor_pos]  # (B, n_anchors)
    CS2_j = CumSum2[:, anchor_pos]  # (B, n_anchors)
    x_j = x_sorted[anchor_pos].unsqueeze(0)  # (1, n_anchors)

    integral_vals = (CS1_max - CS1_j) - x_j * (CS2_max - CS2_j)  # (B, n_anchors)

    c_H = slope.unsqueeze(-1)  # (B, 1)
    S_int = 1.0 + c_H * (x_j - 1.0) + integral_vals  # (B, n_anchors)
    S_nn = S_sorted[:, anchor_pos]  # (B, n_anchors)

    loss = torch.mean(torch.abs(S_nn - S_int) ** 2)
    return loss
