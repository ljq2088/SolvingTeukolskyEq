"""
F_y = dF/dy derivative residual for Stage-1 near-infinity refinement.

Mathematical background:
  F(y) = D2(y)*S_yy + D1(y)*S_y + D0(y)*S = 0

  F_y = D2*S_yyy + (D2_y + D1)*S_yy + (D1_y + D0)*S_y + D0_y*S = 0

Near infinity (y→-1), D2→0 weakens S_yy constraint in F=0.
F_y restores curvature constraint because (D2_y + D1) stays O(1).
"""
import torch
from physical_ansatz.transform_y import (
    horizon_regularity_slope,
    compose_reduced_shape_from_f,
)
from physical_ansatz.teukolsky_coeffs import coeffs_x


def chebyshev_half_points(n_points, y_min=-0.9999, y_max=0.9999, device='cpu', dtype=torch.float64):
    """
    Half-Chebyshev collocation points clustered near y_min (infinity).

    y_ref(k) = 3 - 2*cos(pi*k/(2N)),  k = 0, ..., N-1
    y(k) = y_min + (y_max - y_min) * (y_ref(k) - 1) / 2
    """
    if n_points <= 0:
        return torch.empty(0, device=device, dtype=dtype)
    if n_points == 1:
        return torch.tensor([0.5 * (y_min + y_max)], device=device, dtype=dtype)

    k = torch.arange(n_points, device=device, dtype=dtype)
    y_ref = 3.0 - 2.0 * torch.cos(torch.pi * k / (2.0 * n_points))
    y = y_min + (y_max - y_min) * (y_ref - 1.0) / 2.0
    return torch.sort(y).values


def compute_S_derivatives_order3(
    model, a_batch, omega_batch, y_batch,
    u_batch=None, v_batch=None, lambda_batch=None,
    M=1.0, s=-2, m=2,
):
    """Compute S, S_y, S_yy, S_yyy via batched autograd (single backward pass).

    If model.output_type == "S": uses model output directly as S.
    Otherwise: composes S = g*(h1*f + 1) + 1 from model output f(y).

    Returns: S, S_y, S_yy, S_yyy  each (B, N)
    """
    output_type = getattr(model, "output_type", "f")
    B = a_batch.shape[0]

    # Ensure y_batch is 2D (B, N) and a leaf for autograd
    if y_batch.ndim == 1:
        y_batch = y_batch.unsqueeze(0).expand(B, -1)
    y_leaf = y_batch.detach().clone().requires_grad_(True)

    if output_type == "S":
        S = model(a_batch, omega_batch, y_leaf, u=u_batch, v=v_batch)
    else:
        f = model(a_batch, omega_batch, y_leaf, u=u_batch, v=v_batch)
        slope = horizon_regularity_slope(
            a=a_batch, omega=omega_batch, lambda_=lambda_batch, m=m, M=M, s=s,
        )
        S = compose_reduced_shape_from_f(f, y_leaf, slope)

    # Batched autograd: grad of sum over all elements w.r.t. y_leaf gives
    # per-element gradients because each y_leaf[i,j] only affects S[i,j].
    real = S.real
    imag = S.imag

    # S_y
    Sy_re = torch.autograd.grad(real.sum(), y_leaf, create_graph=True, retain_graph=True)[0]
    Sy_im = torch.autograd.grad(imag.sum(), y_leaf, create_graph=True, retain_graph=True)[0]
    Sy = torch.complex(Sy_re, Sy_im)

    # S_yy
    Syy_re = torch.autograd.grad(Sy_re.sum(), y_leaf, create_graph=True, retain_graph=True)[0]
    Syy_im = torch.autograd.grad(Sy_im.sum(), y_leaf, create_graph=True, retain_graph=True)[0]
    Syy = torch.complex(Syy_re, Syy_im)

    # S_yyy
    Syyy_re = torch.autograd.grad(Syy_re.sum(), y_leaf, create_graph=True, retain_graph=True)[0]
    Syyy_im = torch.autograd.grad(Syy_im.sum(), y_leaf, create_graph=True, retain_graph=True)[0]
    Syyy = torch.complex(Syyy_re, Syyy_im)

    return S, Sy, Syy, Syyy


def compute_S_reduced_coeffs_y(a_batch, omega_batch, lambda_batch, y_batch,
                                M=1.0, s=-2, m=2):
    """
    Compute S-space coefficients D2, D1, D0 at given y points.

    F = D2*S_yy + D1*S_y + D0*S = 0

    coeffs_x already returns the P-extracted S(x)-equation coefficients:
        A2*S_xx + A1*S_x + A0*S = 0   where R = P*S.
    The x=(y+1)/2 coordinate transform gives S_x=2*S_y, S_xx=4*S_yy, so:
        D2 = 4*A2,  D1 = 2*A1,  D0 = A0.
    Do NOT call transform_coeffs_x_to_y_S here — P is already extracted.

    Returns: D2, D1, D0  each (B, N) complex, detached.
    """
    B = a_batch.shape[0]
    x_batch = (y_batch + 1.0) / 2.0

    D2_list, D1_list, D0_list = [], [], []

    for i in range(B):
        A2_i, A1_i, A0_i = coeffs_x(
            x=x_batch, a=a_batch[i:i+1], omega=omega_batch[i:i+1],
            m=m, lambda_=lambda_batch[i:i+1], s=s, M=M,
        )

        D2_list.append((4.0 * A2_i).detach())
        D1_list.append((2.0 * A1_i).detach())
        D0_list.append(A0_i.detach())

    D2 = torch.cat(D2_list, dim=0)
    D1 = torch.cat(D1_list, dim=0)
    D0 = torch.cat(D0_list, dim=0)

    return D2, D1, D0


def compute_coeff_derivatives_y(a_batch, omega_batch, lambda_batch, y_batch,
                                 M=1.0, s=-2, m=2):
    """Compute y-derivatives of S-space coefficients via per-sample autograd.

    Returns: D2, D1, D0, D2_y, D1_y, D0_y  each (B, N) complex, detached.
    """
    B = a_batch.shape[0]

    if y_batch.ndim == 1:
        y_1d = y_batch
    else:
        y_1d = y_batch[0]

    D2_list, D1_list, D0_list, D2_y_list, D1_y_list, D0_y_list = [], [], [], [], [], []

    for i in range(B):
        y_i = y_1d.clone().detach().requires_grad_(True)
        y_i_2d = y_i.unsqueeze(0)

        x_i = (y_i_2d + 1.0) / 2.0
        A2_i, A1_i, A0_i = coeffs_x(
            x=x_i, a=a_batch[i:i+1], omega=omega_batch[i:i+1],
            m=m, lambda_=lambda_batch[i:i+1], s=s, M=M,
        )

        D2_i = 4.0 * A2_i
        D1_i = 2.0 * A1_i
        D0_i = A0_i

        def _grad_y(t):
            if t.is_complex():
                r = torch.autograd.grad(t.real.sum(), y_i, create_graph=False, retain_graph=True)[0]
                im = torch.autograd.grad(t.imag.sum(), y_i, create_graph=False, retain_graph=True)[0]
                return torch.complex(r, im)
            return torch.autograd.grad(t.sum(), y_i, create_graph=False, retain_graph=True)[0]

        D2_y_i = _grad_y(D2_i)
        D1_y_i = _grad_y(D1_i)
        D0_y_i = _grad_y(D0_i)

        D2_list.append(D2_i.detach())
        D1_list.append(D1_i.detach())
        D0_list.append(D0_i.detach())
        D2_y_list.append(D2_y_i.detach().unsqueeze(0))
        D1_y_list.append(D1_y_i.detach().unsqueeze(0))
        D0_y_list.append(D0_y_i.detach().unsqueeze(0))

    D2 = torch.cat(D2_list, dim=0)
    D1 = torch.cat(D1_list, dim=0)
    D0 = torch.cat(D0_list, dim=0)
    D2_y = torch.cat(D2_y_list, dim=0)
    D1_y = torch.cat(D1_y_list, dim=0)
    D0_y = torch.cat(D0_y_list, dim=0)

    return D2, D1, D0, D2_y, D1_y, D0_y


def compute_Fy_residual(S, S_y, S_yy, S_yyy, D2, D1, D0, D2_y, D1_y, D0_y):
    """
    Compute F_y = dF/dy residual.

    F_y = D2*S_yyy + (D2_y + D1)*S_yy + (D1_y + D0)*S_y + D0_y*S

    All inputs: (B, N) complex tensors.
    Returns: Fy  (B, N) complex.
    """
    Fy = (
        D2 * S_yyy
        + (D2_y + D1) * S_yy
        + (D1_y + D0) * S_y
        + D0_y * S
    )
    return Fy


def compute_Fy_loss(
    model, a_batch, omega_batch, lambda_batch, y_fy,
    u_batch=None, v_batch=None,
    M=1.0, s=-2, m=2,
    return_components=False,
    normalize=True,
):
    """
    Compute F_y derivative residual loss.

    L_{F_y} = mean(|F_y|^2)  (no normalization)
    or
    L_{F_y} = mean(|F_y / norm|^2)  (coefficient-magnitude normalization)

    where norm = sqrt(|D2|^2 + |D2_y+D1|^2 + |D1_y+D0|^2 + |D0_y|^2)

    Normalization makes the loss dimensionless and uniform across y-points,
    avoiding domination by near-infinity points where S_yyy explodes.
    """
    # Ensure y_fy is 2D (B, N_fy) for per-sample processing
    if y_fy.ndim == 1:
        y_fy_2d = y_fy.unsqueeze(0).expand(a_batch.shape[0], -1)
    else:
        y_fy_2d = y_fy

    S, S_y, S_yy, S_yyy = compute_S_derivatives_order3(
        model, a_batch, omega_batch, y_fy_2d,
        u_batch=u_batch, v_batch=v_batch, lambda_batch=lambda_batch,
        M=M, s=s, m=m,
    )

    D2, D1, D0, D2_y, D1_y, D0_y = compute_coeff_derivatives_y(
        a_batch, omega_batch, lambda_batch, y_fy,
        M=M, s=s, m=m,
    )

    Fy = compute_Fy_residual(S, S_y, S_yy, S_yyy, D2, D1, D0, D2_y, D1_y, D0_y)

    if normalize:
        norm = torch.sqrt(
            torch.abs(D2)**2 + torch.abs(D2_y + D1)**2 +
            torch.abs(D1_y + D0)**2 + torch.abs(D0_y)**2
        )
        Fy = Fy / (norm + 1e-8)

    loss_fy = torch.mean(torch.abs(Fy) ** 2)

    if return_components:
        with torch.no_grad():
            t3 = torch.abs(D2 * S_yyy / (norm + 1e-8) if normalize else D2 * S_yyy).mean().item()
            t2 = torch.abs((D2_y + D1) * S_yy / (norm + 1e-8) if normalize else (D2_y + D1) * S_yy).mean().item()
            t1 = torch.abs((D1_y + D0) * S_y / (norm + 1e-8) if normalize else (D1_y + D0) * S_y).mean().item()
            t0 = torch.abs(D0_y * S / (norm + 1e-8) if normalize else D0_y * S).mean().item()
            fy_abs = torch.abs(Fy)
        info = {
            "loss_fy": float(loss_fy.detach().cpu().item()),
            "fy_mean_abs": float(fy_abs.mean().detach().cpu().item()),
            "fy_max_abs": float(fy_abs.max().detach().cpu().item()),
            "fy_term3_mean": float(t3),
            "fy_term2_mean": float(t2),
            "fy_term1_mean": float(t1),
            "fy_term0_mean": float(t0),
        }
        return loss_fy, info

    return loss_fy
