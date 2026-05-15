"""Stage-3 u-basis residual: compute Teukolsky ODE residual for R_up, R_down.

R_up(y)   = A_up(r(y))   * u_up(y)
R_down(y) = A_down(r(y)) * u_down(y)

where u_up/u_down are composed via the infinity-boundary ansatz:
    u(y) = 1 + c_inf*(y+1) + (y+1)^2 * f(y)

The Teukolsky residual in r-space:
    Delta * R_rr + (s+1) * Delta_r * R_r + V * R = 0

Derivatives R_y, R_yy are computed via autograd (real/imag split, same pattern
as residual_pinn.compute_f_derivatives_autograd), then converted to r-space
using the chain rule through y → x → r.
"""

import torch
from physical_ansatz.mapping import r_plus
from physical_ansatz.prefactor import delta, delta_r, V_of_r
from physical_ansatz.u_basis import (
    compose_u_from_f,
    infinity_slopes_y,
    A_up,
    A_down,
)


def _ensure_2d(t, name):
    """Ensure tensor has shape (B,1) for broadcasting. Accept (B,) or (B,1)."""
    if t is None:
        return None
    if t.ndim == 1:
        return t.unsqueeze(-1)
    return t


def compute_u_basis_residual(model, a, omega, y, lambda_, u=None, v=None,
                              M=1.0, s=-2, m=2, basis="up"):
    """Compute Teukolsky residual for one u-basis (up or down).

    Args:
        model:  AutoencoderTeukolskyPINN
        a:      (B,) or (B,1) float
        omega:  (B,) or (B,1) float
        y:      (B,N) float — collocation points in [-1,1]
        lambda_:(B,) complex — angular eigenvalue
        u, v:   (B,) or (B,1) float or None — chart coordinates
        M, s, m: scalar physics parameters
        basis:  "up" or "down"

    Returns:
        R:        (B,N) complex — the full radial solution
        residual: (B,N) complex — Teukolsky residual at each point
        pointwise:(B,N) float   — |residual|^2
    """
    if basis not in ("up", "down"):
        raise ValueError(f"basis must be 'up' or 'down', got '{basis}'")

    if not y.requires_grad:
        y = y.clone().requires_grad_(True)

    B, N = y.shape
    a2 = _ensure_2d(a, "a")
    omega2 = _ensure_2d(omega, "omega")
    lambda2 = _ensure_2d(lambda_, "lambda_")
    u2 = _ensure_2d(u, "u")
    v2 = _ensure_2d(v, "v")

    # ---- predict free function f(y) from decoder ----
    if basis == "up":
        f = model.predict_u_up(a2, omega2, y, u2, v2)
    else:
        f = model.predict_u_down(a2, omega2, y, u2, v2)

    # ---- infinity slopes & boundary ansatz ----
    c_up, c_down = infinity_slopes_y(a2, omega2, lambda2, m=m, M=M)
    c_inf = c_up if basis == "up" else c_down
    u_val = compose_u_from_f(f, y, c_inf)  # (B,N) complex

    # ---- asymptotic factor & full R ----
    rp = r_plus(a2, M)
    x = (y + 1.0) / 2.0
    r = rp / x  # (B,N), avoids r_from_x which can't handle batched rp

    if basis == "up":
        A = A_up(r, a2, omega2, M)
    else:
        A = A_down(r, a2, omega2, M)

    R = A * u_val  # (B,N) complex

    # ---- R_y, R_yy via autograd (real/imag split) ----
    # Pointwise network: ∂R[i,j]/∂y[k,l] = 0 for (i,j)≠(k,l), so grad(sum, y)
    # gives the per-element gradient directly.  4 autograd calls per basis,
    # all with retain_graph=True — the outer total_loss.backward() frees them.
    R_y = torch.complex(
        torch.autograd.grad(R.real.sum(), y, create_graph=True, retain_graph=True)[0],
        torch.autograd.grad(R.imag.sum(), y, create_graph=True, retain_graph=True)[0],
    )  # (B,N)

    R_yy = torch.complex(
        torch.autograd.grad(R_y.real.sum(), y, create_graph=False, retain_graph=True)[0],
        torch.autograd.grad(R_y.imag.sum(), y, create_graph=False, retain_graph=True)[0],
    )  # (B,N)

    # ---- convert to r-space via chain rule ----
    # r = rp / x,  x = (y+1)/2
    # dy/dr = -(y+1)^2 / (2*rp)
    # d^2y/dr^2 = (y+1)^3 / (2*rp^2)
    y1 = y + 1.0  # (B,N)
    dy_dr = -y1 ** 2 / (2.0 * rp)       # (B,N) / (B,1) → (B,N)
    d2y_dr2 = y1 ** 3 / (2.0 * rp ** 2)

    R_r = R_y * dy_dr
    R_rr = R_yy * dy_dr ** 2 + R_y * d2y_dr2

    # ---- Teukolsky residual ----
    Delta = delta(r, a2, M)
    Delta_r = delta_r(r, M)
    V = V_of_r(r, a2, omega2, m, s, lambda2, M)

    residual = Delta * R_rr + (s + 1) * Delta_r * R_r + V * R
    pointwise = torch.abs(residual) ** 2

    return R, residual, pointwise


def compute_stage3_loss(model, a, omega, y, lambda_, u=None, v=None,
                         M=1.0, s=-2, m=2,
                         weight_up=1.0, weight_down=1.0,
                         normalize_residual=False, eps=1e-12):
    """Compute Stage-3 loss: L_up + L_down.

    If normalize_residual=True, each pointwise residual is divided by
    |R|^2 at that point to prevent large-r regions from dominating.

    Returns:
        total_loss: scalar
        info: dict with loss_up, loss_down, etc.
    """
    R_up, res_up, pw_up = compute_u_basis_residual(
        model, a, omega, y, lambda_, u, v, M, s, m, basis="up")
    R_down, res_down, pw_down = compute_u_basis_residual(
        model, a, omega, y, lambda_, u, v, M, s, m, basis="down")

    if normalize_residual:
        scale_up = torch.abs(R_up.detach()) ** 2 + eps
        scale_down = torch.abs(R_down.detach()) ** 2 + eps
        pw_up = pw_up / scale_up
        pw_down = pw_down / scale_down

    loss_up = torch.mean(pw_up)
    loss_down = torch.mean(pw_down)
    total_loss = weight_up * loss_up + weight_down * loss_down

    info = {
        "loss_up": float(loss_up.detach().cpu().item()),
        "loss_down": float(loss_down.detach().cpu().item()),
        "total_loss": float(total_loss.detach().cpu().item()),
    }
    return total_loss, info
