"""Stage-3 u-basis residual: compute Teukolsky ODE residual for u_up, u_down.

Two modes:
  - u_equation (training default): analytically extract A_up/A_down from the ODE,
    yielding coefficients C2, C1, C0 that act directly on u.  No large-r blow-up.
  - direct_R (debug): construct R = A * u, compute Teukolsky residual on R.
    Kept for reference; numerically unstable at large r for the up basis.

u-equation derivation
---------------------
Start from the Teukolsky r-space ODE:
    Delta * R_rr + (s+1) * Delta_r * R_r + V * R = 0

Substitute R = A * u:
    R_r   = A_r u + A u_r
    R_rr  = A_rr u + 2 A_r u_r + A u_rr

Divide through by A and define q_r = A_r / A, q_rr = d(q_r)/dr:
    Delta * u_rr + [2*Delta*q_r + (s+1)*Delta_r] * u_r
    + [Delta*(q_rr + q_r^2) + (s+1)*Delta_r*q_r + V] * u = 0

Convert to y-space (u_r = u_y * y_r, u_rr = u_yy * y_r^2 + u_y * y_rr):
    C2 * u_yy + C1 * u_y + C0 * u = 0

with:
    C2 = Delta * y_r^2
    C1 = Delta * y_rr + [2*Delta*q_r + (s+1)*Delta_r] * y_r
    C0 = Delta*(q_rr + q_r^2) + (s+1)*Delta_r*q_r + V

y_r, y_rr from r = r_+/x, x = (y+1)/2:
    y_r  = -(y+1)^2 / (2*r_+)     or equivalently  y_r  = -2*r_+/r^2
    y_rr =  (y+1)^3 / (2*r_+^2)   or                 y_rr =  4*r_+/r^3

q_r, q_rr for each basis
-------------------------
Let rstar_r = (r^2 + a^2) / Delta,  rstar_rr = [2r*Delta - (r^2+a^2)*Delta_r] / Delta^2.

up (A_up = r^3 * exp(+i*omega*r_*)):
    q_r   =  3/r  + i*omega*rstar_r
    q_rr  = -3/r^2 + i*omega*rstar_rr

down (A_down = r^{-1} * exp(-i*omega*r_*)):
    q_r   = -1/r  - i*omega*rstar_r
    q_rr  =  1/r^2 - i*omega*rstar_rr
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
    if t is None:
        return None
    if t.ndim == 1:
        return t.unsqueeze(-1)
    return t


# ============================================================
# q_r / q_rr  (analytic, complex)
# ============================================================

def compute_q_terms(r, a, omega, basis, M=1.0):
    """Compute q_r = A_r/A and q_rr = d(q_r)/dr for up or down basis.

    Args:
        r:     (B,N) float
        a:     (B,1) float
        omega: (B,1) float
        basis: "up" or "down"
        M:     scalar

    Returns:
        q_r:  (B,N) complex
        q_rr: (B,N) complex
    """
    Delta = delta(r, a, M)
    Delta_r = delta_r(r, M)

    rstar_r = (r ** 2 + a ** 2) / Delta
    rstar_rr = (2.0 * r * Delta - (r ** 2 + a ** 2) * Delta_r) / (Delta ** 2)

    iw = 1j * omega  # (B,1) complex → broadcasts to (B,N)

    if basis == "up":
        q_r = 3.0 / r + iw * rstar_r
        q_rr = -3.0 / (r ** 2) + iw * rstar_rr
    else:
        q_r = -1.0 / r - iw * rstar_r
        q_rr = 1.0 / (r ** 2) - iw * rstar_rr

    return q_r, q_rr


# ============================================================
# u-equation residual  (training default)
# ============================================================

def compute_u_equation_residual(model, a, omega, y, lambda_, u=None, v=None,
                                M=1.0, s=-2, m=2, basis="up"):
    """Compute Teukolsky residual via the u-equation (no A_up/A_down blow-up).

    Returns:
        u_val:    (B,N) complex — the u-basis function
        residual: (B,N) complex — C2*u_yy + C1*u_y + C0*u
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

    # ---- u_y, u_yy via autograd (both create_graph=True for training) ----
    # Pointwise network so grad(sum, y) gives per-element gradient.
    u_y = torch.complex(
        torch.autograd.grad(u_val.real.sum(), y, create_graph=True, retain_graph=True)[0],
        torch.autograd.grad(u_val.imag.sum(), y, create_graph=True, retain_graph=True)[0],
    )  # (B,N)

    u_yy = torch.complex(
        torch.autograd.grad(u_y.real.sum(), y, create_graph=True, retain_graph=True)[0],
        torch.autograd.grad(u_y.imag.sum(), y, create_graph=True, retain_graph=True)[0],
    )  # (B,N)

    # ---- compute C2, C1, C0 in y-space ----
    rp = r_plus(a2, M)
    x = (y + 1.0) / 2.0
    r = rp / x

    # y_r = dy/dr, y_rr = d^2y/dr^2
    y1 = y + 1.0
    y_r = -y1 ** 2 / (2.0 * rp)
    y_rr = y1 ** 3 / (2.0 * rp ** 2)

    Delta = delta(r, a2, M)
    Delta_r_v = delta_r(r, M)
    V = V_of_r(r, a2, omega2, m, s, lambda2, M)

    q_r, q_rr = compute_q_terms(r, a2, omega2, basis, M)

    C2 = Delta * (y_r ** 2)
    C1 = Delta * y_rr + (2.0 * Delta * q_r + (s + 1.0) * Delta_r_v) * y_r
    C0 = Delta * (q_rr + q_r ** 2) + (s + 1.0) * Delta_r_v * q_r + V

    residual = C2 * u_yy + C1 * u_y + C0 * u_val
    pointwise = torch.abs(residual) ** 2

    return u_val, residual, pointwise


def compute_stage3_loss_u_equation(model, a, omega, y, lambda_, u=None, v=None,
                                   M=1.0, s=-2, m=2,
                                   weight_up=1.0, weight_down=1.0):
    """Compute Stage-3 loss using u-equation residual.

    Returns:
        total_loss: scalar
        info: dict with loss_up, loss_down, etc.
    """
    _, res_up, pw_up = compute_u_equation_residual(
        model, a, omega, y, lambda_, u, v, M, s, m, basis="up")
    _, res_down, pw_down = compute_u_equation_residual(
        model, a, omega, y, lambda_, u, v, M, s, m, basis="down")

    loss_up = torch.mean(pw_up)
    loss_down = torch.mean(pw_down)
    total_loss = weight_up * loss_up + weight_down * loss_down

    info = {
        "loss_up": float(loss_up.detach().cpu().item()),
        "loss_down": float(loss_down.detach().cpu().item()),
        "total_loss": float(total_loss.detach().cpu().item()),
    }
    return total_loss, info


# ============================================================
# direct R residual  (debug only)
# ============================================================

def compute_u_basis_residual(model, a, omega, y, lambda_, u=None, v=None,
                              M=1.0, s=-2, m=2, basis="up"):
    """[DEBUG] Compute Teukolsky residual via direct R = A * u.

    Numerically unstable at large r for the up basis (A_up ~ r^3).
    Use compute_u_equation_residual for training.
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

    if basis == "up":
        f = model.predict_u_up(a2, omega2, y, u2, v2)
    else:
        f = model.predict_u_down(a2, omega2, y, u2, v2)

    c_up, c_down = infinity_slopes_y(a2, omega2, lambda2, m=m, M=M)
    c_inf = c_up if basis == "up" else c_down
    u_val = compose_u_from_f(f, y, c_inf)

    rp = r_plus(a2, M)
    x = (y + 1.0) / 2.0
    r = rp / x

    if basis == "up":
        A = A_up(r, a2, omega2, M)
    else:
        A = A_down(r, a2, omega2, M)

    R = A * u_val

    R_y = torch.complex(
        torch.autograd.grad(R.real.sum(), y, create_graph=True, retain_graph=True)[0],
        torch.autograd.grad(R.imag.sum(), y, create_graph=True, retain_graph=True)[0],
    )
    R_yy = torch.complex(
        torch.autograd.grad(R_y.real.sum(), y, create_graph=True, retain_graph=True)[0],
        torch.autograd.grad(R_y.imag.sum(), y, create_graph=True, retain_graph=True)[0],
    )

    y1 = y + 1.0
    dy_dr = -y1 ** 2 / (2.0 * rp)
    d2y_dr2 = y1 ** 3 / (2.0 * rp ** 2)

    R_r = R_y * dy_dr
    R_rr = R_yy * dy_dr ** 2 + R_y * d2y_dr2

    Delta = delta(r, a2, M)
    Delta_r_v = delta_r(r, M)
    V = V_of_r(r, a2, omega2, m, s, lambda2, M)

    residual = Delta * R_rr + (s + 1.0) * Delta_r_v * R_r + V * R
    pointwise = torch.abs(residual) ** 2

    return R, residual, pointwise


def compute_stage3_loss(model, a, omega, y, lambda_, u=None, v=None,
                         M=1.0, s=-2, m=2,
                         weight_up=1.0, weight_down=1.0,
                         normalize_residual=False, eps=1e-12):
    """[DEBUG] Compute Stage-3 loss via direct R residual.

    Use compute_stage3_loss_u_equation for training.
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
