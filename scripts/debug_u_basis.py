#!/usr/bin/env python3
"""
Boundary math tests for u_basis.py.

Checks:
    - infinity_slopes_y returns du/dy, not du/dx
    - compose_u_from_f: u(y=-1) == 1
    - autograd: du/dy|_{y=-1} == c_inf (real + complex)
    - A_up/A_down import works (r_star from prefactor)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
torch.manual_seed(42)

from physical_ansatz.u_basis import (
    infinity_slope_up_x,
    infinity_slope_down_x,
    infinity_slopes_y,
    compose_u_from_f,
    A_up,
    A_down,
)


def check(desc, condition):
    assert condition, f"FAIL: {desc}"
    print(f"   OK — {desc}")


# ---- test params ----
B = 3
a = torch.tensor([0.1, 0.5, 0.9])
omega = torch.tensor([0.1, 1.0, 5.0])
lambda_ = torch.tensor([3.0 + 0.1j, 4.0 + 0.2j, 5.0 - 0.1j])
m = 2
M = 1.0

N = 64
y = torch.linspace(-1.0, 1.0, N, requires_grad=True)

print("1. Slope conversion check: du/dy = (1/2) du/dx...")
c_up_x = infinity_slope_up_x(a, omega, lambda_, m=m, M=M)
c_down_x = infinity_slope_down_x(a, omega, lambda_, m=m, M=M)
c_up_y, c_down_y = infinity_slopes_y(a, omega, lambda_, m=m, M=M)

err_up = (c_up_y - c_up_x / 2.0).abs().max().item()
err_down = (c_down_y - c_down_x / 2.0).abs().max().item()
check(f"c_up_y = c_up_x/2  (max err={err_up:.2e})", err_up < 1e-10)
check(f"c_down_y = c_down_x/2  (max err={err_down:.2e})", err_down < 1e-10)

print("2. compose_u_from_f boundary value u(y=-1)=1...")
# use a dummy f(y) that is non-zero at y=-1
y_batch = y.unsqueeze(0).expand(B, -1).clone().detach().requires_grad_(True)
f_dummy = torch.sin(3.0 * y_batch) + 1j * torch.cos(2.0 * y_batch)
c_inf = c_up_y  # (B,) complex

u = compose_u_from_f(f_dummy, y_batch, c_inf)
u_at_inf = u[:, 0]  # y=-1 is index 0 since y is linspace(-1,1,N)
err_u1 = (u_at_inf - 1.0).abs().max().item()
check(f"u(y=-1)=1 (max err={err_u1:.2e})", err_u1 < 1e-10)

print("3. autograd: du/dy|_{y=-1} == c_inf...")
# Compute du/dy at y=-1 using autograd
u_real = u.real.sum()
grad_u = torch.autograd.grad(u_real, y_batch, create_graph=True)[0]
# du/dy at y=-1 (index 0)
u_y_at_inf = grad_u[:, 0]

# For complex c_inf, we need to check real and imag separately
# The real part of u depends on the real part of the ansatz
# Use a simpler check: compute with finite diff
# Actually, let's just check: the ansatz u = 1 + c*(y+1) + (y+1)^2*f
# du/dy = c + 2*(y+1)*f + (y+1)^2*f_y
# At y=-1: du/dy = c  (since (y+1)=0)
# So we should have du/dy = c_inf

# Use a dummy f that's zero everywhere (simplest test)
f_zero = torch.zeros_like(y_batch)
u_zero = compose_u_from_f(f_zero, y_batch, c_down_y)
# du/dy should be c_down_y everywhere (since f=0)
# Let's check at a few points using autograd
y_test = y_batch[:1].clone().detach().requires_grad_(True)
u_test = compose_u_from_f(
    torch.zeros_like(y_test),
    y_test,
    c_down_y[:1].detach(),
)
du_dy_real = torch.autograd.grad(u_test.real.sum(), y_test, create_graph=True)[0]
du_dy_imag = torch.autograd.grad(u_test.imag.sum(), y_test, create_graph=True)[0]

du_dy_pred = du_dy_real + 1j * du_dy_imag
c_target = c_down_y[0]

# Check at y=-1 (index 0)
du_dy_at_inf = du_dy_pred[0, 0]
err_slope_real = abs(du_dy_at_inf.real - c_target.real)
err_slope_imag = abs(du_dy_at_inf.imag - c_target.imag)
check(f"du_down/dy real match (err={err_slope_real:.2e})", err_slope_real < 1e-10)
check(f"du_down/dy imag match (err={err_slope_imag:.2e})", err_slope_imag < 1e-10)

# Check at multiple y points that du/dy = c for f=0
full_err = (du_dy_pred - c_target).abs().max().item()
check(f"du/dy = c_inf everywhere for f=0 (err={full_err:.2e})", full_err < 1e-10)

print("4. A_up/A_down import works (r_star from prefactor)...")
r = torch.tensor([2.0, 10.0, 100.0])
a0 = torch.tensor(0.5)
omega0 = torch.tensor(1.0)
Aup = A_up(r, a0, omega0)
Adown = A_down(r, a0, omega0)
check("A_up finite", torch.all(torch.isfinite(Aup.abs())))
check("A_down finite", torch.all(torch.isfinite(Adown.abs())))
print(f"   A_up[0] = {Aup[0]:.4e}, A_down[0] = {Adown[0]:.4e}")

print("5. complex c_inf boundary check...")
# Test with non-trivial complex f and check du/dy at y=-1
y_c = y_batch[:1].detach().requires_grad_(True)
f_test = torch.randn(1, N, dtype=torch.complex64) * 0.1
# Use a small f so ansatz is dominated by linear term
c_test = c_up_y[0].detach()
u_c = compose_u_from_f(f_test, y_c, c_test)
# u = 1 + c*(y+1) + (y+1)^2*f
# du/dy = c + 2*(y+1)*f + (y+1)^2*f_y
# At y=-1: du/dy = c
# Use autograd on both real and imag
du_dy_c_re = torch.autograd.grad(u_c.real.sum(), y_c, retain_graph=True)[0]
du_dy_c_im = torch.autograd.grad(u_c.imag.sum(), y_c, create_graph=True)[0]
du_dy_c = du_dy_c_re + 1j * du_dy_c_im
err_c = (du_dy_c[0, 0] - c_test).abs().item()
check(f"complex c_inf boundary derivative (err={err_c:.2e})", err_c < 1e-10)

print()
print("=" * 60)
print("ALL U_BASIS CHECKS PASSED")
print("=" * 60)
