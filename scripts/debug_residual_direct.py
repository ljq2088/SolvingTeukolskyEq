#!/usr/bin/env python3
"""Direct PDE residual comparison: r-space (gold) vs f(y)-ansatz path.

Computes the Teukolsky residual Δ*R_rr + (s+1)*Δ_r*R_r + V*R = 0
directly from pybhpt R(r) using finite differences, then compares against
the f(y)-ansatz residual computation used in training.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from physical_ansatz.mapping import r_plus, r_from_x, dx_dr_from_x, d2x_dr2_from_x
from physical_ansatz.prefactor import (
    delta, delta_r, V_of_r,
    build_prefactor_primitives, Leaver_prefactors,
)
from physical_ansatz.teukolsky_coeffs import coeffs_x
from physical_ansatz.transform_y import (
    transform_coeffs_x_to_y,
    horizon_regularity_slope,
    h_factor,
    compose_reduced_shape_from_f,
    h1_factor, g_factor,
)
from utils.compute_lambda_usage import compute_lambda


def compute_direct_r_residual(r, R, a, omega, lam, m=2, s=-2, M=1.0):
    """Compute Teukolsky residual directly in r-space using finite differences.

    Residual = Δ*R_rr + (s+1)*Δ_r*R_r + V*R

    Returns pointwise |residual|^2.
    """
    dr = np.diff(r)
    dr = np.concatenate([dr, dr[-1:]])  # pad last point

    # First derivative: central differences interior, forward/backward at edges
    R_r = np.zeros_like(R, dtype=np.complex128)
    R_r[1:-1] = (R[2:] - R[:-2]) / (r[2:] - r[:-2])
    R_r[0] = (R[1] - R[0]) / (r[1] - r[0])
    R_r[-1] = (R[-1] - R[-2]) / (r[-1] - r[-2])

    # Second derivative
    R_rr = np.zeros_like(R, dtype=np.complex128)
    R_rr[1:-1] = (R_r[2:] - R_r[:-2]) / (r[2:] - r[:-2])
    R_rr[0] = (R_r[1] - R_r[0]) / (r[1] - r[0])
    R_rr[-1] = (R_r[-1] - R_r[-2]) / (r[-1] - r[-2])

    Delta = delta(torch.tensor(r), torch.tensor([a]), M).numpy()
    Delta_r = delta_r(torch.tensor(r), M).numpy()
    V = V_of_r(torch.tensor(r), torch.tensor([a]), torch.tensor([omega]),
               m, s, torch.tensor([lam]), M).numpy()

    residual = Delta * R_rr + (s + 1) * Delta_r * R_r + V * R
    return np.abs(residual)**2


def compute_f_ansatz_residual(r, R, a, omega, lam, m=2, s=-2, M=1.0):
    """Compute residual through the f(y) ansatz path (what the model sees).

    Reconstructs f(y) from R(r), then computes B2*f_yy + B1*f_y + B0*f - rhs.
    """
    dtype = torch.float64
    cdtype = torch.complex128

    a_t = torch.tensor([a], dtype=dtype)
    omega_t = torch.tensor([omega], dtype=dtype)
    lam_t = torch.tensor([lam], dtype=cdtype)
    r_t = torch.tensor(r, dtype=dtype)

    # Compute S = R / (P * h2)
    rp = r_plus(a_t, M)
    _, rm, _, _, _ = build_prefactor_primitives(r_t, a_t, M=M, need_rs=False)
    P, P_r, P_rr = Leaver_prefactors(r_t, a_t, omega_t, m=m, M=M, s=s, rp=rp, rm=rm)
    h2 = h_factor(a_t, omega_t, m=m, M=M, s=s)
    S = torch.tensor(R, dtype=cdtype) / (P * h2)

    # Convert to x, y coordinates
    x = rp / r_t
    y = 2.0 * x - 1.0

    # Compute slope
    slope = horizon_regularity_slope(a=a_t, omega=omega_t, lambda_=lam_t, m=m, M=M, s=s)

    # Invert ansatz to get f: S = g*(h1*f + 1) + 1 => f = (S - g - 1) / (g*h1)
    h1, _, _ = h1_factor(x)
    g, _, _ = g_factor(x, slope)
    # f = (S - g - 1) / (g * h1)
    denom = g * h1
    f = (S - g - 1.0) / denom

    # Compute f_y, f_yy via finite differences on y-grid
    y_np = y.numpy()
    f_np = f.detach().numpy()

    dy = np.diff(y_np)
    dy = np.concatenate([dy, dy[-1:]])

    f_y = np.zeros_like(f_np, dtype=np.complex128)
    f_y[1:-1] = (f_np[2:] - f_np[:-2]) / (y_np[2:] - y_np[:-2])
    f_y[0] = (f_np[1] - f_np[0]) / (y_np[1] - y_np[0])
    f_y[-1] = (f_np[-1] - f_np[-2]) / (y_np[-1] - y_np[-2])

    f_yy = np.zeros_like(f_np, dtype=np.complex128)
    f_yy[1:-1] = (f_y[2:] - f_y[:-2]) / (y_np[2:] - y_np[:-2])
    f_yy[0] = (f_y[1] - f_y[0]) / (y_np[1] - y_np[0])
    f_yy[-1] = (f_y[-1] - f_y[-2]) / (y_np[-1] - y_np[-2])

    f_y_t = torch.tensor(f_y, dtype=cdtype)
    f_yy_t = torch.tensor(f_yy, dtype=cdtype)

    # Compute transformed coefficients
    A2, A1, A0 = coeffs_x(x=x, a=a_t, omega=omega_t, m=m, lambda_=lam_t, s=s, M=M)
    B2, B1, B0, rhs = transform_coeffs_x_to_y(A2, A1, A0, y.unsqueeze(0), slope=slope)

    B2_np = B2.squeeze(0).detach().numpy()
    B1_np = B1.squeeze(0).detach().numpy()
    B0_np = B0.squeeze(0).detach().numpy()
    rhs_np = rhs.squeeze(0).detach().numpy()

    residual = B2_np * f_yy_t.numpy() + B1_np * f_y_t.numpy() + B0_np * f_np - rhs_np
    pointwise = np.abs(residual)**2

    return pointwise, {
        "f": f_np, "f_y": f_y, "f_yy": f_yy,
        "B2": B2_np, "B1": B1_np, "B0": B0_np, "rhs": rhs_np,
        "S": S.detach().numpy(),
    }


def compute_Sspace_residual(r, R, a, omega, lam, m=2, s=-2, M=1.0):
    """Compute residual in S-space: A2*S_xx + A1*S_x + A0*S (x-coordinates).

    Bypasses the f(y) ansatz entirely. S = R/(P*h2), then finite-diff in x.
    """
    dtype = torch.float64
    cdtype = torch.complex128

    a_t = torch.tensor([a], dtype=dtype)
    omega_t = torch.tensor([omega], dtype=dtype)
    lam_t = torch.tensor([lam], dtype=cdtype)
    r_t = torch.tensor(r, dtype=dtype)

    rp = r_plus(a_t, M)
    _, rm, _, _, _ = build_prefactor_primitives(r_t, a_t, M=M, need_rs=False)
    P, P_r, P_rr = Leaver_prefactors(r_t, a_t, omega_t, m=m, M=M, s=s, rp=rp, rm=rm)
    h2 = h_factor(a_t, omega_t, m=m, M=M, s=s)
    S = torch.tensor(R, dtype=cdtype) / (P * h2)

    x = (rp / r_t).detach().numpy()
    S_np = S.detach().numpy()

    # Finite differences in x
    dx = np.diff(x)
    dx = np.concatenate([dx, dx[-1:]])

    S_x = np.zeros_like(S_np, dtype=np.complex128)
    S_x[1:-1] = (S_np[2:] - S_np[:-2]) / (x[2:] - x[:-2])
    S_x[0] = (S_np[1] - S_np[0]) / (x[1] - x[0])
    S_x[-1] = (S_np[-1] - S_np[-2]) / (x[-1] - x[-2])

    S_xx = np.zeros_like(S_np, dtype=np.complex128)
    S_xx[1:-1] = (S_x[2:] - S_x[:-2]) / (x[2:] - x[:-2])
    S_xx[0] = (S_x[1] - S_x[0]) / (x[1] - x[0])
    S_xx[-1] = (S_x[-1] - S_x[-2]) / (x[-1] - x[-2])

    # Compute A2, A1, A0
    x_t = torch.tensor(x, dtype=dtype).unsqueeze(0)
    A2, A1, A0 = coeffs_x(x=x_t, a=a_t, omega=omega_t, m=m, lambda_=lam_t, s=s, M=M)
    A2_np = A2.squeeze(0).detach().numpy()
    A1_np = A1.squeeze(0).detach().numpy()
    A0_np = A0.squeeze(0).detach().numpy()

    residual = A2_np * S_xx + A1_np * S_x + A0_np * S_np
    return np.abs(residual)**2, {"S": S_np, "S_x": S_x, "S_xx": S_xx,
                                  "A2": A2_np, "A1": A1_np, "A0": A0_np}


def main():
    a, omega = 0.5, 0.0257
    ell, m, s = 2, 2, -2
    M = 1.0
    lam = compute_lambda(a, omega, ell, m, s=s)

    print(f"a={a}, omega={omega}, lambda={lam:.10f}")

    # Get pybhpt solution
    from pybhpt_usage.compute_solution import compute_pybhpt_solution
    rp = float(M + np.sqrt(M**2 - a**2))
    r_grid = np.logspace(np.log10(rp + 0.001), 2.0, 256)
    r_ref, R_ref = compute_pybhpt_solution(a, omega, ell=ell, m=m, r_grid=r_grid, timeout=30.0)
    R_ref = np.asarray(R_ref, dtype=np.complex128)

    print(f"\nr range: [{r_ref[0]:.4f}, {r_ref[-1]:.4f}], {len(r_ref)} points")

    # 1. Direct r-space residual (gold standard)
    res_r = compute_direct_r_residual(r_ref, R_ref, a, omega, lam, m=m, s=s, M=M)
    print(f"\n=== Direct r-space residual (Δ*R_rr + (s+1)*Δ_r*R_r + V*R) ===")
    print(f"  mean |res|^2: {np.mean(res_r):.6e}")
    print(f"  median:       {np.median(res_r):.6e}")
    print(f"  max:          {np.max(res_r):.6e}")
    print(f"  min:          {np.min(res_r):.6e}")

    # 2. S-space residual (x-coordinates, bypassing f(y))
    res_S, info_S = compute_Sspace_residual(r_ref, R_ref, a, omega, lam, m=m, s=s, M=M)
    print(f"\n=== S-space residual (A2*S_xx + A1*S_x + A0*S) ===")
    print(f"  mean |res|^2: {np.mean(res_S):.6e}")
    print(f"  median:       {np.median(res_S):.6e}")
    print(f"  max:          {np.max(res_S):.6e}")
    print(f"  min:          {np.min(res_S):.6e}")
    print(f"  S(1)  = {info_S['S'][-1]}  (expected 1+0j)")
    print(f"  S_x(1)= {info_S['S_x'][-1]}")

    # 3. f(y)-ansatz residual (what the model minimizes)
    res_f, info_f = compute_f_ansatz_residual(r_ref, R_ref, a, omega, lam, m=m, s=s, M=M)
    print(f"\n=== f(y)-ansatz residual (B2*f_yy + B1*f_y + B0*f - rhs) ===")
    print(f"  mean |res|^2: {np.mean(res_f):.6e}")
    print(f"  median:       {np.median(res_f):.6e}")
    print(f"  max:          {np.max(res_f):.6e}")
    print(f"  min:          {np.min(res_f):.6e}")

    # 4. Show where f(y) inversion is most singular
    y = 2.0 * rp / r_ref - 1.0
    denom = info_f.get("B2", None)
    if denom is not None:
        sort_idx = np.argsort(y)
        print(f"\n=== Top-5 largest residual points (f-ansatz) ===")
        top5 = np.argsort(res_f)[-5:][::-1]
        for i in top5:
            print(f"  y={y[i]:.4f}, r={r_ref[i]:.4f}, |res|^2={res_f[i]:.2e}, "
                  f"|B2|={np.abs(info_f['B2'][i]):.2e}, |f|={np.abs(info_f['f'][i]):.2e}")

    # 5. Coefficient magnitude analysis
    print(f"\n=== Coefficient magnitudes (f-ansatz) ===")
    for name in ["B2", "B1", "B0", "rhs"]:
        vals = np.abs(info_f[name])
        print(f"  |{name}|: mean={np.mean(vals):.4e}, median={np.median(vals):.4e}, "
              f"min={np.min(vals):.4e}, max={np.max(vals):.4e}")

    # 6. Consistency check: does S(y) from f(y) match S_direct from R/(P*h2)?
    S_from_f = info_f.get("S")
    S_direct = info_S.get("S")
    if S_from_f is not None and S_direct is not None:
        # S_from_f wasn't computed from f, it was just S=R/(P*h2)
        # Let's compute S from the reconstructed f
        dtype = torch.float64
        cdtype = torch.complex128
        a_t = torch.tensor([a], dtype=dtype)
        omega_t = torch.tensor([omega], dtype=dtype)
        lam_t = torch.tensor([lam], dtype=cdtype)
        slope = horizon_regularity_slope(a=a_t, omega=omega_t, lambda_=lam_t, m=m, M=M, s=s)
        y_t = torch.tensor(y, dtype=dtype)
        f_from_R = torch.tensor(info_f["f"], dtype=cdtype)
        S_reconstructed = compose_reduced_shape_from_f(
            f=f_from_R.unsqueeze(0), y=y_t.unsqueeze(0), slope=slope,
        ).squeeze(0).detach().numpy()
        S_diff = np.abs(S_reconstructed - S_direct)
        print(f"\n=== S reconstruction consistency ===")
        print(f"  |S_reconstructed - S_direct|: mean={np.mean(S_diff):.4e}, max={np.max(S_diff):.4e}")


if __name__ == "__main__":
    main()
