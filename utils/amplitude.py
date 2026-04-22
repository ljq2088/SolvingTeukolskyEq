from __future__ import annotations
import math
from typing import Dict, Tuple

import numpy as np

from utils.mode import InAmplitudesResult, KerrMode


def _safe_complex_ratio(numer: complex, denom: complex, *, tol: float = 1.0e-30) -> complex | None:
    if abs(denom) <= tol:
        return None
    return numer / denom


def Delta(r, mode: KerrMode):
    return (r - mode.rp) * (r - mode.rm)


def Delta_p(r, mode: KerrMode):
    return 2.0 * r - 2.0 * mode.M


def K_of_r(r, mode: KerrMode):
    return (r * r + mode.a * mode.a) * mode.omega - mode.a * mode.m


def K_p(r, mode: KerrMode):
    return 2.0 * r * mode.omega


def r_star(r, mode: KerrMode):
    d = mode.delta_h
    rp = mode.rp
    rm = mode.rm
    return r + 2.0 * rp / d * np.log((r - rp) / 2.0) - 2.0 * rm / d * np.log((r - rm) / 2.0)


def drstar_dr(r, mode: KerrMode):
    return (r * r + mode.a * mode.a) / Delta(r, mode)


def d_dr_drstar(r, mode: KerrMode):
    P = r * r + mode.a * mode.a
    D = Delta(r, mode)
    Dp = Delta_p(r, mode)
    return (2.0 * r * D - P * Dp) / (D * D)


def phase_int_K_over_Delta(r, mode: KerrMode):
    d = mode.delta_h
    return mode.omega * r_star(r, mode) - mode.a * mode.m / d * np.log((r - mode.rp) / (r - mode.rm))


def r_of_z(z, mode: KerrMode):
    return mode.rp / z


def dz_dr(z, mode: KerrMode):
    return -(z * z) / mode.rp


def d2z_dr2(z, mode: KerrMode):
    return 2.0 * z**3 / (mode.rp**2)


def A_down(r, mode: KerrMode):
    return r ** (-1.0) * np.exp(-1j * mode.omega * r_star(r, mode))


def A_up(r, mode: KerrMode):
    return r ** 3 * np.exp(+1j * mode.omega * r_star(r, mode))


def A_in(r, mode: KerrMode):
    return Delta(r, mode) ** 2 * np.exp(-1j * mode.k_hor * r_star(r, mode))

def A_out(r, mode: KerrMode):
    return np.exp(+1j * mode.k_hor * r_star(r, mode))

def q_and_qp(r, mode: KerrMode, basis: str):
    rst = drstar_dr(r, mode)
    rst_p = d_dr_drstar(r, mode)
    D = Delta(r, mode)
    Dp = Delta_p(r, mode)
    K = K_of_r(r, mode)
    Kp = K_p(r, mode)
    if basis == 'down':
        q = -1.0 / r - 1j * mode.omega * rst
        qp = 1.0 / (r * r) - 1j * mode.omega * rst_p
    elif basis == 'up':
        q = 3.0 / r + 1j * mode.omega * rst
        qp = -3.0 / (r * r) + 1j * mode.omega * rst_p
    elif basis == 'in':
        rst = drstar_dr(r, mode)
        rst_p = d_dr_drstar(r, mode)
        q = 2.0 * Dp / D - 1j * mode.k_hor * rst
        qp = 2.0 * (2.0 * D - Dp * Dp) / (D * D) - 1j * mode.k_hor * rst_p
    elif basis == 'out':
        rst = drstar_dr(r, mode)
        rst_p = d_dr_drstar(r, mode)
        q = 1j * mode.k_hor * rst
        qp = 1j * mode.k_hor * rst_p
    else:
        raise ValueError
    return q, qp
def boundary_du_exact(mode: KerrMode, basis: str, side: str) -> complex:
    rp = mode.rp
    rm = mode.rm
    d = mode.delta_h
    Pp = rp * rp + mode.a * mode.a
    om = mode.omega
    lam = mode.lambda_value
    am = mode.a * mode.m
    kH = mode.k_hor

    if side == 'left':
        if basis == 'down':
            return (
                -2.0 * (rp + rm) / rp
                + 1j * (2.0 - am * om - 0.5 * lam) / (om * rp)
            )
        elif basis == 'up':
            return 1j * (am * om + 0.5 * lam) / (om * rp)
        else:
            raise ValueError("left boundary only supports 'down' and 'up'")

    elif side == 'right':
        if basis == 'in':
            return (
                rp * (
                    d * (lam - 4.0 + 1j * (6.0 * kH * rp + 4.0 * om * rp))
                    - 4.0 * am * kH * rp
                )
                / (d * (2.0j * kH * Pp - 3.0 * d))
            )
        elif basis == 'out':
            return (
                rp * (
                    4.0 * am * kH * rp
                    - d * (lam + 1j * (2.0 * kH * rp + 4.0 * om * rp))
                )
                / (d * (2.0j * kH * Pp - d))
            )
        else:
            raise ValueError("right boundary only supports 'in' and 'out'")

    else:
        raise ValueError

def coeffs_numeric(z, mode: KerrMode, basis: str):
    r = r_of_z(z, mode)
    zr = dz_dr(z, mode)
    zrr = d2z_dr2(z, mode)
    D = Delta(r, mode)
    Dp = Delta_p(r, mode)
    K = K_of_r(r, mode)
    V = (K * K + 4j * (r - mode.M) * K) / D - 8j * mode.omega * r - mode.lambda_value
    q, qp = q_and_qp(r, mode, basis)
    B2 = D * zr * zr
    B1 = D * (zrr + 2.0 * q * zr) - Dp * zr
    B0 = D * (qp + q * q) - Dp * q + V
    return B2.astype(complex), B1.astype(complex), B0.astype(complex)


def cheb_D(N: int, a: float, b: float) -> Tuple[np.ndarray, np.ndarray]:
    xi = np.cos(np.pi * np.arange(N + 1) / N)
    c = np.ones(N + 1)
    c[0] = c[-1] = 2.0
    D = np.zeros((N + 1, N + 1), dtype=float)
    for i in range(N + 1):
        for j in range(N + 1):
            if i != j:
                D[i, j] = (c[i] / c[j]) * ((-1) ** (i + j)) / (xi[i] - xi[j])
    D[np.diag_indices(N + 1)] = -np.sum(D, axis=1)
    z = a + (b - a) * (1.0 - xi) / 2.0
    D = (-2.0 / (b - a)) * D
    return D, z


def solve_basis_domain(mode: KerrMode, basis: str, N: int, z_a: float, z_b: float, bc_side: str):
    D, z = cheb_D(N, z_a, z_b)
    D2 = D @ D

    A = np.zeros((N + 1, N + 1), dtype=complex)
    b = np.zeros(N + 1, dtype=complex)

    if bc_side == 'left':
        # 真端点在 z=0
        du0 = boundary_du_exact(mode, basis, 'left')

        # BC 1: u(0)=1
        A[0, :] = 0.0
        A[0, 0] = 1.0
        b[0] = 1.0

        # BC 2: u_z(0)=du0
        A[1, :] = D[0, :]
        b[1] = du0

        # PDE rows: i = 2, ..., N
        idx = np.arange(2, N + 1)
        B2, B1, B0 = coeffs_numeric(z[idx], mode, basis)
        A[idx, :] = B2[:, None] * D2[idx, :] + B1[:, None] * D[idx, :]
        A[idx, idx] += B0

    elif bc_side == 'right':
        # 真端点在 z=1
        du1 = boundary_du_exact(mode, basis, 'right')

        # PDE rows: i = 0, ..., N-2
        idx = np.arange(0, N - 1)
        B2, B1, B0 = coeffs_numeric(z[idx], mode, basis)
        A[idx, :] = B2[:, None] * D2[idx, :] + B1[:, None] * D[idx, :]
        A[idx, idx] += B0

        # BC 1: u_z(1)=du1
        A[-2, :] = D[-1, :]
        b[-2] = du1

        # BC 2: u(1)=1
        A[-1, :] = 0.0
        A[-1, -1] = 1.0
        b[-1] = 1.0

    else:
        raise ValueError

    u = np.linalg.solve(A, b)
    uz = D @ u
    return {'z': z, 'u': u, 'uz': uz}
    

def basis_values_at_match(mode: KerrMode, basis: str, sol, side: str):
    z = sol['z']; u = sol['u']; uz = sol['uz']
    if side == 'left':
        zm, um, uzm = z[0], u[0], uz[0]
    else:
        zm, um, uzm = z[-1], u[-1], uz[-1]
    r = r_of_z(zm, mode)
    dzdr = dz_dr(zm, mode)
    if basis == 'in':
        F = A_in(r, mode)
    elif basis == 'out':
        F = A_out(r, mode)
    elif basis == 'down':
        F = A_down(r, mode)
    elif basis == 'up':
        F = A_up(r, mode)
    else:
        raise ValueError
    q, _ = q_and_qp(r, mode, basis)
    Rm = F * um
    Rm_r = F * (dzdr * uzm + q * um)
    return complex(Rm), complex(Rm_r)
# def boundary_du_true(mode: KerrMode, basis: str, side: str, eps: float):
#     """
#     在真边界附近拟合 u_z 的局部展开：

#     左端 z=0:
#         u_z(z) ≈ c0 + c1 z + c2 z^2
#     右端 y=1-z=0:
#         u_z(y) ≈ c0 + c1 y + c2 y^2

#     返回 (c0, c1)，这样既能构造
#         u(eps)
#     也能构造
#         u_z(eps)
#     """
#     ss = eps * np.array([0.25, 0.5, 0.75, 1.0], dtype=float)

#     if side == 'left':
#         z_s = ss
#         _, B1, B0 = coeffs_numeric(z_s, mode, basis)
#         du_s = -(B0 / B1)

#         V = np.vstack([np.ones_like(ss), ss, ss**2]).T
#         c, *_ = np.linalg.lstsq(V, du_s, rcond=None)
#         return complex(c[0]), complex(c[1])

#     elif side == 'right':
#         y_s = ss
#         z_s = 1.0 - y_s
#         _, B1, B0 = coeffs_numeric(z_s, mode, basis)
#         du_s = -(B0 / B1)

#         V = np.vstack([np.ones_like(y_s), y_s, y_s**2]).T
#         c, *_ = np.linalg.lstsq(V, du_s, rcond=None)
#         return complex(c[0]), complex(c[1])

#     else:
#         raise ValueError
# from typing import Callable
def abel_constant_from_pair(
    Ra: complex,
    Ra_r: complex,
    Rb: complex,
    Rb_r: complex,
    r: float,
    mode: KerrMode,
) -> complex:
    """
    Abel-type bilinear invariant for the current Teukolsky radial equation:
        C[Ra,Rb] = (Ra*Rb_r - Ra_r*Rb) / Delta(r)
    which is constant for any two solutions Ra, Rb of the same ODE.
    """
    D = Delta(r, mode)
    return (Ra * Rb_r - Ra_r * Rb) / D


def outer_abel_theory(mode: KerrMode) -> complex:
    """
    Theoretical constant for the outer basis pair (down, up):
        C_out = W(R_down, R_up) / Delta = 2 i omega
    """
    return 2.0j * mode.omega


def inner_abel_theory(mode: KerrMode) -> complex:
    """
    Theoretical constant for the inner basis pair (in, out):
        C_in = W(R_in, R_out) / Delta
             = 2 i k_H (r_+^2 + a^2) - 2 (r_+ - r_-)
    """
    return 2.0j * mode.k_hor * (mode.rp * mode.rp + mode.a * mode.a) - 2.0 * mode.delta_h


def rel_complex_residual(val: complex, ref: complex, floor: float = 1.0e-30) -> float:
    return abs(val - ref) / max(abs(ref), floor)

def leaver_factor(r, mode: KerrMode):
    """
    Project-consistent Leaver prefactor:
        P(r) = exp(i ω r) (r-r_+)^pp (r-r_-)^pm
    with
        pp = -s - i σ_p
        pm = -1 - s + 2 i ω + i σ_p
        σ_p = (2 ω r_+ - a m)/(r_+ - r_-)
    """
    r = np.asarray(r, dtype=np.float64)
    sigma_p = (2.0 * mode.omega * mode.rp - mode.a * mode.m) / (mode.rp - mode.rm)
    pp = -mode.s - 1j * sigma_p
    pm = -1.0 - mode.s + 2.0j * mode.omega + 1j * sigma_p

    drp = (r - mode.rp).astype(np.complex128)
    drm = (r - mode.rm).astype(np.complex128)
    return np.exp(1j * mode.omega * r) * (drp ** pp) * (drm ** pm)


def _basis_full_profile(mode: KerrMode, basis: str, sol, *, drop_left=False, drop_right=False):
    """
    Return nodal full-R profile on one spectral patch.
    """
    z = np.asarray(sol["z"], dtype=np.float64)
    u = np.asarray(sol["u"], dtype=np.complex128)

    mask = np.ones_like(z, dtype=bool)
    if drop_left:
        mask[0] = False
    if drop_right:
        mask[-1] = False

    z = z[mask]
    u = u[mask]
    r = r_of_z(z, mode)

    if basis == "in":
        F = A_in(r, mode)
    elif basis == "out":
        F = A_out(r, mode)
    elif basis == "down":
        F = A_down(r, mode)
    elif basis == "up":
        F = A_up(r, mode)
    else:
        raise ValueError(f"Unknown basis={basis}")

    R = F * u
    return {"z": z, "r": r, "u": u, "R": R}


def _complex_interp_linear(x_query, x_nodes, y_nodes):
    """
    Complex-valued piecewise linear interpolation.
    """
    xq = np.asarray(x_query, dtype=np.float64)
    yn = np.asarray(y_nodes, dtype=np.complex128)
    xr = np.asarray(x_nodes, dtype=np.float64)

    re = np.interp(xq, xr, yn.real)
    im = np.interp(xq, xr, yn.imag)
    out = re + 1j * im
    if np.ndim(x_query) == 0:
        return complex(out)
    return out


def _one_sided_linear_extrap(x_target, x1, x2, y1, y2):
    """
    One-sided linear extrapolation for complex values.
    """
    return y2 + (x_target - x2) * (y2 - y1) / (x2 - x1)


class InModeLeaverInterpolant:
    """
    Piecewise interpolant for the true in-mode solution.

    Internally interpolate:
        Psi(z) = R_in(z) / P_Leaver(r(z))

    Then reconstruct:
        R_in(z) = Psi(z) * P_Leaver(r(z))

    Notes
    -----
    * psi_of_z(z): defined on [0,1]
    * R_of_z(z): intended for 0 < z < 1
    * R_of_r(r): intended for finite r > r_+
    """
    def __init__(
        self,
        mode: KerrMode,
        z_outer: np.ndarray,
        psi_outer: np.ndarray,
        z_inner: np.ndarray,
        psi_inner: np.ndarray,
        z_match: float,
    ):
        self.mode = mode
        self.z_outer = np.asarray(z_outer, dtype=np.float64)
        self.psi_outer = np.asarray(psi_outer, dtype=np.complex128)
        self.z_inner = np.asarray(z_inner, dtype=np.float64)
        self.psi_inner = np.asarray(psi_inner, dtype=np.complex128)
        self.z_match = float(z_match)

    def psi_of_z(self, z):
        zq = np.asarray(z, dtype=np.float64)
        out = np.empty_like(zq, dtype=np.complex128)

        mask_outer = zq <= self.z_match
        if np.any(mask_outer):
            out[mask_outer] = _complex_interp_linear(
                zq[mask_outer], self.z_outer, self.psi_outer
            )
        if np.any(~mask_outer):
            out[~mask_outer] = _complex_interp_linear(
                zq[~mask_outer], self.z_inner, self.psi_inner
            )

        if np.ndim(z) == 0:
            return complex(out)
        return out

    __call__ = psi_of_z

    def R_of_z(self, z):
        zq = np.asarray(z, dtype=np.float64)
        if np.any(zq <= 0.0) or np.any(zq >= 1.0):
            raise ValueError("R_of_z is intended for 0 < z < 1. Use psi_of_z on [0,1].")
        r = r_of_z(zq, self.mode)
        return self.psi_of_z(zq) * leaver_factor(r, self.mode)

    def psi_of_r(self, r):
        rq = np.asarray(r, dtype=np.float64)
        if np.any(rq <= self.mode.rp):
            raise ValueError(f"psi_of_r requires r > r_+ = {self.mode.rp}")
        z = self.mode.rp / rq
        return self.psi_of_z(z)

    def R_of_r(self, r):
        rq = np.asarray(r, dtype=np.float64)
        if np.any(rq <= self.mode.rp):
            raise ValueError(f"R_of_r requires r > r_+ = {self.mode.rp}")
        z = self.mode.rp / rq
        return self.R_of_z(z)
def compute_smatrix(
    mode: KerrMode,
    N_in: int = 80,
    N_out: int = 80,
    z_m: float = 0.3,
    *,
    return_profile: bool = False,
):
    # outer domain: [0, z_m]
    sol_down = solve_basis_domain(mode, 'down', N_out, 0.0, z_m, 'left')
    sol_up   = solve_basis_domain(mode, 'up',   N_out, 0.0, z_m, 'left')

    R_down_m, Rr_down_m = basis_values_at_match(mode, 'down', sol_down, 'right')
    R_up_m,   Rr_up_m   = basis_values_at_match(mode, 'up',   sol_up,   'right')
    Mmatch = np.array([[R_down_m, R_up_m], [Rr_down_m, Rr_up_m]], dtype=complex)

    # inner domain: [z_m, 1]
    sol_in  = solve_basis_domain(mode, 'in',  N_in, z_m, 1.0, 'right')
    sol_out = solve_basis_domain(mode, 'out', N_in, z_m, 1.0, 'right')

    R_in_m,  Rr_in_m  = basis_values_at_match(mode, 'in',  sol_in,  'left')
    R_out_m, Rr_out_m = basis_values_at_match(mode, 'out', sol_out, 'left')

    Cin_down, Cin_up   = np.linalg.solve(Mmatch, np.array([R_in_m,  Rr_in_m],  dtype=complex))
    Cout_down, Cout_up = np.linalg.solve(Mmatch, np.array([R_out_m, Rr_out_m], dtype=complex))

    b_inc = complex(Cin_down)
    b_ref = complex(Cin_up)
    b_trans = 1.0 + 0.0j
    ratio_ref_over_inc = _safe_complex_ratio(b_ref, b_inc)
    ratio_inc_over_ref = _safe_complex_ratio(b_inc, b_ref)

    result = {
        'S': np.array([[Cin_down, Cin_up], [Cout_down, Cout_up]], dtype=complex),
        'B_inc': b_inc,
        'B_ref': b_ref,
        'B_trans': b_trans,
        'ratio_ref_over_inc': ratio_ref_over_inc,
        'ratio_inc_over_ref': ratio_inc_over_ref,
        'B_trans_over_B_inc': _safe_complex_ratio(b_trans, b_inc),
        'B_ref_over_B_inc': ratio_ref_over_inc,
    }

    if not return_profile:
        return result

    # ---------------------------------------------------------
    # Assemble true R_in on both patches, then interpolate R_in / P_Leaver
    # ---------------------------------------------------------

    # Outer patch: true in-solution = B_inc * down + B_ref * up
    prof_down = _basis_full_profile(mode, 'down', sol_down, drop_left=True, drop_right=False)
    prof_up   = _basis_full_profile(mode, 'up',   sol_up,   drop_left=True, drop_right=False)

    z_outer = prof_down["z"]
    r_outer = prof_down["r"]
    Rin_outer = b_inc * prof_down["R"] + b_ref * prof_up["R"]
    P_outer = leaver_factor(r_outer, mode)
    psi_outer = Rin_outer / P_outer

    # z -> 0 exact reduced limit:
    # A_up / P_Leaver -> 1, A_down / P_Leaver -> 0, so Psi(0) = B_ref
    z_outer_full = np.concatenate(([0.0], z_outer))
    psi_outer_full = np.concatenate(([b_ref], psi_outer))

    # Inner patch: true in-solution = B_trans * in, here B_trans = 1
    prof_in = _basis_full_profile(mode, 'in', sol_in, drop_left=False, drop_right=True)

    z_inner = prof_in["z"]
    r_inner = prof_in["r"]
    Rin_inner = b_trans * prof_in["R"]
    P_inner = leaver_factor(r_inner, mode)
    psi_inner = Rin_inner / P_inner

    # z -> 1 reduced limit by one-sided extrapolation
    if len(z_inner) >= 2:
        psi_h = _one_sided_linear_extrap(1.0, z_inner[-2], z_inner[-1], psi_inner[-2], psi_inner[-1])
    else:
        psi_h = psi_inner[-1]

    z_inner_full = np.concatenate((z_inner, [1.0]))
    psi_inner_full = np.concatenate((psi_inner, [psi_h]))

    profile = InModeLeaverInterpolant(
        mode=mode,
        z_outer=z_outer_full,
        psi_outer=psi_outer_full,
        z_inner=z_inner_full,
        psi_inner=psi_inner_full,
        z_match=z_m,
    )

    result["profile"] = profile
    result["z_outer"] = z_outer_full
    result["psi_outer"] = psi_outer_full
    result["z_inner"] = z_inner_full
    result["psi_inner"] = psi_inner_full

    return result

def compute_smatrix_with_abel(mode: KerrMode, N_in: int = 80, N_out: int = 80, z_m: float = 0.3):
    """
    Same transfer-matrix computation as compute_smatrix, but also returns
    Abel-invariant diagnostics suitable for Kerr Teukolsky with complex potential.

    Returned diagnostics:
      - outer_abel_num, outer_abel_th, outer_abel_residual
      - inner_abel_num, inner_abel_th, inner_abel_residual
      - detS_num, detS_th, detS_residual

    Notes
    -----
    * outer_abel checks the pair (R_down, R_up)
    * inner_abel checks the pair (R_in, R_out)
    * detS check validates the whole matching / amplitude extraction process
    """

    # outer domain: [0, z_m]
    sol_down = solve_basis_domain(mode, 'down', N_out, 0.0, z_m, 'left')
    sol_up   = solve_basis_domain(mode, 'up',   N_out, 0.0, z_m, 'left')

    R_down_m, Rr_down_m = basis_values_at_match(mode, 'down', sol_down, 'right')
    R_up_m,   Rr_up_m   = basis_values_at_match(mode, 'up',   sol_up,   'right')
    Mmatch = np.array([[R_down_m, R_up_m], [Rr_down_m, Rr_up_m]], dtype=complex)

    # inner domain: [z_m, 1]
    sol_in  = solve_basis_domain(mode, 'in',  N_in, z_m, 1.0, 'right')
    sol_out = solve_basis_domain(mode, 'out', N_in, z_m, 1.0, 'right')

    R_in_m,  Rr_in_m  = basis_values_at_match(mode, 'in',  sol_in,  'left')
    R_out_m, Rr_out_m = basis_values_at_match(mode, 'out', sol_out, 'left')

    # first column: R_in expressed in outer basis
    Cin_down, Cin_up = np.linalg.solve(
        Mmatch, np.array([R_in_m, Rr_in_m], dtype=complex)
    )

    # second column: R_out expressed in outer basis
    Cout_down, Cout_up = np.linalg.solve(
        Mmatch, np.array([R_out_m, Rr_out_m], dtype=complex)
    )

    S = np.array([[Cin_down, Cin_up], [Cout_down, Cout_up]], dtype=complex)

    b_inc = complex(Cin_down)
    b_ref = complex(Cin_up)
    b_trans = 1.0 + 0.0j

    ratio_ref_over_inc = _safe_complex_ratio(b_ref, b_inc)
    ratio_inc_over_ref = _safe_complex_ratio(b_inc, b_ref)

    # match-point radius
    r_m_outer = r_of_z(sol_down['z'][-1], mode)
    r_m_inner = r_of_z(sol_in['z'][0], mode)

    # Abel invariants on each side
    outer_abel_num = abel_constant_from_pair(
        R_down_m, Rr_down_m, R_up_m, Rr_up_m, r_m_outer, mode
    )
    outer_abel_th = outer_abel_theory(mode)
    outer_abel_residual = rel_complex_residual(outer_abel_num, outer_abel_th)

    inner_abel_num = abel_constant_from_pair(
        R_in_m, Rr_in_m, R_out_m, Rr_out_m, r_m_inner, mode
    )
    inner_abel_th = inner_abel_theory(mode)
    inner_abel_residual = rel_complex_residual(inner_abel_num, inner_abel_th)

    # determinant relation:
    # det(S) * C_outer = C_inner
    detS_num = np.linalg.det(S)
    detS_th = inner_abel_th / outer_abel_th
    detS_residual = rel_complex_residual(detS_num, detS_th)

    return {
        # original outputs
        'S': S,
        'B_inc': b_inc,
        'B_ref': b_ref,
        'B_trans': b_trans,
        'ratio_ref_over_inc': ratio_ref_over_inc,
        'ratio_inc_over_ref': ratio_inc_over_ref,
        'B_trans_over_B_inc': _safe_complex_ratio(b_trans, b_inc),
        'B_ref_over_B_inc': ratio_ref_over_inc,

        # expose both columns explicitly
        'Cin_down': complex(Cin_down),
        'Cin_up': complex(Cin_up),
        'Cout_down': complex(Cout_down),
        'Cout_up': complex(Cout_up),

        # Abel diagnostics
        'r_match_outer': float(r_m_outer),
        'r_match_inner': float(r_m_inner),

        'outer_abel_num': complex(outer_abel_num),
        'outer_abel_th': complex(outer_abel_th),
        'outer_abel_residual': float(outer_abel_residual),

        'inner_abel_num': complex(inner_abel_num),
        'inner_abel_th': complex(inner_abel_th),
        'inner_abel_residual': float(inner_abel_residual),

        'detS_num': complex(detS_num),
        'detS_th': complex(detS_th),
        'detS_residual': float(detS_residual),
    }
class TeukRadAmplitudeIn(object):
    def __init__(self, mode: KerrMode, N_in: int = 80, N_out: int = 80, z_m: float = 0.3):
        self.mode = mode
        self.M = mode.M
        self.a = mode.a
        self.omega = mode.omega
        
        self.ell = mode.ell
        self.m = mode.m
        self.N_in = N_in
        self.N_out = N_out
        self.z_m = z_m
        self.lam = mode.lam
        self._smatrix: Dict[str, complex | np.ndarray | None] | None = None
        self._result: InAmplitudesResult | None = None

    def __call__(self) -> InAmplitudesResult:
        return self.to_result()

    @property
    def smatrix(self) -> Dict[str, complex | np.ndarray | None]:
        if self._smatrix is None:
            self._smatrix = compute_smatrix(self.mode, N_in=self.N_in, N_out=self.N_out, z_m=self.z_m)
        return self._smatrix

    @property
    def B_inc(self) -> complex:
        return complex(self.smatrix["B_inc"])

    @property
    def B_ref(self) -> complex:
        return complex(self.smatrix["B_ref"])

    @property
    def B_trans(self) -> complex:
        return complex(self.smatrix["B_trans"])

    @property
    def ratio_ref_over_inc(self) -> complex | None:
        return self.smatrix["ratio_ref_over_inc"]

    @property
    def ratio_inc_over_ref(self) -> complex | None:
        return self.smatrix["ratio_inc_over_ref"]

    def to_result(self) -> InAmplitudesResult:
        if self._result is None:
            lam = self.mode.lambda_value
            self.lam = lam
            self._result = InAmplitudesResult(
                l=self.mode.ell,
                m=self.mode.m,
                s=self.mode.s,
                a=self.a,
                omega=self.omega,
                lam=lam,
                B_inc=self.B_inc,
                B_ref=self.B_ref,
                B_trans=self.B_trans,
                N_in=self.N_in,
                N_out=self.N_out,
                z_m=self.z_m,
                ratio_ref_over_inc=self.ratio_ref_over_inc,
                ratio_inc_over_ref=self.ratio_inc_over_ref,
            )
        return self._result

    @property
    def result(self) -> InAmplitudesResult:
        return self.to_result()

    def __repr__(self) -> str:
        return (
            f"TeukRadAmplitudeIn(l={self.ell}, m={self.m}, a={self.a}, omega={self.omega}, "
            f"N_in={self.N_in}, N_out={self.N_out}, z_m={self.z_m})"
        )

class TeukRadAmplitudeInWithAbelChecks(TeukRadAmplitudeIn):
    """
    Same amplitude interface as TeukRadAmplitudeIn, but the underlying computation
    also returns Abel-type invariant diagnostics that remain valid for the complex-
    potential Kerr Teukolsky radial equation.
    """

    @property
    def smatrix(self) -> Dict[str, complex | np.ndarray | None]:
        if self._smatrix is None:
            self._smatrix = compute_smatrix_with_abel(
                self.mode,
                N_in=self.N_in,
                N_out=self.N_out,
                z_m=self.z_m,
            )
        return self._smatrix

    @property
    def outer_abel_residual(self) -> float:
        return float(self.smatrix["outer_abel_residual"])

    @property
    def inner_abel_residual(self) -> float:
        return float(self.smatrix["inner_abel_residual"])

    @property
    def detS_residual(self) -> float:
        return float(self.smatrix["detS_residual"])

class TeukRadAmplitudeInWithInterpolant(TeukRadAmplitudeIn):
    """
    Same amplitudes as TeukRadAmplitudeIn, but also exposes
    a piecewise interpolant of the true in-mode solution
    reconstructed via:
        Psi = R_in / P_Leaver
        R_in = Psi * P_Leaver
    """
    def __init__(self, mode: KerrMode, N_in: int = 80, N_out: int = 80, z_m: float = 0.3):
        super().__init__(mode=mode, N_in=N_in, N_out=N_out, z_m=z_m)
        self._profile = None

    @property
    def smatrix(self) -> Dict[str, complex | np.ndarray | None]:
        if self._smatrix is None:
            self._smatrix = compute_smatrix(
                self.mode,
                N_in=self.N_in,
                N_out=self.N_out,
                z_m=self.z_m,
                return_profile=True,
            )
        return self._smatrix

    @property
    def profile(self) -> InModeLeaverInterpolant:
        if self._profile is None:
            self._profile = self.smatrix["profile"]
        return self._profile
