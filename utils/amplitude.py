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
    """
    在真边界附近拟合 u_z 的局部展开：

    左端 z=0:
        u_z(z) ≈ c0 + c1 z + c2 z^2
    右端 y=1-z=0:
        u_z(y) ≈ c0 + c1 y + c2 y^2

    返回 (c0, c1)，这样既能构造
        u(eps)
    也能构造
        u_z(eps)
    """
    ss = eps * np.array([0.25, 0.5, 0.75, 1.0], dtype=float)

    if side == 'left':
        z_s = ss
        _, B1, B0 = coeffs_numeric(z_s, mode, basis)
        du_s = -(B0 / B1)

        V = np.vstack([np.ones_like(ss), ss, ss**2]).T
        c, *_ = np.linalg.lstsq(V, du_s, rcond=None)
        return complex(c[0]), complex(c[1])

    elif side == 'right':
        y_s = ss
        z_s = 1.0 - y_s
        _, B1, B0 = coeffs_numeric(z_s, mode, basis)
        du_s = -(B0 / B1)

        V = np.vstack([np.ones_like(y_s), y_s, y_s**2]).T
        c, *_ = np.linalg.lstsq(V, du_s, rcond=None)
        return complex(c[0]), complex(c[1])

    else:
        raise ValueError

def compute_smatrix(mode: KerrMode, N_in: int = 80, N_out: int = 80, z_m: float = 0.3):
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

    return {
        'S': np.array([[Cin_down, Cin_up], [Cout_down, Cout_up]], dtype=complex),
        'B_inc': b_inc,
        'B_ref': b_ref,
        'B_trans': b_trans,
        'ratio_ref_over_inc': ratio_ref_over_inc,
        'ratio_inc_over_ref': ratio_inc_over_ref,
        'B_trans_over_B_inc': _safe_complex_ratio(b_trans, b_inc),
        'B_ref_over_B_inc': ratio_ref_over_inc,
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
        self.lam = mode.lambda_value
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
            self._result = InAmplitudesResult(
                l=self.mode.ell,
                m=self.mode.m,
                s=self.mode.s,
                a=self.a,
                omega=self.omega,
                lam=self.lam,
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
