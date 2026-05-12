from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import mpmath as mp

from utils.mode import InAmplitudesResult, KerrMode
from utils.amplitude import (
    _safe_complex_ratio,
    Delta,
    Delta_p,
    K_of_r,
    r_of_z,
    dz_dr,
    d2z_dr2,
    A_down,
    A_up,
    A_in,
    A_out,
    q_and_qp as q_and_qp_two_patch,
    cheb_D,
    solve_basis_domain,
    basis_values_at_match,
    abel_constant_from_pair,
    outer_abel_theory,
    inner_abel_theory,
    rel_complex_residual,
    r_star,
    drstar_dr,
    d_dr_drstar,
)
from utils.hp_linear import (
    use_mp_backend,
    solve_2x2_np,
    solve_2x2_mp,
    solve_scaled_2x2_np,
    solve_scaled_2x2_mp,
    det_2x2_mp,
)


# ---------------------------------------------------------------------------
# mpmath high-precision spectral helpers
# ---------------------------------------------------------------------------

def _to_mpc(z) -> mp.mpc:
    zc = complex(z)
    return mp.mpc(zc.real, zc.imag)


def _to_complex(z: mp.mpc) -> complex:
    return complex(float(mp.re(z)), float(mp.im(z)))


def cheb_D_mp(N: int, a: float, b: float, dps: int = 100):
    mp.mp.dps = dps
    xi = [mp.cos(mp.pi * i / N) for i in range(N + 1)]
    c = [mp.mpf(2) if i == 0 or i == N else mp.mpf(1) for i in range(N + 1)]
    D = mp.matrix(N + 1, N + 1)
    for i in range(N + 1):
        for j in range(N + 1):
            if i != j:
                D[i, j] = (c[i] / c[j]) * ((-1) ** (i + j)) / (xi[i] - xi[j])
    for i in range(N + 1):
        D[i, i] = -sum(D[i, j] for j in range(N + 1) if j != i)
    a_mp, b_mp = mp.mpf(a), mp.mpf(b)
    z_nodes = [a_mp + (b_mp - a_mp) * (mp.mpf(1) - xi[i]) / mp.mpf(2) for i in range(N + 1)]
    scale = mp.mpf(-2) / (b_mp - a_mp)
    D = scale * D
    return D, z_nodes


def _coeffs_numeric_mp(z_val: mp.mpf, mode: KerrMode, basis: str):
    r = mode.rp / float(z_val)
    zr = -float(z_val)**2 / mode.rp
    zrr = 2.0 * float(z_val)**3 / (mode.rp**2)
    D = (r - mode.rp) * (r - mode.rm)
    Dp = 2.0 * r - 2.0 * mode.M
    K = (r * r + mode.a * mode.a) * mode.omega - mode.a * mode.m
    V = (K * K + 4j * (r - mode.M) * K) / D - 8j * mode.omega * r - mode.lambda_value
    q, qp = q_and_qp_two_patch(r, mode, basis)
    B2_val = complex(D * zr * zr)
    B1_val = complex(D * (zrr + 2.0 * q * zr) - Dp * zr)
    B0_val = complex(D * (qp + q * q) - Dp * q + V)
    return (mp.mpc(B2_val.real, B2_val.imag),
            mp.mpc(B1_val.real, B1_val.imag),
            mp.mpc(B0_val.real, B0_val.imag))


def solve_basis_domain_mp(
    mode: KerrMode,
    basis: str,
    N: int,
    z_a: float,
    z_b: float,
    bc_side: str,
    dps: int = 100,
):
    """Spectral solve of the transformed ODE using mpmath high precision."""
    mp.mp.dps = dps
    D_mat, z_nodes = cheb_D_mp(N, z_a, z_b, dps)
    D2 = D_mat @ D_mat
    A = mp.matrix(N + 1, N + 1)
    b = mp.matrix(N + 1, 1)

    from utils.amplitude import boundary_du_exact

    if bc_side == "left":
        du0 = boundary_du_exact(mode, basis, "left")
        du0_mp = mp.mpc(du0.real, du0.imag)
        for j in range(N + 1):
            A[0, j] = mp.mpc(0)
        A[0, 0] = mp.mpc(1)
        b[0] = mp.mpc(1)
        for j in range(N + 1):
            A[1, j] = D_mat[0, j]
        b[1] = du0_mp
        for i in range(2, N + 1):
            z_i = float(z_nodes[i])
            B2, B1, B0 = _coeffs_numeric_mp(z_i, mode, basis)
            for j in range(N + 1):
                A[i, j] = B2 * D2[i, j] + B1 * D_mat[i, j]
            A[i, i] += B0
            b[i] = mp.mpc(0)
    elif bc_side == "right":
        du1 = boundary_du_exact(mode, basis, "right")
        du1_mp = mp.mpc(du1.real, du1.imag)
        for i in range(N - 1):
            z_i = float(z_nodes[i])
            B2, B1, B0 = _coeffs_numeric_mp(z_i, mode, basis)
            for j in range(N + 1):
                A[i, j] = B2 * D2[i, j] + B1 * D_mat[i, j]
            A[i, i] += B0
            b[i] = mp.mpc(0)
        for j in range(N + 1):
            A[N - 1, j] = D_mat[N, j]
        b[N - 1] = du1_mp
        for j in range(N + 1):
            A[N, j] = mp.mpc(0)
        A[N, N] = mp.mpc(1)
        b[N] = mp.mpc(1)
    else:
        raise ValueError("bc_side must be 'left' or 'right'")

    x = mp.lu_solve(A, b)
    uz_vals_mp = []
    for i in range(N + 1):
        s = mp.mpc(0)
        for j in range(N + 1):
            s += D_mat[i, j] * x[j, 0]
        uz_vals_mp.append(s)

    z_np = np.array([float(z_nodes[i]) for i in range(N + 1)])
    u_np = np.array([_to_complex(x[i, 0]) for i in range(N + 1)], dtype=complex)
    uz_np = np.array([_to_complex(uz_vals_mp[i]) for i in range(N + 1)], dtype=complex)
    return {"z": z_np, "u": u_np, "uz": uz_np}


def q_and_qp_three_patch(r, mode: KerrMode, basis: str):
    """
    Extend the existing q_and_qp by adding a raw basis:
        R = u   (i.e. no prefactor extracted)
    """
    if basis == "raw":
        return 0.0 + 0.0j, 0.0 + 0.0j
    return q_and_qp_two_patch(r, mode, basis)


def coeffs_numeric_three_patch(z, mode: KerrMode, basis: str):
    """
    Same transformed ODE coefficients as the current code, but allows basis='raw'.
    For basis='raw', q = qp = 0, i.e. the full-R equation is solved directly on the middle patch.
    """
    r = r_of_z(z, mode)
    zr = dz_dr(z, mode)
    zrr = d2z_dr2(z, mode)
    D = Delta(r, mode)
    Dp = Delta_p(r, mode)
    K = K_of_r(r, mode)

    # same V as in current amplitude.py
    V = (K * K + 4j * (r - mode.M) * K) / D - 8j * mode.omega * r - mode.lambda_value

    q, qp = q_and_qp_three_patch(r, mode, basis)

    B2 = D * zr * zr
    B1 = D * (zrr + 2.0 * q * zr) - Dp * zr
    B0 = D * (qp + q * q) - Dp * q + V
    return B2.astype(complex), B1.astype(complex), B0.astype(complex)


def solve_basis_domain_custom(
    mode: KerrMode,
    basis: str,
    N: int,
    z_a: float,
    z_b: float,
    bc_side: str,
    u_bc: complex,
    uz_bc: complex,
):
    """
    Generic spectral solve on [z_a, z_b] with user-specified boundary data on one side:
        u = u_bc,
        u_z = uz_bc
    This is used for the middle raw patch.
    """
    D, z = cheb_D(N, z_a, z_b)
    D2 = D @ D

    A = np.zeros((N + 1, N + 1), dtype=complex)
    b = np.zeros(N + 1, dtype=complex)

    if bc_side == "left":
        # BC 1: u(z_a) = u_bc
        A[0, :] = 0.0
        A[0, 0] = 1.0
        b[0] = u_bc

        # BC 2: u_z(z_a) = uz_bc
        A[1, :] = D[0, :]
        b[1] = uz_bc

        idx = np.arange(2, N + 1)
        B2, B1, B0 = coeffs_numeric_three_patch(z[idx], mode, basis)
        A[idx, :] = B2[:, None] * D2[idx, :] + B1[:, None] * D[idx, :]
        A[idx, idx] += B0

    elif bc_side == "right":
        idx = np.arange(0, N - 1)
        B2, B1, B0 = coeffs_numeric_three_patch(z[idx], mode, basis)
        A[idx, :] = B2[:, None] * D2[idx, :] + B1[:, None] * D[idx, :]
        A[idx, idx] += B0

        # BC 1: u_z(z_b) = uz_bc
        A[-2, :] = D[-1, :]
        b[-2] = uz_bc

        # BC 2: u(z_b) = u_bc
        A[-1, :] = 0.0
        A[-1, -1] = 1.0
        b[-1] = u_bc

    else:
        raise ValueError("bc_side must be 'left' or 'right'")

    u = np.linalg.solve(A, b)
    uz = D @ u
    return {"z": z, "u": u, "uz": uz}


def basis_values_at_match_three_patch(mode: KerrMode, basis: str, sol, side: str):
    """
    Same as basis_values_at_match, but supports basis='raw'.
    """
    if basis != "raw":
        return basis_values_at_match(mode, basis, sol, side)

    z = sol["z"]
    u = sol["u"]
    uz = sol["uz"]

    if side == "left":
        zm, um, uzm = z[0], u[0], uz[0]
    else:
        zm, um, uzm = z[-1], u[-1], uz[-1]

    dzdr = dz_dr(zm, mode)

    # raw basis means R = u
    Rm = um
    Rm_r = dzdr * uzm
    return complex(Rm), complex(Rm_r)


def solve_middle_raw_pair(
    mode: KerrMode,
    N_mid: int,
    z1: float,
    z2: float,
):
    """
    Build two raw full-R basis solutions on the middle patch [z1, z2].

    They are normalized so that at z=z1:
        mid_val : [R, R_r] = [1, 0]
        mid_der : [R, R_r] = [0, 1]

    Because the spectral solver uses u_z, the second one uses
        u_z(z1) = 1 / (dz/dr)(z1).
    """
    dzdr1 = dz_dr(z1, mode)

    sol_mid_val = solve_basis_domain_custom(
        mode=mode,
        basis="raw",
        N=N_mid,
        z_a=z1,
        z_b=z2,
        bc_side="left",
        u_bc=1.0 + 0.0j,
        uz_bc=0.0 + 0.0j,
    )

    sol_mid_der = solve_basis_domain_custom(
        mode=mode,
        basis="raw",
        N=N_mid,
        z_a=z1,
        z_b=z2,
        bc_side="left",
        u_bc=0.0 + 0.0j,
        uz_bc=1.0 / dzdr1,
    )

    R0_z2, R0r_z2 = basis_values_at_match_three_patch(mode, "raw", sol_mid_val, "right")
    R1_z2, R1r_z2 = basis_values_at_match_three_patch(mode, "raw", sol_mid_der, "right")

    # state transfer from z1 to z2 in [R, R_r] variables
    T_mid = np.array([[R0_z2, R1_z2], [R0r_z2, R1r_z2]], dtype=complex)

    return {
        "sol_mid_val": sol_mid_val,
        "sol_mid_der": sol_mid_der,
        "T_mid": T_mid,
    }

def _solve_scaled_2x2(M: np.ndarray, y: np.ndarray, floor: float = 1.0e-300):
    """
    Solve M x = y with simple column scaling + least squares.
    This is more robust than a raw solve when the 2x2 system is badly scaled.
    """
    M = np.asarray(M, dtype=complex)
    y = np.asarray(y, dtype=complex)

    col_norms = np.linalg.norm(M, axis=0)
    col_norms = np.where(col_norms > floor, col_norms, 1.0)

    Ms = M / col_norms[None, :]
    x_scaled, *_ = np.linalg.lstsq(Ms, y, rcond=None)
    x = x_scaled / col_norms

    relres = np.linalg.norm(M @ x - y) / max(np.linalg.norm(y), floor)
    svals = np.linalg.svd(M, compute_uv=False)

    diag = {
        "relres": float(relres),
        "cond_raw": float(np.linalg.cond(M)),
        "cond_scaled": float(np.linalg.cond(Ms)),
        "smax": float(np.abs(svals[0])),
        "smin": float(np.abs(svals[-1])),
        "col_norm_0": float(col_norms[0]),
        "col_norm_1": float(col_norms[1]),
    }
    return x, diag 

def _auto_z1(omega: float, z_min: float = 0.001, z_max: float = 0.5) -> float:
    """Heuristic z1 based on ω.

    Three regimes:
      - |ω| < 1e-2: low ω, z1 = clamp(15·ω, z_min, 0.2)
      - 1e-2 ≤ |ω| ≤ 3: normal ω, z1 = 0.1
      - |ω| > 3: high ω, z1 increases to reduce cond(M_left) from r³ growth
        Formula: z1 = min(0.5, 0.1 + 0.057·(ω-3))
    """
    omega = abs(omega)
    if omega < 1.0e-2:
        z1 = max(float(15.0 * omega), z_min)
        return min(z1, 0.2)
    if omega > 3.0:
        z1 = 0.1 + 0.057 * (omega - 3.0)
        return min(max(z1, 0.1), z_max)
    return 0.1


def _auto_z2(omega: float) -> float:
    """z2=0.5 for low ω (|ω| < 1e-2), 0.9 for normal ω.
    For high ω (>3), keep middle patch small: z2 = max(0.6, z1+0.15)."""
    omega = abs(omega)
    if omega < 1.0e-2:
        return 0.5
    return 0.9


def _auto_n_mid_subdomains(z1: float, z2: float,
                           max_span_ratio: float = 60.0) -> int:
    """Estimate subdomains so each has span ratio < max_span_ratio."""
    span_ratio = z2 / z1
    if span_ratio <= max_span_ratio:
        return 1
    return int(np.ceil(np.log(span_ratio) / np.log(max_span_ratio)))


def _compute_mid_transport_chain(
    mode: KerrMode,
    N_mid: int,
    z1: float,
    z2: float,
    n_subdomains: int = 1,
    *,
    store_subdomain_sols: bool = False,
) -> dict:
    """
    Compute total T_mid by splitting [z1, z2] into n_subdomains log-spaced
    sub-intervals and chaining the 2x2 transport matrices.

    When store_subdomain_sols=True, also return per-subdomain raw pairs and
    z_breaks for profile reconstruction.
    """
    if n_subdomains <= 1:
        mid = solve_middle_raw_pair(mode, N_mid, z1, z2)
        result: dict = {"T_mid": mid["T_mid"], "n_subdomains": 1}
        if store_subdomain_sols:
            result["z_breaks"] = np.array([z1, z2])
            result["sub_sols"] = [mid]
            result["T_prefixes"] = [np.eye(2, dtype=complex)]
        return result

    z_breaks = np.logspace(np.log10(z1), np.log10(z2), n_subdomains + 1)
    T_total = np.eye(2, dtype=complex)
    T_prefixes: list[np.ndarray] = [np.eye(2, dtype=complex)]
    sub_sols: list[dict] = []

    for i in range(n_subdomains):
        mid = solve_middle_raw_pair(mode, N_mid, z_breaks[i], z_breaks[i + 1])
        T_total = mid["T_mid"] @ T_total
        if store_subdomain_sols:
            T_prefixes.append(T_total.copy())
            sub_sols.append(mid)

    result = {
        "T_mid": T_total,
        "n_subdomains": n_subdomains,
        "z_breaks": z_breaks,
    }
    if store_subdomain_sols:
        result["sub_sols"] = sub_sols
        result["T_prefixes"] = T_prefixes
    return result


def extract_ref_ratio_from_logderivative(mode: KerrMode, R: complex, Rr: complex, z: float) -> complex:
    """
    Extract B_ref / B_inc from the logarithmic derivative Y = R_r / R.

    The full solution is  R = B_inc * A_down * u_down + B_ref * A_up * u_up.
    Where the bases are nearly pure asymptotically, u_down ≈ u_up ≈ 1, so
        Y = R_r / R ≈ (B_inc*A'_down + B_ref*A'_up) / (B_inc*A_down + B_ref*A_up).

    Solving for ratio = B_ref / B_inc gives:
        ratio = (A_down / A_up) * (q_down - Y) / (Y - q_up)

    where q_down = A'_down / A_down, q_up = A'_up / A_up are the logarithmic
    derivatives of the asymptotic prefactors.
    """
    r = r_of_z(z, mode)
    A_d = A_down(r, mode)
    A_u = A_up(r, mode)
    q_d, _ = q_and_qp_two_patch(r, mode, "down")
    q_u, _ = q_and_qp_two_patch(r, mode, "up")

    Y = Rr / R
    ratio = (A_d / A_u) * (q_d - Y) / (Y - q_u)
    return complex(ratio)


def _reconstruct_inmode_on_left_patch(
    mode: KerrMode,
    B_inc: complex,
    B_ref: complex,
    sol_down: dict,
    sol_up: dict,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Reconstruct the in-mode R(z) and R_r(z) on the left-patch Chebyshev nodes
    using the S-matrix coefficients and the pre-computed basis solutions.
    """
    z_vals = sol_down["z"]
    u_d = sol_down["u"]
    uz_d = sol_down["uz"]
    u_u = sol_up["u"]
    uz_u = sol_up["uz"]

    R_vals = np.zeros(len(z_vals), dtype=complex)
    Rr_vals = np.zeros(len(z_vals), dtype=complex)

    for j, zj in enumerate(z_vals):
        rj = r_of_z(zj, mode)
        dzdrj = dz_dr(zj, mode)
        q_d_j, _ = q_and_qp_two_patch(rj, mode, "down")
        q_u_j, _ = q_and_qp_two_patch(rj, mode, "up")
        F_d = A_down(rj, mode)
        F_u = A_up(rj, mode)

        R_d = F_d * u_d[j]
        Rr_d = F_d * (dzdrj * uz_d[j] + q_d_j * u_d[j])
        R_u = F_u * u_u[j]
        Rr_u = F_u * (dzdrj * uz_u[j] + q_u_j * u_u[j])

        R_vals[j] = B_inc * R_d + B_ref * R_u
        Rr_vals[j] = B_inc * Rr_d + B_ref * Rr_u

    return z_vals, R_vals, Rr_vals


def _compute_ratio_plateau(
    mode: KerrMode,
    z_vals: np.ndarray,
    R_vals: np.ndarray,
    Rr_vals: np.ndarray,
    *,
    z_cut: float | None = None,
    min_points: int = 4,
) -> dict:
    """
    Compute B_ref/B_inc from log-derivative at each z, return plateau statistics.

    Optionally restrict to z <= z_cut to avoid the match-point region where
    the asymptotic approximation u_down≈u_up breaks down.
    """
    ratios = []
    zs = []
    for j, zj in enumerate(z_vals):
        if z_cut is not None and zj > z_cut:
            continue
        if abs(R_vals[j]) < 1e-300:
            continue
        try:
            r = extract_ref_ratio_from_logderivative(mode, complex(R_vals[j]), complex(Rr_vals[j]), float(zj))
            ratios.append(r)
            zs.append(float(zj))
        except (ZeroDivisionError, ValueError):
            continue

    if len(ratios) < min_points:
        return {
            "n_points": len(ratios),
            "median_abs": float("nan"),
            "median_phase": float("nan"),
            "logabs_std": float("nan"),
            "phase_std": float("nan"),
            "ratio_median": complex(float("nan"), float("nan")),
        }

    ratios_arr = np.array(ratios)
    logabs = np.log10(np.abs(ratios_arr))
    phases = np.angle(ratios_arr)

    median_abs = float(10.0 ** np.median(logabs))
    median_phase = float(np.median(phases))
    logabs_std = float(np.std(logabs))
    phase_std = float(np.std(phases))

    # median in complex plane: take the point closest to the median of abs and phase
    ratio_median = complex(median_abs * np.cos(median_phase), median_abs * np.sin(median_phase))

    return {
        "n_points": len(ratios),
        "z_min": float(min(zs)),
        "z_max": float(max(zs)),
        "median_abs": median_abs,
        "median_phase": median_phase,
        "logabs_std": logabs_std,
        "phase_std": phase_std,
        "ratio_median": ratio_median,
    }


# -----------------------------------------------------------------------------
# Enhanced middle-patch factor support
# -----------------------------------------------------------------------------

def _resolve_middle_basis(omega: float, middle_basis: str) -> str:
    """Resolve the middle propagation variable.

    Supported values:
        raw     : R = U, original implementation.
        up      : R = A_up U = r^3 exp(+i omega r_*) U.
        down    : R = A_down U = r^-1 exp(-i omega r_*) U.
        custom  : R = r^p exp(i sigma omega r_*) U.
        auto    : raw for |omega| < 3, custom for |omega| >= 3.
    """
    b = str(middle_basis).lower()
    if b == "auto":
        return "custom" if abs(omega) >= 3.0 else "raw"
    if b in {"raw", "up", "down", "custom"}:
        return b
    raise ValueError(f"Unknown middle_basis={middle_basis!r}")


def _middle_factor(r, mode: KerrMode, basis: str, *, middle_p: float = 3.4, middle_sigma: float = 1.0):
    """Return F(r) in R = F(r) U for the requested middle variable."""
    b = str(basis).lower()
    if b == "raw":
        return np.ones_like(np.asarray(r, dtype=np.float64), dtype=np.complex128)
    if b == "up":
        return A_up(r, mode)
    if b == "down":
        return A_down(r, mode)
    if b == "custom":
        r_arr = np.asarray(r, dtype=np.float64)
        return (r_arr ** float(middle_p)) * np.exp(1j * float(middle_sigma) * mode.omega * r_star(r_arr, mode))
    raise ValueError(f"Unknown middle factor basis={basis!r}")


def q_and_qp_three_patch(
    r,
    mode: KerrMode,
    basis: str,
    *,
    middle_p: float = 3.4,
    middle_sigma: float = 1.0,
):
    """Logarithmic derivatives q=F_r/F and q_p=dq/dr for R=F u."""
    b = str(basis).lower()
    if b == "raw":
        return 0.0 + 0.0j, 0.0 + 0.0j
    if b in {"up", "down", "in", "out"}:
        return q_and_qp_two_patch(r, mode, b)
    if b == "custom":
        rst = drstar_dr(r, mode)
        rst_p = d_dr_drstar(r, mode)
        p = float(middle_p)
        sig = float(middle_sigma)
        q = p / r + 1j * sig * mode.omega * rst
        qp = -p / (r * r) + 1j * sig * mode.omega * rst_p
        return q, qp
    raise ValueError(f"Unknown basis={basis!r}")


def coeffs_numeric_three_patch(
    z,
    mode: KerrMode,
    basis: str,
    *,
    middle_p: float = 3.4,
    middle_sigma: float = 1.0,
):
    """Transformed ODE coefficients for R=F u on the z-domain."""
    r = r_of_z(z, mode)
    zr = dz_dr(z, mode)
    zrr = d2z_dr2(z, mode)
    D = Delta(r, mode)
    Dp = Delta_p(r, mode)
    K = K_of_r(r, mode)

    V = (K * K + 4j * (r - mode.M) * K) / D - 8j * mode.omega * r - mode.lambda_value
    q, qp = q_and_qp_three_patch(
        r,
        mode,
        basis,
        middle_p=middle_p,
        middle_sigma=middle_sigma,
    )

    B2 = D * zr * zr
    B1 = D * (zrr + 2.0 * q * zr) - Dp * zr
    B0 = D * (qp + q * q) - Dp * q + V
    return B2.astype(complex), B1.astype(complex), B0.astype(complex)


def solve_basis_domain_custom(
    mode: KerrMode,
    basis: str,
    N: int,
    z_a: float,
    z_b: float,
    bc_side: str,
    u_bc: complex,
    uz_bc: complex,
    *,
    middle_p: float = 3.4,
    middle_sigma: float = 1.0,
):
    """Generic spectral solve on [z_a,z_b] for R=F u with custom boundary data."""
    D, z = cheb_D(N, z_a, z_b)
    D2 = D @ D

    A = np.zeros((N + 1, N + 1), dtype=complex)
    b = np.zeros(N + 1, dtype=complex)

    if bc_side == "left":
        A[0, :] = 0.0
        A[0, 0] = 1.0
        b[0] = u_bc

        A[1, :] = D[0, :]
        b[1] = uz_bc

        idx = np.arange(2, N + 1)
        B2, B1, B0 = coeffs_numeric_three_patch(
            z[idx], mode, basis, middle_p=middle_p, middle_sigma=middle_sigma
        )
        A[idx, :] = B2[:, None] * D2[idx, :] + B1[:, None] * D[idx, :]
        A[idx, idx] += B0

    elif bc_side == "right":
        idx = np.arange(0, N - 1)
        B2, B1, B0 = coeffs_numeric_three_patch(
            z[idx], mode, basis, middle_p=middle_p, middle_sigma=middle_sigma
        )
        A[idx, :] = B2[:, None] * D2[idx, :] + B1[:, None] * D[idx, :]
        A[idx, idx] += B0

        A[-2, :] = D[-1, :]
        b[-2] = uz_bc

        A[-1, :] = 0.0
        A[-1, -1] = 1.0
        b[-1] = u_bc

    else:
        raise ValueError("bc_side must be 'left' or 'right'")

    u = np.linalg.solve(A, b)
    uz = D @ u
    return {"z": z, "u": u, "uz": uz}


def _side_z(sol: dict, side: str) -> float:
    return float(sol["z"][0] if side == "left" else sol["z"][-1])


def _R_state_to_factor_state(
    mode: KerrMode,
    z: float,
    R: complex,
    Rr: complex,
    factor_basis: str,
    *,
    middle_p: float = 3.4,
    middle_sigma: float = 1.0,
) -> np.ndarray:
    """Convert [R,R_r] to [U,U_r] for R=F U."""
    r = r_of_z(z, mode)
    F = _middle_factor(r, mode, factor_basis, middle_p=middle_p, middle_sigma=middle_sigma)
    q, _ = q_and_qp_three_patch(
        r, mode, factor_basis, middle_p=middle_p, middle_sigma=middle_sigma
    )
    U = R / F
    Ur = Rr / F - q * U
    return np.array([complex(U), complex(Ur)], dtype=complex)


def _factor_state_to_R_state(
    mode: KerrMode,
    z: float,
    U: complex,
    Ur: complex,
    factor_basis: str,
    *,
    middle_p: float = 3.4,
    middle_sigma: float = 1.0,
) -> np.ndarray:
    """Convert [U,U_r] to [R,R_r] for R=F U."""
    r = r_of_z(z, mode)
    F = _middle_factor(r, mode, factor_basis, middle_p=middle_p, middle_sigma=middle_sigma)
    q, _ = q_and_qp_three_patch(
        r, mode, factor_basis, middle_p=middle_p, middle_sigma=middle_sigma
    )
    R = F * U
    Rr = F * (Ur + q * U)
    return np.array([complex(R), complex(Rr)], dtype=complex)


def _basis_state_in_factor_space(
    mode: KerrMode,
    physical_basis: str,
    sol: dict,
    side: str,
    factor_basis: str,
    *,
    middle_p: float = 3.4,
    middle_sigma: float = 1.0,
) -> np.ndarray:
    """Evaluate a physical basis solution and express it in the middle factor state [U,U_r]."""
    R, Rr = basis_values_at_match(mode, physical_basis, sol, side)
    z = _side_z(sol, side)
    return _R_state_to_factor_state(
        mode, z, R, Rr, factor_basis, middle_p=middle_p, middle_sigma=middle_sigma
    )


def solve_middle_factor_pair(
    mode: KerrMode,
    N_mid: int,
    z1: float,
    z2: float,
    *,
    middle_basis: str = "raw",
    middle_p: float = 3.4,
    middle_sigma: float = 1.0,
):
    """Build middle-patch transfer matrix in [U,U_r] for R=F U.

    The two basis solutions are normalized at z=z1 as
        [U,U_r] = [1,0], [0,1].
    """
    dzdr1 = dz_dr(z1, mode)

    sol_mid_val = solve_basis_domain_custom(
        mode=mode,
        basis=middle_basis,
        N=N_mid,
        z_a=z1,
        z_b=z2,
        bc_side="left",
        u_bc=1.0 + 0.0j,
        uz_bc=0.0 + 0.0j,
        middle_p=middle_p,
        middle_sigma=middle_sigma,
    )

    sol_mid_der = solve_basis_domain_custom(
        mode=mode,
        basis=middle_basis,
        N=N_mid,
        z_a=z1,
        z_b=z2,
        bc_side="left",
        u_bc=0.0 + 0.0j,
        uz_bc=1.0 / dzdr1,
        middle_p=middle_p,
        middle_sigma=middle_sigma,
    )

    z_right = sol_mid_val["z"][-1]
    dzdr_right = dz_dr(z_right, mode)
    U0 = sol_mid_val["u"][-1]
    U0r = dzdr_right * sol_mid_val["uz"][-1]
    U1 = sol_mid_der["u"][-1]
    U1r = dzdr_right * sol_mid_der["uz"][-1]

    T_mid = np.array([[U0, U1], [U0r, U1r]], dtype=complex)
    return {"sol_mid_val": sol_mid_val, "sol_mid_der": sol_mid_der, "T_mid": T_mid}


# Keep the old public helper name for raw compatibility.
def solve_middle_raw_pair(mode: KerrMode, N_mid: int, z1: float, z2: float):
    return solve_middle_factor_pair(mode, N_mid, z1, z2, middle_basis="raw")


def _auto_z2(omega: float, z1: float | None = None) -> float:
    """Adaptive z2.

    For high frequencies, a short middle patch is preferred when using the
    scaled/custom middle variable.  This fixes the old behavior where the
    function comment said high-frequency z2 should be close to z1 but the code
    always returned 0.9.
    """
    omega = abs(omega)
    if omega < 1.0e-2:
        return 0.5
    if z1 is None:
        z1 = _auto_z1(omega)
    if omega <= 1.0:
        return 0.9
    if omega <= 3.0:
        return min(0.9, max(0.7, z1 + 0.2))
    return min(0.95, z1 + 0.10)


def _compute_mid_transport_chain(
    mode: KerrMode,
    N_mid: int,
    z1: float,
    z2: float,
    n_subdomains: int = 1,
    *,
    store_subdomain_sols: bool = False,
    middle_basis: str = "raw",
    middle_p: float = 3.4,
    middle_sigma: float = 1.0,
) -> dict:
    """Compute total middle transfer matrix in the chosen [U,U_r] state."""
    if n_subdomains <= 1:
        mid = solve_middle_factor_pair(
            mode,
            N_mid,
            z1,
            z2,
            middle_basis=middle_basis,
            middle_p=middle_p,
            middle_sigma=middle_sigma,
        )
        result: dict = {"T_mid": mid["T_mid"], "n_subdomains": 1}
        if store_subdomain_sols:
            result["z_breaks"] = np.array([z1, z2])
            result["sub_sols"] = [mid]
            result["T_prefixes"] = [np.eye(2, dtype=complex)]
        return result

    z_breaks = np.logspace(np.log10(z1), np.log10(z2), n_subdomains + 1)
    T_total = np.eye(2, dtype=complex)
    T_prefixes: list[np.ndarray] = [np.eye(2, dtype=complex)]
    sub_sols: list[dict] = []

    for i in range(n_subdomains):
        mid = solve_middle_factor_pair(
            mode,
            N_mid,
            float(z_breaks[i]),
            float(z_breaks[i + 1]),
            middle_basis=middle_basis,
            middle_p=middle_p,
            middle_sigma=middle_sigma,
        )
        T_total = mid["T_mid"] @ T_total
        if store_subdomain_sols:
            T_prefixes.append(T_total.copy())
            sub_sols.append(mid)

    result = {"T_mid": T_total, "n_subdomains": n_subdomains, "z_breaks": z_breaks}
    if store_subdomain_sols:
        result["sub_sols"] = sub_sols
        result["T_prefixes"] = T_prefixes
    return result


def compute_smatrix_three_patch_with_abel(
    mode: KerrMode,
    N_left: int = 64,
    N_mid: int = 96,
    N_right: int = 64,
    z1: float | None = None,
    z2: float | None = None,
    *,
    n_mid_subdomains: int | None = None,
    return_details: bool = False,
    omega_mp_cut: float = 1.0e-2,
    mp_dps_loww: int = 200,
    use_logderivative_reflection: bool | None = None,
    middle_basis: str = "auto",
    middle_p: float = 3.4,
    middle_sigma: float = 1.0,
    use_mp_spectral: bool | None = None,
    mp_dps_highw: int = 200,
):
    """Three-patch Teukolsky radial solver with a scalable middle variable.

    The physical amplitudes are still defined by the outer bases
        R = B_inc R_down + B_ref R_up.

    The middle patch may use a different exact factor
        R = F_mid(r) U,
    with F_mid chosen to keep the propagated quantities at comparable scale.
    For high-frequency reflection extraction, the tested effective choice is
        middle_basis='custom', middle_p≈3.4, middle_sigma=+1,
    i.e. F_mid = r^p exp(+i omega r_*).
    """
    is_high_omega = abs(mode.omega) >= 2.5
    middle_basis_resolved = _resolve_middle_basis(mode.omega, middle_basis)

    if use_mp_spectral is None:
        use_mp_spectral = abs(mode.omega) >= 3.0

    if z1 is None:
        z1 = _auto_z1(mode.omega)
    if z2 is None:
        z2 = _auto_z2(mode.omega, z1=z1)
    if n_mid_subdomains is None or n_mid_subdomains < 1:
        n_mid_subdomains = _auto_n_mid_subdomains(z1, z2)

    # Do not force high N when the high-frequency scaled middle variable is used.
    # The scans showed that moderate N with a short middle patch is more reliable.
    # When mpmath spectral is enabled, use larger N to reduce ODE residual below
    # the B_ref signal level.
    if middle_basis_resolved == "raw":
        scale = max(1.0, abs(mode.omega) / 2.5)
        N_mid_min = max(48, min(int(48 * scale), 80))
        N_left_min = max(32, min(int(32 * scale), 64))
        N_right_min = max(32, min(int(32 * scale), 64))
    elif use_mp_spectral:
        N_mid_min = 64
        N_left_min = 32
        N_right_min = 48
    else:
        N_mid_min = 16
        N_left_min = 16
        N_right_min = 16

    N_mid = max(N_mid, N_mid_min)
    N_left = max(N_left, N_left_min)
    N_right = max(N_right, N_right_min)

    if not (0.0 < z1 < z2 < 1.0):
        raise ValueError(f"Require 0 < z1 < z2 < 1, got z1={z1}, z2={z2}")

    # ---------------------------------------------------------
    # left patch: physical outer asymptotic basis on [0, z1]
    # ---------------------------------------------------------
    sol_down = solve_basis_domain(mode, "down", N_left, 0.0, z1, "left")
    sol_up = solve_basis_domain(mode, "up", N_left, 0.0, z1, "left")

    R_down_z1, Rr_down_z1 = basis_values_at_match(mode, "down", sol_down, "right")
    R_up_z1, Rr_up_z1 = basis_values_at_match(mode, "up", sol_up, "right")

    M_left_R = np.array(
        [[R_down_z1, R_up_z1], [Rr_down_z1, Rr_up_z1]],
        dtype=complex,
    )

    # Express the same physical left basis in the middle state [U,U_r].
    col_down_U = _basis_state_in_factor_space(
        mode,
        "down",
        sol_down,
        "right",
        middle_basis_resolved,
        middle_p=middle_p,
        middle_sigma=middle_sigma,
    )
    col_up_U = _basis_state_in_factor_space(
        mode,
        "up",
        sol_up,
        "right",
        middle_basis_resolved,
        middle_p=middle_p,
        middle_sigma=middle_sigma,
    )
    M_left = np.column_stack([col_down_U, col_up_U]).astype(complex)

    # ---------------------------------------------------------
    # middle patch(s): transport in [U,U_r], R=F_mid U
    # ---------------------------------------------------------
    mid_chain = _compute_mid_transport_chain(
        mode,
        N_mid,
        z1,
        z2,
        n_subdomains=n_mid_subdomains,
        middle_basis=middle_basis_resolved,
        middle_p=middle_p,
        middle_sigma=middle_sigma,
    )
    T_mid = mid_chain["T_mid"]
    n_mid_actual = mid_chain["n_subdomains"]
    M_outer_at_z2 = T_mid @ M_left

    # ---------------------------------------------------------
    # right patch: physical horizon asymptotic basis on [z2, 1]
    # ---------------------------------------------------------
    if use_mp_spectral:
        sol_in = solve_basis_domain_mp(mode, "in", N_right, z2, 1.0, "right", dps=mp_dps_highw)
        sol_out = solve_basis_domain_mp(mode, "out", N_right, z2, 1.0, "right", dps=mp_dps_highw)
    else:
        sol_in = solve_basis_domain(mode, "in", N_right, z2, 1.0, "right")
        sol_out = solve_basis_domain(mode, "out", N_right, z2, 1.0, "right")

    R_in_z2, Rr_in_z2 = basis_values_at_match(mode, "in", sol_in, "left")
    R_out_z2, Rr_out_z2 = basis_values_at_match(mode, "out", sol_out, "left")

    y_in_z2 = _R_state_to_factor_state(
        mode,
        z2,
        R_in_z2,
        Rr_in_z2,
        middle_basis_resolved,
        middle_p=middle_p,
        middle_sigma=middle_sigma,
    )
    y_out_z2 = _R_state_to_factor_state(
        mode,
        z2,
        R_out_z2,
        Rr_out_z2,
        middle_basis_resolved,
        middle_p=middle_p,
        middle_sigma=middle_sigma,
    )

    # ---------------------------------------------------------
    # Back-propagate right states to z1 in the same middle state,
    # then decompose on the left physical down/up basis expressed in that state.
    # ---------------------------------------------------------
    use_mp = use_mp_backend(mode.omega, omega_mp_cut) or is_high_omega or use_mp_spectral
    mp_dps = mp_dps_highw if use_mp_spectral else mp_dps_loww

    if use_mp:
        state_in_z1_back, diag_back_in = solve_2x2_mp(T_mid, y_in_z2, dps=mp_dps)
        state_out_z1_back, diag_back_out = solve_2x2_mp(T_mid, y_out_z2, dps=mp_dps)
        coef_in, diag_in = solve_scaled_2x2_mp(M_left, state_in_z1_back, dps=mp_dps)
        coef_out, diag_out = solve_scaled_2x2_mp(M_left, state_out_z1_back, dps=mp_dps)
    else:
        state_in_z1_back, diag_back_in = solve_2x2_np(T_mid, y_in_z2)
        state_out_z1_back, diag_back_out = solve_2x2_np(T_mid, y_out_z2)
        coef_in, diag_in = solve_scaled_2x2_np(M_left, state_in_z1_back)
        coef_out, diag_out = solve_scaled_2x2_np(M_left, state_out_z1_back)

    Cin_down, Cin_up = coef_in
    Cout_down, Cout_up = coef_out

    state_in_z1_recon = M_left @ np.array([Cin_down, Cin_up], dtype=complex)
    state_out_z1_recon = M_left @ np.array([Cout_down, Cout_up], dtype=complex)

    state_in_z1_relerr = float(
        np.linalg.norm(state_in_z1_recon - state_in_z1_back)
        / max(np.linalg.norm(state_in_z1_back), 1.0e-300)
    )
    state_out_z1_relerr = float(
        np.linalg.norm(state_out_z1_recon - state_out_z1_back)
        / max(np.linalg.norm(state_out_z1_back), 1.0e-300)
    )

    S = np.array([[Cin_down, Cin_up], [Cout_down, Cout_up]], dtype=complex)
    b_inc = complex(Cin_down)
    b_ref = complex(Cin_up)
    b_trans = 1.0 + 0.0j

    ratio_ref_over_inc = _safe_complex_ratio(b_ref, b_inc)
    ratio_inc_over_ref = _safe_complex_ratio(b_inc, b_ref)

    # Legacy log-derivative fallback is disabled by default for scaled middle variables;
    # it was designed for raw-R post-processing and was not stable at high frequency.
    if use_logderivative_reflection is None:
        use_logderivative_reflection = False if middle_basis_resolved != "raw" else abs(mode.omega) >= 3.0
    plateau_diag = None
    b_ref_logderiv: complex | None = None

    # ---------------------------------------------------------
    # Abel diagnostics in physical [R,R_r] variables
    # ---------------------------------------------------------
    r_match_outer = r_of_z(z1, mode)
    r_match_inner = r_of_z(z2, mode)

    outer_abel_num = abel_constant_from_pair(
        R_down_z1, Rr_down_z1, R_up_z1, Rr_up_z1, r_match_outer, mode
    )
    outer_abel_th = outer_abel_theory(mode)
    outer_abel_residual = rel_complex_residual(outer_abel_num, outer_abel_th)

    inner_abel_num = abel_constant_from_pair(
        R_in_z2, Rr_in_z2, R_out_z2, Rr_out_z2, r_match_inner, mode
    )
    inner_abel_th = inner_abel_theory(mode)
    inner_abel_residual = rel_complex_residual(inner_abel_num, inner_abel_th)

    if use_mp:
        detS_num = det_2x2_mp(S, dps=mp_dps_loww)
    else:
        detS_num = np.linalg.det(S)
    detS_th = inner_abel_th / outer_abel_th
    detS_residual = rel_complex_residual(detS_num, detS_th)

    result = {
        "S": S,
        "B_inc": b_inc,
        "B_ref": b_ref,
        "B_ref_matrix": complex(Cin_up),
        "B_ref_logderiv": b_ref_logderiv,
        "B_trans": b_trans,
        "ratio_ref_over_inc": ratio_ref_over_inc,
        "ratio_inc_over_ref": ratio_inc_over_ref,
        "B_trans_over_B_inc": _safe_complex_ratio(b_trans, b_inc),
        "B_ref_over_B_inc": ratio_ref_over_inc,

        "middle_basis": middle_basis_resolved,
        "middle_p": float(middle_p),
        "middle_sigma": float(middle_sigma),
        "use_logderivative_reflection": bool(use_logderivative_reflection),
        "plateau_diag": plateau_diag,

        "Cin_down": complex(Cin_down),
        "Cin_up": complex(Cin_up),
        "Cout_down": complex(Cout_down),
        "Cout_up": complex(Cout_up),

        "outer_abel_num": complex(outer_abel_num),
        "outer_abel_th": complex(outer_abel_th),
        "outer_abel_residual": float(outer_abel_residual),

        "inner_abel_num": complex(inner_abel_num),
        "inner_abel_th": complex(inner_abel_th),
        "inner_abel_residual": float(inner_abel_residual),

        "detS_num": complex(detS_num),
        "detS_th": complex(detS_th),
        "detS_residual": float(detS_residual),

        "z1": float(z1),
        "z2": float(z2),
        "r_match_outer": float(r_match_outer),
        "r_match_inner": float(r_match_inner),

        "n_mid_subdomains": int(n_mid_actual),
        "N_left": int(N_left),
        "N_mid": int(N_mid),
        "N_right": int(N_right),

        "cond_M_left": float(np.linalg.cond(M_left)),
        "cond_M_left_R": float(np.linalg.cond(M_left_R)),
        "cond_T_mid": float(np.linalg.cond(T_mid)),
        "cond_M_outer_at_z2": float(np.linalg.cond(M_outer_at_z2)),

        "solve_in_relres": float(diag_in["relres"]),
        "solve_out_relres": float(diag_out["relres"]),
        "solve_in_cond_raw": float(diag_in["cond_raw"]),
        "solve_out_cond_raw": float(diag_out["cond_raw"]),
        "solve_in_cond_scaled": float(diag_in["cond_scaled"]),
        "solve_out_cond_scaled": float(diag_out["cond_scaled"]),
        "solve_in_smax": float(diag_in["smax"]),
        "solve_in_smin": float(diag_in["smin"]),
        "solve_out_smax": float(diag_out["smax"]),
        "solve_out_smin": float(diag_out["smin"]),

        "state_in_z1_relerr": state_in_z1_relerr,
        "state_out_z1_relerr": state_out_z1_relerr,
        "state_in_z1_back": state_in_z1_back,
        "state_out_z1_back": state_out_z1_back,
        "state_in_z1_recon": state_in_z1_recon,
        "state_out_z1_recon": state_out_z1_recon,

        "use_mp_backend": bool(use_mp),
        "is_high_omega": bool(is_high_omega),
        "use_mp_spectral": bool(use_mp_spectral),
        "omega_mp_cut": float(omega_mp_cut),
        "mp_dps": int(mp_dps if use_mp else 0),
        "back_in_relres": float(diag_back_in["relres"]),
        "back_out_relres": float(diag_back_out["relres"]),
    }

    if return_details:
        state_in_z1_R = _factor_state_to_R_state(
            mode,
            z1,
            state_in_z1_recon[0],
            state_in_z1_recon[1],
            middle_basis_resolved,
            middle_p=middle_p,
            middle_sigma=middle_sigma,
        )
        extra = {
            "sol_down": sol_down,
            "sol_up": sol_up,
            "sol_in": sol_in,
            "sol_out": sol_out,
            "M_left": M_left,
            "M_left_R": M_left_R,
            "T_mid": T_mid,
            "M_outer_at_z2": M_outer_at_z2,
            "state_in_z1": state_in_z1_recon,
            "state_in_z1_R": state_in_z1_R,
        }
        if n_mid_actual == 1:
            mid = solve_middle_factor_pair(
                mode,
                N_mid,
                z1,
                z2,
                middle_basis=middle_basis_resolved,
                middle_p=middle_p,
                middle_sigma=middle_sigma,
            )
            extra["sol_mid_val"] = mid["sol_mid_val"]
            extra["sol_mid_der"] = mid["sol_mid_der"]
        else:
            chain_detailed = _compute_mid_transport_chain(
                mode,
                N_mid,
                z1,
                z2,
                n_subdomains=n_mid_actual,
                store_subdomain_sols=True,
                middle_basis=middle_basis_resolved,
                middle_p=middle_p,
                middle_sigma=middle_sigma,
            )
            extra["z_breaks"] = chain_detailed["z_breaks"]
            extra["sub_sols"] = chain_detailed["sub_sols"]
            extra["T_prefixes"] = chain_detailed["T_prefixes"]
        result.update(extra)

    return result


class TeukRadAmplitudeIn3Patch:
    def __init__(
        self,
        mode: KerrMode,
        N_left: int = 64,
        N_mid: int = 96,
        N_right: int = 64,
        z1: float | None = None,
        z2: float | None = None,
        *,
        n_mid_subdomains: int | None = None,
        omega_mp_cut: float = 1.0e-2,
        mp_dps_loww: int = 200,
        use_logderivative_reflection: bool | None = None,
        middle_basis: str = "auto",
        middle_p: float = 3.4,
        middle_sigma: float = 1.0,
        use_mp_spectral: bool | None = None,
        mp_dps_highw: int = 200,
    ):
        self.mode = mode
        self.M = mode.M
        self.a = mode.a
        self.omega = mode.omega
        self.ell = mode.ell
        self.m = mode.m

        self.N_left = N_left
        self.N_mid = N_mid
        self.N_right = N_right
        self.z1_param = z1
        self.z2 = z2
        self.n_mid_subdomains = n_mid_subdomains

        self.lam = mode.lam
        self._smatrix: Dict[str, complex | np.ndarray | None] | None = None
        self._result: InAmplitudesResult | None = None
        self.omega_mp_cut = omega_mp_cut
        self.mp_dps_loww = mp_dps_loww
        self.mp_dps_highw = mp_dps_highw
        self.use_logderivative_reflection = use_logderivative_reflection
        self.middle_basis = middle_basis
        self.middle_p = middle_p
        self.middle_sigma = middle_sigma
        self.use_mp_spectral = use_mp_spectral
        self.middle_p = middle_p
        self.middle_sigma = middle_sigma

    def __call__(self) -> InAmplitudesResult:
        return self.to_result()

    @property
    def z1(self) -> float:
        if self._smatrix is not None:
            return float(self._smatrix.get("z1", 0.1))
        return float(self.z1_param) if self.z1_param is not None else 0.1

    @property
    def smatrix(self) -> Dict[str, complex | np.ndarray | None]:
        if self._smatrix is None:
            self._smatrix = compute_smatrix_three_patch_with_abel(
                self.mode,
                N_left=self.N_left,
                N_mid=self.N_mid,
                N_right=self.N_right,
                z1=self.z1_param,
                z2=self.z2,
                n_mid_subdomains=self.n_mid_subdomains,
                return_details=False,
                omega_mp_cut=self.omega_mp_cut,
                mp_dps_loww=self.mp_dps_loww,
                use_logderivative_reflection=self.use_logderivative_reflection,
                middle_basis=self.middle_basis,
                middle_p=self.middle_p,
                middle_sigma=self.middle_sigma,
                use_mp_spectral=self.use_mp_spectral,
                mp_dps_highw=self.mp_dps_highw,
            )
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
                N_in=self.N_right,
                N_out=self.N_left,
                z_m=self.z1,
                ratio_ref_over_inc=self.ratio_ref_over_inc,
                ratio_inc_over_ref=self.ratio_inc_over_ref,
            )
        return self._result

    @property
    def result(self) -> InAmplitudesResult:
        return self.to_result()

    def __repr__(self) -> str:
        return (
            f"TeukRadAmplitudeIn3Patch(l={self.ell}, m={self.m}, a={self.a}, omega={self.omega}, "
            f"N_left={self.N_left}, N_mid={self.N_mid}, N_right={self.N_right}, "
            f"z1={self.z1}, z2={self.z2}, middle_basis={self.middle_basis}, "
            f"middle_p={self.middle_p}, middle_sigma={self.middle_sigma})"
        )


class TeukRadAmplitudeIn3PatchWithAbelChecks(TeukRadAmplitudeIn3Patch):
    @property
    def outer_abel_residual(self) -> float:
        return float(self.smatrix["outer_abel_residual"])

    @property
    def inner_abel_residual(self) -> float:
        return float(self.smatrix["inner_abel_residual"])

    @property
    def detS_residual(self) -> float:
        return float(self.smatrix["detS_residual"])

