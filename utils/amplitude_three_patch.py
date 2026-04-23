from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

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
)


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

def compute_smatrix_three_patch_with_abel(
    mode: KerrMode,
    N_left: int = 64,
    N_mid: int = 96,
    N_right: int = 64,
    z1: float = 0.1,
    z2: float = 0.9,
    *,
    return_details: bool = False,
):
    """
    Three-patch version:

        left  : [0.0, z1]   with down/up bases
        mid   : [z1,  z2]   with raw full-R transport basis
        right : [z2,  1.0]  with in/out bases

    Matching is done twice:
        (i)  left -> middle at z=z1
        (ii) middle -> right at z=z2

    The final amplitudes are still the coefficients of R_in / R_out
    in the outer (down/up) basis.
    """

    if not (0.0 < z1 < z2 < 1.0):
        raise ValueError("Require 0 < z1 < z2 < 1")

    # ---------------------------------------------------------
    # left patch: outer asymptotic basis on [0, z1]
    # ---------------------------------------------------------
    sol_down = solve_basis_domain(mode, "down", N_left, 0.0, z1, "left")
    sol_up = solve_basis_domain(mode, "up", N_left, 0.0, z1, "left")

    R_down_z1, Rr_down_z1 = basis_values_at_match(mode, "down", sol_down, "right")
    R_up_z1, Rr_up_z1 = basis_values_at_match(mode, "up", sol_up, "right")

    M_left = np.array(
        [[R_down_z1, R_up_z1], [Rr_down_z1, Rr_up_z1]],
        dtype=complex,
    )

    # ---------------------------------------------------------
    # middle patch: raw full-R transport on [z1, z2]
    # ---------------------------------------------------------
    mid = solve_middle_raw_pair(mode, N_mid, z1, z2)
    sol_mid_val = mid["sol_mid_val"]
    sol_mid_der = mid["sol_mid_der"]
    T_mid = mid["T_mid"]

    # Propagate the two outer basis solutions from z1 to z2:
    # columns are the [R, R_r] states of down/up at z2 after mid transport
    M_outer_at_z2 = T_mid @ M_left

    # ---------------------------------------------------------
    # right patch: horizon asymptotic basis on [z2, 1]
    # ---------------------------------------------------------
    sol_in = solve_basis_domain(mode, "in", N_right, z2, 1.0, "right")
    sol_out = solve_basis_domain(mode, "out", N_right, z2, 1.0, "right")

    R_in_z2, Rr_in_z2 = basis_values_at_match(mode, "in", sol_in, "left")
    R_out_z2, Rr_out_z2 = basis_values_at_match(mode, "out", sol_out, "left")

    y_in_z2 = np.array([R_in_z2, Rr_in_z2], dtype=complex)
    y_out_z2 = np.array([R_out_z2, Rr_out_z2], dtype=complex)

    # ---------------------------------------------------------
    # IMPORTANT CHANGE:
    # instead of decomposing directly on M_outer_at_z2 (often badly conditioned),
    # first propagate the right states backward to z1 through T_mid^{-1},
    # then decompose on the original left outer basis M_left.
    # ---------------------------------------------------------
    state_in_z1_back = np.linalg.solve(T_mid, y_in_z2)
    state_out_z1_back = np.linalg.solve(T_mid, y_out_z2)

    coef_in, diag_in = _solve_scaled_2x2(M_left, state_in_z1_back)
    coef_out, diag_out = _solve_scaled_2x2(M_left, state_out_z1_back)

    Cin_down, Cin_up = coef_in
    Cout_down, Cout_up = coef_out

    # reconstructed states at z1 for debugging
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

    # ---------------------------------------------------------
    # Abel diagnostics
    # outer Abel constant is checked on the left patch at z1
    # inner Abel constant is checked on the right patch at z2
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

    detS_num = np.linalg.det(S)
    detS_th = inner_abel_th / outer_abel_th
    detS_residual = rel_complex_residual(detS_num, detS_th)

    result = {
        "S": S,
        "B_inc": b_inc,
        "B_ref": b_ref,
        "B_trans": b_trans,
        "ratio_ref_over_inc": ratio_ref_over_inc,
        "ratio_inc_over_ref": ratio_inc_over_ref,
        "B_trans_over_B_inc": _safe_complex_ratio(b_trans, b_inc),
        "B_ref_over_B_inc": ratio_ref_over_inc,

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

        "cond_M_left": float(np.linalg.cond(M_left)),
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
    }

    if return_details:
        # True in-mode state at z1 in [R, R_r] variables
        state_in_z1 = M_left @ np.array([Cin_down, Cin_up], dtype=complex)

        result.update(
            {
                "sol_down": sol_down,
                "sol_up": sol_up,
                "sol_mid_val": sol_mid_val,
                "sol_mid_der": sol_mid_der,
                "sol_in": sol_in,
                "sol_out": sol_out,
                "M_left": M_left,
                "T_mid": T_mid,
                "M_outer_at_z2": M_outer_at_z2,
                "state_in_z1": state_in_z1,
            }
        )

    return result


class TeukRadAmplitudeIn3Patch:
    def __init__(
        self,
        mode: KerrMode,
        N_left: int = 64,
        N_mid: int = 96,
        N_right: int = 64,
        z1: float = 0.1,
        z2: float = 0.9,
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
        self.z1 = z1
        self.z2 = z2

        self.lam = mode.lam
        self._smatrix: Dict[str, complex | np.ndarray | None] | None = None
        self._result: InAmplitudesResult | None = None

    def __call__(self) -> InAmplitudesResult:
        return self.to_result()

    @property
    def smatrix(self) -> Dict[str, complex | np.ndarray | None]:
        if self._smatrix is None:
            self._smatrix = compute_smatrix_three_patch_with_abel(
                self.mode,
                N_left=self.N_left,
                N_mid=self.N_mid,
                N_right=self.N_right,
                z1=self.z1,
                z2=self.z2,
                return_details=False,
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
                N_in=self.N_right,   # keep compatibility with existing dataclass fields
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
            f"z1={self.z1}, z2={self.z2})"
        )


class TeukRadAmplitudeIn3PatchWithAbelChecks(TeukRadAmplitudeIn3Patch):
    @property
    def smatrix(self) -> Dict[str, complex | np.ndarray | None]:
        if self._smatrix is None:
            self._smatrix = compute_smatrix_three_patch_with_abel(
                self.mode,
                N_left=self.N_left,
                N_mid=self.N_mid,
                N_right=self.N_right,
                z1=self.z1,
                z2=self.z2,
                return_details=False,
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