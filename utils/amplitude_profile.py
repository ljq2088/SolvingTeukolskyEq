from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from utils.mode import KerrMode
from utils.amplitude import (
    A_down,
    A_in,
    A_out,
    A_up,
    _safe_complex_ratio,
    basis_values_at_match,
    r_of_z,
    solve_basis_domain,
)


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


@dataclass(slots=True)
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

    mode: KerrMode
    z_outer: np.ndarray
    psi_outer: np.ndarray
    z_inner: np.ndarray
    psi_inner: np.ndarray
    z_match: float

    def psi_of_z(self, z):
        zq = np.asarray(z, dtype=np.float64)
        scalar = np.ndim(z) == 0
        zq1 = np.atleast_1d(zq)
        out = np.empty_like(zq1, dtype=np.complex128)

        mask_outer = zq1 <= self.z_match
        if np.any(mask_outer):
            out[mask_outer] = _complex_interp_linear(
                zq1[mask_outer], self.z_outer, self.psi_outer
            )
        if np.any(~mask_outer):
            out[~mask_outer] = _complex_interp_linear(
                zq1[~mask_outer], self.z_inner, self.psi_inner
            )

        if scalar:
            return complex(out[0])
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


def compute_smatrix_with_profile(mode: KerrMode, N_in: int = 80, N_out: int = 80, z_m: float = 0.3):
    # outer domain: [0, z_m]
    sol_down = solve_basis_domain(mode, 'down', N_out, 0.0, z_m, 'left')
    sol_up = solve_basis_domain(mode, 'up', N_out, 0.0, z_m, 'left')

    R_down_m, Rr_down_m = basis_values_at_match(mode, 'down', sol_down, 'right')
    R_up_m, Rr_up_m = basis_values_at_match(mode, 'up', sol_up, 'right')
    Mmatch = np.array([[R_down_m, R_up_m], [Rr_down_m, Rr_up_m]], dtype=complex)

    # inner domain: [z_m, 1]
    sol_in = solve_basis_domain(mode, 'in', N_in, z_m, 1.0, 'right')
    sol_out = solve_basis_domain(mode, 'out', N_in, z_m, 1.0, 'right')

    R_in_m, Rr_in_m = basis_values_at_match(mode, 'in', sol_in, 'left')
    R_out_m, Rr_out_m = basis_values_at_match(mode, 'out', sol_out, 'left')

    Cin_down, Cin_up = np.linalg.solve(Mmatch, np.array([R_in_m, Rr_in_m], dtype=complex))
    Cout_down, Cout_up = np.linalg.solve(Mmatch, np.array([R_out_m, Rr_out_m], dtype=complex))

    b_inc = complex(Cin_down)
    b_ref = complex(Cin_up)
    b_trans = 1.0 + 0.0j
    ratio_ref_over_inc = _safe_complex_ratio(b_ref, b_inc)
    ratio_inc_over_ref = _safe_complex_ratio(b_inc, b_ref)

    # Outer patch: true in-solution = B_inc * down + B_ref * up
    prof_down = _basis_full_profile(mode, 'down', sol_down, drop_left=True, drop_right=False)
    prof_up = _basis_full_profile(mode, 'up', sol_up, drop_left=True, drop_right=False)

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
        psi_h = _one_sided_linear_extrap(
            1.0, z_inner[-2], z_inner[-1], psi_inner[-2], psi_inner[-1]
        )
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

    return {
        "S": np.array([[Cin_down, Cin_up], [Cout_down, Cout_up]], dtype=complex),
        "B_inc": b_inc,
        "B_ref": b_ref,
        "B_trans": b_trans,
        "ratio_ref_over_inc": ratio_ref_over_inc,
        "ratio_inc_over_ref": ratio_inc_over_ref,
        "B_trans_over_B_inc": _safe_complex_ratio(b_trans, b_inc),
        "B_ref_over_B_inc": ratio_ref_over_inc,
        "profile": profile,
        "z_outer": z_outer_full,
        "psi_outer": psi_outer_full,
        "z_inner": z_inner_full,
        "psi_inner": psi_inner_full,
        "sol_down": sol_down,
        "sol_up": sol_up,
        "sol_in": sol_in,
        "sol_out": sol_out,
    }


class TeukRadAmplitudeInWithProfile:
    """
    Amplitude calculator + piecewise interpolant of the true in-mode solution
    reconstructed through the Leaver-reduced function Psi = R_in / P_Leaver.
    """

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
        self.lam = mode.lam if mode.lam is not None else None
        self._smatrix: Dict[str, complex | np.ndarray | None] | None = None

    @property
    def smatrix(self) -> Dict[str, complex | np.ndarray | None]:
        if self._smatrix is None:
            self._smatrix = compute_smatrix_with_profile(
                self.mode,
                N_in=self.N_in,
                N_out=self.N_out,
                z_m=self.z_m,
            )
        return self._smatrix

    @property
    def profile(self) -> InModeLeaverInterpolant:
        return self.smatrix["profile"]

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

    def __repr__(self) -> str:
        return (
            f"TeukRadAmplitudeInWithProfile(l={self.ell}, m={self.m}, a={self.a}, omega={self.omega}, "
            f"N_in={self.N_in}, N_out={self.N_out}, z_m={self.z_m})"
        )