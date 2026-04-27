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
    r_of_z,
)
from utils.amplitude_three_patch import (
    compute_smatrix_three_patch_with_abel,
)


def leaver_factor(r, mode: KerrMode):
    """
    Same Leaver prefactor as the current profile code.
    """
    r = np.asarray(r, dtype=np.float64)
    sigma_p = (2.0 * mode.omega * mode.rp - mode.a * mode.m) / (mode.rp - mode.rm)
    pp = -mode.s - 1j * sigma_p
    pm = -1.0 - mode.s + 2.0j * mode.omega + 1j * sigma_p

    drp = (r - mode.rp).astype(np.complex128)
    drm = (r - mode.rm).astype(np.complex128)
    return np.exp(1j * mode.omega * r) * (drp ** pp) * (drm ** pm)


def _basis_full_profile_three(mode: KerrMode, basis: str, sol, *, drop_left=False, drop_right=False):
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
    elif basis == "raw":
        F = np.ones_like(r, dtype=np.complex128)
    else:
        raise ValueError(f"Unknown basis={basis}")

    R = F * u
    return {"z": z, "r": r, "u": u, "R": R}


def _complex_interp_linear(x_query, x_nodes, y_nodes):
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
    return y2 + (x_target - x2) * (y2 - y1) / (x2 - x1)


@dataclass(slots=True)
class InModeLeaverInterpolant3Patch:
    mode: KerrMode
    z_left: np.ndarray
    psi_left: np.ndarray
    z_mid: np.ndarray
    psi_mid: np.ndarray
    z_right: np.ndarray
    psi_right: np.ndarray
    z1: float
    z2: float

    def psi_of_z(self, z):
        zq = np.asarray(z, dtype=np.float64)
        scalar = np.ndim(z) == 0
        zq1 = np.atleast_1d(zq)
        out = np.empty_like(zq1, dtype=np.complex128)

        m_left = zq1 <= self.z1
        m_mid = (zq1 > self.z1) & (zq1 <= self.z2)
        m_right = zq1 > self.z2

        if np.any(m_left):
            out[m_left] = _complex_interp_linear(zq1[m_left], self.z_left, self.psi_left)
        if np.any(m_mid):
            out[m_mid] = _complex_interp_linear(zq1[m_mid], self.z_mid, self.psi_mid)
        if np.any(m_right):
            out[m_right] = _complex_interp_linear(zq1[m_right], self.z_right, self.psi_right)

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


def compute_smatrix_three_patch_with_profile(
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
):
    sm = compute_smatrix_three_patch_with_abel(
        mode,
        N_left=N_left,
        N_mid=N_mid,
        N_right=N_right,
        z1=z1,
        z2=z2,
        n_mid_subdomains=n_mid_subdomains,
        omega_mp_cut=omega_mp_cut,
        mp_dps_loww=mp_dps_loww,
        return_details=True,
    )

    b_inc = complex(sm["B_inc"])
    b_ref = complex(sm["B_ref"])
    b_trans = complex(sm["B_trans"])

    # ---------------------------------------------------------
    # left patch: true in-solution from down/up basis
    # ---------------------------------------------------------
    prof_down = _basis_full_profile_three(mode, "down", sm["sol_down"], drop_left=True, drop_right=False)
    prof_up   = _basis_full_profile_three(mode, "up",   sm["sol_up"],   drop_left=True, drop_right=False)

    z_left = prof_down["z"]
    r_left = prof_down["r"]
    Rin_left = b_inc * prof_down["R"] + b_ref * prof_up["R"]
    psi_left = Rin_left / leaver_factor(r_left, mode)

    z_left_full = np.concatenate(([0.0], z_left))
    psi_left_full = np.concatenate(([b_ref], psi_left))

    # ---------------------------------------------------------
    # middle patch: true in-solution (handles multi-subdomain)
    # ---------------------------------------------------------
    n_mid_actual = int(sm["n_mid_subdomains"])
    state_in_z1 = np.asarray(sm["state_in_z1"], dtype=np.complex128)

    if n_mid_actual == 1:
        prof_mid_val = _basis_full_profile_three(
            mode, "raw", sm["sol_mid_val"], drop_left=False, drop_right=False,
        )
        prof_mid_der = _basis_full_profile_three(
            mode, "raw", sm["sol_mid_der"], drop_left=False, drop_right=False,
        )
        Rin_mid = state_in_z1[0] * prof_mid_val["R"] + state_in_z1[1] * prof_mid_der["R"]
        z_mid = prof_mid_val["z"]
    else:
        z_breaks = sm.get("z_breaks")
        sub_sols = sm.get("sub_sols")
        T_prefixes = sm.get("T_prefixes")
        if z_breaks is None or sub_sols is None:
            raise RuntimeError("Multi-subdomain details missing (z_breaks/sub_sols)")

        z_parts, Rin_parts = [], []
        for i, (sd, Tpre) in enumerate(zip(sub_sols, T_prefixes)):
            st = Tpre @ state_in_z1
            pv = _basis_full_profile_three(mode, "raw", sd["sol_mid_val"], drop_left=False, drop_right=False)
            pd = _basis_full_profile_three(mode, "raw", sd["sol_mid_der"], drop_left=False, drop_right=False)
            z_parts.append(pv["z"])
            Rin_parts.append(st[0] * pv["R"] + st[1] * pd["R"])
        z_mid = np.concatenate(z_parts)
        Rin_mid = np.concatenate(Rin_parts)

    r_mid = r_of_z(z_mid, mode)
    psi_mid = Rin_mid / leaver_factor(r_mid, mode)

    # ---------------------------------------------------------
    # right patch: true in-solution from in/out basis
    # ---------------------------------------------------------
    prof_in = _basis_full_profile_three(mode, "in", sm["sol_in"], drop_left=False, drop_right=True)

    z_right = prof_in["z"]
    r_right = prof_in["r"]
    Rin_right = b_trans * prof_in["R"]
    psi_right = Rin_right / leaver_factor(r_right, mode)

    if len(z_right) >= 2:
        psi_h = _one_sided_linear_extrap(1.0, z_right[-2], z_right[-1], psi_right[-2], psi_right[-1])
    else:
        psi_h = psi_right[-1]

    z_right_full = np.concatenate((z_right, [1.0]))
    psi_right_full = np.concatenate((psi_right, [psi_h]))

    profile = InModeLeaverInterpolant3Patch(
        mode=mode,
        z_left=z_left_full,
        psi_left=psi_left_full,
        z_mid=z_mid,
        psi_mid=psi_mid,
        z_right=z_right_full,
        psi_right=psi_right_full,
        z1=float(sm["z1"]),
        z2=float(sm["z2"]),
    )

    sm["profile"] = profile
    sm["z_left"] = z_left_full
    sm["psi_left"] = psi_left_full
    sm["z_mid"] = z_mid
    sm["psi_mid"] = psi_mid
    sm["z_right"] = z_right_full
    sm["psi_right"] = psi_right_full

    return sm


class TeukRadAmplitudeIn3PatchWithProfile:
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
        self.z2_param = z2
        self.n_mid_subdomains = n_mid_subdomains
        self.omega_mp_cut = omega_mp_cut
        self.mp_dps_loww = mp_dps_loww

        self.lam = mode.lam if mode.lam is not None else None
        self._smatrix: Dict[str, complex | np.ndarray | None] | None = None

    @property
    def smatrix(self) -> Dict[str, complex | np.ndarray | None]:
        if self._smatrix is None:
            self._smatrix = compute_smatrix_three_patch_with_profile(
                self.mode,
                N_left=self.N_left,
                N_mid=self.N_mid,
                N_right=self.N_right,
                z1=self.z1_param,
                z2=self.z2_param,
                n_mid_subdomains=self.n_mid_subdomains,
                omega_mp_cut=self.omega_mp_cut,
                mp_dps_loww=self.mp_dps_loww,
            )
        return self._smatrix

    @property
    def profile(self) -> InModeLeaverInterpolant3Patch:
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
            f"TeukRadAmplitudeIn3PatchWithProfile(l={self.ell}, m={self.m}, a={self.a}, omega={self.omega}, "
            f"N_left={self.N_left}, N_mid={self.N_mid}, N_right={self.N_right}, "
            f"z1={self.z1_param}, z2={self.z2_param})"
        )