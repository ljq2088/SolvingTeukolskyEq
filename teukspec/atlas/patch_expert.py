from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from teukspec.amplitude.multipoint_ls import solve_binc_bref_from_branches
from teukspec.decoders.param_cheb_decoder import ParamChebDecoder
from utils.amplitude import A_in, coeffs_numeric, r_of_z
from utils.mode import KerrMode


class PatchExpert:
    def __init__(self, patch_dir: str | Path, enable_corrector: bool = False):
        if enable_corrector:
            raise NotImplementedError("FiLM correctors are not implemented in the first milestone.")
        self.patch_dir = Path(patch_dir)
        with open(self.patch_dir / "patch_config.json") as f:
            self.config = json.load(f)
        self.decoders = {
            basis: ParamChebDecoder.load_npz(self.patch_dir / f"decoder_{basis}.npz")
            for basis in ("in", "down", "up")
            if (self.patch_dir / f"decoder_{basis}.npz").exists()
        }

    def mode(self, a: float, omega: float) -> KerrMode:
        return KerrMode(M=1.0, a=float(a), omega=float(omega), ell=self.config.get("ell", 2), m=self.config.get("m", 2), s=self.config.get("s", -2))

    def eval_branches(self, a: float, omega: float, z_values: np.ndarray) -> dict[str, np.ndarray]:
        logw = float(np.log10(omega))
        return {basis: decoder.eval_u(z_values, a, logw) for basis, decoder in self.decoders.items()}

    def eval_branch_derivatives(self, a: float, omega: float, branch: str, z_values: np.ndarray):
        return self.decoders[branch].eval_u_derivatives(z_values, a, float(np.log10(omega)))

    def eval_R_in(self, a: float, omega: float, r_values: np.ndarray) -> np.ndarray:
        mode = self.mode(a, omega)
        z = mode.rp / np.asarray(r_values, dtype=float)
        u = self.decoders["in"].eval_u(z, a, float(np.log10(omega)))
        return A_in(np.asarray(r_values, dtype=float), mode) * u

    def solve_amplitudes(self, a: float, omega: float, z_values: np.ndarray | None = None) -> dict:
        if z_values is None:
            z_values = self.default_match_z_values()
        mode = self.mode(a, omega)
        branches = self.eval_branches(a, omega, z_values)
        return solve_binc_bref_from_branches(mode, z_values, branches["in"], branches["down"], branches["up"])

    def default_match_z_values(self, n: int = 32) -> np.ndarray:
        z_left = self.decoders["in"].z_domain[0]
        z_right = self.decoders["down"].z_domain[1]
        lo = max(z_left, 1.0e-8)
        hi = min(z_right, 1.0 - 1.0e-8)
        return np.linspace(lo, hi, n)

    def residual_scan(self, a: float, omega: float, branch: str, z_values: np.ndarray) -> np.ndarray:
        mode = self.mode(a, omega)
        u, uz, uzz = self.eval_branch_derivatives(a, omega, branch, z_values)
        B2, B1, B0 = coeffs_numeric(np.asarray(z_values, dtype=float), mode, branch)
        res = B2 * uzz + B1 * uz + B0 * u
        den = np.maximum.reduce([np.abs(B2 * uzz), np.abs(B1 * uz), np.abs(B0 * u)])
        return np.abs(res) / np.maximum(den, 1.0e-300)
