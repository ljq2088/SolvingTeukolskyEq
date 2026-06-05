from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from teukspec.core.chebyshev import clenshaw_eval, eval_tensor_decoder, z_to_comp_xi
from utils.amplitude import _domain_D
from utils.mode import KerrMode


class ParamChebDecoder:
    def __init__(
        self,
        decoder_coeffs: np.ndarray,
        a_center: float,
        a_half_width: float,
        logw_center: float,
        logw_half_width: float,
        z_domain: tuple[float, float],
        n_z: int,
        deg_a: int,
        deg_logw: int,
        basis: str,
        grid_kind: str,
        ell: int = 2,
        m: int = 2,
        s: int = -2,
        domain: str | None = None,
    ):
        self.decoder_coeffs = np.asarray(decoder_coeffs, dtype=np.complex128)
        self.a_center = float(a_center)
        self.a_half_width = float(a_half_width)
        self.logw_center = float(logw_center)
        self.logw_half_width = float(logw_half_width)
        self.z_domain = (float(z_domain[0]), float(z_domain[1]))
        self.n_z = int(n_z)
        self.deg_a = int(deg_a)
        self.deg_logw = int(deg_logw)
        self.basis = str(basis)
        self.grid_kind = str(grid_kind)
        self.ell = int(ell)
        self.m = int(m)
        self.s = int(s)
        self.domain = domain or ("inner" if basis == "in" else "outer")

    def xi_a(self, a: np.ndarray | float) -> np.ndarray:
        return (np.asarray(a, dtype=float) - self.a_center) / self.a_half_width

    def xi_logw(self, logw: np.ndarray | float) -> np.ndarray:
        return (np.asarray(logw, dtype=float) - self.logw_center) / self.logw_half_width

    def eval_coeff(self, a: float, logw: float) -> np.ndarray:
        out = eval_tensor_decoder(self.decoder_coeffs, np.array([self.xi_a(a)]), np.array([self.xi_logw(logw)]))
        return out[0, 0].astype(np.complex128)

    def eval_u(self, z: np.ndarray | float, a: float, logw: float) -> np.ndarray:
        coeff = self.eval_coeff(a, logw)
        mode = KerrMode(M=1.0, a=float(a), omega=10.0 ** float(logw), ell=self.ell, m=self.m, s=self.s)
        xi = z_to_comp_xi(z, *self.z_domain, mode, self.domain, self.grid_kind)
        return clenshaw_eval(coeff, xi)

    def eval_u_derivatives(self, z_nodes: np.ndarray, a: float, logw: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        z_nodes = np.asarray(z_nodes, dtype=float)
        mode = KerrMode(M=1.0, a=float(a), omega=10.0 ** float(logw), ell=self.ell, m=self.m, s=self.s)
        coeff = self.eval_coeff(a, logw)
        xi_nodes = z_to_comp_xi(z_nodes, *self.z_domain, mode, self.domain, self.grid_kind)
        u = np.asarray(clenshaw_eval(coeff, xi_nodes), dtype=np.complex128)
        D, z_grid = _domain_D(mode, self.n_z, self.z_domain[0], self.z_domain[1], grid_kind=self.grid_kind, domain=self.domain)
        xi_grid = z_to_comp_xi(z_grid, *self.z_domain, mode, self.domain, self.grid_kind)
        u_grid = np.asarray(clenshaw_eval(coeff, xi_grid), dtype=np.complex128)
        uz_grid = D @ u_grid
        uzz_grid = D @ uz_grid
        uz_coeff = np.polynomial.chebyshev.chebfit(xi_grid, uz_grid, self.n_z)
        uzz_coeff = np.polynomial.chebyshev.chebfit(xi_grid, uzz_grid, self.n_z)
        uz = clenshaw_eval(uz_coeff, xi_nodes)
        uzz = clenshaw_eval(uzz_coeff, xi_nodes)
        return u, np.asarray(uz, dtype=np.complex128), np.asarray(uzz, dtype=np.complex128)

    def metadata(self) -> dict:
        return {
            "a_center": self.a_center,
            "a_half_width": self.a_half_width,
            "logw_center": self.logw_center,
            "logw_half_width": self.logw_half_width,
            "z_left": self.z_domain[0],
            "z_right": self.z_domain[1],
            "n_z": self.n_z,
            "deg_a": self.deg_a,
            "deg_logw": self.deg_logw,
            "basis": self.basis,
            "grid_kind": self.grid_kind,
            "ell": self.ell,
            "m": self.m,
            "s": self.s,
            "domain": self.domain,
        }

    def save_npz(self, path: str | Path) -> None:
        path = Path(path)
        np.savez(path, decoder_coeffs=self.decoder_coeffs, metadata=json.dumps(self.metadata()))

    @classmethod
    def load_npz(cls, path: str | Path) -> "ParamChebDecoder":
        data = np.load(path, allow_pickle=False)
        meta = json.loads(str(data["metadata"]))
        return cls(
            data["decoder_coeffs"],
            meta["a_center"],
            meta["a_half_width"],
            meta["logw_center"],
            meta["logw_half_width"],
            (meta["z_left"], meta["z_right"]),
            meta["n_z"],
            meta["deg_a"],
            meta["deg_logw"],
            meta["basis"],
            meta["grid_kind"],
            ell=meta.get("ell", 2),
            m=meta.get("m", 2),
            s=meta.get("s", -2),
            domain=meta.get("domain"),
        )
