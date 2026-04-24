from __future__ import annotations

import numpy as np
import mpmath as mp


def use_mp_backend(omega: float, omega_cut: float = 1.0e-2) -> bool:
    return abs(float(omega)) < float(omega_cut)


def _to_mpc(z) -> mp.mpc:
    zc = complex(z)
    return mp.mpc(zc.real, zc.imag)


def _to_complex(z: mp.mpc) -> complex:
    return complex(float(mp.re(z)), float(mp.im(z)))


def solve_2x2_np(M: np.ndarray, y: np.ndarray, floor: float = 1.0e-300):
    M = np.asarray(M, dtype=complex)
    y = np.asarray(y, dtype=complex)
    x = np.linalg.solve(M, y)
    relres = np.linalg.norm(M @ x - y) / max(np.linalg.norm(y), floor)
    return x, {"relres": float(relres)}


def solve_2x2_mp(M: np.ndarray, y: np.ndarray, dps: int = 80, floor: float = 1.0e-300):
    M = np.asarray(M, dtype=complex)
    y = np.asarray(y, dtype=complex)

    with mp.workdps(dps):
        a = _to_mpc(M[0, 0])
        b = _to_mpc(M[0, 1])
        c = _to_mpc(M[1, 0])
        d = _to_mpc(M[1, 1])

        y0 = _to_mpc(y[0])
        y1 = _to_mpc(y[1])

        det = a * d - b * c
        x0 = (d * y0 - b * y1) / det
        x1 = (-c * y0 + a * y1) / det

    x = np.array([_to_complex(x0), _to_complex(x1)], dtype=complex)
    relres = np.linalg.norm(M @ x - y) / max(np.linalg.norm(y), floor)
    return x, {"relres": float(relres)}


def solve_scaled_2x2_np(M: np.ndarray, y: np.ndarray, floor: float = 1.0e-300):
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


def solve_scaled_2x2_mp(M: np.ndarray, y: np.ndarray, dps: int = 80, floor: float = 1.0e-300):
    M = np.asarray(M, dtype=complex)
    y = np.asarray(y, dtype=complex)

    col_norms = np.linalg.norm(M, axis=0)
    col_norms = np.where(col_norms > floor, col_norms, 1.0)

    Ms = M / col_norms[None, :]
    x_scaled, _ = solve_2x2_mp(Ms, y, dps=dps, floor=floor)
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


def det_2x2_mp(M: np.ndarray, dps: int = 80) -> complex:
    M = np.asarray(M, dtype=complex)
    with mp.workdps(dps):
        a = _to_mpc(M[0, 0])
        b = _to_mpc(M[0, 1])
        c = _to_mpc(M[1, 0])
        d = _to_mpc(M[1, 1])
        val = a * d - b * c
    return _to_complex(val)