from __future__ import annotations

import numpy as np

from utils.amplitude import A_down, A_in, A_up, r_of_z
from utils.mode import KerrMode


def solve_scaled_lstsq(A: np.ndarray, b: np.ndarray, floor: float = 1.0e-300) -> tuple[np.ndarray, float, float]:
    A = np.asarray(A, dtype=np.complex128)
    b = np.asarray(b, dtype=np.complex128)
    row_norm = np.linalg.norm(A, axis=1)
    row_scale = np.where(row_norm > floor, row_norm, 1.0)
    Ar = A / row_scale[:, None]
    br = b / row_scale
    col_norm = np.linalg.norm(Ar, axis=0)
    col_scale = np.where(col_norm > floor, col_norm, 1.0)
    As = Ar / col_scale[None, :]
    x_scaled, *_ = np.linalg.lstsq(As, br, rcond=None)
    x = x_scaled / col_scale
    residual = np.linalg.norm(A @ x - b) / max(np.linalg.norm(b), floor)
    cond = float(np.linalg.cond(As))
    return x, float(residual), cond


def solve_binc_bref_from_branches(
    mode: KerrMode,
    z_values: np.ndarray,
    u_in: np.ndarray,
    u_down: np.ndarray,
    u_up: np.ndarray,
) -> dict:
    z_values = np.asarray(z_values, dtype=float)
    r = r_of_z(z_values, mode)
    R_in = A_in(r, mode) * np.asarray(u_in, dtype=np.complex128)
    R_down = A_down(r, mode) * np.asarray(u_down, dtype=np.complex128)
    R_up = A_up(r, mode) * np.asarray(u_up, dtype=np.complex128)
    coef, residual, cond = solve_scaled_lstsq(np.column_stack([R_down, R_up]), R_in)
    return {
        "B_inc": complex(coef[0]),
        "B_ref": complex(coef[1]),
        "residual": residual,
        "cond": cond,
    }
