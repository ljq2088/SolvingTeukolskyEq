"""Stage-1 R_over_P reconstruction from network output f_R(y).

R_in = P(r) * h2 * S(x)
R_over_P = h2 * S(x)

Where:
  f_R = model.predict_Rin(a, omega, y, u, v)
  c_H = horizon_regularity_slope(a, omega, lambda_, ...)
  S(x) = compose_reduced_shape_from_f(f_R, y, c_H)
  h2 = h_factor(a, omega, ...)
"""

import torch

from .transform_y import compose_reduced_shape_from_f, horizon_regularity_slope, h_factor


def compute_R_over_P_from_model(model, a, omega, y, lambda_, u=None, v=None,
                                m=2, M=1.0, s=-2):
    """Compute R_in / P from network output f_R(y).

    Args:
        model: AutoencoderTeukolskyPINN
        a: (B, 1) spin parameter
        omega: (B, 1) orbital frequency
        y: (B, N) compactified radial coordinate
        lambda_: (B, 1) angular eigenvalue
        u, v: optional (B, 1) local chart coordinates
        m, M, s: physical parameters

    Returns:
        R_over_P: (B, N) complex tensor = R_in(r) / P(r)
    """
    f_R = model.predict_Rin(a, omega, y, u, v)  # (B, N) complex
    c_H = horizon_regularity_slope(a, omega, lambda_, m=m, M=M, s=s)
    S = compose_reduced_shape_from_f(f_R, y, c_H)
    h2 = h_factor(a, omega, m=m, M=M, s=s)
    return h2 * S
