"""计算 Kerr 黑洞引力波散射振幅比"""
import sys
import os
import math
import warnings

# 添加 kerr_matcher 路径
_KERR_MATCHER_PATH = "/home/ljq/code/radial_flow/spec_flow_method_Kerr/kerr_matcher_project/src"
if _KERR_MATCHER_PATH not in sys.path:
    sys.path.insert(0, _KERR_MATCHER_PATH)

from .compute_lambda import compute_lambda

def compute_amplitude_ratio(a, omega, l, m, lambda_sep=None, r_match=8.0, n_cheb=32, s=-2):
    """
    计算Kerr黑洞引力波散射振幅比

    参数:
        a: 自旋参数 (0 <= a < M, M=1)
        omega: 频率
        l: 角量子数 (l >= 2)
        m: 方位角量子数 (|m| <= l)
        lambda_sep: 球旋分离常数 (None则自动计算)
        r_match: 匹配半径
        n_cheb: Chebyshev多项式阶数
        s: 自旋权重 (默认 -2)

    返回:
        dict: {
            'ratio': complex,  # B_inc/B_ref (入射/反射)
            'lambda': float,  # 分离常数
            'ratio_abs': float,  # |ratio|
            'ratio_arg': float   # arg(ratio)
        }
    """
    # 如果未提供 lambda_sep，则自动计算
    if lambda_sep is None:
        lambda_sep = compute_lambda(a, omega, l, m, s)

    try:
        from kerr_matcher.params import SolverParams
        from kerr_matcher.solver import solve_case
    except ImportError as e:
        raise ImportError(f"无法导入 kerr_matcher 模块: {e}")

    params = SolverParams(
        M=1.0, a=a, omega=omega, ell=l, m_mode=m,
        lambda_sep=lambda_sep, r_match=r_match, n_cheb=n_cheb, flow_eps=1e-6
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        try:
            result = solve_case(params)
        except RuntimeWarning as exc:
            raise FloatingPointError(
                f"kerr_matcher flow failed for a={a}, omega={omega}, l={l}, m={m}: {exc}"
            ) from exc

    # ratio_up_over_um = R_+/R_- = 反射/入射
    # 所以 B_inc/B_ref = 1/ratio_up_over_um
    ratio_ref_over_inc = result.spectral.ratio_up_over_um
    if not (math.isfinite(ratio_ref_over_inc.real) and math.isfinite(ratio_ref_over_inc.imag)):
        raise FloatingPointError(
            f"kerr_matcher returned non-finite reflection/incidence ratio for "
            f"a={a}, omega={omega}, l={l}, m={m}"
        )
    if abs(ratio_ref_over_inc) == 0.0:
        raise FloatingPointError(
            f"kerr_matcher returned zero reflection/incidence ratio for "
            f"a={a}, omega={omega}, l={l}, m={m}"
        )
    ratio_inc_over_ref = 1.0 / ratio_ref_over_inc
    if not (math.isfinite(ratio_inc_over_ref.real) and math.isfinite(ratio_inc_over_ref.imag)):
        raise FloatingPointError(
            f"kerr_matcher returned non-finite incidence/reflection ratio for "
            f"a={a}, omega={omega}, l={l}, m={m}"
        )

    return {
        'ratio': ratio_inc_over_ref,
        'lambda': params.lambda_value,
        'ratio_abs': abs(ratio_inc_over_ref),
        'ratio_arg': ratio_inc_over_ref.real / abs(ratio_inc_over_ref) if abs(ratio_inc_over_ref) > 0 else 0
    }
