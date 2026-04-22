"""计算角向分离常数 λ 和振幅比值的接口函数"""
import sys
import os

# 添加 GremLinEqRe 路径
_GREMLINEQRE_PATH = "/home/ljq/code/Teukolsky_based/GremLinEqRe"
if _GREMLINEQRE_PATH not in sys.path:
    sys.path.insert(0, _GREMLINEQRE_PATH)

def compute_lambda(a, omega, l, m, s=-2):
    """
    计算角向分离常数 λ

    参数:
        a: 黑洞自旋参数 (0 <= a < 1)
        omega: 频率 (M*ω)
        l: 角量子数 l (l >= |s|)
        m: 角量子数 m (|m| <= l)
        s: 自旋权重 (默认 -2，用于引力波)

    返回:
        lambda_val: 角向分离常数
    """
    try:
        from GremLinEqRe import _core
    except ImportError as e:
        raise ImportError(f"无法导入 GremLinEqRe._core 模块: {e}\n请确保已编译 GremLinEqRe")

    swsh = _core.SWSH(s, l, m, a * omega)
    return swsh.m_lambda

# 使用示例
if __name__ == "__main__":
    a = 0.5
    omega = 1.0
    l = 2
    m = 1
    s = -2
    lambda_val = compute_lambda(a, omega, l, m, s)
    print(f"角向分离常数 λ = {lambda_val}")