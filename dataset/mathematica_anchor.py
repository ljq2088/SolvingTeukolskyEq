"""
Mathematica数据锚点：用于PINN训练的监督信号
"""
import torch
import numpy as np
from wolframclient.evaluation import WolframLanguageSession
from wolframclient.language import wlexpr

KERNEL_PATH = r"/mnt/f/mma/WolframKernel.exe"
WL_PATH_WIN = r"F:/EMRI/Radial_flow/Radial_Function.wl"

# 全局session，避免每次调用都启动新窗口
_global_session = None

def _get_session():
    """获取或创建全局Mathematica session"""
    global _global_session
    if _global_session is None:
        _global_session = WolframLanguageSession(kernel=KERNEL_PATH)
        _global_session.evaluate(wlexpr(rf'Get["{WL_PATH_WIN}"]'))
    return _global_session

def close_mathematica_session():
    """关闭全局session"""
    global _global_session
    if _global_session is not None:
        _global_session.terminate()
        _global_session = None


def get_mathematica_Rin(a, omega, l, m, s, r_points):
    """
    从Mathematica获取指定 r 点上的 R_in(r)

    Args:
        a, omega, l, m, s: 物理参数
        r_points: numpy array, 需要采样的 r 坐标点（顺序会被保留）

    Returns:
        R_mma: numpy array (complex), 与 r_points 一一对应的 R_in(r)
    """
    session = _get_session()

    r_points = np.asarray(r_points, dtype=float).reshape(-1)

    # 安全检查：验证参数合法性，防止Mathematica卡死
    if len(r_points) == 0:
        raise ValueError("r_points is empty")
    if len(r_points) > 1000:
        raise ValueError(f"Too many points ({len(r_points)}), max 1000 to prevent Mathematica hang")
    if not np.all(np.isfinite(r_points)):
        raise ValueError(f"r_points contains non-finite values: {r_points}")

    # 检查物理参数合法性
    if not (0 <= abs(a) < 1.0):
        raise ValueError(f"Invalid spin parameter a = {a}, must be in [0, 1)")
    if not np.isfinite(omega):
        raise ValueError(f"Invalid omega = {omega}")

    # 计算视界半径
    M = 1.0
    r_plus = M + np.sqrt(M**2 - a**2)

    # 检查r必须 >= r_+ (允许0.1%误差)
    if np.any(r_points < r_plus * 0.999):
        raise ValueError(f"r_points contains values below horizon r_+ = {r_plus:.6f}: min(r) = {np.min(r_points):.6f}")

    # 构造成 Mathematica 列表 {r1, r2, ..., rn}
    rlist_str = ", ".join(f"{float(r):.16g}" for r in r_points)

    expr = (
        f"SampleRinAtPoints[{s}, {l}, {m}, "
        f"{a:.16g}, {omega:.16g}, "
        f"{{{rlist_str}}}]"
    )

    result = session.evaluate(wlexpr(expr))
    arr = np.array(result, dtype=complex)

    if arr.ndim != 2 or arr.shape[1] < 3:
        raise ValueError(f"Unexpected shape from Mathematica: {arr.shape}")

    # 第一列是 Mathematica 实际求值时使用的 r，做一次一致性检查
    r_eval = arr[:, 0]
    if len(r_eval) != len(r_points):
        raise ValueError(
            f"Length mismatch: Mathematica returned {len(r_eval)} points, "
            f"but Python requested {len(r_points)} points."
        )

    max_diff = np.max(np.abs(r_eval - r_points))
    if max_diff > 1e-10:
        raise ValueError(
            f"r_points mismatch between Python and Mathematica, max diff = {max_diff:.3e}"
        )

    R_mma = arr[:, 1] + 1j * arr[:, 2]
    return R_mma


def sample_anchor_points_gaussian_clusters(
    n_clusters=2,
    n_points_per_cluster=10,
    sigma=0.1,
    device='cpu',
    dtype=torch.float64
):
    """
    高斯聚类采样：随机选择中心点，在其周围高斯采样

    Args:
        n_clusters: 聚类中心数量
        n_points_per_cluster: 每个聚类的点数
        sigma: 高斯分布标准差

    Returns:
        y_anchors: (n_clusters * n_points_per_cluster,)

    IMPORTANT: y必须在(-0.99, 0.99)内，避免x=0或x=1
    """
    y_anchors_list = []

    # 安全边界
    y_safe_min = -0.98
    y_safe_max = 0.98

    for _ in range(n_clusters):
        # 随机选择中心点 y ∈ [-0.8, 0.8]（避开边界）
        center = -0.8 + 1.6 * torch.rand(1, device=device, dtype=dtype)

        # 在中心周围高斯采样
        offsets = sigma * torch.randn(n_points_per_cluster, device=device, dtype=dtype)
        cluster_points = center + offsets

        # 截断到安全范围
        cluster_points = torch.clamp(cluster_points, min=y_safe_min, max=y_safe_max)

        y_anchors_list.append(cluster_points)

    y_anchors = torch.cat(y_anchors_list, dim=0)
    return y_anchors
