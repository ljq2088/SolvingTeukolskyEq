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
    """
    y_anchors_list = []

    for _ in range(n_clusters):
        # 随机选择中心点 y ∈ [-0.9, 0.9]（避开边界）
        center = -0.9 + 1.8 * torch.rand(1, device=device, dtype=dtype)

        # 在中心周围高斯采样
        offsets = sigma * torch.randn(n_points_per_cluster, device=device, dtype=dtype)
        cluster_points = center + offsets

        # 截断到 [-1, 1]
        # cluster_points = torch.clamp(cluster_points, -1.0, 1.0)
        cluster_points = torch.clamp(cluster_points, min=-0.99, max=0.99)

        y_anchors_list.append(cluster_points)

    y_anchors = torch.cat(y_anchors_list, dim=0)
    return y_anchors
