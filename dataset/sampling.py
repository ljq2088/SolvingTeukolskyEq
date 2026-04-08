"""
PINN采样策略
类似Luna，边界附近采点更密集
"""
import torch
import numpy as np


def sample_points_luna_style(
    n_interior,
    n_boundary,
    boundary_layer_width=0.1,
    device='cpu',
    dtype=torch.float64,
):
    """
    Luna风格采样：边界附近密集采样

    Args:
        n_interior: 内点数量
        n_boundary: 边界点数量（每侧）
        boundary_layer_width: 边界层宽度（相对于[-1,1]）
        device: torch device
        dtype: torch dtype

    Returns:
        y_interior: (n_interior,) 内点
        y_boundary: (2*n_boundary,) 边界点
    """
    # 边界点：在 y=-1 和 y=1 附近密集采样
    y_left_bd = -1.0 + boundary_layer_width * torch.rand(
        n_boundary, device=device, dtype=dtype
    )
    y_right_bd = 1.0 - boundary_layer_width * torch.rand(
        n_boundary, device=device, dtype=dtype
    )
    y_boundary = torch.cat([y_left_bd, y_right_bd], dim=0)
    

    # 内点：在整个区域均匀随机采样
    y_interior = -1.0 + 2.0 * torch.rand(n_interior, device=device, dtype=dtype)

    return y_interior, y_boundary


def sample_points_adaptive(
    n_interior,
    n_boundary_each,
    n_boundary_layer_each,
    boundary_layer_width=0.05,
    device='cpu',
    dtype=torch.float64,
):
    """
    自适应采样：
    - 边界点：精确在 y=±1
    - 边界层点：在 y=±1 附近
    - 内点：在整个区域

    Args:
        n_interior: 内点数量
        n_boundary_each: 每侧边界点数量
        n_boundary_layer_each: 每侧边界层点数量
        boundary_layer_width: 边界层宽度
        device, dtype

    Returns:
        y_interior: (n_interior,)
        y_boundary: (2*n_boundary_each,)
        y_boundary_layer: (2*n_boundary_layer_each,)
    """
    # 精确边界点
    y_left_exact = torch.full(
        (n_boundary_each,), -1.0, device=device, dtype=dtype
    )
    y_right_exact = torch.full(
        (n_boundary_each,), 1.0, device=device, dtype=dtype
    )
    y_boundary = torch.cat([y_left_exact, y_right_exact], dim=0)

    # 边界层点
    y_left_layer = -1.0 + boundary_layer_width * torch.rand(
        n_boundary_layer_each, device=device, dtype=dtype
    )
    y_right_layer = 1.0 - boundary_layer_width * torch.rand(
        n_boundary_layer_each, device=device, dtype=dtype
    )
    y_boundary_layer = torch.cat([y_left_layer, y_right_layer], dim=0)

    # 内点
    y_interior = -1.0 + 2.0 * torch.rand(n_interior, device=device, dtype=dtype)

    return y_interior, y_boundary, y_boundary_layer


def sample_parameters(
    batch_size,
    a_center=0.1,
    a_range=0.01,
    omega_center=0.1,
    omega_range=0.01,
    device='cpu',
    dtype=torch.float64,
):
    """
    采样参数 (a, ω)

    Args:
        batch_size: batch大小
        a_center: a的中心值
        a_range: a的变化范围（±）
        omega_center: ω的中心值
        omega_range: ω的变化范围（±）

    Returns:
        a_batch: (batch_size,)
        omega_batch: (batch_size,)
    """
    a_batch = a_center + (2 * torch.rand(batch_size, device=device, dtype=dtype) - 1) * a_range
    omega_batch = omega_center + (2 * torch.rand(batch_size, device=device, dtype=dtype) - 1) * omega_range

    return a_batch, omega_batch


def sample_anchor_points(n_anchors, y_min=-0.8, y_max=0.8, device='cpu', dtype=torch.float64):
    """
    采样锚点（避免边界）

    Args:
        n_anchors: 锚点数量
        y_min, y_max: y坐标范围（避开±1边界）

    Returns:
        y_anchors: (n_anchors,)
    """
    y = y_min + (y_max - y_min) * torch.rand(n_anchors, device=device, dtype=dtype)
    return y


def sample_parameters_sobol(
    batch_size,
    a_center=0.1,
    a_range=0.01,
    omega_center=0.1,
    omega_range=0.01,
    device='cpu',
    dtype=torch.float64,
    seed=1234,
    skip=0,
):
    engine = torch.quasirandom.SobolEngine(dimension=2, scramble=True, seed=seed)
    if skip > 0:
        engine.fast_forward(skip)

    u = engine.draw(batch_size).to(device=device, dtype=dtype)  # (B, 2) in [0,1]
    a_batch = a_center + (2.0 * u[:, 0] - 1.0) * a_range
    omega_batch = omega_center + (2.0 * u[:, 1] - 1.0) * omega_range
    return a_batch, omega_batch


def build_candidate_pool_1d(
    n_points,
    method="sobol",
    device='cpu',
    dtype=torch.float64,
    seed=2026,
):
    if method == "sobol":
        engine = torch.quasirandom.SobolEngine(dimension=1, scramble=True, seed=seed)
        y = 2.0 * engine.draw(n_points).squeeze(-1).to(device=device, dtype=dtype) - 1.0
    elif method == "chebyshev":
        k = torch.arange(n_points, device=device, dtype=dtype)
        y = torch.cos(torch.pi * (2.0 * k + 1.0) / (2.0 * n_points))
    else:
        y = -1.0 + 2.0 * torch.rand(n_points, device=device, dtype=dtype)

    return torch.sort(y).values


def sample_points_rard(
    candidate_y,
    residual_score,
    n_select,
    adaptive_frac=0.7,
):
    device = candidate_y.device
    n_select = min(n_select, candidate_y.numel())
    n_adapt = max(1, int(adaptive_frac * n_select))
    n_uniform = max(0, n_select - n_adapt)

    score = residual_score.detach().clone().flatten()
    score = torch.clamp(score, min=0.0)
    if torch.all(score <= 0):
        score = torch.ones_like(score)

    probs = score / score.sum().clamp_min(1e-12)
    idx_adapt = torch.multinomial(probs, n_adapt, replacement=False)

    mask = torch.ones(candidate_y.numel(), dtype=torch.bool, device=device)
    mask[idx_adapt] = False
    remain = torch.nonzero(mask, as_tuple=False).squeeze(-1)

    if n_uniform > 0 and remain.numel() > 0:
        perm = remain[torch.randperm(remain.numel(), device=device)]
        idx_uniform = perm[:n_uniform]
        idx = torch.cat([idx_adapt, idx_uniform], dim=0)
    else:
        idx = idx_adapt

    y = candidate_y[idx]
    return y[torch.randperm(y.numel(), device=device)]


def get_sentinel_anchor_points(device='cpu', dtype=torch.float64):
    return torch.tensor(
        [-0.95, -0.75, -0.45, -0.15, 0.15, 0.45, 0.75, 0.95],
        device=device,
        dtype=dtype,
    )