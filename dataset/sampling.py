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

    IMPORTANT: y范围必须严格在(-1, 1)内，避免x=0或x=1导致r无穷大或r=r_+
    """
    # 安全边界：避开±1，防止x=0或x=1
    y_safe_min = -0.99
    y_safe_max = 0.99

    # 边界点：在 y=-1 和 y=1 附近密集采样，但不能到达±1
    y_left_bd = y_safe_min + boundary_layer_width * torch.rand(
        n_boundary, device=device, dtype=dtype
    )
    y_right_bd = y_safe_max - boundary_layer_width * torch.rand(
        n_boundary, device=device, dtype=dtype
    )
    y_boundary = torch.cat([y_left_bd, y_right_bd], dim=0)


    # 内点：在安全区域均匀随机采样
    y_interior = y_safe_min + (y_safe_max - y_safe_min) * torch.rand(
        n_interior, device=device, dtype=dtype
    )

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
    - 边界点：接近但不到达 y=±1
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
    # 安全边界：避开±1
    y_safe_min = -0.99
    y_safe_max = 0.99

    # 边界点：接近±0.99但不到达
    y_left_exact = torch.full(
        (n_boundary_each,), y_safe_min, device=device, dtype=dtype
    )
    y_right_exact = torch.full(
        (n_boundary_each,), y_safe_max, device=device, dtype=dtype
    )
    y_boundary = torch.cat([y_left_exact, y_right_exact], dim=0)

    # 边界层点
    y_left_layer = y_safe_min + boundary_layer_width * torch.rand(
        n_boundary_layer_each, device=device, dtype=dtype
    )
    y_right_layer = y_safe_max - boundary_layer_width * torch.rand(
        n_boundary_layer_each, device=device, dtype=dtype
    )
    y_boundary_layer = torch.cat([y_left_layer, y_right_layer], dim=0)

    # 内点
    y_interior = y_safe_min + (y_safe_max - y_safe_min) * torch.rand(
        n_interior, device=device, dtype=dtype
    )

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
        y_min, y_max: y坐标范围（必须在(-0.99, 0.99)内避开边界）

    Returns:
        y_anchors: (n_anchors,)
    """
    # 确保范围安全
    y_min = max(y_min, -0.98)
    y_max = min(y_max, 0.98)

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
    """
    构建候选点池，范围严格在(-0.99, 0.99)内
    """
    if method == "sobol":
        engine = torch.quasirandom.SobolEngine(dimension=1, scramble=True, seed=seed)
        # 映射到(-0.99, 0.99)而不是(-1, 1)
        y = 1.98 * engine.draw(n_points).squeeze(-1).to(device=device, dtype=dtype) - 0.99
    elif method == "chebyshev":
        if n_points <= 1:
            y = torch.tensor([0.0], device=device, dtype=dtype)
        else:
            k = torch.arange(n_points, device=device, dtype=dtype)
            y_raw = torch.cos(torch.pi * k / (n_points - 1))   # Lobatto
            y = 0.99 * y_raw
    else:
        y = -0.99 + 1.98 * torch.rand(n_points, device=device, dtype=dtype)

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


def sample_points_uniform_grid(
    n_points,
    y_min=-0.99,
    y_max=0.99,
    device='cpu',
    dtype=torch.float64,
    shuffle=False,
):
    if n_points <= 1:
        y = torch.tensor([(y_min + y_max) * 0.5], device=device, dtype=dtype)
    else:
        y = torch.linspace(y_min, y_max, n_points, device=device, dtype=dtype)
    if shuffle:
        y = y[torch.randperm(y.numel(), device=device)]
    return y


def sample_points_random_uniform(
    n_points,
    y_min=-0.99,
    y_max=0.99,
    device='cpu',
    dtype=torch.float64,
):
    return y_min + (y_max - y_min) * torch.rand(n_points, device=device, dtype=dtype)


def sample_points_sobol_random(
    n_points,
    y_min=-0.99,
    y_max=0.99,
    device='cpu',
    dtype=torch.float64,
    seed=2026,
):
    engine = torch.quasirandom.SobolEngine(dimension=1, scramble=True, seed=seed)
    y = engine.draw(n_points).squeeze(-1).to(device=device, dtype=dtype)
    return y_min + (y_max - y_min) * y


def sample_points_chebyshev_grid(
    n_points,
    y_min=-0.9999,
    y_max=0.9999,
    device='cpu',
    dtype=torch.float64,
):
    """
    安全版 Chebyshev-Lobatto 采点。
    特点：
    - 在 y_min / y_max 附近更密
    - 不触碰真正的 ±1，只在安全区间 [y_min, y_max] 内
    - 默认返回升序点列
    """
    if n_points <= 0:
        return torch.empty(0, device=device, dtype=dtype)

    if n_points == 1:
        return torch.tensor(
            [0.5 * (y_min + y_max)],
            device=device,
            dtype=dtype,
        )

    k = torch.arange(n_points, device=device, dtype=dtype)
    # Lobatto nodes in [-1, 1]
    y_ref = torch.cos(torch.pi * k / (n_points - 1))

    # 映射到 [y_min, y_max]
    y = 0.5 * (y_max - y_min) * y_ref + 0.5 * (y_max + y_min)

    # 升序，便于后续处理
    return torch.sort(y).values


def sample_points_chebyshev_horizon(
    n_points,
    y_min=-0.9999,
    y_max=0.9999,
    device='cpu',
    dtype=torch.float64,
):
    """
    视界侧集中的 Chebyshev 采点：y = 2*cos(k*pi/(2N)) - 1, k=0,...,N-1.
    点聚集在 y=1（视界）附近，y=-1（无穷远）侧较稀疏.
    """
    if n_points <= 0:
        return torch.empty(0, device=device, dtype=dtype)
    if n_points == 1:
        return torch.tensor([0.5 * (y_min + y_max)], device=device, dtype=dtype)

    k = torch.arange(n_points, device=device, dtype=dtype)
    y_ref = 2.0 * torch.cos(torch.pi * (2.0 * n_points - k) / (2.0 * n_points)) + 1.0

    y = 0.5 * (y_max - y_min) * y_ref + 0.5 * (y_max + y_min)
    return torch.sort(y).values


def sample_points_normal_truncated(
    n_points: int,
    sigma: float = 0.3,
    y_min: float = -0.999,
    y_max: float = 0.999,
    device: str = 'cpu',
    dtype: torch.dtype = torch.float64,
    seed: int = None,
) -> torch.Tensor:
    """Sample from truncated normal centered at y=-1 (infinity endpoint).

    Concentrates points near y=-1 where the solution is hardest to fit.
    Distribution is N(loc=-1, scale=sigma) truncated to [y_min, y_max].

    Args:
        n_points: number of collocation points
        sigma: standard deviation; smaller = tighter concentration near y=-1
        y_min: left truncation bound (default -0.999, avoids exact y=-1)
        y_max: right truncation bound (default 0.999)
        device, dtype: output tensor properties
        seed: optional numpy random seed for reproducibility

    Returns:
        y: (n_points,) sorted tensor
    """
    from scipy.stats import truncnorm

    if seed is not None:
        np.random.seed(seed)

    loc = -1.0
    a = (y_min - loc) / sigma
    b = (y_max - loc) / sigma
    rng = truncnorm(a, b, loc=loc, scale=sigma)
    y_np = rng.rvs(size=n_points)
    y = torch.from_numpy(y_np).to(device=device, dtype=dtype)
    return torch.sort(y).values


def sample_points_residual_weighted(
    candidate_y: torch.Tensor,
    residual_profile: torch.Tensor,
    n_select: int,
    uniform_frac: float = 0.3,
    temperature: float = 1.0,
    device: str = 'cpu',
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Sample collocation points based on PDE residual distribution.

    Mixes residual-based sampling (focusing on hard regions) with uniform
    sampling (maintaining domain coverage), following FI-PINN approach.

    Args:
        candidate_y: (M,) dense evaluation grid
        residual_profile: (M,) |PDE_residual| on candidate_y
        n_select: number of points to select
        uniform_frac: fraction from uniform sampling (0.3 = 30%)
        temperature: softmax temperature; smaller = sharper focus on peaks
        device, dtype: output tensor properties

    Returns:
        y_selected: (n_select,) sorted tensor
    """
    n_select = min(n_select, candidate_y.numel())
    n_uniform = max(0, int(uniform_frac * n_select))
    n_residual = n_select - n_uniform

    # Pure uniform: no residual sampling needed
    if n_residual <= 0:
        perm = torch.randperm(candidate_y.numel(), device=device)
        y = candidate_y[perm[:n_select]]
        return torch.sort(y).values

    # Pure residual: sample all from residual distribution
    if n_uniform <= 0:
        profile = residual_profile.detach().clone().flatten().to(dtype=torch.float64)
        profile = torch.clamp(profile, min=0.0)
        if profile.sum() < 1e-30:
            profile = torch.ones_like(profile)
        logits = profile / max(temperature, 1e-8)
        probs = torch.softmax(logits, dim=0)
        probs = probs / probs.sum().clamp_min(1e-12)
        idx = torch.multinomial(probs, n_select, replacement=True)
        y = candidate_y[idx]
        return torch.sort(y).values

    # Mixed: residual + uniform

    # Residual-based sampling
    profile = residual_profile.detach().clone().flatten().to(dtype=torch.float64)
    profile = torch.clamp(profile, min=0.0)
    if profile.sum() < 1e-30:
        profile = torch.ones_like(profile)
    logits = profile / max(temperature, 1e-8)
    probs = torch.softmax(logits, dim=0)
    probs = probs / probs.sum().clamp_min(1e-12)
    idx_residual = torch.multinomial(probs, n_residual, replacement=True)

    # Uniform sampling
    idx_remain = torch.ones(candidate_y.numel(), dtype=torch.bool, device=device)
    idx_remain[idx_residual] = False
    available = torch.nonzero(idx_remain, as_tuple=False).squeeze(-1)
    if available.numel() >= n_uniform:
        perm = available[torch.randperm(available.numel(), device=device)]
        idx_uniform = perm[:n_uniform]
    else:
        idx_uniform = available

    idx = torch.cat([idx_residual, idx_uniform], dim=0)
    y = candidate_y[idx]
    return torch.sort(y).values


def sample_interior_points(
    strategy,
    n_points,
    device='cpu',
    dtype=torch.float64,
    article_cfg=None,
    boundary_layer_width=0.1,
):
    article_cfg = article_cfg or {}
    y_min = article_cfg.get("y_min", -0.9999)
    y_max = article_cfg.get("y_max", 0.9999)
    shuffle = article_cfg.get("shuffle", False)

    if strategy == "article_uniform":
        return sample_points_uniform_grid(
            n_points=n_points,
            y_min=y_min,
            y_max=y_max,
            device=device,
            dtype=dtype,
            shuffle=shuffle,
        )

    if strategy == "random_uniform":
        return sample_points_random_uniform(
            n_points=n_points,
            y_min=y_min,
            y_max=y_max,
            device=device,
            dtype=dtype,
        )

    if strategy == "sobol_random":
        return sample_points_sobol_random(
            n_points=n_points,
            y_min=y_min,
            y_max=y_max,
            device=device,
            dtype=dtype,
        )

    if strategy == "chebyshev":
        return sample_points_chebyshev_grid(
            n_points=n_points,
            y_min=y_min,
            y_max=y_max,
            device=device,
            dtype=dtype,
        )

    if strategy == "chebyshev_horizon":
        return sample_points_chebyshev_horizon(
            n_points=n_points,
            y_min=y_min,
            y_max=y_max,
            device=device,
            dtype=dtype,
        )

    if strategy == "luna":
        y_interior, _ = sample_points_luna_style(
            n_interior=n_points,
            n_boundary=0,
            boundary_layer_width=boundary_layer_width,
            device=device,
            dtype=dtype,
        )
        return y_interior

    raise ValueError(f"Unknown interior sampling strategy: {strategy}")