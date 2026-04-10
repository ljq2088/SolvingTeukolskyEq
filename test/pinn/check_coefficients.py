"""
检查ODE系数B2, B1, B0在y=[-1,1]的值
"""
import sys
sys.path.append("/home/ljq/code/PINN/SolvingTeukolsky")
import torch
import numpy as np
import yaml
from physical_ansatz.residual import AuxCache, get_lambda_from_cfg, get_ramp_and_p_from_cfg
from physical_ansatz.teukolsky_coeffs import coeffs_x

# 加载配置
with open('config/pinn_config.yaml', 'r') as f:
    cfg = yaml.safe_load(f)

# 参数
a = cfg['a_center']
omega = cfg['omega_center']
M = float(cfg["problem"].get("M", 1.0))
s = int(cfg["problem"].get("s", -2))
m = int(cfg["problem"].get("m", 2))

print(f"参数: a={a}, ω={omega}, l=2, m={m}, s={s}\n")

# 转换为torch
device = torch.device('cpu')
dtype = torch.float64
a_t = torch.tensor(a, dtype=dtype, device=device)
omega_t = torch.tensor(omega, dtype=dtype, device=device)

# 计算lambda和ramp
cache = AuxCache()
lambda_val = get_lambda_from_cfg(cfg, cache, a_t, omega_t)
p, ramp_val = get_ramp_and_p_from_cfg(cfg, cache, a_t, omega_t)

# y网格
y_points = np.linspace(-1, 1, 21)
print(f"{'y':<8} {'x':<10} {'B2 (real)':<15} {'B2 (imag)':<15} {'B1 (real)':<15} {'B1 (imag)':<15} {'B0 (real)':<15} {'B0 (imag)':<15}")
print("=" * 120)

for y in y_points:
    x = (y + 1.0) / 2.0
    x_t = torch.tensor(x, dtype=dtype, device=device)

    # 计算A2, A1, A0
    A2, A1, A0 = coeffs_x(
        x_t, a_t, omega_t, m, p, ramp_val, lambda_val, s, M
    )

    # 转换为B系数（y坐标）
    # dx/dy = 1/2, R'_x = 2*R'_y, R'_xx = 4*R'_yy
    B2 = A2 * 4.0
    B1 = A1 * 2.0
    B0 = A0

    # 转换为复数（如果是实数）
    if not torch.is_complex(B2):
        B2 = B2.to(torch.complex128)
    if not torch.is_complex(B1):
        B1 = B1.to(torch.complex128)
    if not torch.is_complex(B0):
        B0 = B0.to(torch.complex128)

    print(f"{y:<8.2f} {x:<10.4f} {B2.real.item():<15.6e} {B2.imag.item():<15.6e} "
          f"{B1.real.item():<15.6e} {B1.imag.item():<15.6e} "
          f"{B0.real.item():<15.6e} {B0.imag.item():<15.6e}")

print("\n注: ODE方程为 B2*R''_yy + B1*R'_y + B0*R' = 0")
