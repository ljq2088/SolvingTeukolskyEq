"""
测试PINN模型预测（不依赖Mathematica）
"""
import sys
sys.path.append("/home/ljq/code/PINN/SolvingTeukolsky")
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib.pyplot as plt
from model.pinn_mlp import PINN_MLP
from physical_ansatz.prefactor import U_factor
from physical_ansatz.mapping import r_plus
from physical_ansatz.residual import AuxCache, get_ramp_and_p_from_cfg
import yaml

def g_func(x):
    """g(x) = exp(x-1) - 1"""
    return torch.exp(x - 1.0) - 1.0

def pinn_f_to_R(f_pred, a, omega, r_grid, cfg):
    """将PINN的f(y)转换为R(r)"""
    device = torch.device('cpu')
    dtype = torch.float64
    M = float(cfg["problem"].get("M", 1.0))

    # 转换为torch
    a_t = torch.tensor(a, dtype=dtype, device=device)
    omega_t = torch.tensor(omega, dtype=dtype, device=device)
    r_t = torch.tensor(r_grid, dtype=dtype, device=device)

    # 计算 r_+ 和 x
    rp = r_plus(a_t, M)
    x_t = rp / r_t  # x = r_+/r

    # 计算 g(x)
    g_t = g_func(x_t)

    # 计算 R' = g(x)*f(y) + 1
    Rprime_t = g_t * f_pred + 1.0

    # 获取 U(r) = Q(r) * I(r)
    cache = AuxCache()
    p, ramp_t = get_ramp_and_p_from_cfg(cfg, cache, a_t, omega_t)
    m = int(cfg["problem"].get("m", 2))
    s = int(cfg["problem"].get("s", -2))

    U_t = U_factor(r_t, a_t, omega_t, p, ramp_t, m, s, M)

    # R = U * R'
    R_t = U_t * Rprime_t

    return R_t.detach().cpu().numpy(), Rprime_t.detach().cpu().numpy()

# 加载模型
checkpoint = torch.load('outputs/pinn/pinn_model.pt')
cfg = checkpoint['cfg']

# 重建模型
hidden_dims = cfg.get('hidden_dims', [64, 64, 64, 64])
model = PINN_MLP(hidden_dims=hidden_dims, activation='tanh')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
model = model.to(dtype=torch.float64)

print("模型加载成功")
print(f"参数范围: a={cfg['a_center']}±{cfg['a_range']}, ω={cfg['omega_center']}±{cfg['omega_range']}")

# 测试参数
a_test = cfg['a_center']
omega_test = cfg['omega_center']
l_test = int(cfg['problem'].get('l', 2))
m_test = int(cfg['problem'].get('m', 2))
s_test = int(cfg['problem'].get('s', -2))
M = float(cfg['problem'].get('M', 1.0))

print(f"\n测试参数: a={a_test}, ω={omega_test}, l={l_test}, m={m_test}, s={s_test}")

# 定义r网格
a_torch = torch.tensor(a_test, dtype=torch.float64)
rp_val = float(r_plus(a_torch, M).item())
r_min = rp_val + 0.1
r_max = 100.0
n_points = 200

# 在x坐标上均匀采样
x_grid = np.linspace(rp_val/r_max, rp_val/r_min, n_points)
r_grid = rp_val / x_grid
y_grid = 2.0 * x_grid - 1.0

print(f"r范围: [{r_grid.min():.3f}, {r_grid.max():.3f}]")

# PINN预测
print("\n计算PINN预测...")
a_batch = torch.tensor([a_test], dtype=torch.float64)
omega_batch = torch.tensor([omega_test], dtype=torch.float64)
y_torch = torch.tensor(y_grid, dtype=torch.float64)

with torch.no_grad():
    f_pred = model(a_batch, omega_batch, y_torch)[0]

# 转换为R(r)和R'(x)
R_pinn, Rprime_pinn = pinn_f_to_R(f_pred, a_test, omega_test, r_grid, cfg)

print(f"PINN预测:")
print(f"  |f|范围 = [{np.abs(f_pred.numpy()).min():.4e}, {np.abs(f_pred.numpy()).max():.4e}]")
print(f"  |R'|范围 = [{np.abs(Rprime_pinn).min():.4e}, {np.abs(Rprime_pinn).max():.4e}]")
print(f"  |R|范围 = [{np.abs(R_pinn).min():.4e}, {np.abs(R_pinn).max():.4e}]")

# 绘图
print("\n生成可视化...")
fig = plt.figure(figsize=(16, 10))

# 图1: f(y) - PINN内部函数
ax1 = plt.subplot(3, 3, 1)
ax1.plot(y_grid, f_pred.real.numpy(), 'b-', lw=2, label='Re[f]')
ax1.plot(y_grid, f_pred.imag.numpy(), 'r-', lw=2, label='Im[f]')
ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
ax1.axvline(x=-1, color='gray', linestyle='--', alpha=0.3)
ax1.axvline(x=1, color='gray', linestyle='--', alpha=0.3)
ax1.set_xlabel('y')
ax1.set_ylabel('f(y)')
ax1.set_title('PINN Internal Function f(y)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 图2: |f(y)|
ax2 = plt.subplot(3, 3, 2)
ax2.plot(y_grid, np.abs(f_pred.numpy()), 'g-', lw=2)
ax2.set_xlabel('y')
ax2.set_ylabel('|f(y)|')
ax2.set_title('Magnitude of f(y)')
ax2.grid(True, alpha=0.3)

# 图3: R'(r) 实部
ax3 = plt.subplot(3, 3, 3)
ax3.plot(r_grid, Rprime_pinn.real, 'b-', lw=2)
ax3.set_xlabel('r / M')
ax3.set_ylabel("Re[R'(r)]")
ax3.set_title("Regular Part R' = g(x)f(y) + 1")
ax3.grid(True, alpha=0.3)

# 图4: R'(r) 虚部
ax4 = plt.subplot(3, 3, 4)
ax4.plot(r_grid, Rprime_pinn.imag, 'r-', lw=2)
ax4.set_xlabel('r / M')
ax4.set_ylabel("Im[R'(r)]")
ax4.set_title("Imaginary Part of R'")
ax4.grid(True, alpha=0.3)

# 图5: |R'(r)|
ax5 = plt.subplot(3, 3, 5)
ax5.semilogy(r_grid, np.abs(Rprime_pinn), 'g-', lw=2)
ax5.set_xlabel('r / M')
ax5.set_ylabel("|R'(r)|")
ax5.set_title("Magnitude of R' (log scale)")
ax5.grid(True, alpha=0.3, which='both')

# 图6: R(r) 实部
ax6 = plt.subplot(3, 3, 6)
ax6.plot(r_grid, R_pinn.real, 'b-', lw=2)
ax6.set_xlabel('r / M')
ax6.set_ylabel('Re[R(r)]')
ax6.set_title('Full Solution R = U·R\'')
ax6.grid(True, alpha=0.3)

# 图7: R(r) 虚部
ax7 = plt.subplot(3, 3, 7)
ax7.plot(r_grid, R_pinn.imag, 'r-', lw=2)
ax7.set_xlabel('r / M')
ax7.set_ylabel('Im[R(r)]')
ax7.set_title('Imaginary Part of R')
ax7.grid(True, alpha=0.3)

# 图8: |R(r)| 线性
ax8 = plt.subplot(3, 3, 8)
ax8.plot(r_grid, np.abs(R_pinn), 'g-', lw=2)
ax8.set_xlabel('r / M')
ax8.set_ylabel('|R(r)|')
ax8.set_title('Magnitude of R (linear)')
ax8.grid(True, alpha=0.3)

# 图9: |R(r)| 对数
ax9 = plt.subplot(3, 3, 9)
ax9.semilogy(r_grid, np.abs(R_pinn), 'g-', lw=2)
ax9.set_xlabel('r / M')
ax9.set_ylabel('|R(r)|')
ax9.set_title('Magnitude of R (log scale)')
ax9.grid(True, alpha=0.3, which='both')

plt.suptitle(f'PINN Prediction: a={a_test}, ω={omega_test}, l={l_test}, m={m_test}, s={s_test}', fontsize=14)
plt.tight_layout()
plt.savefig('outputs/pinn/pinn_full_prediction.png', dpi=150)
print("完整预测图已保存到: outputs/pinn/pinn_full_prediction.png")

# 边界条件检查
print(f"\n边界条件检查:")
print(f"  f(-1) = {f_pred[0]:.6e}")
print(f"  f(+1) = {f_pred[-1]:.6e}")
print(f"  |f(-1)| = {abs(f_pred[0]):.6e}")
print(f"  |f(+1)| = {abs(f_pred[-1]):.6e}")

# 保存数据
np.savez(
    'outputs/pinn/pinn_prediction_data.npz',
    r=r_grid,
    y=y_grid,
    f=f_pred.numpy(),
    Rprime=Rprime_pinn,
    R=R_pinn,
    a=a_test,
    omega=omega_test,
    l=l_test,
    m=m_test,
    s=s_test,
)
print("\n数值数据已保存到: outputs/pinn/pinn_prediction_data.npz")
print("\n提示: 如需与Mathematica对比，请运行: python test/test_pinn_vs_mathematica.py")
