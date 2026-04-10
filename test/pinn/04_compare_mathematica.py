"""
测试PINN模型预测并与Mathematica对比
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
from physical_ansatz.mapping import r_plus, r_from_x
from physical_ansatz.residual import AuxCache, get_ramp_and_p_from_cfg
import yaml

from wolframclient.evaluation import WolframLanguageSession
from wolframclient.language import wlexpr

# Mathematica配置
KERNEL_PATH = r"/mnt/f/mma/WolframKernel.exe"
WL_PATH_WIN = r"F:/EMRI/Radial_flow/Radial_Function.wl"

def get_mathematica_solution(a, omega, l, m, s, r_grid):
    """从Mathematica获取真实解"""
    session = WolframLanguageSession(kernel=KERNEL_PATH)
    try:
        session.evaluate(wlexpr(rf'Get["{WL_PATH_WIN}"]'))

        npts = len(r_grid)
        rmin = r_grid[0]
        rmax = r_grid[-1]

        expr = (
            f"SampleRinOnGrid[{s}, {l}, {m}, "
            f"{a:.16g}, {omega:.16g}, "
            f"{rmin:.16g}, {rmax:.16g}, {npts}]"
        )
        result = session.evaluate(wlexpr(expr))
        arr = np.array(result, dtype=float)

        if arr.ndim != 2 or arr.shape[1] < 3:
            raise ValueError(f"Unexpected Mathematica output shape: {arr.shape}")

        r_mma = arr[:, 0]
        R_mma = arr[:, 1] + 1j * arr[:, 2]
        return r_mma, R_mma
    finally:
        session.terminate()

def pinn_Rprime_to_R(Rprime_pred, a, omega, r_grid, cfg):
    """将PINN的R'直接转换为R(r)"""
    device = torch.device('cpu')
    dtype = torch.float64
    M = float(cfg["problem"].get("M", 1.0))

    # 转换为torch
    a_t = torch.tensor(a, dtype=dtype, device=device)
    omega_t = torch.tensor(omega, dtype=dtype, device=device)
    r_t = torch.tensor(r_grid, dtype=dtype, device=device)

    # 获取 U(r) = Q(r) * Inf_prefactor(r)
    cache = AuxCache()
    p, ramp_t = get_ramp_and_p_from_cfg(cfg, cache, a_t, omega_t)
    m = int(cfg["problem"].get("m", 2))
    s = int(cfg["problem"].get("s", -2))

    U_t = U_factor(r_t, a_t, omega_t, p, ramp_t, m, s, M)

    # R = U * R'
    R_t = U_t * Rprime_pred

    return R_t.detach().cpu().numpy()


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

# 定义r网格（从视界附近到远场）
a_torch = torch.tensor(a_test, dtype=torch.float64)
rp_val = float(r_plus(a_torch, M).item())
r_min = rp_val + 0.1  # 视界外一点
r_max = 100.0
n_points = 200

# 在x坐标上均匀采样（靠近视界采点密）
x_grid = np.linspace(rp_val/r_max, rp_val/r_min, n_points)
r_grid = rp_val / x_grid

# 转换到y坐标
y_grid = 2.0 * x_grid - 1.0

print(f"r范围: [{r_grid.min():.3f}, {r_grid.max():.3f}]")

# ========== PINN预测 ==========
print("\n计算PINN预测...")
a_batch = torch.tensor([a_test], dtype=torch.float64)
omega_batch = torch.tensor([omega_test], dtype=torch.float64)
y_torch = torch.tensor(y_grid, dtype=torch.float64)

with torch.no_grad():
    Rprime_pred = model(a_batch, omega_batch, y_torch)[0]  # (N,)

# 转换为R(r)
R_pinn = pinn_Rprime_to_R(Rprime_pred, a_test, omega_test, r_grid, cfg)

print(f"PINN: |R|范围 = [{np.abs(R_pinn).min():.4e}, {np.abs(R_pinn).max():.4e}]")

# ========== Mathematica真实解 ==========
print("\n从Mathematica获取真实解...")
r_mma, R_mma = get_mathematica_solution(
    a_test, omega_test, l_test, m_test, s_test, r_grid
)

print(f"Mathematica: |R|范围 = [{np.abs(R_mma).min():.4e}, {np.abs(R_mma).max():.4e}]")

# ========== 计算误差 ==========
abs_error = np.abs(R_pinn - R_mma)
rel_error = abs_error / (np.abs(R_mma) + 1e-10)

print(f"\n误差统计:")
print(f"  绝对误差: mean={abs_error.mean():.4e}, max={abs_error.max():.4e}")
print(f"  相对误差: mean={rel_error.mean():.4e}, max={rel_error.max():.4e}")


# ========== 绘图对比 ==========
print("\n生成对比图...")

fig = plt.figure(figsize=(16, 10))

# 图1: R的实部对比
ax1 = plt.subplot(3, 2, 1)
ax1.plot(r_grid, R_mma.real, 'b-', lw=2, label='Mathematica', alpha=0.7)
ax1.plot(r_grid, R_pinn.real, 'r--', lw=2, label='PINN', alpha=0.7)
ax1.set_ylabel('Re[R(r)]')
ax1.set_title(f'Real Part: a={a_test}, ω={omega_test}')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 图2: R的虚部对比
ax2 = plt.subplot(3, 2, 2)
ax2.plot(r_grid, R_mma.imag, 'b-', lw=2, label='Mathematica', alpha=0.7)
ax2.plot(r_grid, R_pinn.imag, 'r--', lw=2, label='PINN', alpha=0.7)
ax2.set_ylabel('Im[R(r)]')
ax2.set_title('Imaginary Part')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 图3: |R|对比（对数坐标）
ax3 = plt.subplot(3, 2, 3)
ax3.semilogy(r_grid, np.abs(R_mma), 'b-', lw=2, label='Mathematica', alpha=0.7)
ax3.semilogy(r_grid, np.abs(R_pinn), 'r--', lw=2, label='PINN', alpha=0.7)
ax3.set_ylabel('|R(r)|')
ax3.set_title('Magnitude (log scale)')
ax3.legend()
ax3.grid(True, alpha=0.3, which='both')

# 图4: 绝对误差
ax4 = plt.subplot(3, 2, 4)
ax4.semilogy(r_grid, abs_error, 'g-', lw=2)
ax4.set_ylabel('|R_PINN - R_Mma|')
ax4.set_title('Absolute Error')
ax4.grid(True, alpha=0.3, which='both')

# 图5: 相对误差
ax5 = plt.subplot(3, 2, 5)
ax5.semilogy(r_grid, rel_error, 'm-', lw=2)
ax5.set_xlabel('r / M')
ax5.set_ylabel('|R_PINN - R_Mma| / |R_Mma|')
ax5.set_title('Relative Error')
ax5.grid(True, alpha=0.3, which='both')

# 图6: R'(y)的预测（PINN内部函数）
ax6 = plt.subplot(3, 2, 6)
ax6.plot(y_grid, Rprime_pred.real.numpy(), 'b-', lw=2, label="Re[R']")
ax6.plot(y_grid, Rprime_pred.imag.numpy(), 'r-', lw=2, label="Im[R']")
ax6.axhline(y=0, color='k', linestyle='--', alpha=0.3)
ax6.axvline(x=-1, color='gray', linestyle='--', alpha=0.3)
ax6.axvline(x=1, color='gray', linestyle='--', alpha=0.3)
ax6.set_xlabel('y')
ax6.set_ylabel("R'(y)")
ax6.set_title("PINN Internal Function R'(y)")
ax6.legend()
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/pinn/comparison_with_mathematica.png', dpi=150)
print("对比图已保存到: outputs/pinn/comparison_with_mathematica.png")

# ========== 边界条件检查 ==========
print(f"\n边界条件检查 (R'(y)):")
print(f"  R'(-1) = {Rprime_pred[0]:.6e}")
print(f"  R'(+1) = {Rprime_pred[-1]:.6e}")
print(f"  |R'(-1)| = {abs(Rprime_pred[0]):.6e}")
print(f"  |R'(+1)| = {abs(Rprime_pred[-1]):.6e}")

# ========== 保存数值结果 ==========
np.savez(
    'outputs/pinn/comparison_data.npz',
    r=r_grid,
    R_mma=R_mma,
    R_pinn=R_pinn,
    abs_error=abs_error,
    rel_error=rel_error,
    Rprime_pred=Rprime_pred.numpy(),
    y=y_grid,
)
print("\n数值数据已保存到: outputs/pinn/comparison_data.npz")

