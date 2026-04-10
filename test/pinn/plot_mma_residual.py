"""
绘制Mathematica有限差分计算的残差
"""
import numpy as np
import matplotlib.pyplot as plt

# 加载数据
data = np.load('outputs/mma_fd_residual/mma_fd_residual_data.npz')

y = data['y']
r = data['r']
residual_abs = data['residual_abs']
residual_real = data['residual_real']
residual_imag = data['residual_imag']

# 绘图
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 图1: 残差绝对值 vs y
ax1 = axes[0, 0]
ax1.semilogy(y, residual_abs, 'b-', lw=2)
ax1.set_xlabel('y')
ax1.set_ylabel('|Residual|')
ax1.set_title('Residual Magnitude vs y')
ax1.grid(True, alpha=0.3)
ax1.axvline(x=-1, color='r', linestyle='--', alpha=0.5, label='Boundary')
ax1.axvline(x=1, color='r', linestyle='--', alpha=0.5)
ax1.legend()

# 图2: 残差绝对值 vs r
ax2 = axes[0, 1]
ax2.loglog(r, residual_abs, 'g-', lw=2)
ax2.set_xlabel('r / M')
ax2.set_ylabel('|Residual|')
ax2.set_title('Residual Magnitude vs r (log-log)')
ax2.grid(True, alpha=0.3, which='both')

# 图3: 残差实部和虚部 vs y
ax3 = axes[1, 0]
ax3.plot(y, residual_real, 'b-', lw=2, label='Real', alpha=0.7)
ax3.plot(y, residual_imag, 'r-', lw=2, label='Imag', alpha=0.7)
ax3.set_xlabel('y')
ax3.set_ylabel('Residual')
ax3.set_title('Residual Components vs y')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 图4: 统计信息
ax4 = axes[1, 1]
ax4.axis('off')
stats_text = f"""
Mathematica有限差分残差统计

数据点数: {len(y)}
y范围: [{y.min():.3f}, {y.max():.3f}]
r范围: [{r.min():.3f}, {r.max():.3f}]

残差统计:
  Mean: {residual_abs.mean():.6e}
  Max:  {residual_abs.max():.6e}
  Min:  {residual_abs.min():.6e}
  Std:  {residual_abs.std():.6e}

边界附近残差 (|y|>0.95):
  Mean: {residual_abs[np.abs(y)>0.95].mean():.6e}
  Max:  {residual_abs[np.abs(y)>0.95].max():.6e}

内部残差 (|y|<0.8):
  Mean: {residual_abs[np.abs(y)<0.8].mean():.6e}
  Max:  {residual_abs[np.abs(y)<0.8].max():.6e}
"""
ax4.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
         verticalalignment='center')

plt.tight_layout()
plt.savefig('outputs/pinn/mma_residual_analysis.png', dpi=150)
print("残差分析图已保存到: outputs/pinn/mma_residual_analysis.png")
