"""
查看PINN训练结果
"""
import torch
import matplotlib.pyplot as plt
import numpy as np

# 加载模型
checkpoint = torch.load('outputs/pinn/pinn_model.pt')
print("训练配置:")
print(f"  Steps: {len(checkpoint['step_history'])}")
print(f"  Final loss: {checkpoint['loss_history'][-1]:.6e}")

# 绘制loss曲线
plt.figure(figsize=(10, 5))
plt.semilogy(checkpoint['step_history'], checkpoint['loss_history'])
plt.xlabel('Step')
plt.ylabel('Loss')
plt.title('PINN Training Loss')
plt.grid(True, alpha=0.3)
plt.savefig('outputs/pinn/loss_analysis.png', dpi=150)
print("\nLoss曲线已保存到: outputs/pinn/loss_analysis.png")

# 显示loss统计
losses = np.array(checkpoint['loss_history'])
print(f"\nLoss统计:")
print(f"  最小值: {losses.min():.6e}")
print(f"  最大值: {losses.max():.6e}")
print(f"  最终值: {losses[-1]:.6e}")
print(f"  下降比例: {(1 - losses[-1]/losses[0])*100:.2f}%")
