---
name: teu-recover
description: 当训练崩溃、loss爆炸、导入报错、结果异常或需要从失败状态恢复时触发。
---

# Teukolsky 任务失败恢复

## 1. 训练崩溃 / Loss 爆炸

```bash
# 查看最后日志
tail -100 outputs/pinn_train.log | grep -E "loss|Error|nan|inf"
```

恢复步骤：
1. 从最近 checkpoint 恢复
```python
ckpt = torch.load('saved_models/pinn_model.pt')
model.load_state_dict(ckpt['model_state_dict'])
optimizer.load_state_dict(ckpt['optimizer_state_dict'])
start_step = ckpt['step']
```
2. 降低学习率：`lr: 0.0001`（原来的1/10）
3. 增加梯度裁剪：`max_norm=0.1`

## 2. NaN / Inf 定位

```python
# 在 residual_from_nodes 前后加检查
assert not torch.isnan(f).any(), f"f has NaN at step {step}"
assert not torch.isinf(res).any(), f"residual has Inf"
```

常见原因：
- x 接近 0（r→∞）导致 r=r_+/x 溢出 → 检查 `mapping.py:r_from_x`
- Δ(r)=0 处除零 → 检查 `prefactor.py:delta`
- omega 为复数但接口只接受实数 → 见 `residual.py` 中的 ValueError

## 3. 导入错误

```bash
cd /home/ljq/code/PINN/SolvingTeukolsky
python -c "from physical_ansatz.residual import residual_from_nodes; print('OK')"
python -c "from dataset.grids import chebyshev_grid_bundle; print('OK')"
```

常见修复：
```bash
export PYTHONPATH=/home/ljq/code/PINN/SolvingTeukolsky:$PYTHONPATH
```

## 4. λ / R_amp 计算失败

```python
# 降级：改用 given 模式跳过外部求解器
# config/teukolsky_radial.yaml:
lambda:
  mode: given
  value: 4.0   # Schwarzschild 近似值

R_amp:
  mode: off    # 关闭反射项
```

## 5. 谱方法不收敛

检查系数衰减：
```python
coeffs = torch.load('outputs/single_case_coeffs.pt')
import matplotlib.pyplot as plt
plt.semilogy(abs(coeffs))  # 应指数衰减
```
若不衰减：增大 `spectral.order`（64→128）

## 6. 快速重置重跑

```bash
# 清除旧结果，重新训练
rm -f saved_models/pinn_model.pt outputs/pinn_train.log
python test/test_pinn_training.py
```

## 常见错误速查

| 错误 | 原因 | 修复 |
|------|------|------|
| `x must be > 0` | 采样点含 x=0 | 调整采样范围下界 |
| `only accepts real omega` | 复频传入 | 固定 omega.imag=0 |
| `lambda.value=null` | 配置未填 | 改 mode=compute 或填值 |
| Loss 不下降 | lr 太小或边界权重不足 | 调 lr/weight_boundary |
| CUDA OOM | batch 太大 | 减小 n_interior/batch_size |
