# PINN超参数配置说明

## 物理参数 (problem)

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `M` | 黑洞质量 | 1.0 |
| `s` | 自旋权重 | -2 |
| `l` | 角量子数 | 2 |
| `m` | 磁量子数 | 2 |
| `lambda.mode` | λ计算模式：`compute`或`fixed` | compute |
| `ramp.mode` | R_amp计算模式：`compute`或`fixed` | compute |

## 参数范围

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `a_center` | 黑洞自旋参数中心值 | 0.1 |
| `a_range` | a的变化范围（±） | 0.01 |
| `omega_center` | 频率中心值 | 0.1 |
| `omega_range` | ω的变化范围（±） | 0.01 |

**说明**：训练时参数在 [center-range, center+range] 范围内随机采样

## 网络结构

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `hidden_dims` | 隐藏层维度列表 | [64, 64, 64, 64] |
| `activation` | 激活函数：`tanh`, `relu`, `gelu`, `silu` | tanh |

**说明**：
- 输入：3维 (a, ω, y)
- 输出：2维 (Re[R'], Im[R'])
- 可调整层数和每层神经元数

## 采样配置

| 参数 | 说明 | 默认值 | 建议范围 |
|------|------|--------|----------|
| `n_interior` | 内点数量 | 50 | 50-200 |
| `n_boundary` | 每侧边界点数量 | 10 | 10-50 |
| `batch_size` | 参数batch大小 | 2 | 1-8 |

**说明**：
- 内点：在y∈[-1,1]均匀随机采样
- 边界点：在y≈±1附近密集采样（boundary_layer_width=0.1）

## 批梯度下降配置

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `use_batch_gd` | 是否使用批梯度下降 | true |
| `n_param_samples` | 参数空间总采样数 | 20 |
| `n_epochs` | epoch数量 | 100 |

**模式对比**：
- `use_batch_gd=false`：随机采样模式，每步随机采样新参数
- `use_batch_gd=true`：批梯度下降，固定参数集，按epoch遍历

## 训练配置

| 参数 | 说明 | 默认值 | 建议范围 |
|------|------|--------|----------|
| `n_steps` | 训练步数（随机采样模式） | 1000 | 1000-50000 |
| `lr` | 学习率 | 0.001 | 1e-4 ~ 1e-2 |
| `weight_interior` | 内点loss权重 | 1.0 | 0.1-10 |
| `weight_boundary` | 边界loss权重 | 10.0 | 1-100 |
| `weight_anchor` | 锚点loss权重 | 1.0 | 0.1-10 |

**说明**：
- 总loss = weight_interior × loss_interior + weight_boundary × loss_boundary + weight_anchor × loss_anchor
- 边界权重较大：强制边界条件R'≈0

## 验证配置

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `val_freq` | 每N步/epochs进行一次验证 | 50 |
| `val_n_points` | 验证时r网格点数 | 200 |
| `save_best_freq` | 每N步/epochs保存最佳模型 | 50 |

**说明**：
- 验证在全r网格上计算PDE残差
- 保存最佳val_loss的模型到pinn_best.pt
- 随机采样模式：以步数为单位
- 批梯度下降模式：以epoch为单位

## 数据锚点配置

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `n_anchors` | 每次采样的锚点数量 | 10 |
| `anchor_freq` | 每N步使用一次锚点 | 100 |

**锚点采样策略**：
- 60%在边界层：y∈[-1+0.01, -1+0.15] 和 [0.85, 0.99]
- 40%在内部：y∈[-0.85, 0.85]
- 避开正好的边界点y=±1

## 早停配置

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `early_stop_patience` | 容忍步数/epochs | 500 |
| `early_stop_threshold` | loss改善阈值 | 1.0e-6 |

**说明**：
- 随机采样模式：每100步检查一次
- 批梯度下降模式：每epoch检查一次
- 连续patience次无改善则触发早停

## 调参建议

### 快速测试
```yaml
n_interior: 50
n_boundary: 10
batch_size: 2
n_param_samples: 10
n_epochs: 50
```

### 高精度训练
```yaml
n_interior: 200
n_boundary: 50
batch_size: 4
n_param_samples: 50
n_epochs: 500
weight_boundary: 50.0
```

### 边界敏感问题
```yaml
weight_boundary: 100.0
n_boundary: 50
boundary_layer_width: 0.05
```

### 参数空间覆盖
```yaml
a_range: 0.05  # 扩大范围
omega_range: 0.05
n_param_samples: 100  # 增加采样
```
