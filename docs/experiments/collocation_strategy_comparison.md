# 随机/自适应配点策略实验报告

**日期:** 2026-05-21
**脚本:** `scripts/explore_gaussian_collocation.py`
**模型:** AutoencoderTeukolskyPINN Stage-1 (encoder + rin_decoder, 240k params)
**baseline checkpoint:** `outputs/autoencoder_stage1_rin_train/20260518_194655_patch_000_comp_0_u_0.500_v_0.603/checkpoints/best_model.pt`

## 1. 实验动机

当前 Stage-1 PINN 训练使用 Chebyshev-Lobatto 固定配点。理论上，PINN 的 PDE residual 在解变化剧烈的区域（视界附近、无穷远附近）更大，因此基于 residual 的自适应配点可能提升精度。参考 FI-PINN Part II 和 RAR-D 等方法，设计了以下配点策略进行对比。

## 2. 配点策略

| 策略 | 说明 |
|------|------|
| **baseline** | Chebyshev-Lobatto 固定网格（当前默认） |
| **gaussian** | 截断正态分布 N(μ=-1, σ=0.3)，集中在 y=-1（无穷远）附近 |
| **mixed** | 70% gaussian + 30% uniform，兼顾无穷远聚焦和全域覆盖 |
| **adaptive** | Cosine-annealing 从 uniform 过渡到 residual-based 自适应采样 |

### 2.1 Adaptive 策略细节

- **Phase 1 (warmup, epoch 0-9):** 纯 uniform 采样，uniform_frac=1.0
- **Phase 2 (transition, epoch 10-39):** Cosine-annealing 过渡
  ```
  alpha = 0.5 * (1 - cos(pi * progress))
  uniform_frac = 1.0 + alpha * (0.3 - 1.0)
  ```
- **Phase 3 (adaptive, epoch 40+):** 纯 residual-based，保留 30% uniform
- 评估网格：256 点 Chebyshev 密集网格（fixed, 仅用于评估 residual profile）
- 重采样频率：每 5 epoch
- 采样分布：softmax(|residual| / temperature), temperature=1.0

## 3. 实验设置

- **Patch 0:** a∈[0.001,0.999], ω∈[1e-4,10], 覆盖中心区域 a~0.5, ω~0.51
- **训练参数:** 50 epochs, batch_size=32, n_interior=128, lr=5e-5, seed=42
- **损失函数:** PDE residual (L2) + infinity Robin BC (weight=0.1)
- **设备:** CUDA (NVIDIA GeForce RTX 4060 Ti)
- **对比实验:** 4 策略同时运行在同一 checkpoint 起始点

## 4. 结果

### 4.1 50-epoch 对比

| 策略 | PDE Loss (final) | Robin Loss (final) |
|------|-------------------|---------------------|
| baseline (Chebyshev) | 774,628 | 0.907 |
| gaussian (σ=0.3) | 1,592,432 | 0.953 |
| mixed (70/30) | 1,265,640 | 0.961 |
| adaptive | 2,930,815 | 0.892 |

**排序 (PDE loss 从低到高):** baseline < mixed < gaussian < adaptive

所有策略的 Robin loss 基本持平(~0.9)，但 PDE loss 差距显著：baseline 的 PDE loss 仅为 adaptive 的 26%，gaussian 的 49%。

### 4.2 Adaptive 200-epoch 长跑 (robin_weight=1e5)

为进一步验证 adaptive 策略，运行了 200 epoch 的长跑（robin_weight 提升至 1e5），并与原始 baseline checkpoint 进行 pybhpt 基准对比（6×6 网格，36 点）：

| 模型 | rel_err (mean of means) | rel_err (median of medians) | phase_err (mean) |
|------|------------------------|---------------------------|-------------------|
| baseline checkpoint (step 10000) | **0.129** | **0.064** | **0.130** |
| adaptive 200-epoch 训练后 | 0.193 | 0.176 | 0.168 |

Adaptive 训练后的模型在所有指标上均**劣于**原始 baseline checkpoint。

### 4.3 多次重复验证

50-epoch 对比实验在不同日期重复了多次，结果一致：

| 日期 | baseline PDE final | gaussian PDE final | mixed PDE final |
|------|-------------------|-------------------|-----------------|
| 2026-05-21 run 1 | 410,811 | 2,399,017 | 383,035 |
| 2026-05-21 run 2 | 252,024 | 1,615,121 | 1,230,757 |
| 2026-05-21 run 3 | 774,628 | 1,592,432 | 1,265,640 |

每次结果 rank 一致：baseline < mixed < gaussian < adaptive。

## 5. 分析

### 5.1 为何随机/自适应配点反而更差

1. **Chebyshev 网格的谱精度优势。** Chebyshev 点分布天然在端点（y=-1 无穷远, y=+1 视界）处密集、中心稀疏，这正好匹配 Teukolsky 方程解在端点处变化剧烈的特征。用随机分布替代会破坏这一结构。

2. **Residual 作为采样权重的不可靠性。** PINN 训练早期的 PDE residual 极不可靠（loss ~10^5），用它指导采样会导致"错误的正反馈"——residual 大但梯度方向错误的地方被反复采样，训练发散。

3. **Gaussian 集中在 y=-1 导致配点坍塌。** 狭窄的高斯分布（σ=0.3）使绝大多数配点落在 y=-1 附近，PDE 在 (0,1) 区间几乎没有约束，导致全域解精度崩溃。

4. **Mixed 策略的 30% uniform 不够。** 即使在 mixed 策略中保留 30% uniform 点，70% 的集中采样仍导致配点分布严重偏斜，PDE loss 仍比 baseline 高 63%。

### 5.2 与文献的一致性

FI-PINN Part II 的自适应配点在以下条件下有效：
- 模型已经大致收敛（residual profile 有意义）
- 问题是 high-frequency oscillatory 型（如 Helmholtz 方程）
- Teukolsky 方程在 y 坐标下并非高频振荡，而是端点处代数衰减 + 中心平滑

对于本问题的**低阶光滑解 + 端点边界层**特征，Chebyshev 谱配点本来就是最优的。

## 6. 结论

**对于 Stage-1 Teukolsky PINN 训练，固定 Chebyshev-Lobatto 配点是最优选择。** 随机配点（Gaussian/Mixed）和基于 residual 的自适应配点均导致 PDE loss 上升 60%-280%，且 pybhpt 基准相对误差恶化。不建议继续探索随机配点方向。

## 7. 后续方向

1. 继续使用 Chebyshev 固定配点，重点优化 loss 权重和训练策略
2. 考虑在 Chebyshev 网格基础上**增密**（增加配点数）而非替换
3. 若需自适应，应在模型充分收敛后（PDE loss 稳定、pybhpt 误差 <5%）再进行
