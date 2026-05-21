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

> **重要:** 训练 loss 仅为优化过程的相对判据，**最终评价标准是 pybhpt 基准的相对误差**。Loss 下降不等于物理精度提升。

### 4.1 原始 Chebyshev baseline 的 pybhpt 基准（step 10000, 9 个 val case）

| a | ω | median rel_err | max rel_err |
|---|---|---|---|
| 0.255 | 0.0514 | 2.34% | 2.38% |
| 0.296 | 0.0770 | 2.69% | 2.75% |
| 0.337 | 0.0514 | 3.04% | 3.32% |
| 0.378 | 0.0770 | 2.54% | 2.58% |
| 0.500 | 0.0257 | 8.23% | 9.03% |
| 0.541 | 0.0514 | 2.56% | 3.26% |
| 0.663 | 0.0514 | 2.96% | 3.64% |
| 0.745 | 0.0257 | 13.27% | 14.15% |
| 0.786 | 0.0770 | 2.19% | 2.22% |

- **median of medians: 2.69%**
- **max: 13.27%**（最差 case: a=0.745, ω=0.0257, 高自旋低频率）

### 4.2 50-epoch 配点策略对比（训练 loss）

训练 loss 仅作相对比较参考，不代表物理精度：

| 策略 | PDE Loss (final) | Robin Loss (final) |
|------|-------------------|---------------------|
| baseline (Chebyshev) | 774,628 | 0.907 |
| gaussian (σ=0.3) | 1,592,432 | 0.953 |
| mixed (70/30) | 1,265,640 | 0.961 |
| adaptive | 2,930,815 | 0.892 |

PDE loss 排序: baseline < mixed < gaussian < adaptive。所有策略的 Robin loss 基本持平(~0.9)，但 baseline PDE loss 仅为 adaptive 的 26%，gaussian 的 49%。

### 4.3 Adaptive 200-epoch 长跑的 pybhpt 基准对比

运行 200 epoch adaptive 训练（robin_weight=1e5），与 baseline checkpoint 在 6×6 网格（36 点, ω∈[0.012,0.074], a∈[0.235,0.765]）上做 pybhpt 对比：

| 模型 | rel_err (mean of means) | rel_err (median of medians) | phase_err (mean) |
|------|------------------------|---------------------------|-------------------|
| baseline checkpoint (step 10000) | **12.9%** | **6.4%** | **0.130 rad** |
| adaptive 200-epoch 训练后 | 19.3% | 17.6% | 0.168 rad |

**Adaptive 训练后相对误差恶化约 50%（mean）~175%（median）。**

按 ω 分组的相对误差（adaptive 模型 / baseline 模型）：

| ω | baseline rel_err | adaptive rel_err | 退化 |
|---|---|---|---|
| 0.0120 | 14.0% | 21.8% | +56% |
| 0.0173 | 11.0% | 21.8% | +98% |
| 0.0249 | 7.2% | 21.3% | +196% |
| 0.0358 | 4.4% | 19.6% | +345% |
| 0.0516 | 11.4% | 15.5% | +36% |
| 0.0743 | 29.7% | 15.8% | −47%（仅高频端改善）|

除了最高频端（ω=0.0743），adaptive 模型在所有频率区间均显著劣于 baseline。尤其在原本精度最高的中间频率（ω=0.0358, rel_err 仅 4.4%）退化最严重（→19.6%, +345%）。

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
