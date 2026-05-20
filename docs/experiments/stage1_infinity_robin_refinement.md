# Stage-1 Infinity Robin Refinement 实验报告

**日期:** 2026-05-18 ~ 2026-05-19  
**分支:** `feature/autoencoder-asymptotic-decoder2`  
**目标:** 通过 analytic infinity Robin 条件约束，改善 Stage-1 PINN 模型在近无穷远 (y=-1) 处的渐近行为

## 1. 实验背景

patch 0（中心 patch, a∈[0.001,0.999], ω∈[1e-4,10]）训练后的 Stage-1 模型在 y=-1 处不满足正确的渐近 Robin 条件：

```
Sy(-1) = c_inf * S(-1)
```

其中 `c_inf` 是文献 `docs/instruction/inf_coeff.md` 给出的解析表达式：

```
c_inf = 1/2 + 1/r_+ - i*(8*r_+*omega^2 - 4*a*m*omega - lambda) / (4*omega*r_+)
```

## 2. 实验设计

按 `docs/instruction/stage1_refine.md` 执行两阶段 refinement：

**Phase 1 — Adam 短程 fine-tune:**
- 仅训练 encoder + rin_decoder（AmplitudeNet/UpDecoder/DownDecoder 冻结）
- 4 项 loss: PDE + infinity Robin (weight=1.0) + near-infinity PDE + drift
- lr_encoder=2e-7, lr_rin_decoder=1e-6

**Phase 2 — L-BFGS fixed-grid refinement:**
- 固定配点 grid（不重采样），strong_wolfe line search
- 每 10 epoch restart（重新采样并重建 optimizer）
- history_size=50, max_iter=20

**新增文件:**
| 文件 | 用途 |
|------|------|
| `physical_ansatz/stage1_endpoint.py` | 计算 S(-1), Sy(-1), c_inf 及 Robin residual |
| `config/autoencoder_stage1_infinity_robin.yaml` | refinement 配置 |
| `scripts/train_stage1_with_infinity_robin.py` | Adam + L-BFGS 训练脚本 |
| `scripts/debug_stage1_infinity_robin.py` | 18 项 debug 检查 |
| `scripts/plot_stage1_infinity_robin_effect.py` | before/after 对比可视化 |

## 3. 实验过程

### 3.1 Preflight 诊断

**Checkpoint:** `outputs/autoencoder_stage1_rin_train/20260518_194655_patch_000_comp_0_u_0.500_v_0.603/checkpoints/best_model.pt`

在 8 个诊断点上的 Robin 相对残差：

| 标签 | a | ω | Robin rel |
|------|---|---|-----------|
| difficult_point | 0.5127 | 0.0198 | 0.252 |
| low_a | 0.1 | 0.1 | 0.500 |
| center | 0.5 | 0.1 | 0.486 |
| high_a | 0.9 | 0.1 | 0.424 |
| low_omega | 0.5 | 0.02 | 0.247 |
| mid_omega | 0.5 | 0.5 | **0.921** |
| moderate | 0.7 | 0.05 | 0.229 |
| moderate_low | 0.3 | 0.03 | 0.131 |

- **中位数:** 0.338
- **最大值:** 0.921 (mid_omega: a=0.5, ω=0.5)

结论：ω 较高的点 Robin 残差最大，是主要优化目标。

### 3.2 Debug 检查

运行 `scripts/debug_stage1_infinity_robin.py`，全部检查通过（checkpoint 加载、c_inf finite、y=-1 forward pass、S(-1)/Sy(-1) finite、Robin/PDE/drift loss finite、freeze 策略正确、optimizer.step 只改 encoder+rin_decoder、L-BFGS closure 可重复、无 NaN/Inf）。

### 3.3 Adam 20-epoch Smoke

在 checkpoint `20260519_130434_adam_refine` 上进行 20-epoch Adam fine-tune：

| Epoch | 总 Loss | PDE Loss | Robin Loss | Drift | Grad Norm |
|-------|---------|----------|------------|-------|-----------|
| 0 | 385,888 | 385,888 | 0.678 | 0.0 | 1,584,906 |
| 19 | 473,377 | 473,377 | 0.531 | ~0.0 | 1,880,534 |

- Robin loss 从 0.678 降到 0.531 (~22% 改善)
- **但 PDE loss 从 385k 增长到 473k，梯度范数 1.88M —— 模型正在发散**

### 3.4 Adam 500-epoch（中止）

继续 500-epoch Adam，结果恶化：

| Epoch | 总 Loss | PDE Loss | Robin Loss | Grad Norm |
|-------|---------|----------|------------|-----------|
| 0 | 650,119 | 650,119 | 0.725 | 2,544,566 |
| 75 | 1,580,181 | 1,580,181 | 0.793 | 6,281,071 |

- PDE loss 从 650k 爆炸到 1.58M
- Robin loss 反而从 0.725 升到 0.793
- 梯度范数达 6.28M —— **完全失控**

用户决定切换到 L-BFGS。

### 3.5 Autograd 内存问题修复

在准备 L-BFGS 时遇到 CUDA OOM 和 CPU OOM。根本原因在 `physical_ansatz/residual_pinn.py` 的 `compute_f_derivatives_autograd`：

**旧代码问题：** 对每个 sample i 调用 `torch.autograd.grad(f[i].sum(), y_points)` 对完整 `y_points` (B×N) 求导，产生 (B,N) 梯度 → stack B 个得 (B,B,N) → 内存 O(B²N) 且 `retain_graph=True` 在 L-BFGS strong_wolfe line search 下反复保留图。

**修复：** 改为逐样本处理，每个 sample 独立做 forward + backward，`y_i` 维度为 (1,N) 而非 (B,N)。图保留大小降低 B 倍。

### 3.6 L-BFGS 5-epoch Smoke

使用修复后的 autograd，在 GPU 上成功运行 5-epoch L-BFGS：

| Epoch | 总 Loss | PDE Loss | Robin Loss | Drift | Grad Norm |
|-------|---------|----------|------------|-------|-----------|
| 0 | 763.8 | 763.5 | 0.199 | 0.923 | 45,024 |
| 4 | 64.3 | 64.0 | 0.177 | 0.866 | 3,362 |

Loss 从 764 降到 64 (11.9×)，Robin 略有改善。

### 3.7 L-BFGS 50-epoch Full Run

继续 50-epoch，restart_every=10：

| Epoch | 总 Loss | PDE Loss | Robin Loss | Drift | Grad Norm | 备注 |
|-------|---------|----------|------------|-------|-----------|------|
| 0 | 12.8 | 12.6 | 0.278 | 0.063 | 232 | 从 smoke best 继续 |
| 5 | 7.4 | 7.0 | 0.329 | 0.213 | 587 | |
| **9** | **7.24** | - | - | - | - | **最佳点** |
| 10 | 388.1 | 387.5 | 0.434 | 2.255 | 19,133 | restart 导致 spike |
| 15 | 102.9 | 102.5 | 0.335 | 0.833 | 3,884 | 恢复中 |
| 20 | 166.0 | 165.2 | 0.676 | 0.969 | 45,181 | restart, 停在原地 |
| 30 | 48.7 | 48.2 | 0.360 | 1.358 | 21,456 | |
| 40 | 54.7 | 53.8 | 0.795 | 1.175 | 6,747 | |
| 49 | 50.8 | 50.0 | 0.755 | 1.056 | 4,211 | 最终 |

**关键观察：**
- 每 10 epoch restart 重采样固定配点并重建 L-BFGS optimizer，导致 loss 剧烈震荡
- epoch 10: loss 从 7.4 飙升到 388（50×），随后缓慢恢复
- epoch 20: restart 后 L-BFGS 卡在原地（epoch 20 和 25 数据完全相同）
- Robin loss 最终（0.755）比开始（0.278）更差

## 4. 最终对比

在 5 个代表性 (a,ω) 点上对比原始模型与 L-BFGS 最佳模型：

### 4.1 Robin 相对残差

| a | ω | Original | L-BFGS best | 变化 |
|---|---|---------|------------|------|
| 0.10 | 0.20 | 0.716 | **0.998** | -39.4% ⬆ |
| 0.30 | 0.35 | 0.857 | 0.784 | +8.5% |
| 0.50 | 0.51 | 0.924 | 0.827 | +10.5% |
| 0.70 | 0.65 | 0.949 | 0.927 | +2.3% |
| 0.90 | 0.80 | 0.956 | 0.888 | +7.1% |

| 指标 | Original | L-BFGS |
|------|----------|--------|
| Robin 中位数 | 0.924 | 0.889 |
| Robin 最大值 | 0.956 | 0.998 |

**改善仅 4%（中位数），低 a 区域反而显著变差。**

### 4.2 S(-1) 和 Sy(-1) 幅值变化

| 指标 | Original | L-BFGS | 变化 |
|------|----------|--------|------|
| \|S(-1)\| 均值 | 42.4 | 1.45 | **-29×** |
| \|Sy(-1)\| 均值 | 501 | 2.25 | **-223×** |

模型学会了将 S 和 Sy 同时缩小，但 **未学到正确的比值关系**。Robin 相对残差 |Sy - c_inf*S| / (|Sy| + |c_inf*S|) 几乎不变。

### 4.3 对比图

图片保存在 `outputs/stage1_infinity_robin_compare/20260519_234711/plots/`：

| 图片 | 内容 |
|------|------|
| `re_im_S_comparison.png` | Re(S) 和 Im(S) 沿 y∈[-0.999, 0.999] 对比（5 个参数点） |
| `abs_arg_S_comparison.png` | \|S\| 和 arg(S) 对比 |
| `robin_residual_bar.png` | 各点 Robin 相对残差柱状图 |
| `Sy_vs_cS_scatter.png` | Sy(-1) vs c_inf*S(-1) 散点图 |
| `near_pde_residual.png` | y∈[-0.999, -0.95] 近无穷 PDE 残差 |
| `full_pde_residual.png` | 全 y 范围 PDE 残差 |

## 5. 失败原因分析

### 5.1 Robin loss 权重不足

Robin relative loss 在总 loss 中占比极小：

```
weight_inf_robin=1.0 时: Robin(~0.2) / PDE(~64~760) ≈ 0.03% ~ 0.3%
```

模型主要被 PDE loss 驱动，Robin 约束几乎不影响梯度方向。

### 5.2 模型学到"缩小"而非"校斜"

Robin absolute loss = |Sy - c_inf*S|²。模型的最简单降 loss 方式是把 |S| 和 |Sy| 同步压小，而非让 Sy/S 逼近 c_inf。L-BFGS 做到了前者（幅值降 29-223 倍），没做到后者（比值改善仅 4%）。

### 5.3 L-BFGS restart 机制不稳定

- 固定配点在 restart 时重采样，改变了 loss landscape
- L-BFGS 的 history_size=50 积累的历史梯度在 restart 后完全失效
- 每次 restart 后需要大量迭代才能恢复

### 5.4 Adam divergence 的原因

- lr_encoder=2e-7 看似很小，但 Robin loss 的梯度方向与 PDE loss 冲突
- PDE 残差在 near-infinity 区域本已很大（~10⁶~10⁸），加 Robin 约束后模型为满足 Robin 条件而调整 y=-1 附近的行为，破坏了内部 PDE 解

### 5.5 单点约束的局限性

Robin 条件仅在 y=-1 施加，但模型在 y=-1 的 f, fy 值依赖于整个编码器-解码器映射。单点约束无法充分传播到网络的参数更新中。

## 6. 教训与建议

1. **Robin loss 需大幅提权。** weight_inf_robin 从 1.0 提到 100~1000 才可能让 Robin 项占总 loss 的 10-50%。

2. **使用绝对（非相对）Robin loss。** 相对 loss 的分母 |Sy| + |c_inf*S| 在 S 缩小时也缩小，导致"压小 S 就能降 loss"的漏洞。绝对 loss = |Sy - c_inf*S|² 直接惩罚残差。

3. **或改用 Robin angle loss。** 定义 cos(θ) = |Sy · c_inf*S| / (|Sy|·|c_inf*S|)，惩罚 Sy 和 c_inf*S 的方向差异而非大小。

4. **启用 near-infinity PDE loss。** 当前 weight_near_pde=2.0 但配置中被跳过（loss_near_pde=0）。在 y∈[-0.999, -0.95] 区间加 PDE 约束可直接改善近无穷解质量。

5. **Adam 阶段需要更保守的 lr 或 grad clipping。** 当前 grad_clip=0.1 但梯度范数达 6M，几乎所有梯度被 clip 到相同大小，失去方向信息。

6. **L-BFGS 不应频繁 restart。** 建议 restart_every 设得更大（如 25），或不 restart 直接用更大 fixed_batch_size 覆盖参数空间。

7. **可考虑在 Stage-1 训练初期就加入 Robin loss。** 从 scratch 训练时用较小 weight（如 0.01），让模型从一开始就学习正确的渐近行为，而非事后纠偏。

## 7. 结论

**按 `stage1_refine.md` 执行的两阶段 refinement（Adam + L-BFGS）未达到改善 Infinity Robin 条件的目标。** Robin 相对残差中位数从 0.924 微降至 0.889（4% 改善），低 a 区域反而恶化。主要原因是 Robin loss 权重过低、模型学会通过缩小 S 幅值而非校斜比值来降低 loss，以及 L-BFGS restart 导致训练不稳定。

下一步应优先试验：大幅提权 Robin loss（100×）+ 启用 near-infinity PDE loss + 改进 loss 形式（绝对或 angle-based）。

## 8. 附录

### 基准 checkpoint
```
outputs/autoencoder_stage1_rin_train/20260518_194655_patch_000_comp_0_u_0.500_v_0.603/checkpoints/best_model.pt
```

### L-BFGS best checkpoint
```
outputs/stage1_infinity_robin/20260519_195041_adam_refine/checkpoints/best_model.pt
```

### 对比图目录
```
outputs/stage1_infinity_robin_compare/20260519_234711/
├── plots/
│   ├── re_im_S_comparison.png
│   ├── abs_arg_S_comparison.png
│   ├── robin_residual_bar.png
│   ├── Sy_vs_cS_scatter.png
│   ├── near_pde_residual.png
│   └── full_pde_residual.png
├── metrics.json
└── comparison_summary.md
```

### 所有实验 run 目录
```
outputs/stage1_infinity_robin/
├── 20260519_125259_adam_refine/    # 早期尝试
├── 20260519_125447_adam_refine/    # 早期尝试
├── 20260519_130200_adam_refine/    # 早期尝试
├── 20260519_130434_adam_refine/    # Adam 20-epoch (smoke, 有 best)
├── 20260519_185243_adam_refine/    # Adam 500-epoch (diverged, 有 best)
├── 20260519_185824_adam_refine/    # Adam 500-epoch continuation
├── 20260519_193039_adam_refine/    # L-BFGS 尝试 (OOM)
├── 20260519_193634_adam_refine/    # L-BFGS 尝试 (OOM)
├── 20260519_193820_adam_refine/    # L-BFGS 尝试 (graph freed)
├── 20260519_193917_adam_refine/    # L-BFGS 尝试 (CPU OOM)
├── 20260519_194528_adam_refine/    # L-BFGS 尝试 (graph error)
├── 20260519_194859_adam_refine/    # L-BFGS 5-epoch smoke (成功)
└── 20260519_195041_adam_refine/    # L-BFGS 50-epoch full run
```
