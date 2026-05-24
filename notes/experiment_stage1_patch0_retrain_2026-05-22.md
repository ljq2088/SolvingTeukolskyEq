# Stage-1 Patch 0 干净重训练实验记录

**日期**: 2026-05-22  
**分支**: `feature/autoencoder-asymptotic-decoder2` (commit `fc661a3`)  
**目标**: 在 patch 0 (log-ω, a∈[0.206,0.794], ω∈[0.0085,0.0777]) 重训 Stage-1 autoencoder 达到 v2 水平精度 (val<0.02)

## 基线

- **模型**: AutoencoderTeukolskyPINN (4×128, SiLU, Fourier 2, FiLM, residual)
- **Loss**: 纯 PDE residual (B2·f_yy + B1·f_y + B0·f - RHS)
- **配点**: 128 点固定配点
- **参数**: 48 train / 24 val, Sobol 采样
- **硬件**: 单 GPU (RTX 3090), bfloat16

## 实验矩阵

| # | 热启动 | 配点策略 | Anchor | Robin | LR | Grad Clip | 结果 |
|---|--------|---------|--------|-------|-----|-----------|------|
| 1 | 冷启动 | chebyshev | 无 | 无 | 5e-5 | 0.5 | 爆炸 @ step ~18k, PDE 4.5→6764 |
| 2 | retrain best (val=8.27) | chebyshev_horizon | 无 | 无 | 5e-5 | 0.5 | 爆炸 @ step ~18k, PDE 3305↔9524 |
| 3 | 冷启动 | chebyshev_horizon | 有 (weight=1e-4) | 无 | 5e-5 | 0.5 | anchor loss ~500k-1M 主导, grad_norm=9000 |

## 关键发现

1. **纯 PDE 在低 ω patch 0 无法收敛** — 无论冷/热启动、无论配点策略，均在 step ~18k 附近发生梯度爆炸（grad_norm > 100k）。Teukolsky 方程在 ω→0 时趋近静态极限，视界附近系数急剧变化，loss landscape 极端崎岖。

2. **chebyshev_horizon 配点无显著改善** — 将配点聚集在 y=1（视界）侧不能解决根本的不稳定性问题。

3. **Anchor loss 尺度不匹配** — Mathematica 参考 f 值 |f|~128-2473，导致 anchor MSE ~500k-1M，需要极小的 weight 才能不失衡，但此时 anchor 约束过弱。

4. **Phase 4 pinn_mlp 在 patch 0 有优秀结果** — 找到多个 val<0.02 的 checkpoint:
   - `phase4_sgdr`: val=**0.00375** (step=26000)
   - `phase4_integral`: val=**0.01074** (step=15000)
   - `phase4_joint_hybrid`: val=**0.074** (step=20000)

## 结论

纯 PDE loss 策略在低 ω 区域不可行。Phase 4 的 pinn_mlp 模型通过**积分一致性约束 + Robin + 复杂 loss 权重平衡**成功稳定训练。下一步应考虑:

- **方案 A**: 从 Phase 4 pinn_mlp checkpoint 权重迁移到 autoencoder encoder (结构完全一致，有 `copy_pinn_mlp_to_autoencoder()` 函数)
- **方案 B**: 在 Stage-1 重训中加入额外约束（积分一致性、弱 anchor、或 Robin）
- **方案 C**: 提升 LR 到原先 v2 成功用的 1e-4，配合更强的正则化组合
