# Enhanced PINN Training - Phase 1 优化方案

## 目标
在不修改现有代码的前提下，通过配置文件调参提升模型精度。

## 优化策略（按优先级）

### Phase 1: 纯调参优化（无需改代码）
**配置文件**: `config/pinn_config_phase1.yaml`

1. **增大网络容量**
   - `hidden_dims: [256, 512, 512, 256]` (原: [128,128,128,128])
   - `param_embed_dim: 128` (原: 64)

2. **增强 Fourier features**
   - `fourier_num_freqs: 8` (原: 2)
   - `fourier_scale: 2.0` (原: 1.0)

3. **提高 anchor loss 权重**
   - `weight_anchor: 5.0e+2` (原: 1.0e+2)

4. **增加训练资源**
   - `steps: 20000` (原: 10000)
   - `batch_size: 16` (原: 8)
   - `n_interior: 128` (原: 64)

5. **启用 curriculum learning**
   - `curriculum.parameter.enabled: true`

6. **更保守的学习率**
   - `lr: 5.0e-4` (原: 1.0e-3)

### Phase 2: 代码增强（可选，需修改）
如果 Phase 1 效果不够，再考虑：
- Multi-scale Fourier features (需修改 FourierFeature1D)
- Attention mechanism (需添加新模块)
- Knowledge distillation from GSN (需修改 loss)

## 使用方法

### 测试 Phase 1（纯调参）
```bash
# 在 patch 0 上测试
python trainer/atlas_patch_trainer.py \
  --cfg config/pinn_config_phase1.yaml \
  --patch-id 0 \
  --device cuda

# 完整 benchmark
python benchmark/scripts/benchmark_9patch_vs_gsn_spectral.py \
  --registry outputs/atlas_multipatch_phase1/atlas_registry_phase1.json
```

### 回退到原版
```bash
# 使用原配置
python trainer/atlas_patch_trainer.py \
  --cfg config/pinn_config.yaml \
  --patch-id 0
```

## 预期效果
- Patch 0: 2.8% → < 1%
- Patch 5/6/7: 200-400% → < 50%（理想 < 10%）

## 风险评估
- **低风险**: 纯调参，不改代码逻辑
- **可回退**: 随时切换回原配置
- **资源消耗**: 训练时间 2x，显存 1.5x
