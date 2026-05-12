# 重构大纲：机器精度 Teukolsky 求解器

## 目标
- 相对误差：< 1e-12（逼近 float64 机器精度 ~1e-16）
- 推理速度：单点 < 1μs（Chebyshev 多项式求值），批量更快
- 覆盖范围：ω ∈ [1e-4, 1.0], a ∈ [0, 0.999]
- 不依赖 anchor loss
- 最高优先级：patch 5,6,7（高 ω），其次 2,8（极端 a）

## 当前瓶颈
- MLP 精度天花板在 1e-4~1e-8，加大网络几乎无效（Hazy Research 2025 已证明）
- Fourier features 只有 2 个频率，不足以捕获高 ω 振荡
- FiLM 调制对极端参数泛化能力不足

## 方案：Chebyshev 系数网络（Spectral-Neural Hybrid）

### 核心思路
放弃 MLP 在 y 方向的表达，改为 Chebyshev 多项式基函数：
```
f(y; a,ω) = Σ_{n=0}^{N-1} c_n(a,ω) * T_n(y)
```
- T_n(y) 是 Chebyshev 多项式（y ∈ [-1,1]）
- c_n(a,ω) 由参数网络输出（复数，N 个系数）
- 导数通过 Chebyshev 递推精确计算

### 为什么能达到机器精度
1. **谱收敛**：光滑函数的 Chebyshev 展开是指数收敛的，N=64 可达 1e-14
2. **无谱偏置**：Chebyshev 基是全局的，不存在 MLP 的低频优先问题
3. **优化**：系数网络只需学习参数→系数的映射，比 MLP 直接拟合函数容易得多
4. **BWLer**（Stanford 2025）用类似方法在 PINN 上达到 1e-12

### 推理速度
- N=64 个 Chebyshev 系数的多项式求值：~64 次乘加 → < 0.1μs
- 参数网络：一次前向传播，但只做一次（不随 y 点数变化）
- 总体：单点 < 1μs，1000 点 < 10μs

## 实施步骤

### Phase 1：新模型架构 `ChebCoeffNet`
- [ ] 实现 Chebyshev 多项式基函数模块（含递推求导）
- [ ] 实现系数网络：`(a,ω,u,v) → [c_0_re, c_0_im, ..., c_{N-1}_re, c_{N-1}_im]`
- [ ] 实现 PDE residual loss（用 Chebyshev 导数）
- [ ] 实现快速推理接口 `predict_f(a, omega, y)`
- [ ] 支持 chart_uv 局部坐标模式

### Phase 2：分片重组
- [ ] 分析当前 9 片在哪些区域不足
- [ ] 方案 A：保持 9 片，但针对 patch 5/6/7 增大 N
- [ ] 方案 B：减少到 4-6 片（新架构泛化能力更强），每片更大覆盖
- [ ] 方案 C：a=0 单独处理（λ = l(l+1)-s(s+1) 即 λ=2）

### Phase 3：优化策略
- [ ] Adam 预热 + L-BFGS 精调（二阶优化对谱方法关键）
- [ ] 损失加权：PDE 残差 + 边界条件（y=-1 和 y=1）
- [ ] Curriculum：先低 ω 后高 ω，先中心后边界

### Phase 4：训练 & 验证
- [ ] 对每个 patch 训练 ChebCoeffNet
- [ ] 用 GSN.jl 作为参考基准验证
- [ ] 和当前 PINN_MLP 结果对比
- [ ] 推理速度 benchmark（vs spectral, vs GSN）

### Phase 5：部署
- [ ] TorchScript 导出，确保纯推理无 Python 开销
- [ ] 实现 AtlasPredictor 适配器，兼容现有接口
