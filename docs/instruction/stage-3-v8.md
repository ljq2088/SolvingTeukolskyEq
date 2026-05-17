当前分支：feature/autoencoder-asymptotic-decoder2。

现在不要继续训练，也不要直接加入 spectral anchor。当前图中 Stage-3 decoder 与谱方法差异非常大，尤其 u_up。下一步必须先严格校验 u-equation 系数是否与谱方法 coeffs_numeric 完全一致。

核心问题：
Stage-3 训练使用 physical_ansatz/u_residual.py 中的 y-space u-equation residual；
谱方法使用 utils/amplitude.py 中的 z-space coeffs_numeric。
二者应满足：

y = 2z - 1

C2_y = 4 * B2_z
C1_y = 2 * B1_z
C0_y = B0_z

如果这个关系不成立，Stage-3 residual 方程就是错的；
如果成立，说明方程系数主推导没错，问题更可能是训练区间、decoder capacity、loss scaling 或缺少 anchor。

------------------------------------------------------------
任务 1：确认 git 状态
------------------------------------------------------------

运行：

git branch --show-current
git pull
git status
git log --oneline -5

确认当前分支为：

feature/autoencoder-asymptotic-decoder2

确认包含最近的 Stage-3 v7 commit 5e31bd5 或更新。

------------------------------------------------------------
任务 2：新增 u-equation 系数对照脚本
------------------------------------------------------------

新增脚本：

scripts/check_u_equation_coefficients_against_spectral.py

目标：
直接比较 physical_ansatz/u_residual.py 的 y-space 系数和 utils/amplitude.py 的 z-space 系数。

对 patch 0 calibration points：

outputs/spectral_calibration/patch_000_logw_v2/patch0_calibration_points.json

或者如果该文件不存在，就从：

outputs/stage1_artifacts/patch_000_logw_v2/rpred_cache.npz

重新选取 center、low_omega、high_omega、low_a、high_a 五个点。

对每个点，使用相同的：

a, omega, lambda, s=-2, ell=2, m=2, M=1

构造 KerrMode。

选择若干 z 点：

z in [1e-4, 1e-3, 1e-2, 0.05, 0.1, 0.3, 0.5]

但要避免 z=0 和 z=1 精确端点。

对 basis in ["up", "down"]：

1. 用 utils.amplitude.coeffs_numeric(z, mode, basis) 得到：
   B2_z, B1_z, B0_z

2. 用 physical_ansatz/u_residual.py 中新提取出的 helper 得到：
   C2_y, C1_y, C0_y

如果当前 u_residual.py 没有单独返回系数的函数，请先新增：

compute_u_equation_coefficients(a, omega, y, lambda_, basis, M=1.0, s=-2, m=2)

它只返回 C2, C1, C0，不需要 model，不需要 autograd。

3. 检查：

rel_err_C2 = |C2_y - 4*B2_z| / |4*B2_z|
rel_err_C1 = |C1_y - 2*B1_z| / |2*B1_z|
rel_err_C0 = |C0_y - B0_z| / |B0_z|

输出每个点、每个 basis、每个 z 的误差。

通过标准：
所有 rel_err 应该在 1e-10 到 1e-8 量级；靠近 z=1e-4 如果数值病态，可放宽到 1e-6，但不能是 O(1)。

保存：

outputs/spectral_calibration/patch_000_logw_v2/u_equation_coeff_check.json
outputs/spectral_calibration/patch_000_logw_v2/u_equation_coeff_check.md

------------------------------------------------------------
任务 3：检查 residual 是否能让谱方法解消失
------------------------------------------------------------

新增或扩展：

scripts/check_spectral_solution_in_u_residual.py

目标：
把谱方法解 u_up/u_down 代入 Stage-3 的 u-equation residual，检查 residual 是否接近 0。

对每个 calibration point：

1. 用 solve_basis_domain(mode, basis, N=80 or 96, z_a=0, z_b=z1, bc_side="left") 得到谱方法 u(z)；
2. 转换到 y=2z-1；
3. 用谱方法给出的 u、u_z、u_zz 或 Chebyshev D/D2 计算：
   - u_y = u_z / 2
   - u_yy = u_zz / 4
4. 用 physical_ansatz/u_residual.py 的 C2,C1,C0 计算：
   residual_y = C2*u_yy + C1*u_y + C0*u

注意：
这里不要用 PyTorch autograd，因为谱方法 u 不是模型输出。直接用谱方法的 Chebyshev 导数矩阵或已有 uz/D2。

报告：
- residual median abs；
- residual max abs；
- relative residual = |res| / (|C0*u| + |C1*u_y| + |C2*u_yy| + eps)；
- 按 z/y 画 residual 曲线。

通过标准：
谱方法解代入 Stage-3 系数后 residual 应该很小，至少 relative residual 不应 O(1)。

如果这个测试失败，说明 u-equation 系数或者坐标导数转换有误。

------------------------------------------------------------
任务 4：分区间重算 Stage-3 vs spectral 误差
------------------------------------------------------------

当前图把 left patch 画到 y≈0.25，也就是 z≈0.625。这不是纯 near-infinity 区域，且 u_up 在这个范围内会快速增长。

请修改或新增：

scripts/compare_stage3_ubasis_with_spectral.py

让它输出分区间误差：

intervals:
1. near_core:
   y in [-0.999, -0.95]

2. near_extended:
   y in [-0.999, -0.5]

3. train_nearinf:
   y in [-0.999, -0.001]

4. left_patch_full:
   y in [-0.999, 0.25]

分别输出：
- median_rel_err_up/down；
- max_rel_err_up/down；
- median_abs_err_up/down；
- max_abs_err_up/down。

注意：
如果 Stage-4 只用 near_core，那么 near_core 的误差比 left_patch_full 更重要。
不要再只用整段 left_patch_full 的 median_rel_err 判断训练成败。

------------------------------------------------------------
任务 5：检查 compare 脚本是否真正传入 u/v/lambda
------------------------------------------------------------

必须确认：

model.predict_u_up(a, omega, y, u, v)
model.predict_u_down(a, omega, y, u, v)

而不是：

model.predict_u_up(a, omega, y)

同时确认 lambda 来自 calibration point，而不是默认硬编码。

如果脚本没有 u/v/lambda，就必须报错，不允许 silent fallback。

------------------------------------------------------------
任务 6：根据系数检查结果做判断
------------------------------------------------------------

如果任务 2 和任务 3 失败：

不要训练。
不要 anchor。
先修方程系数。

重点检查：
- Delta_r 项符号；
- y_r/y_rr 转换；
- C1 系数是否应为 2*B1_z；
- C2 是否应为 4*B2_z；
- C0 是否和 B0_z 一致；
- lambda 是否一致；
- s=-2 是否固定一致。

如果任务 2 和任务 3 通过：

说明 Stage-3 residual 方程本身正确。
那么当前差异的主要原因不是方程系数，而是训练策略或模型表达能力。

进入任务 7。

------------------------------------------------------------
任务 7：如果方程正确，先不要立即训练，先给出三种方案排序
------------------------------------------------------------

如果系数和 residual 验证通过，请报告下面三个方案的优先级：

方案 A：加入少量 spectral anchor loss
- 在 3-5 个 calibration points；
- 只在 near_core 或 train_nearinf 区间；
- loss:
  L = L_residual + epsilon * L_anchor
- epsilon = 1e-3 或 1e-2；
- 仍然只训练 up/down decoder；
- 不动 encoder/Rin/AmplitudeNet。

方案 B：增加 decoder 表达能力
- 把 up_decoder/down_decoder 从单层 TaylorComplexDecoder 换成小型 MLP decoder；
- 例如每个 decoder 用 2 层 hidden_dim=64；
- 保持 encoder frozen；
- 这可能是必要的，因为当前 up/down decoder 只有少量参数，难以表示增长很快的 u_up。

方案 C：只训练 near-infinity 核心区
- y in [-0.999, -0.95] 或 [-0.999, -0.5]；
- 不追求 left_patch_full；
- 只服务 Stage-4 consistency。

在报告中说明：
如果 near_core 误差已经可接受，不需要拟合整个 left patch。
如果 near_core 误差也大，则优先方案 A，再考虑 B。

------------------------------------------------------------
任务 8：本轮不做长训练
------------------------------------------------------------

本轮只做校验和诊断，不跑 2000 epoch。

可以跑很短的测试，但不要做正式训练。

------------------------------------------------------------
任务 9：提交
------------------------------------------------------------

提交新增诊断脚本和必要 helper：

git add physical_ansatz/u_residual.py \
        scripts/check_u_equation_coefficients_against_spectral.py \
        scripts/check_spectral_solution_in_u_residual.py \
        scripts/compare_stage3_ubasis_with_spectral.py

git commit -m "Validate Stage-3 u-equation coefficients against spectral basis"
git push

大型 outputs 不提交。

最终报告必须包含：

1. 当前 commit hash；
2. C2_y vs 4B2_z 的最大误差；
3. C1_y vs 2B1_z 的最大误差；
4. C0_y vs B0_z 的最大误差；
5. 谱方法 u 代入 Stage-3 residual 的 relative residual；
6. 分区间 Stage-3 vs spectral 误差；
7. 是否确认方程系数正确；
8. 如果正确，推荐下一步是 spectral anchor、decoder capacity 还是缩小训练区间。