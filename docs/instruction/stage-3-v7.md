当前分支：feature/autoencoder-asymptotic-decoder2。

现在暂停 Stage-3 继续训练。不要继续 nearinf 2000 epoch，也不要进入 Stage-4。

原因：
之前 Stage-3 vs spectral 对比使用了 a=0.1, omega=10.0，这是错误的。当前 patch 0 的参数池 omega 大约在 [8.5e-3, 7.8e-2]，omega=10 完全超出 patch 0 训练域。这个比较不能用于判断 Stage-3 decoder 是否失败。

当前最高优先级：
先校准谱方法在 patch 0 参数点上的正确性，包括：
1. patch 0 真实参数点选择；
2. 谱方法 z 分段位置和谱阶数收敛；
3. B_inc / B_ref 提取；
4. 与 Mathematica 或 pybhpt 的一致性；
5. 再用校准后的谱方法结果和 Stage-3 decoder 的 u_up/u_down 做比较；
6. 最后才决定是否继续训练。

------------------------------------------------------------
任务 1：确认 Git 状态
------------------------------------------------------------

运行：

git branch --show-current
git pull
git status
git log --oneline -5

确认当前分支是：

feature/autoencoder-asymptotic-decoder2

确认当前包含最近提交 f37c773 或更新。

------------------------------------------------------------
任务 2：明确 patch 0 的真实参数域和测试点
------------------------------------------------------------

不要再使用默认的 a=0.1, omega=10.0。

从以下文件读取 patch 0 真实参数池：

outputs/stage1_artifacts/patch_000_logw_v2/rpred_cache.npz

读取：

- a
- omega
- u
- v
- lambda_

输出：

- a_min, a_max
- omega_min, omega_max
- log10omega_min, log10omega_max
- u_min, u_max
- v_min, v_max
- lambda range

选择 5 个校准点：

1. patch center；
2. low omega edge；
3. high omega edge；
4. low a edge；
5. high a edge。

所有点必须来自 rpred_cache.npz 或 patch_cover_l2_m2_logw_v2.json 中实际覆盖的 safe points，不要手动猜。

保存为：

outputs/spectral_calibration/patch_000_logw_v2/patch0_calibration_points.json

每个点保存：

a, omega, u, v, lambda, source_index, label

------------------------------------------------------------
任务 3：修复 compare 脚本必须传入 u/v
------------------------------------------------------------

当前 scripts/compare_stage3_ubasis_with_spectral.py 调用：

model.predict_u_up(a_t, omega_t, y_t)
model.predict_u_down(a_t, omega_t, y_t)

这是不够的。Stage-3 使用的是 patch-local chart，必须传入 u/v：

model.predict_u_up(a_t, omega_t, y_t, u_t, v_t)
model.predict_u_down(a_t, omega_t, y_t, u_t, v_t)

请修改 compare 脚本：

1. 新增参数：
   --u
   --v
   --lambda
   或者：
   --point-json outputs/spectral_calibration/patch_000_logw_v2/patch0_calibration_points.json
   --point-label center

2. 如果给了 point-json，就从文件读取 a, omega, u, v, lambda；
3. 如果没有 u/v，脚本必须报 warning，不要静默 fallback；
4. 默认值不要再用 omega=10.0；
5. 默认应该要求用户显式给 point-json 或 a/omega/u/v/lambda。

------------------------------------------------------------
任务 4：先校准谱方法收敛性
------------------------------------------------------------

新增脚本：

scripts/calibrate_spectral_patch0.py

目标：
对 patch 0 的 5 个 calibration points，系统检查谱方法收敛。

使用仓库已有：

utils.amplitude.solve_basis_domain
utils.amplitude.compute_smatrix_with_abel
utils.amplitude.TeukRadAmplitudeInWithAbelChecks

不要重写谱方法。

对每个参数点，扫描：

谱阶数：
N_out, N_in ∈ [24, 32, 48, 64, 80, 96]

匹配位置：
z_m ∈ [0.20, 0.30, 0.40, 0.50]

如果使用 basis plot 的 left patch / right patch，也扫描：
z1 ∈ [0.40, 0.50, 0.625]
z2 ∈ [0.70, 0.80, 0.90]

每组输出：

- B_inc
- B_ref
- B_trans
- outer_abel_residual
- inner_abel_residual
- detS_residual
- solve_in_relres
- solve_out_relres
- solve_in_cond_scaled
- solve_out_cond_scaled

收敛判据：

1. Abel residual 随 N 增大稳定下降或保持小；
2. B_inc, B_ref 随 N 和 z_m 变化稳定；
3. detS_residual 不大；
4. condition number 不爆炸；
5. 若 low omega 需要 mp backend，确认 use_mp_backend=true 并记录 mp_dps。

保存：

outputs/spectral_calibration/patch_000_logw_v2/
  spectral_convergence_summary.json
  spectral_convergence_summary.md
  spectral_convergence_cases.csv
  plots/
    B_ref_convergence_*.png
    abel_residual_*.png
    condition_number_*.png

不要训练模型。

------------------------------------------------------------
任务 5：和 Mathematica / pybhpt benchmark 对比
------------------------------------------------------------

先搜索现有 Mathematica 调用脚本：

grep -R "WolframLanguageSession" -n .
grep -R "SampleRinOnGrid" -n .
grep -R "TeukolskyRadial" -n .
grep -R "Amplitudes" -n .
grep -R "Radial_Function.wl" -n .

当前已知脚本包括：

mma/plot_rin_radial.py
mma/plot_divide_by_P.py
mma/plot_regular_part.py
mma/batch_scan_regular_part.py
mma/batch_scan_regular_part_quick.py

请不要依赖硬编码 a=0.1, omega=0.1 的旧脚本。新增统一脚本：

scripts/compare_spectral_with_mma_patch0.py

功能：

对 patch 0 calibration points：

1. 调用 Mathematica/Wolfram 采样 R_in(r)；
2. 使用与 Python 谱方法相同的 r_* 和 A_up/A_down convention；
3. 在大 r 区域拟合：

R_in(r) ≈ B_inc * A_down(r) + B_ref * A_up(r)

注意顺序：
B_inc 对应 A_down
B_ref 对应 A_up

4. 比较 Python spectral 的 B_inc/B_ref 和 Mathematica 拟合出的 B_inc/B_ref；
5. 如果 pybhpt 已经能给 B_inc/B_ref，也一起比较 pybhpt；
6. 输出三者差异。

输出：

outputs/spectral_calibration/patch_000_logw_v2/
  mma_comparison_summary.json
  mma_comparison_summary.md
  mma_vs_spectral_cases.csv

如果本地 Mathematica kernel 不可用：
- 脚本应优雅失败；
- 报告 Mathematica unavailable；
- 继续使用 pybhpt 或 spectral Abel diagnostics；
- 不要让整个流程卡死。

------------------------------------------------------------
任务 6：校准 basis 函数本身
------------------------------------------------------------

新增脚本：

scripts/diagnose_spectral_basis_patch0.py

对每个 calibration point：

1. 用谱方法计算 u_down/u_up；
2. 检查边界：
   u_down(z=0)=1
   u_up(z=0)=1
   du_down/dz(0)=boundary_du_exact(down)
   du_up/dz(0)=boundary_du_exact(up)

3. 检查 basis residual：
   C2 u_zz + C1 u_z + C0 u
   或使用现有 coeffs_numeric；
4. 输出 residual norm；
5. 检查不同 N 下 u(z) 的收敛；
6. 保存 basis profiles：

outputs/spectral_calibration/patch_000_logw_v2/basis_profiles/
  point_center_N80.npz
  point_lowomega_N80.npz
  ...

每个 npz 包含：

z, y, u_down, u_up, uz_down, uz_up, a, omega, u, v, lambda

------------------------------------------------------------
任务 7：再做 Stage-3 decoder vs spectral 比较
------------------------------------------------------------

只有在任务 4-6 通过后，才重新比较 Stage-3 decoder。

修改后的命令类似：

python scripts/compare_stage3_ubasis_with_spectral.py \
  --stage3-checkpoint outputs/autoencoder_stage3_ubasis_nearinf/<run>/checkpoints/best_model.pt \
  --config config/autoencoder_stage3_ubasis_nearinf.yaml \
  --point-json outputs/spectral_calibration/patch_000_logw_v2/patch0_calibration_points.json \
  --point-label center \
  --spectral-profile outputs/spectral_calibration/patch_000_logw_v2/basis_profiles/point_center_N80.npz \
  --device cuda \
  --output-dir outputs/stage3_spectral_compare/patch_000_logw_v2_calibrated/center

如果还没有 nearinf 训练后的 checkpoint，可以先用当前 Stage-3 best：

outputs/autoencoder_stage3_ubasis_refine/20260515_222026_stage3_ubasis/checkpoints/best_model.pt

比较时必须：

- 传入 a, omega, u, v, lambda；
- 使用 patch 0 真实参数；
- 使用谱方法已校准 profile；
- 不再调用 omega=10；
- 不再使用默认 lambda。

输出每个 calibration point 的：

- median_rel_err_up
- median_rel_err_down
- max_rel_err_up
- max_rel_err_down
- median_abs_err_up/down
- boundary exact check
- 图像路径

保存总表：

outputs/stage3_spectral_compare/patch_000_logw_v2_calibrated/stage3_vs_spectral_summary.md

------------------------------------------------------------
任务 8：根据比较结果决定训练策略
------------------------------------------------------------

如果校准后发现当前 Stage-3 在 patch 0 点上已经接近谱方法：

- 不需要继续 Stage-3；
- 回到 Stage-4 前先修 checkpoint lineage 和 complex P 的问题。

如果当前 Stage-3 仍然偏差大：

执行 near-infinity-only 训练，但要先用真实 patch 0 参数点和正确 u/v/lambda 监控。

训练命令：

python scripts/train_autoencoder_stage3_ubasis.py \
  --config config/autoencoder_stage3_ubasis_nearinf.yaml \
  --checkpoint outputs/autoencoder_stage3_ubasis_refine/20260515_222026_stage3_ubasis/checkpoints/best_model.pt \
  --device cuda \
  --epochs 500 \
  --verbose

训练期间每 50 epoch 自动或手动运行：

scripts/compare_stage3_ubasis_with_spectral.py

至少监控 center、lowomega、highomega 三个点。

不要只看 PDE loss。
必须同时看：

- loss_up/loss_down；
- u_up/u_down vs spectral median_rel_err；
- max|u_up|；
- max|u_down|；
- boundary exact check；
- 是否 NaN/Inf。

如果 u_up spectral error 不下降，停止，不要继续 2000 epoch。

------------------------------------------------------------
任务 9：如果 PDE residual 下降但 spectral error 不下降
------------------------------------------------------------

如果出现：

loss_up 下降，
但 u_up vs spectral error 不下降，

说明 PDE residual 训练不够约束网络到正确物理解，或者参数/坐标输入仍错。

此时不要继续只靠 residual 训练。提出但先不要实现：

A. 加少量 spectral anchor loss：
   L = L_residual + epsilon * L_spectral_u
   只在少数校准点和 near-infinity 区域加入；
   epsilon 取 1e-3 到 1e-2。

B. 只训练 up_decoder：
   train_up=true
   train_down=false

C. 增加 near-infinity 采样权重：
   y in [-0.999, -0.95] 更密。

D. 检查 encoder local coordinates：
   确保 predict_u_up/down 传入 u/v；
   确保 u/v 来自同一个 logomega patch chart。

------------------------------------------------------------
任务 10：提交
------------------------------------------------------------

本轮先提交校准和诊断脚本，不提交大型 outputs：

git add scripts/calibrate_spectral_patch0.py \
        scripts/compare_spectral_with_mma_patch0.py \
        scripts/diagnose_spectral_basis_patch0.py \
        scripts/compare_stage3_ubasis_with_spectral.py

git commit -m "Calibrate spectral basis and patch-0 Stage-3 diagnostics"
git push

最终报告必须包含：

1. 当前 commit hash；
2. patch 0 calibration points；
3. spectral N/z_m 收敛表；
4. Abel residual / determinant residual；
5. B_inc/B_ref 收敛结果；
6. Mathematica 或 pybhpt 对比结果；
7. 谱方法 basis residual 与边界检查；
8. Stage-3 decoder vs spectral 的 patch 0 点误差；
9. 是否建议继续 nearinf 训练；
10. 如果建议训练，给出具体训练命令和监控点。