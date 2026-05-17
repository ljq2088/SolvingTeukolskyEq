当前分支：feature/autoencoder-asymptotic-decoder2。

最新 Stage-4 commit 2eb0fd5 已经加入 spectral-basis Stage-4 consistency polishing，并完成 20-epoch smoke。报告显示：
- L_cons 无明显下降；
- median_rel_err 仍约 1.6–2.8；
- drift 很小；
- debug 47/47 通过；
- 不建议继续 200 epoch；
- pre-stage4 bundle 已保留。

但是现在不要直接下结论说方案失败。需要先检查 Stage-4 loss 的定义是否正确。

我怀疑当前 Stage-4 代码把 model.predict_Rin() 当成了 R_model/P，但 AutoencoderTeukolskyPINN.predict_Rin() 返回的是 f_R(y)，不是 S(y)，也不是 R/P。

Stage-1 正确关系是：

R_in = P * h2 * S(y)

S(y) = compose_reduced_shape_from_f(f_R, y, c_H)

因此：

R_in / P = h2 * S(y)

当前任务是 Stage-4 v2 consistency audit。不要继续训练。

------------------------------------------------------------
任务 1：确认当前状态
------------------------------------------------------------

运行：

git branch --show-current
git pull
git status
git log --oneline -5

确认当前分支是：

feature/autoencoder-asymptotic-decoder2

确认包含最新 commit：

2eb0fd5 Add spectral-basis Stage-4 consistency polishing

------------------------------------------------------------
任务 2：审计 Stage-4 中 R_model_over_P 的定义
------------------------------------------------------------

检查：

scripts/train_autoencoder_stage4_spectral_consistency.py
scripts/debug_stage4_spectral_consistency.py
scripts/diagnose_stage4_spectral_consistency.py
scripts/diagnose_stage4_spectral_consistency.py 或同类脚本

查找所有：

model.predict_Rin(...)

如果它被直接命名为 R_over_P / R_model_over_P / R_stage1_over_P，则这是错误的。

正确流程必须是：

1. f_R = model.predict_Rin(a, omega, y, u, v)

2. 计算 horizon regularity slope：

c_H = horizon_regularity_slope(a, omega, lambda, m=2, M=1, s=-2)

3. 构造 reduced shape：

S = compose_reduced_shape_from_f(f_R, y, c_H)

4. 计算 h2：

h2 = h_factor(a, omega, m=2, M=1, s=-2)

5. 得到：

R_over_P = h2 * S

请新增 helper，例如：

compute_R_over_P_from_model(model, a, omega, y, u, v, lambda_, m=2, M=1, s=-2)

放在一个合适模块里，例如：

scripts/stage4_utils.py

或者：

physical_ansatz/stage1_reconstruction.py

不要在每个脚本里复制一套。

------------------------------------------------------------
任务 3：修正 Stage-4 loss 和 drift loss
------------------------------------------------------------

在 train_autoencoder_stage4_spectral_consistency.py 中：

错误形式：

R_over_P = model.predict_Rin(...)

改成：

R_over_P = compute_R_over_P_from_model(...)

同样，frozen Stage-1 drift 也要改：

R_stage1_over_P = compute_R_over_P_from_model(stage1_model, ...)

Drift 应该约束真正的 R/P，而不是只约束 f_R。

注意：
这个 helper 需要 lambda_，所以 Stage-4 batch 必须包含 lambda_。

------------------------------------------------------------
任务 4：Stage-4 spectral cache 和训练参数必须一一对应
------------------------------------------------------------

当前代码从 rpred_cache.npz 中随机取参数点，然后用最近邻找 spectral cache：

dist = |a_cache-a| + |omega_cache-omega|

这不够严谨。

请新增训练模式：

stage4:
  parameter_source: spectral_cache

默认 Stage-4 v2 使用 spectral cache 中自带的参数：

a_cache, omega_cache, u_cache, v_cache, lambda_cache

训练 batch 直接从 spectral cache 的 index 里采样。

也就是说，如果 spectral cache 有 64 个点，就只在这 64 个点上做 Stage-4 consistency training。不要再从更大的 rpred pool 随机取点并用最近邻匹配。

如果以后要覆盖更多点，就先重建更大的 spectral cache，而不是最近邻。

------------------------------------------------------------
任务 5：确保 spectral_basis_cache.npz 包含 lambda_
------------------------------------------------------------

检查：

outputs/stage4_spectral_basis_cache/patch_000_logw_v2/spectral_basis_cache.npz

必须包含：

lambda_

如果没有，修改 scripts/build_stage4_spectral_basis_cache.py，让它保存：

lambda_=lam_samp

训练脚本必须读取 lambda_cache，并传给 compute_R_over_P_from_model。

------------------------------------------------------------
任务 6：重新做 Stage-4 preflight，不训练
------------------------------------------------------------

新增或修改：

scripts/diagnose_stage4_spectral_consistency.py

用修正后的 R_over_P 和 spectral_cache 参数点重新计算：

R_model_over_P = h2*S_model

R_combo_over_P =
(B_ref_net * u_up_spec * A_up + B_inc_net * u_down_spec * A_down) / P

注意顺序：
B_ref 配 u_up*A_up
B_inc 配 u_down*A_down

输出：

- median_rel_err；
- max_rel_err；
- per-point error；
- low_omega/high_omega/center；
- before/after 和旧版本对比；
- 是否仍然约 1.6。

如果修正后 median_rel_err 大幅下降，说明之前 Stage-4 失败主要是因为 loss 写错。

如果修正后仍然很大，才说明 Stage-1 Rin、Stage-2 amplitude、spectral basis 之间本身不自洽。

------------------------------------------------------------
任务 7：加一个对照诊断，分解误差来源
------------------------------------------------------------

新增脚本或在 diagnose 中加入：

1. spectral self-consistency:
   用 spectral B_inc/B_ref 和 spectral u_up/u_down 构造 R_combo_spec/P；
   和 spectral R_in/P 比较。
   这应该很小，用于验证 spectral cache、A_up/A_down、P 全部一致。

2. amplitude-net consistency:
   用 B_net + spectral u 构造 R_combo_net/P；
   和 spectral R_in/P 比较。
   如果这里很小，说明 AmplitudeNet 没问题。

3. Stage-1 consistency:
   用 R_stage1_model/P 和 spectral R_in/P 比较。
   如果这里很大，说明 Stage-1 Rin 在无穷远端并不准确。

4. actual Stage-4 consistency:
   R_stage1_model/P vs B_net*spectral_u*A/P。

这样可以判断问题到底来自：
- Stage-1 Rin；
- Stage-2 B；
- spectral cache；
- P/A convention；
- 或 Stage-4 代码。

------------------------------------------------------------
任务 8：不要继续 Stage-4 训练
------------------------------------------------------------

在上述 audit 完成前：

不要：
- 不要继续 200 epoch；
- 不要提高 LR；
- 不要解除 encoder freeze；
- 不要增加训练模块；
- 不要采用当前 Stage-4 checkpoint。

当前 Stage-4 结果应视为 rejected。
采用 pre-stage4 bundle 作为回退点。

------------------------------------------------------------
任务 9：如果修正后 preflight 变好，再做 20 epoch v2 smoke
------------------------------------------------------------

只有当修正后 preflight 满足：

median_rel_err 明显低于旧值 1.6，例如 < 0.3，

才跑新的 20 epoch smoke。

否则不要训练，直接报告误差来源。

如果可以训练，使用非常保守设置：

train_encoder: false
train_rin_decoder: true
train_amplitude_net: true

lr_rin_decoder: 1e-7
lr_amplitude_net: 5e-7

如果 20 epoch 仍无改善，再停止。

------------------------------------------------------------
任务 10：提交
------------------------------------------------------------

提交 Stage-4 audit 修正代码：

git add scripts/ physical_ansatz/ config/
git commit -m "Fix Stage-4 R-over-P reconstruction and spectral-cache parameter alignment"
git push

最终报告必须包含：

1. 当前 commit hash；
2. 是否确认旧 Stage-4 错把 f_R 当成 R/P；
3. 修正后的 R_over_P helper；
4. spectral_cache 是否包含 lambda；
5. 是否取消最近邻参数匹配；
6. 修正前后 preflight median_rel_err 对比；
7. 误差来源分解：
   - spectral self-consistency；
   - amplitude-net consistency；
   - Stage-1 consistency；
   - actual Stage-4 consistency；
8. 是否仍建议 Stage-4 training；
9. 如果不建议，具体瓶颈是什么。