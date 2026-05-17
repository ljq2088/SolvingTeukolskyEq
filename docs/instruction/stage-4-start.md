当前分支：feature/autoencoder-asymptotic-decoder2。

现在调整路线。不要继续训练 Stage-3 learned u_up/u_down decoders，也不要实现 spectral anchor 去训练这两个 decoder。

原因：
Stage-3 v8 已经证明：
1. u-equation 方程系数正确；
2. 谱方法解满足 Stage-3 residual；
3. 但 learned up/down decoder 与谱方法 basis 差异很大；
4. 纯 PDE residual 无法把冻结 encoder + 小 decoder 拉到正确 basis。

因此现在采用新方案：

废弃 Stage-3 learned u_up/u_down decoder 在 Stage-4 中的作用。
保留代码但冻结并 bypass。
Stage-4 使用谱方法计算得到的固定 u_up/u_down basis cache。

目标：
把 Stage-1 的 Rin PINN、Stage-2 的 AmplitudeNet、谱方法的 u_up/u_down basis 结合起来做 Stage-4 low-learning-rate consistency polishing。

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

确认包含 commit d4ac27a 或更新。

------------------------------------------------------------
任务 2：保留 Stage-4 前模型，建立回退点
------------------------------------------------------------

在任何 Stage-4 训练之前，必须保存 pre-stage4 artifact。

新增脚本：

scripts/export_pre_stage4_bundle.py

输入：

--stage1-artifact outputs/stage1_artifacts/patch_000_logw_v2
--stage2-checkpoint outputs/autoencoder_stage2_amplitude_train/20260515_192625_stage2_amplitude/checkpoints/best_model.pt
--output-dir outputs/pre_stage4_bundle/patch_000_logw_v2

输出：

outputs/pre_stage4_bundle/patch_000_logw_v2/
  metadata.json
  stage1_best_model.pt
  stage2_best_model.pt
  rpred_cache.npz 或 symlink 信息
  rollback_instructions.md

metadata 必须包含：

- git commit；
- stage1 checkpoint；
- stage2 checkpoint；
- patch_id；
- omega_chart_mode；
- scattering convention；
- 明确写：
  Stage-3 learned decoders are bypassed in Stage-4 spectral-basis mode.

不要覆盖已有 Stage-1 / Stage-2 输出。

------------------------------------------------------------
任务 3：构建 Stage-4 spectral basis cache
------------------------------------------------------------

新增脚本：

scripts/build_stage4_spectral_basis_cache.py

目标：
对 patch 0 的训练参数点，使用谱方法生成固定的：

u_up_spec(a,omega,y)
u_down_spec(a,omega,y)

用于 Stage-4 consistency loss。

输入：

--stage1-artifact outputs/stage1_artifacts/patch_000_logw_v2
--calibration-dir outputs/spectral_calibration/patch_000_logw_v2
--n-param 64
--y-interval near_core
--y-min -0.999
--y-max -0.95
--n-y 128
--N-out 80
--z-b 0.05
--output-dir outputs/stage4_spectral_basis_cache/patch_000_logw_v2

输出：

outputs/stage4_spectral_basis_cache/patch_000_logw_v2/
  spectral_basis_cache.npz
  spectral_basis_cache_summary.md
  metadata.json

npz 至少包含：

- a
- omega
- u
- v
- lambda
- y
- z
- u_up
- u_down
- u_up_y 或 u_up_z，可选
- u_down_y 或 u_down_z，可选
- basis convention
- N_out
- z_b

注意：
1. 参数点必须来自 patch 0 的 rpred_cache.npz 或 patch0_calibration_points.json，不要手动猜；
2. 不允许再使用 omega=10；
3. 必须包含 u/v 局部坐标；
4. 谱方法 basis 必须满足：
   u_up(z=0)=1
   u_down(z=0)=1
5. cache 只覆盖靠近无穷远区域，初始使用 y in [-0.999,-0.95]。

先对少量点做 pilot，例如 n-param=16；确认无误后再生成 n-param=64。

------------------------------------------------------------
任务 4：Stage-4 preflight，不训练
------------------------------------------------------------

新增脚本：

scripts/diagnose_stage4_spectral_consistency.py

这个脚本只诊断，不更新参数。

输入：

--pre-stage4-bundle outputs/pre_stage4_bundle/patch_000_logw_v2
--stage2-checkpoint outputs/autoencoder_stage2_amplitude_train/20260515_192625_stage2_amplitude/checkpoints/best_model.pt
--spectral-basis-cache outputs/stage4_spectral_basis_cache/patch_000_logw_v2/spectral_basis_cache.npz
--config config/autoencoder_stage3_ubasis_refine.yaml
--device cuda
--output-dir outputs/stage4_spectral_preflight/patch_000_logw_v2

计算：

1. 用 Stage-1 模型重新计算或从 artifact 读取：

R_model_over_P

优先重新计算：
R_model_over_P = h2 * S_model

2. 用 Stage-2 amplitude net 计算：

B_inc_net, B_ref_net

3. 从 spectral cache 读取：

u_up_spec, u_down_spec

4. 计算：

R_combo_over_P =
(B_ref_net * u_up_spec * A_up + B_inc_net * u_down_spec * A_down) / P

5. 计算 consistency error：

rel_err =
|R_model_over_P - R_combo_over_P|
/
(|R_model_over_P| + eps)

输出：

- median_rel_err
- max_rel_err
- per-point median/max
- low_omega/high_omega/center cases
- up/down contribution ratio
- 是否 NaN/Inf
- plots

注意：
这一步仍然不训练。
如果 preflight error 已经很小，就不一定需要 Stage-4 微调。

------------------------------------------------------------
任务 5：Stage-4 训练配置
------------------------------------------------------------

新增配置：

config/autoencoder_stage4_spectral_consistency.yaml

建议：

stage4:
  basis_source: spectral_cache
  spectral_basis_cache: outputs/stage4_spectral_basis_cache/patch_000_logw_v2/spectral_basis_cache.npz
  pre_stage4_bundle: outputs/pre_stage4_bundle/patch_000_logw_v2

  train_encoder: false
  train_rin_decoder: true
  train_amplitude_net: true
  train_up_decoder: false
  train_down_decoder: false

  optimizer:
    lr_rin_decoder: 1.0e-7
    lr_amplitude_net: 5.0e-7
    lr_encoder: 0.0
    weight_decay: 0.0

  epochs: 500
  batch_size: 8
  grad_clip: 0.1

  loss:
    weight_consistency: 1.0
    weight_rin_drift: 50.0
    weight_amp_drift: 50.0
    eps: 1.0e-12

  y_region:
    y_min: -0.999
    y_max: -0.95

  output_root: outputs/autoencoder_stage4_spectral_consistency

runtime:
  dtype: float64

重要：
初始不要训练 encoder。
Stage-4 只允许 RinDecoder 和 AmplitudeNet 以极小学习率微调。
如果 100 epoch 后效果很好，再考虑是否给 encoder 一个极小 lr，比如 1e-8。但默认 encoder 冻结。

------------------------------------------------------------
任务 6：Stage-4 loss
------------------------------------------------------------

新增训练脚本：

scripts/train_autoencoder_stage4_spectral_consistency.py

loss 定义：

L_cons =
mean |R_model_over_P - R_combo_over_P|^2 / mean(|R_model_over_P|^2 + eps)

其中：

R_combo_over_P =
(B_ref_net * u_up_spec * A_up + B_inc_net * u_down_spec * A_down) / P

Drift regularization：

L_rin_drift =
mean |R_model_over_P - R_stage1_frozen_over_P|^2
/
mean(|R_stage1_frozen_over_P|^2 + eps)

L_amp_drift =
mean |B_inc_net - B_inc_stage2|^2 / mean(|B_inc_stage2|^2 + eps)
+
mean |B_ref_net - B_ref_stage2|^2 / mean(|B_ref_stage2|^2 + eps)

Total:

L =
L_cons
+ weight_rin_drift * L_rin_drift
+ weight_amp_drift * L_amp_drift

必须注意：
- up_decoder/down_decoder 不参与；
- 不调用 predict_u_up/predict_u_down；
- spectral u 是固定 cache；
- B_ref 配 u_up A_up；
- B_inc 配 u_down A_down；
- 训练前后都要记录 drift。

------------------------------------------------------------
任务 7：debug 脚本
------------------------------------------------------------

新增：

scripts/debug_stage4_spectral_consistency.py

检查：

1. pre-stage4 bundle 可加载；
2. spectral_basis_cache 可加载；
3. Stage-1 / Stage-2 checkpoint 可加载；
4. R_model_over_P shape 正确；
5. B_inc/B_ref shape 正确；
6. u_up/u_down spectral cache shape 正确；
7. A_up/A_down/P shape 正确；
8. R_combo_over_P finite；
9. L_cons finite；
10. drift loss finite；
11. 只有 RinDecoder / AmplitudeNet 有梯度；
12. encoder 无梯度；
13. up/down decoder 无梯度；
14. optimizer.step 后只 RinDecoder / AmplitudeNet 参数变化；
15. 回退 checkpoint 存在。

------------------------------------------------------------
任务 8：先做 20 epoch smoke
------------------------------------------------------------

不要直接 500 epoch。

先运行：

python scripts/debug_stage4_spectral_consistency.py \
  --config config/autoencoder_stage4_spectral_consistency.yaml \
  --device cuda

然后：

python scripts/train_autoencoder_stage4_spectral_consistency.py \
  --config config/autoencoder_stage4_spectral_consistency.yaml \
  --device cuda \
  --epochs 20 \
  --verbose

报告：

- initial L_cons；
- epoch 20 L_cons；
- L_rin_drift；
- L_amp_drift；
- median_rel_err before/after；
- max_rel_err before/after；
- B_inc drift；
- B_ref drift；
- Rin drift；
- 是否 NaN/Inf；
- best checkpoint 路径；
- rollback path。

如果 20 epoch 后 drift 过大，立即停止，不跑更长。

Drift 参考阈值：

- Rin relative drift 不超过 1e-3 到 1e-2；
- B_inc/B_ref relative drift 不超过 1e-3 到 1e-2；
- 如果超过，增大 drift weight 或降低 lr。

------------------------------------------------------------
任务 9：若 20 epoch 有效，再跑 200 epoch
------------------------------------------------------------

只有当：

1. L_cons 下降；
2. median_rel_err 下降；
3. drift 很小；
4. 无 NaN/Inf；

才继续：

python scripts/train_autoencoder_stage4_spectral_consistency.py \
  --config config/autoencoder_stage4_spectral_consistency.yaml \
  --device cuda \
  --epochs 200 \
  --resume-checkpoint <20epoch_best> \
  --verbose

不要超过 200，除非结果非常稳定。

------------------------------------------------------------
任务 10：回退机制
------------------------------------------------------------

必须在训练脚本里实现：

--rollback-to-pre-stage4

或者至少清楚记录：

outputs/pre_stage4_bundle/patch_000_logw_v2/

如果 Stage-4 结果变差，直接回退到：
- Stage-1 best model；
- Stage-2 best model；
- spectral basis cache；
- 不使用 Stage-4 checkpoint。

最终报告必须明确：
是否建议采用 Stage-4 checkpoint；
如果不建议，如何回退。

------------------------------------------------------------
任务 11：提交
------------------------------------------------------------

提交代码，不提交大型 outputs：

git add scripts/export_pre_stage4_bundle.py \
        scripts/build_stage4_spectral_basis_cache.py \
        scripts/diagnose_stage4_spectral_consistency.py \
        scripts/debug_stage4_spectral_consistency.py \
        scripts/train_autoencoder_stage4_spectral_consistency.py \
        config/autoencoder_stage4_spectral_consistency.yaml

git commit -m "Add spectral-basis Stage-4 consistency polishing"
git push

最终报告必须包含：

1. 当前 commit hash；
2. pre-stage4 bundle 路径；
3. spectral basis cache 路径；
4. preflight consistency error；
5. 20 epoch smoke before/after；
6. drift 数值；
7. 是否建议继续 200 epoch；
8. 是否建议采用 Stage-4 checkpoint；
9. 回退路径。