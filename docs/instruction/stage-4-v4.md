当前分支：feature/autoencoder-asymptotic-decoder2。

最新反馈表明：
1. absolute amplitude teacher 已修正；
2. B_inc median relative error 已降到约 0.88%，B_ref median relative error 约 0.69%；
3. 但 Stage-4 preflight median_rel_err 仍约 0.63，没有改善；
4. 因此当前瓶颈不是 AmplitudeNet，而是 Stage-1 RinDecoder 在近无穷远区域的 R_in/P 不准确。

现在不要继续优化 AmplitudeNet 到 1e-5。
不要继续 Stage-4 low-LR polishing。
不要训练 learned u_up/u_down decoders。

下一步做 Stage-1 near-infinity PDE + spectral consistency refinement。

------------------------------------------------------------
任务 1：确认状态
------------------------------------------------------------

运行：

git branch --show-current
git pull
git status
git log --oneline -5

确认当前分支是：

feature/autoencoder-asymptotic-decoder2

确认包含 commit cc24535 或更新。

------------------------------------------------------------
任务 2：先做 oracle-B 诊断，最终确认瓶颈
------------------------------------------------------------

在进入训练前，先新增或扩展诊断脚本：

scripts/diagnose_stage1_nearinf_bottleneck.py

用同一批 spectral cache 点，计算三种 consistency：

A. 使用 AmplitudeNet B：

R_combo_net/P =
(B_ref_net*u_up_spec*A_up + B_inc_net*u_down_spec*A_down)/P

B. 使用谱方法精确 B：

R_combo_spec/P =
(B_ref_spec*u_up_spec*A_up + B_inc_spec*u_down_spec*A_down)/P

C. Stage-1 输出：

R_stage1/P = h2*S_RinDecoder

比较：

err_net =
|R_stage1/P - R_combo_net/P| / |R_stage1/P|

err_spec =
|R_stage1/P - R_combo_spec/P| / |R_stage1/P|

如果 err_net 和 err_spec 都约 0.6，则确认瓶颈是 Stage-1。
如果 err_spec 显著下降而 err_net 高，则说明 AmplitudeNet 仍是瓶颈。

这一步必须输出：
- median_err_net
- median_err_spec
- per-point errors
- low omega cases
- B_inc/B_ref net-vs-spec error

------------------------------------------------------------
任务 3：建立 Stage-1 near-infinity refinement 配置
------------------------------------------------------------

新增配置：

config/autoencoder_stage1_nearinf_consistency_refine.yaml

建议：

stage1_refine:
  stage1_checkpoint: outputs/stage1_artifacts/patch_000_logw_v2/stage1_best_model.pt
  stage2_checkpoint: outputs/autoencoder_stage2_absolute_amplitude_train/<best_run>/checkpoints/best_model.pt
  spectral_basis_cache: outputs/stage4_spectral_basis_cache/patch_000_logw_v2/spectral_basis_cache.npz

  train_encoder: false
  train_rin_decoder: true
  train_amplitude_net: false
  train_up_decoder: false
  train_down_decoder: false

  optimizer:
    lr_rin_decoder: 1.0e-6
    weight_decay: 0.0

  epochs: 500
  batch_size: 8
  grad_clip: 0.1

  loss:
    weight_pde: 1.0
    weight_consistency: 0.1
    weight_rin_drift: 10.0
    eps: 1.0e-12

  sampling:
    near_core:
      y_min: -0.999
      y_max: -0.95
      n_y: 96
    transition:
      y_min: -0.95
      y_max: 0.0
      n_y: 32
    pde_sampling: same_y_grid

  output_root: outputs/autoencoder_stage1_nearinf_consistency_refine

runtime:
  dtype: float64

第一轮只训练 RinDecoder，不动 encoder。
AmplitudeNet 冻结。
spectral u 固定。
不要训练 up/down decoder。

------------------------------------------------------------
任务 4：实现联合 loss
------------------------------------------------------------

新增训练脚本：

scripts/train_stage1_nearinf_consistency_refine.py

loss 包括三项：

1. Stage-1 PDE residual：

L_pde =
原 Stage-1 的 Teukolsky residual loss

注意：
必须复用现有 Stage-1 trainer/physical_ansatz/residual.py 中的实现。
不要重写一套新方程。
只是换采样点到 near-infinity + transition 区间。

2. Spectral consistency：

R_model/P = h2*S_model

R_combo/P =
(B_ref_net*u_up_spec*A_up + B_inc_net*u_down_spec*A_down)/P

L_cons =
mean |R_model/P - R_combo/P|^2 / mean(|R_model/P|^2 + eps)

3. Rin drift：

R_stage1_frozen/P = h2*S_frozen

L_drift =
mean |R_model/P - R_stage1_frozen/P|^2 / mean(|R_stage1_frozen/P|^2 + eps)

Total:

L =
weight_pde * L_pde
+
weight_consistency * L_cons
+
weight_rin_drift * L_drift

建议初始：
weight_pde = 1
weight_consistency = 0.1
weight_rin_drift = 10

不要一开始把 consistency 权重设太大，否则会破坏 Stage-1 已有解。

------------------------------------------------------------
任务 5：debug 脚本
------------------------------------------------------------

新增：

scripts/debug_stage1_nearinf_consistency_refine.py

检查：

1. Stage-1 checkpoint 可加载；
2. Stage-2 absolute AmplitudeNet checkpoint 可加载；
3. spectral cache 可加载；
4. R_model/P = h2*S_model 计算正确；
5. Stage-1 PDE residual finite；
6. consistency loss finite；
7. drift loss finite；
8. 只有 RinDecoder 有梯度；
9. encoder 无梯度；
10. amplitude_net 无梯度；
11. up/down decoder 无梯度；
12. optimizer.step 后只有 RinDecoder 改变；
13. 无 NaN/Inf。

------------------------------------------------------------
任务 6：先跑 50 epoch smoke
------------------------------------------------------------

不要直接 500。

运行：

python scripts/debug_stage1_nearinf_consistency_refine.py \
  --config config/autoencoder_stage1_nearinf_consistency_refine.yaml \
  --device cuda

然后：

python scripts/train_stage1_nearinf_consistency_refine.py \
  --config config/autoencoder_stage1_nearinf_consistency_refine.yaml \
  --device cuda \
  --epochs 50 \
  --verbose

记录：

- initial L_pde
- initial L_cons
- initial L_drift
- epoch 50 L_pde/L_cons/L_drift
- median_rel_err before/after
- low_omega cases before/after
- Rin drift
- NaN/Inf
- best checkpoint path

通过条件：

1. L_cons 有下降趋势；
2. L_pde 不爆炸；
3. Rin drift < 1e-2；
4. low_omega cases 至少有改善；
5. 无 NaN/Inf。

如果 50 epoch 后 L_cons 完全不动，不要继续 500。

------------------------------------------------------------
任务 7：如果 50 epoch 有效，再跑 500 epoch
------------------------------------------------------------

如果 smoke 通过，再运行：

python scripts/train_stage1_nearinf_consistency_refine.py \
  --config config/autoencoder_stage1_nearinf_consistency_refine.yaml \
  --device cuda \
  --epochs 500 \
  --resume-checkpoint <50epoch_best> \
  --verbose

目标不是让 drift 很大，而是让 near-infinity consistency 从 0.6 明显下降，比如先降到 <0.3。

如果降到 <0.3，再考虑进一步调权重。

------------------------------------------------------------
任务 8：暂时不要做这些
------------------------------------------------------------

不要：
- 不要继续 Stage-4 polishing；
- 不要提高 Stage-4 LR；
- 不要继续追求 AmpNet 到 1e-5；
- 不要训练 encoder；
- 不要训练 up/down decoder；
- 不要重训 Stage-3；
- 不要使用 current Stage-4 checkpoint。

当前有效回退点仍是：
outputs/pre_stage4_bundle/patch_000_logw_v2/

------------------------------------------------------------
任务 9：提交
------------------------------------------------------------

提交代码，不提交大型 outputs：

git add scripts/diagnose_stage1_nearinf_bottleneck.py \
        scripts/debug_stage1_nearinf_consistency_refine.py \
        scripts/train_stage1_nearinf_consistency_refine.py \
        config/autoencoder_stage1_nearinf_consistency_refine.yaml

git commit -m "Add Stage-1 near-infinity consistency refinement"
git push

最终报告必须包含：

1. 当前 commit hash；
2. oracle-B 诊断结果；
3. 是否确认 Stage-1 是瓶颈；
4. 50 epoch smoke 的 L_pde/L_cons/L_drift；
5. before/after median_rel_err；
6. low omega cases before/after；
7. Rin drift；
8. 是否建议继续 500 epoch。