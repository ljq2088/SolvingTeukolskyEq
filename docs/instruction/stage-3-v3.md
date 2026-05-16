当前分支：feature/autoencoder-asymptotic-decoder2。

你已经完成 Stage-3 v2 并推送 commit 08713d9。当前实现已经从 direct R residual 改为 u-equation residual：

C2 * u_yy + C1 * u_y + C0 * u = 0

这符合 Stage-3 的核心原则：训练光滑归一化函数 u_up/u_down，而不是显式训练巨大函数 R_up=A_up*u_up。debug 56/56 通过，500 epoch pilot 中 loss_up 和 loss_down 都明显下降，说明 autograd 和 residual 架构问题已经基本解决。

现在下一步不要进入 Stage-4，不要做 consistency polishing，不要更新 encoder/RinDecoder/AmplitudeNet。继续只训练 UpDecoder/DownDecoder。

但不要直接用原设置跑 5000 epoch。500 epoch pilot 中 loss_up 在 epoch 450 达到 2.04e7 后，epoch 500 回升到 1.07e8，说明训练有过冲或振荡。下一步做 Stage-3 stabilized long run。

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

确认最新 commit 包含：

08713d9 Stabilize Stage-3 u-equation residual for infinity bases

------------------------------------------------------------
任务 2：保存并定位 500 epoch pilot 的 best checkpoint
------------------------------------------------------------

请找到刚才 500 epoch pilot 的 run_dir，例如：

outputs/autoencoder_stage3_ubasis_train/<timestamp>_stage3_ubasis

检查：

summary.json
checkpoints/best_model.pt
checkpoints/latest_model.pt

确认 best epoch 是多少。根据报告，loss_up 在 epoch 450 最好，final epoch 500 有回升，所以后续 continuation 应优先从 best_model.pt 继续，而不是 latest_model.pt。

请输出：
- run_dir
- best_epoch
- best_val_loss
- best loss_up
- best loss_down
- latest loss_up/loss_down
- best_model.pt 路径

------------------------------------------------------------
任务 3：增加 Stage-3 训练稳定性配置
------------------------------------------------------------

新增配置文件：

config/autoencoder_stage3_ubasis_refine.yaml

基于 config/autoencoder_stage3_ubasis.yaml，但修改为更稳的长训练设置：

stage3:
  residual_mode: u_equation

  optimizer:
    lr: 2.0e-5
    weight_decay: 0.0

  scheduler:
    enabled: true
    factor: 0.5
    patience: 200
    min_lr: 1.0e-6

  batch_size: 8
  n_interior: 128
  epochs: 5000
  val_every_epochs: 50
  save_every_epochs: 500

  loss:
    weight_up: 1.0
    weight_down: 1.0

  sampling:
    y_strategy: chebyshev
    near_infinity_extra: true
    n_near_infinity: 32
    y_inf_min: -1.0
    y_inf_max: -0.95
    y_eps: 1.0e-3

  output_root: outputs/autoencoder_stage3_ubasis_refine

runtime:
  dtype: float64

说明：
- lr 从 1e-4 降到 2e-5，避免 loss_up 后段反弹；
- 暂时保持 y_eps=1e-3；
- 不改模型结构；
- 不改 residual 公式。

------------------------------------------------------------
任务 4：给 Stage-3 训练脚本增加 grad clipping 和更完整日志
------------------------------------------------------------

修改 scripts/train_autoencoder_stage3_ubasis.py：

1. 增加 config 字段：

stage3:
  grad_clip: 1.0

2. 在 backward 后、optimizer.step 前加入：

torch.nn.utils.clip_grad_norm_(opt_params, grad_clip)

3. 每次 validation 记录并写入 history：

- train_loss
- val_loss
- val_loss_up
- val_loss_down
- lr
- grad_norm_up
- grad_norm_down
- grad_norm_total
- y_eps
- residual_mode

4. 如果 loss_up 或 loss_down 出现 NaN/Inf，立即停止并保存 debug 信息。

5. 确认 best_model.pt 保存的是 best validation，而不是 latest。

------------------------------------------------------------
任务 5：增加 Stage-3 summary 脚本
------------------------------------------------------------

新增：

scripts/summarize_stage3_ubasis.py

用法：

python scripts/summarize_stage3_ubasis.py \
  --run-dir outputs/autoencoder_stage3_ubasis_refine/<run_dir>

输出：

- best_epoch
- best_val_loss
- best loss_up
- best loss_down
- final loss_up/loss_down
- loss_up 最小值及所在 epoch
- loss_down 最小值及所在 epoch
- 是否发生反弹
- lr 变化
- 保存为 logs/stage3_summary.md 或 run_dir/stage3_summary.md

如果 history 在 summary.json 中，则从 summary.json 读；如果另有 history.jsonl，则优先读 history.jsonl。

------------------------------------------------------------
任务 6：从 500 epoch pilot 的 best checkpoint 继续训练 1000 epoch
------------------------------------------------------------

先不要直接 5000。先做稳定性 continuation 1000 epoch：

python scripts/train_autoencoder_stage3_ubasis.py \
  --config config/autoencoder_stage3_ubasis_refine.yaml \
  --checkpoint <pilot_run_dir>/checkpoints/best_model.pt \
  --device cuda \
  --epochs 1000 \
  --verbose

注意：
这里的 checkpoint 是 Stage-3 pilot best checkpoint，不是 Stage-2 checkpoint。脚本当前如果假设 checkpoint 一定是 Stage-2 best，也要改成兼容：
- 若 checkpoint 含有已训练 up/down decoder，则继续训练；
- 若 checkpoint 是 Stage-2 best，则从未训练 up/down 初始化；
- 两者都应可加载。

训练后运行：

python scripts/summarize_stage3_ubasis.py \
  --run-dir <new_refine_run_dir>

判断标准：

可以继续 5000 epoch 的条件：
- loss_up 继续低于 pilot best 或至少不明显反弹；
- loss_down 保持低量级或继续下降；
- 无 NaN/Inf；
- grad_norm finite；
- best_model.pt 正常保存；
- 只 up/down decoder 参数变化。

------------------------------------------------------------
任务 7：如果 1000 epoch 稳定，再跑 5000 epoch
------------------------------------------------------------

如果 1000 epoch continuation 稳定，再运行：

python scripts/train_autoencoder_stage3_ubasis.py \
  --config config/autoencoder_stage3_ubasis_refine.yaml \
  --checkpoint <best_checkpoint_from_1000_epoch_refine> \
  --device cuda \
  --epochs 5000 \
  --verbose

目标：
- loss_up 比 pilot best \(2.04e7\) 进一步下降，或者至少稳定在同量级；
- loss_down 稳定在 \(10^2\) 或更低；
- 没有 NaN/Inf；
- boundary ansatz 仍然严格满足。

------------------------------------------------------------
任务 8：新增 Stage-3 basis 质量诊断
------------------------------------------------------------

训练完 1000 或 5000 后，新增脚本：

scripts/diagnose_stage3_ubasis_solution.py

输入：

--checkpoint <stage3_best_model.pt>
--artifact-dir outputs/stage1_artifacts/patch_000_logw_v2
--config config/autoencoder_stage3_ubasis_refine.yaml
--device cuda

诊断内容：

1. 对若干参数样本画或保存：
   - |u_up(y)|
   - |u_down(y)|
   - Re/Im u_up
   - Re/Im u_down
   - residual_up(y)
   - residual_down(y)

2. 检查 near-infinity：
   - u_up(-1)=1
   - u_down(-1)=1
   - du_up/dy(-1)=c_up
   - du_down/dy(-1)=c_down

3. 统计：
   - max |u_up|
   - max |u_down|
   - near-infinity residual median
   - whole-domain residual median

4. 保存：
   run_dir/diagnostics/stage3_ubasis_diagnosis.md
   run_dir/diagnostics/*.png

注意：
这一步不是 Stage-4 consistency，只是检查 u-basis 解是否平滑和有无异常振荡。

------------------------------------------------------------
任务 9：暂时不要做这些
------------------------------------------------------------

不要做：

- 不要 Stage-4 consistency polishing；
- 不要加入 R_pred/P consistency loss；
- 不要训练 encoder；
- 不要训练 RinDecoder；
- 不要训练 AmplitudeNet；
- 不要改 B_inc/B_ref；
- 不要把 B_inc/B_ref 和 u_up/u_down 组合；
- 不要改 Stage-1 / Stage-2 checkpoint。

------------------------------------------------------------
任务 10：提交并 push
------------------------------------------------------------

如果 1000 epoch refine 通过，提交代码改动：

git add config/ scripts/
git commit -m "Stabilize Stage-3 u-basis long training"
git push

大型 outputs 不要提交。

最终报告请包含：

1. 当前 commit hash；
2. 使用的起始 checkpoint；
3. 是否从 Stage-3 pilot best checkpoint 继续；
4. 新增 refine config；
5. grad clipping 设置；
6. 1000 epoch refine 的 loss_up/loss_down 变化；
7. 是否反弹；
8. best checkpoint 路径；
9. summarize_stage3_ubasis.py 输出摘要；
10. 是否建议继续跑 5000 epoch；
11. 是否仍阻塞 Stage-4。