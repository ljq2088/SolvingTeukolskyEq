当前分支：feature/autoencoder-asymptotic-decoder2。

你已经完成 Stage-2：
- amplitude teacher convention 已修正为 r_*；
- Stage-2 AmplitudeNet 已训练完成；
- best model: outputs/autoencoder_stage2_amplitude_train/20260515_192625_stage2_amplitude/checkpoints/best_model.pt
- B_inc best complex error ~2.6e-4；
- B_ref best complex error ~8.2e-3。

现在进入 Stage-3：训练 u_up / u_down decoders。

不要进入 Stage-4，不要做 consistency polishing，不要微调 RinDecoder，不要改 Stage-1 R_pred，不要改 AmplitudeNet。Stage-3 只训练 UpDecoder / DownDecoder。

请始终牢记模型核心：

AutoencoderTeukolskyPINN =
1. SharedPINNEncoder：共享物理编码器；
2. RinDecoder：Stage-1，已训练，固定 R_pred；
3. AmplitudeNet：Stage-2，已训练，固定 B_inc/B_ref；
4. UpDecoder / DownDecoder：Stage-3，本轮训练目标；
5. Stage-4：低学习率 consistency polishing，后续再做。

Stage-3 目标：
训练两个归一化无穷远基函数：

u_up = R_up / A_up
u_down = R_down / A_down

其中：

A_up   = r^3 exp(i omega r_*)
A_down = r^{-1} exp(-i omega r_*)

边界 ansatz 必须强制：

u_up(-1) = 1
u_down(-1) = 1
du_up/dy(-1) = c_up
du_down/dy(-1) = c_down

使用 physical_ansatz/u_basis.py 中已有的：

infinity_slopes_y
compose_u_from_f
A_up
A_down

不要重新写一套不一致的 r_star 或 A_up/A_down。

------------------------------------------------------------
任务 1：确认当前 git 状态
------------------------------------------------------------

运行：

git branch --show-current
git pull
git status
git log --oneline -5

确认当前分支是：

feature/autoencoder-asymptotic-decoder2

确认包含最新 commit e164faf 或更新。

------------------------------------------------------------
任务 2：检查 Stage-3 所需输入 checkpoint
------------------------------------------------------------

Stage-3 应从 Stage-2 best model 初始化，因为它包含：

- Stage-1 的 encoder + RinDecoder；
- Stage-2 训练好的 AmplitudeNet；
- 未训练或初始状态的 UpDecoder / DownDecoder。

使用：

outputs/autoencoder_stage2_amplitude_train/20260515_192625_stage2_amplitude/checkpoints/best_model.pt

请先写一个检查：

- checkpoint 可加载；
- encoder 参数存在；
- rin_decoder 参数存在；
- amplitude_net 参数存在；
- up_decoder/down_decoder 参数存在；
- predict_amplitudes 仍给出 Stage-2 训练后的结果；
- predict_Rin 仍能输出；
- predict_u_up / predict_u_down 能输出，但此时还未训练。

如果发现 Stage-2 checkpoint 中 up/down decoder 是未训练随机态，这是正常的。

------------------------------------------------------------
任务 3：实现 Stage-3 冻结策略
------------------------------------------------------------

新增或扩展一个工具函数，例如：

set_autoencoder_stage(model, stage="stage3")

Stage-3 冻结策略：

- freeze encoder；
- freeze rin_decoder；
- freeze amplitude_net；
- train up_decoder；
- train down_decoder。

注意：
不要更新 encoder。虽然 u_up/u_down 也使用 encoder 的 latent，但本阶段应该把 Stage-1 学到的物理表示作为固定坐标系，训练两个新的物理解码头。

打印参数统计：

- total params；
- trainable params；
- encoder trainable = 0；
- rin_decoder trainable = 0；
- amplitude_net trainable = 0；
- up_decoder trainable > 0；
- down_decoder trainable > 0。

------------------------------------------------------------
任务 4：实现 Stage-3 u-basis residual
------------------------------------------------------------

新增文件或模块：

physical_ansatz/u_residual.py

或者如果已有合适位置，就复用已有 residual 模块。

推荐实现方式：优先用 direct R residual，而不是使用旧 transform_coeffs_u_to_y。

对每个 basis：

R_up(y) = A_up(r(y)) * u_up(y)
R_down(y) = A_down(r(y)) * u_down(y)

其中：

u_up(y) = compose_u_from_f(f_up, y, c_up)
u_down(y) = compose_u_from_f(f_down, y, c_down)

然后直接把 R_up / R_down 代入原始 Teukolsky radial ODE residual。

要求：
- 使用和 Stage-1 相同的 y 坐标、r(y)=r_plus/x、x=(y+1)/2；
- 使用现有 teukolsky_coeffs / residual / prefactor 中的物理系数；
- 不要手写一套新的 Teukolsky 方程；
- 如果现有 residual 接口是针对 S 或 f_R 的，需要新增一个针对 R(y) 的 direct residual helper；
- autograd 计算 R_y、R_yy；
- 根据链式法则转换到 r 或使用已有 y-space 系数；
- dtype/device 必须和 trainer 保持一致；
- complex residual 使用 |Re|^2 + |Im|^2 或 torch.abs(residual)**2。

Stage-3 loss：

L_up   = mean |TeukResidual[R_up]|^2
L_down = mean |TeukResidual[R_down]|^2
L_stage3 = L_up + L_down

可选增加 mild normalization：

L_up_norm = mean |res_up|^2 / mean(|R_up|^2 + eps)
L_down_norm = mean |res_down|^2 / mean(|R_down|^2 + eps)

但先不要过度复杂。若 residual 数值尺度爆炸，再加 normalization。

------------------------------------------------------------
任务 5：Stage-3 配置文件
------------------------------------------------------------

新增配置：

config/autoencoder_stage3_ubasis.yaml

建议初始设置：

include:
  physics: config/teukolsky_radial.yaml

autoencoder:
  stage: stage3

stage3:
  checkpoint: outputs/autoencoder_stage2_amplitude_train/20260515_192625_stage2_amplitude/checkpoints/best_model.pt
  artifact_dir: outputs/stage1_artifacts/patch_000_logw_v2

  train_up: true
  train_down: true

  optimizer:
    lr: 1.0e-4
    weight_decay: 0.0

  scheduler:
    enabled: true
    factor: 0.5
    patience: 300
    min_lr: 1.0e-6

  batch_size: 8
  n_interior: 128
  n_param_samples: 64
  epochs: 5000
  val_every_epochs: 50
  save_every_epochs: 500

  loss:
    weight_up: 1.0
    weight_down: 1.0
    normalize_residual: true
    eps: 1.0e-12

  sampling:
    y_strategy: chebyshev
    near_infinity_extra: true
    n_near_infinity: 32
    y_inf_min: -1.0
    y_inf_max: -0.95

  output_root: outputs/autoencoder_stage3_ubasis_train

runtime:
  dtype: float64

说明：
Stage-3 必须加强 near-infinity 采样，因为 u_up/u_down 的边界 ansatz 在 y=-1 强制，但 residual 也需要在靠近无穷远处稳定。

------------------------------------------------------------
任务 6：Stage-3 训练脚本
------------------------------------------------------------

新增：

scripts/train_autoencoder_stage3_ubasis.py

功能：

python scripts/train_autoencoder_stage3_ubasis.py \
  --config config/autoencoder_stage3_ubasis.yaml \
  --checkpoint outputs/autoencoder_stage2_amplitude_train/20260515_192625_stage2_amplitude/checkpoints/best_model.pt \
  --device cuda \
  --epochs 500 \
  --verbose

训练逻辑：

1. 加载 Stage-2 best checkpoint；
2. 构造 AutoencoderTeukolskyPINN；
3. load_state_dict；
4. 设置 Stage-3 freeze 策略；
5. 从 stage1 artifact 或 patch json 中采样 a, omega, u, v, lambda；
6. 对 y 内点采样；
7. 计算 f_up/f_down；
8. 用 infinity_slopes_y 得到 c_up/c_down；
9. compose_u_from_f 得到 u_up/u_down；
10. 构造 R_up=A_up*u_up, R_down=A_down*u_down；
11. 计算 Teukolsky residual；
12. 只更新 up_decoder/down_decoder；
13. 保存 latest/best checkpoint；
14. 写 history.jsonl 和 summary.json。

------------------------------------------------------------
任务 7：Stage-3 debug 脚本
------------------------------------------------------------

新增：

scripts/debug_autoencoder_stage3_ubasis.py

必须检查：

1. Stage-2 checkpoint 可加载；
2. Stage-3 freeze 策略正确；
3. predict_u_up / predict_u_down 输出 shape 正确；
4. u_up(-1)=1；
5. u_down(-1)=1；
6. du_up/dy(-1)=c_up；
7. du_down/dy(-1)=c_down；
8. A_up/A_down finite；
9. R_up/R_down finite；
10. u-basis residual finite；
11. backward 后只有 up_decoder/down_decoder 有梯度；
12. optimizer.step 后只有 up_decoder/down_decoder 参数变化；
13. checkpoint save/load round-trip 正常。

先用 synthetic small batch 测试，不跑长训练。

------------------------------------------------------------
任务 8：先跑 300/500 epoch pilot，不要直接 5000
------------------------------------------------------------

通过 debug 后，先跑 pilot：

python scripts/train_autoencoder_stage3_ubasis.py \
  --config config/autoencoder_stage3_ubasis.yaml \
  --checkpoint outputs/autoencoder_stage2_amplitude_train/20260515_192625_stage2_amplitude/checkpoints/best_model.pt \
  --device cuda \
  --epochs 500 \
  --verbose

记录：

- epoch 1 loss_up/loss_down；
- epoch 50；
- epoch 100；
- epoch 300；
- epoch 500；
- 是否 NaN/Inf；
- up/down residual 是否下降；
- near-infinity residual 是否稳定；
- boundary ansatz 是否始终保持；
- best checkpoint 路径。

判断是否可以跑 5000：

- loss_up 和 loss_down 都明显下降；
- 无 NaN/Inf；
- near-infinity residual 不爆；
- u_up/u_down 不出现巨大振荡；
- checkpoint 能正常保存/加载。

如果 500 epoch 后其中一个 basis 学不会，不要直接 5000。先分析 up/down 哪个难，是否 residual 归一化、采样或 learning rate 需要调整。

------------------------------------------------------------
任务 9：本轮不要做这些
------------------------------------------------------------

不要做：

- 不要 Stage-4 consistency polishing；
- 不要加入 R_pred/P consistency loss；
- 不要更新 encoder；
- 不要更新 RinDecoder；
- 不要更新 AmplitudeNet；
- 不要重新训练 Stage-1；
- 不要重新训练 Stage-2；
- 不要改变 B_inc/B_ref convention；
- 不要把 B_inc 和 B_ref 顺序写反。

------------------------------------------------------------
任务 10：测试与提交
------------------------------------------------------------

运行：

python -m py_compile \
  physical_ansatz/u_residual.py \
  scripts/train_autoencoder_stage3_ubasis.py \
  scripts/debug_autoencoder_stage3_ubasis.py

运行：

python scripts/debug_autoencoder_stage3_ubasis.py \
  --config config/autoencoder_stage3_ubasis.yaml \
  --checkpoint outputs/autoencoder_stage2_amplitude_train/20260515_192625_stage2_amplitude/checkpoints/best_model.pt \
  --device cuda

然后跑 500 epoch pilot。

如果通过：

git add physical_ansatz/ scripts/ config/
git commit -m "Add Stage-3 u-basis decoder training pipeline"
git push

大型 outputs 不要提交。

最终报告必须包含：

1. 当前 commit hash；
2. Stage-3 使用的 checkpoint；
3. freeze 策略参数统计；
4. u_up/u_down 边界 ansatz 测试结果；
5. residual finite 测试结果；
6. 500 epoch pilot 的 loss_up/loss_down 变化；
7. 是否建议继续 5000 epochs；
8. 是否仍阻塞 Stage-4。