当前分支：feature/autoencoder-asymptotic-decoder2。

你已经完成 Stage-3 v5，commit a7cb5cb。谱方法对比显示：
- u_up(-1)=1、u_down(-1)=1，边界值和导数全部精确满足；
- 但和谱方法 basis 对比，u_up median_rel_err = 1.28，u_down median_rel_err = 0.29；
- 因此当前 Stage-3 decoder 还不够好；
- 下一步需要 near-infinity-only Stage-3 重训。

但是在启动训练前，必须先修正两个代码/配置问题：
1. 当前 train_autoencoder_stage3_ubasis.py 可能还没有真正支持 y_strategy=near_infinity_only；
2. config/autoencoder_stage3_ubasis_nearinf.yaml 中 amp_hidden_dim / amp_n_blocks 放在 encoder_kwargs 下是错误的。

本轮目标：
先修 near-infinity-only 训练网格与配置，再跑 200 epoch smoke，不要直接跑 2000 epoch。

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

确认最新包含：

a7cb5cb

------------------------------------------------------------
任务 2：修正 nearinf config 中的 model 字段
------------------------------------------------------------

打开：

config/autoencoder_stage3_ubasis_nearinf.yaml

当前可能写成：

model:
  ...
  encoder_kwargs:
    amp_hidden_dim: 128
    amp_n_blocks: 3

这是不对的。请改成：

model:
  hidden_dims: [128, 128, 128, 128]
  activation: silu
  fourier_num_freqs: 2
  fourier_base_scale: 1.0
  param_embed_dim: 64
  use_film: true
  use_residual: true
  amp_hidden_dim: 128
  amp_n_blocks: 3
  encoder_kwargs: {}

或者如果需要 local_coord / chart_uv 参数，则只把真正属于 encoder 的参数放入 encoder_kwargs。

注意：
amp_hidden_dim 和 amp_n_blocks 属于 AutoencoderTeukolskyPINN / AmplitudeNet 参数，不属于 SharedPINNEncoder 的 local coord 参数。

------------------------------------------------------------
任务 3：让训练脚本真正支持 y_strategy
------------------------------------------------------------

打开：

scripts/train_autoencoder_stage3_ubasis.py

当前 _build_y_grid 逻辑是：
- Chebyshev grid 固定覆盖 y in [-1+y_eps, 1-y_eps]
- 再加 near-infinity extra grid

这意味着即使 config 写了 y_strategy: near_infinity_only，训练仍可能覆盖整个 compact domain。

请修改 _build_y_grid，使它支持：

1. full_domain:
   y in [-1+y_eps, 1-y_eps]

2. near_infinity_only:
   y in [y_min+y_eps, y_max]
   默认 y_min=-1.0, y_max=0.0
   也就是 y in [-1+y_eps, 0.0]

建议接口：

def _build_y_grid(
    n_interior,
    n_near_inf,
    y_inf_min,
    y_inf_max,
    device,
    dtype,
    y_eps=1e-3,
    y_strategy="full_domain",
    y_min=-1.0,
    y_max=1.0,
):
    ...

实现要求：

if y_strategy == "full_domain":
    base grid = Chebyshev on [-1+y_eps, 1-y_eps]

if y_strategy == "near_infinity_only":
    base grid = Chebyshev mapped to [y_min+y_eps, y_max]
    with y_min=-1.0, y_max=0.0 by default

near-infinity extra:
    if near_infinity_extra:
        add points in [max(y_inf_min, -1+y_eps), y_inf_max]
        e.g. [-0.999, -0.95]

最后去重排序，避免重复点太多。

训练启动时必须打印：

[stage3] y_strategy = near_infinity_only
[stage3] base y-range = [...]
[stage3] near-inf y-range = [...]
[stage3] final y min/max = ...

------------------------------------------------------------
任务 4：支持 train_up / train_down 开关
------------------------------------------------------------

nearinf 报告里说 u_up 是主要误差来源。当前可以先训练 up/down 一起，也可以后续只训 up。

请在 train_autoencoder_stage3_ubasis.py 中支持 config：

stage3:
  train_up: true
  train_down: true

冻结策略：

- 先全部 requires_grad=False；
- if train_up: up_decoder requires_grad=True；
- if train_down: down_decoder requires_grad=True。

loss 也要对应：

if train_up and train_down:
    loss = loss_up + loss_down
elif train_up:
    loss = loss_up
elif train_down:
    loss = loss_down
else:
    raise error

第一轮 nearinf smoke 先保持 train_up=true, train_down=true。
如果 u_down 被破坏，再改成只 train_up。

------------------------------------------------------------
任务 5：新增 y-grid debug 脚本或扩展 debug
------------------------------------------------------------

新增或扩展：

scripts/debug_stage3_y_grid.py

用法：

python scripts/debug_stage3_y_grid.py \
  --config config/autoencoder_stage3_ubasis_nearinf.yaml

输出：

- y_strategy；
- y_eps；
- n_interior；
- n_near_infinity；
- base grid min/max；
- near-inf grid min/max；
- final grid min/max；
- final grid 点数；
- 前 10 个 y；
- 后 10 个 y。

通过标准：

对于 nearinf config，必须看到：

final y min ≈ -0.999
final y max ≈ 0.0

而不是 0.999。

------------------------------------------------------------
任务 6：运行 smoke 测试
------------------------------------------------------------

先运行：

python -m py_compile \
  scripts/train_autoencoder_stage3_ubasis.py \
  scripts/debug_stage3_y_grid.py

然后运行：

python scripts/debug_stage3_y_grid.py \
  --config config/autoencoder_stage3_ubasis_nearinf.yaml

确认 nearinf grid 真正是 y in [-0.999, 0.0]。

再运行原 Stage-3 debug：

python scripts/debug_autoencoder_stage3_ubasis.py \
  --config config/autoencoder_stage3_ubasis_nearinf.yaml \
  --checkpoint outputs/autoencoder_stage3_ubasis_refine/20260515_222026_stage3_ubasis/checkpoints/best_model.pt \
  --device cuda

注意：
checkpoint 使用目前 Stage-3 best checkpoint，而不是 Stage-2 checkpoint。

------------------------------------------------------------
任务 7：跑 200 epoch nearinf smoke，不要直接 2000
------------------------------------------------------------

如果 grid 和 debug 都通过，先跑 200 epoch：

python scripts/train_autoencoder_stage3_ubasis.py \
  --config config/autoencoder_stage3_ubasis_nearinf.yaml \
  --checkpoint outputs/autoencoder_stage3_ubasis_refine/20260515_222026_stage3_ubasis/checkpoints/best_model.pt \
  --device cuda \
  --epochs 200 \
  --verbose

记录：

- y_strategy；
- y range；
- train_up/train_down；
- epoch 1 loss_up/loss_down；
- epoch 50；
- epoch 100；
- epoch 200；
- 是否 NaN/Inf；
- grad_norm；
- best checkpoint 路径。

------------------------------------------------------------
任务 8：200 epoch 后立刻做谱方法对比
------------------------------------------------------------

用新的 nearinf best checkpoint 重新跑：

python scripts/compare_stage3_ubasis_with_spectral.py \
  --stage3-checkpoint <nearinf_200epoch_run>/checkpoints/best_model.pt \
  --config config/autoencoder_stage3_ubasis_nearinf.yaml \
  --device cuda \
  --a 0.1 \
  --omega 10.0 \
  --output-dir outputs/stage3_spectral_compare/patch_000_logw_v2_nearinf_200

注意：
参数必须和原 high_omega_debug/basis_functions.png 一致。如果原脚本中 lambda 不是默认值，要显式传入 --lam。

比较目标：

旧结果：
- median_rel_err_up = 1.28
- median_rel_err_down = 0.29

新结果至少应明显下降。
如果 200 epoch 后 up 没有改善，不要跑 2000，先分析 loss 和目标区间。

------------------------------------------------------------
任务 9：是否继续 2000 epoch 的判据
------------------------------------------------------------

可以继续 2000 epoch 的条件：

- median_rel_err_up 明显下降，例如从 1.28 降到 < 0.5；
- median_rel_err_down 不明显变坏；
- boundary exact check 仍为 0；
- loss finite；
- max|u_up| 不爆炸。

如果 u_up 改善而 u_down 变差，下一轮改成：

train_up: true
train_down: false

只 refine up_decoder。

如果 u_up 完全不改善，暂停训练，检查：
- spectral lambda 是否和 decoder 使用的一致；
- local chart u/v 是否传入正确；
- Stage-3 checkpoint 是否来自同一 patch；
- compare 脚本中 z/y 映射是否完全一致。

------------------------------------------------------------
任务 10：提交
------------------------------------------------------------

如果修复了训练脚本和 config：

git add scripts/train_autoencoder_stage3_ubasis.py scripts/debug_stage3_y_grid.py config/autoencoder_stage3_ubasis_nearinf.yaml
git commit -m "Enable true near-infinity Stage-3 training grid"
git push

大型 outputs 和图片不要提交。

最终报告必须包含：

1. 当前 commit hash；
2. nearinf config 修复内容；
3. y_strategy 是否真正生效；
4. debug_stage3_y_grid 输出；
5. 200 epoch smoke 的 loss_up/loss_down；
6. 200 epoch 后谱方法对比误差；
7. 是否建议继续 2000 epoch；
8. 是否建议只训练 up_decoder。