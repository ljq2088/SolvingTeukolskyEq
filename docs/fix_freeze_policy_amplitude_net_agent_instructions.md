# Agent 修正指令：正确理解 freeze_pinn，并修复 AmplitudeNet 只剩 86 个可训练参数的问题

## 0. 背景

当前训练报告显示：

- `freeze_pinn: true`
- “冻结 PINN 后仅 86 个可训练参数”
- 当前目标是 amplitude surrogate 训练
- `loss_amp` 在训练，但可训练参数数量异常偏小
- `PDE loss / integral loss` 仍在日志中出现，但由于 PINN 冻结，它们不能被优化

这个状态说明你很可能误解了“冻结 PINN”的意思。

**冻结 PINN 的意思不是只训练最后几个振幅标量，也不是把 amplitude modulation 相关的参数一起冻结。**

正确含义是：

> 冻结 \(y\)-dependent 的主 PINN 解网络，也就是负责输出 \(f(y)\) 的主体网络；但 amplitude surrogate 自己的参数编码器、latent tokens、FiLM/gated residual blocks、输出头必须全部可训练。

如果 `AmplitudeNet` 只剩 86 个参数可训练，说明至少存在以下问题之一：

1. `AmplitudeNet` 的大部分模块没有被正确注册成 `nn.Module` / `nn.ModuleList`；
2. 冻结逻辑只放开了 token 或最后一小层；
3. `AmplitudeNet` 复用了主 PINN 的 `param_encoder`，而这个 encoder 在 `freeze_pinn` 时被一起冻结；
4. `optimizer` 在设置 `requires_grad=True` 之前已经创建，导致 optimizer 没有拿到正确参数；
5. 代码实际训练的是旧 `amp_head` 或很小的 correction head，而不是新的 absolute `AmplitudeNet`。

---

## 1. 立即停止当前训练

当前 86 个可训练参数的训练没有意义，先停止：

```bash
pkill -f "train_multipatch_parallel.py"
```

或者手动停止当前进程。

---

## 2. 必须先打印可训练参数明细

在创建 optimizer 前，加入以下诊断函数。

建议放到 `trainer/atlas_patch_trainer.py` 里，作为 `AtlasPatchTrainer` 的方法：

```python
def _print_trainable_parameters(self, tag: str = ""):
    total = 0
    rows = []
    for name, p in self.model.named_parameters():
        if p.requires_grad:
            n = p.numel()
            total += n
            rows.append((name, tuple(p.shape), n))

    print("=" * 100)
    print(f"[trainable-params] {tag} total = {total}")
    for name, shape, n in rows:
        print(f"[trainable-params] {name:90s} {str(shape):30s} {n}")
    print("=" * 100)

    return total
```

在 optimizer 创建前调用：

```python
self._apply_freeze_policy()
n_trainable = self._print_trainable_parameters(tag="after freeze policy")

if self.inf_enabled and self.inf_amplitude_mode in ("learned_absolute", "hybrid_warmstart"):
    if n_trainable < 10000:
        raise RuntimeError(
            f"Amplitude surrogate has only {n_trainable} trainable parameters. "
            "This is too small. Check freeze logic and AmplitudeNet registration."
        )

self.optimizer = torch.optim.Adam(
    [p for p in self.model.parameters() if p.requires_grad],
    lr=self.lr,
)
```

注意：**optimizer 必须在 freeze policy 之后创建。**

---

## 3. 正确的 freeze policy

新增或修改 `AtlasPatchTrainer._apply_freeze_policy()`。

### 3.1 配置字段

在 `config/pinn_config_phase4_sgdr.yaml` 的 `atlas_training.infinity_asymptotic` 下使用：

```yaml
  infinity_asymptotic:
    enabled: true
    amplitude_mode: learned_absolute

    freeze_pinn: true
    train_amplitude_net: true
    train_old_amp_head: false
```

解释：

- `freeze_pinn: true`：冻结主 PINN 的 \(y\)-dependent 解网络；
- `train_amplitude_net: true`：训练新的 absolute amplitude surrogate；
- `train_old_amp_head: false`：不要训练旧的 spectral correction head，除非你明确要保留旧模式。

### 3.2 正确冻结逻辑

在 `trainer/atlas_patch_trainer.py` 中加入：

```python
def _apply_freeze_policy(self):
    """
    Correct freeze policy.

    freeze_pinn=True means:
      freeze the y-dependent PINN solver trunk.

    It does NOT mean:
      freeze amplitude_net.
      freeze amplitude_net.param_encoder.
      freeze amplitude_net FiLM/gated residual blocks.
      train only inc/ref tokens.
    """
    inf_cfg = self.atlas_train_cfg.get("infinity_asymptotic", {})
    freeze_pinn = bool(inf_cfg.get("freeze_pinn", False))
    train_amplitude_net = bool(inf_cfg.get("train_amplitude_net", True))
    train_old_amp_head = bool(inf_cfg.get("train_old_amp_head", False))

    if not freeze_pinn:
        # Normal joint training: everything trainable by default.
        for _, p in self.model.named_parameters():
            p.requires_grad = True
        return

    # Freeze everything first.
    for _, p in self.model.named_parameters():
        p.requires_grad = False

    # Unfreeze the new absolute amplitude surrogate.
    if train_amplitude_net:
        if not hasattr(self.model, "amplitude_net"):
            raise RuntimeError(
                "freeze_pinn=True and train_amplitude_net=True, "
                "but model has no amplitude_net."
            )

        for name, p in self.model.named_parameters():
            if name.startswith("amplitude_net."):
                p.requires_grad = True

    # Optional: unfreeze the old correction head.
    # Normally keep this false during learned_absolute pretraining.
    if train_old_amp_head:
        for name, p in self.model.named_parameters():
            if name.startswith("amp_head."):
                p.requires_grad = True

    # Safety check: if learned_absolute or hybrid_warmstart, amplitude_net must be trainable.
    if self.inf_amplitude_mode in ("learned_absolute", "hybrid_warmstart"):
        amp_trainable = sum(
            p.numel()
            for name, p in self.model.named_parameters()
            if name.startswith("amplitude_net.") and p.requires_grad
        )
        if amp_trainable < 10000:
            raise RuntimeError(
                f"amplitude_net trainable params = {amp_trainable}, too small. "
                "Expected at least O(1e4). Check ModuleList registration and freeze policy."
            )
```

然后在 `__init__` 中，创建 model 之后、创建 optimizer 之前调用：

```python
self._apply_freeze_policy()
self._print_trainable_parameters(tag="after freeze policy")

self.optimizer = torch.optim.Adam(
    [p for p in self.model.parameters() if p.requires_grad],
    lr=self.lr,
)
```

删除或替换原先的：

```python
self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
```

---

## 4. 检查 AmplitudeNet 是否正确注册

在 `model/pinn_mlp.py` 中，确认 `AmplitudeNet` 结构必须使用 `nn.ModuleList`，不能使用普通 Python list。

正确示例：

```python
self.inc_blocks = nn.ModuleList([
    ModulatedResidualBlock(hidden_dim, hidden_dim, activation=activation)
    for _ in range(n_blocks)
])

self.ref_blocks = nn.ModuleList([
    ModulatedResidualBlock(hidden_dim, hidden_dim, activation=activation)
    for _ in range(n_blocks)
])
```

错误示例：

```python
# 不要这样写，这样参数不会被 PyTorch 正确注册
self.inc_blocks = [
    ModulatedResidualBlock(hidden_dim, hidden_dim, activation=activation)
    for _ in range(n_blocks)
]
```

并确认 `AmplitudeNet` 作为 `PINN_MLP` 的成员注册：

```python
self.amplitude_net = AmplitudeNet(
    param_in_dim=self.param_in_dim,
    hidden_dim=128,
    n_blocks=3,
    activation=activation,
)
```

不要只写局部变量：

```python
# 错误
amplitude_net = AmplitudeNet(...)
```

---

## 5. 不要复用被冻结的主 PINN param_encoder

如果当前 `AmplitudeNet` 直接复用了主 PINN 的 `self.param_encoder`，请改掉。

错误思路：

```python
cond = self.param_encoder(param_feats)
```

如果 `freeze_pinn=True`，这个主 `param_encoder` 会被冻结。这样 amplitude surrogate 的调制信号也被冻结，表达力会被严重限制。

正确做法：

`AmplitudeNet` 内部必须有自己的参数编码器：

```python
self.param_encoder = nn.Sequential(
    nn.Linear(param_in_dim, hidden_dim),
    _make_activation(activation),
    nn.Linear(hidden_dim, hidden_dim),
    _make_activation(activation),
)
```

forward 中使用：

```python
cond = self.param_encoder(param_feats)
```

这里的 `self.param_encoder` 是 `AmplitudeNet` 自己的，不是 `PINN_MLP.param_encoder`。

---

## 6. AmplitudeNet 的最低结构要求

`AmplitudeNet` 至少应该类似：

```python
class AmplitudeNet(nn.Module):
    def __init__(self, param_in_dim=14, hidden_dim=128, n_blocks=3, activation="silu"):
        super().__init__()

        self.param_encoder = nn.Sequential(
            nn.Linear(param_in_dim, hidden_dim),
            _make_activation(activation),
            nn.Linear(hidden_dim, hidden_dim),
            _make_activation(activation),
        )

        self.inc_token = nn.Parameter(torch.zeros(1, hidden_dim))
        self.ref_token = nn.Parameter(torch.zeros(1, hidden_dim))

        self.inc_blocks = nn.ModuleList([
            ModulatedResidualBlock(hidden_dim, hidden_dim, activation=activation)
            for _ in range(n_blocks)
        ])
        self.ref_blocks = nn.ModuleList([
            ModulatedResidualBlock(hidden_dim, hidden_dim, activation=activation)
            for _ in range(n_blocks)
        ])

        self.out_inc = nn.Linear(hidden_dim, 2)
        self.out_ref = nn.Linear(hidden_dim, 2)
```

如果 `hidden_dim=128, n_blocks=3`，可训练参数应该是几十万量级，至少也应该远大于 10000。

如果打印出来仍然只有几十或几百，说明模块没有注册或冻结策略错误。

---

## 7. 重新定义 amplitude pretraining 的 loss

当前报告说：

- `PDE loss` 固定在约 21；
- `integral loss` 很大；
- 但 PINN 冻结无法优化它们；
- total loss 高主要来自 integral loss。

这说明 amplitude pretraining 阶段不应该把 PDE / integral loss 加入 total loss。

### 7.1 新增训练阶段字段

在 config 中加入：

```yaml
  infinity_asymptotic:
    stage: amplitude_pretrain
```

可选值：

```yaml
stage: amplitude_pretrain
stage: joint_hybrid
stage: joint_learned_absolute
```

### 7.2 amplitude_pretrain 阶段的 total_loss

在训练 loop 中，如果：

```python
self.inf_stage == "amplitude_pretrain"
```

则 total loss 必须只包含 amplitude teacher loss：

```python
total_loss = w_amp * loss_amp
```

不要加入：

```python
loss_pde
loss_integral
loss_inf
```

可以计算它们作为诊断，但不要参与 `total_loss`。

推荐逻辑：

```python
if self.inf_stage == "amplitude_pretrain":
    # Only train amplitude surrogate against spectral teacher.
    total_loss = w_amp * loss_amp

else:
    # Original joint training.
    total_loss = total_loss + w_inf * loss_inf + w_B * loss_B + w_amp * loss_amp
```

为了避免不必要计算，amplitude_pretrain 阶段甚至可以跳过 `pinn_residual_loss()` 和 `compute_integral_consistency_loss()`，只采样参数点并计算 `loss_amp`。

---

## 8. 修正配置：阶段一只训练振幅 surrogate

先使用下面配置：

```yaml
atlas_training:
  infinity_asymptotic:
    enabled: true
    stage: amplitude_pretrain
    amplitude_mode: learned_absolute

    freeze_pinn: true
    train_amplitude_net: true
    train_old_amp_head: false

    weight_amp_init: 1.0
    weight_amp_final: 1.0
    weight_amp_decay_start: 0
    weight_amp_decay_end: 1

    weight_inf_init: 0.0
    weight_inf_final: 0.0
    weight_B_init: 0.0
    weight_B_final: 0.0

    monitor:
      enabled: true
      every: 200
      r_points: [200.0, 300.0, 500.0, 800.0, 1000.0]
      pybhpt_timeout: 20.0
```

这一阶段只看：

```text
loss_amp
inf_amp_rel_Binc
inf_amp_rel_Bref
```

不要看 PDE / integral loss。

---

## 9. 阶段二：joint_hybrid

当 amplitude surrogate 能比较好拟合谱方法振幅后，再进入联合训练。

配置：

```yaml
atlas_training:
  infinity_asymptotic:
    enabled: true
    stage: joint_hybrid
    amplitude_mode: hybrid_warmstart

    freeze_pinn: false
    train_amplitude_net: true
    train_old_amp_head: false

    weight_inf_init: 0.05
    weight_inf_final: 0.2
    weight_inf_ramp_start: 1000
    weight_inf_ramp_end: 8000

    weight_amp_init: 1.0
    weight_amp_final: 0.1
    weight_amp_decay_start: 3000
    weight_amp_decay_end: 15000

    hybrid_eta_init: 0.0
    hybrid_eta_final: 1.0
    hybrid_eta_start: 5000
    hybrid_eta_end: 20000
```

此时：

\[
B_{\rm used}=(1-\eta)B_{\rm spec}+\eta B_{\rm pred}
\]

主 PINN 和 amplitude surrogate 一起训练。

---

## 10. 阶段三：joint_learned_absolute

最终阶段：

```yaml
atlas_training:
  infinity_asymptotic:
    enabled: true
    stage: joint_learned_absolute
    amplitude_mode: learned_absolute

    freeze_pinn: false
    train_amplitude_net: true
    train_old_amp_head: false
```

此时：

\[
B_{\rm used}=B_{\rm pred}
\]

训练和推理都不需要谱方法 runtime 振幅。谱方法只作为训练 teacher / validation reference。

---

## 11. 重新运行前必须通过的检查

### 11.1 语法检查

```bash
python -m py_compile model/pinn_mlp.py
python -m py_compile physical_ansatz/asymptotic_loss.py
python -m py_compile trainer/atlas_patch_trainer.py
```

### 11.2 短训练

```bash
python scripts/train_multipatch_parallel.py \
  --cfg config/pinn_config_phase4_sgdr.yaml \
  --patch_ids 1 \
  --max_parallel 1 \
  --steps 20
```

### 11.3 必须看到的参数明细

输出中必须看到类似：

```text
[trainable-params] amplitude_net.param_encoder.0.weight
[trainable-params] amplitude_net.param_encoder.0.bias
[trainable-params] amplitude_net.param_encoder.2.weight
[trainable-params] amplitude_net.param_encoder.2.bias
[trainable-params] amplitude_net.inc_token
[trainable-params] amplitude_net.ref_token
[trainable-params] amplitude_net.inc_blocks.0.*
[trainable-params] amplitude_net.ref_blocks.0.*
[trainable-params] amplitude_net.out_inc.*
[trainable-params] amplitude_net.out_ref.*
```

并且：

```text
trainable total > 10000
```

如果仍然是 86，禁止继续训练，必须修复注册或冻结逻辑。

---

## 12. 预期结论

正确实现后：

- amplitude_pretrain 阶段可训练参数不应是 86；
- PDE / integral loss 不应进入 total_loss；
- 主 PINN 可以冻结；
- `amplitude_net` 的参数编码器、tokens、调制 residual blocks、输出头必须全部训练；
- `loss_amp` 应该能下降；
- `inf_amp_rel_Binc` 和 `inf_amp_rel_Bref` 应该逐步下降；
- monitor 图会显示网络振幅构造的 \(R_{\rm amp}\) 与 pybhpt \(R_{\rm in}\) 的差距变化。

