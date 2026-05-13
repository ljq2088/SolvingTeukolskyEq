# Agent 实施文档：把 near-infinity 振幅从 spectral correction 升级为 learned absolute surrogate，并加入 PyBHPT 渐近振幅监测

## 0. 当前状态与目标

当前分支：

```bash
git checkout feature/infinity-asymptotic-loss
```

当前训练入口保持不变：

```bash
python scripts/train_multipatch_parallel.py \
  --cfg config/pinn_config_phase4_sgdr.yaml \
  --patch_ids 1 \
  --max_parallel 1
```

当前 `model/pinn_mlp.py` 中已经有一个 `amp_head`。它的作用是对谱方法振幅做乘法修正：

\[
dB=\exp(\rho+i\phi)
\]

\[
B_{\rm net}=B_{\rm spec}dB
\]

这意味着当前推理阶段仍然需要谱方法给出 \(B_{\rm spec}\)。本次任务要把它升级为：

**训练阶段：谱方法振幅只作为 teacher / hotstart / supervision。**

**推理阶段：模型直接输出**

\[
B^{\rm inc}_{\rm pred}(a,\omega),\qquad B^{\rm ref}_{\rm pred}(a,\omega)
\]

**不再调用谱方法。**

同时新增监测功能：

使用网络输出的振幅系数构造无穷远渐近表达式

\[
R_{\rm amp}(r)
=
B^{\rm inc}_{\rm pred} r^{-1}e^{-i\omega r_*}
+
B^{\rm ref}_{\rm pred} r^3e^{i\omega r_*}
\]

并将其与 `pybhpt` 计算的 \(R_{\rm in}(r)\) 在大半径区域作比较，保存图像和误差日志。

---

## 1. 总体设计

### 1.1 保留当前 correction head

当前已有的 `amp_head` 可以保留，作为兼容模式：

```yaml
amplitude_mode: spectral_correction
```

该模式仍然使用：

\[
B_{\rm used}=B_{\rm spec}dB
\]

### 1.2 新增 absolute amplitude surrogate

新增一个更强的振幅网络 `AmplitudeNet`：

\[
p(a,\omega,u,v)\rightarrow B^{\rm inc}_{\rm pred},B^{\rm ref}_{\rm pred}
\]

其中 \(p\) 是当前 `_build_param_features` 得到的 14 维参数特征。

不要只用普通小 MLP。推荐结构：

**参数编码器 + inc/ref amplitude latent tokens + FiLM/gated residual modulation + log-polar complex output**。

输出形式：

\[
B=\exp(\rho)e^{i\phi}
\]

也就是网络输出：

```python
[rho_inc, phi_inc, rho_ref, phi_ref]
```

而不是直接输出 Re/Im。这样更适合振幅模长跨数量级变化的情况。

---

## 2. 修改 `model/pinn_mlp.py`

### 2.1 新增 `ModulatedResidualBlock`

在 `FiLMBlock` 类之后、`PINN_MLP` 类之前加入：

```python
class ModulatedResidualBlock(nn.Module):
    """
    Gated FiLM residual block for parameter-conditioned amplitude tokens.

    h:    (B, hidden_dim)
    cond: (B, cond_dim)
    """
    def __init__(self, hidden_dim: int, cond_dim: int, activation: str = "silu"):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.gamma = nn.Linear(cond_dim, hidden_dim)
        self.beta = nn.Linear(cond_dim, hidden_dim)
        self.gate = nn.Linear(cond_dim, hidden_dim)

        self.act = _make_activation(activation)

    def forward(self, h: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        z = self.norm(h)

        gamma = 1.0 + 0.1 * torch.tanh(self.gamma(cond))
        beta = 0.1 * self.beta(cond)
        gate = torch.sigmoid(self.gate(cond))

        z = gamma * z + beta
        z = self.linear2(self.act(self.linear1(z)))

        return h + gate * z
```

### 2.2 新增 `AmplitudeNet`

仍然放在 `PINN_MLP` 类之前：

```python
class AmplitudeNet(nn.Module):
    """
    Absolute amplitude surrogate:
        p(a, omega, u, v) -> B_inc, B_ref

    Output representation:
        B = exp(rho) * exp(i phi)

    This network does NOT require B_spec at inference time.
    """
    def __init__(
        self,
        param_in_dim: int,
        hidden_dim: int = 128,
        n_blocks: int = 3,
        activation: str = "silu",
    ):
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

        self.out_inc = nn.Linear(hidden_dim, 2)  # [rho_inc, phi_inc]
        self.out_ref = nn.Linear(hidden_dim, 2)  # [rho_ref, phi_ref]

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.normal_(self.inc_token, mean=0.0, std=1.0e-3)
        nn.init.normal_(self.ref_token, mean=0.0, std=1.0e-3)

    def forward(self, param_feats: torch.Tensor):
        """
        Args:
            param_feats: (B, param_in_dim)

        Returns:
            Binc: complex tensor (B,)
            Bref: complex tensor (B,)
            raw: dict containing rho/phase tensors
        """
        B = param_feats.shape[0]
        cond = self.param_encoder(param_feats)

        h_inc = self.inc_token.expand(B, -1)
        h_ref = self.ref_token.expand(B, -1)

        for block in self.inc_blocks:
            h_inc = block(h_inc, cond)
        for block in self.ref_blocks:
            h_ref = block(h_ref, cond)

        out_inc = self.out_inc(h_inc)
        out_ref = self.out_ref(h_ref)

        rho_inc = out_inc[:, 0]
        phi_inc = out_inc[:, 1]
        rho_ref = out_ref[:, 0]
        phi_ref = out_ref[:, 1]

        cdtype = torch.complex128 if param_feats.dtype == torch.float64 else torch.complex64

        Binc = (torch.exp(rho_inc) * torch.exp(1j * phi_inc)).to(cdtype)
        Bref = (torch.exp(rho_ref) * torch.exp(1j * phi_ref)).to(cdtype)

        raw = {
            "rho_inc": rho_inc,
            "phi_inc": phi_inc,
            "rho_ref": rho_ref,
            "phi_ref": phi_ref,
        }
        return Binc, Bref, raw
```

### 2.3 在 `PINN_MLP.__init__` 中加入 `AmplitudeNet`

找到当前的 `amp_head` 定义附近：

```python
self.amp_head = nn.Sequential(
    nn.Linear(self.param_in_dim, 32),
    _make_activation(activation),
    nn.Linear(32, 4),
)
```

保留它，并在其后加入：

```python
# Absolute amplitude surrogate. This is used when
# infinity_asymptotic.amplitude_mode is learned_absolute or hybrid_warmstart.
self.amplitude_net = AmplitudeNet(
    param_in_dim=self.param_in_dim,
    hidden_dim=128,
    n_blocks=3,
    activation=activation,
)
```

### 2.4 新增 `predict_asymptotic_amplitudes`

在 `predict_asymptotic_delta()` 方法后加入：

```python
def predict_asymptotic_amplitudes(self, a, omega, u=None, v=None):
    """
    Directly predict absolute asymptotic amplitudes B_inc and B_ref.

    This function does NOT use spectral coefficients and is intended for final
    inference after training.
    """
    model_param = next(self.parameters())
    target_device = model_param.device
    target_dtype = model_param.dtype

    a = a.to(device=target_device, dtype=target_dtype)
    omega = omega.to(device=target_device, dtype=target_dtype)
    if u is not None:
        u = u.to(device=target_device, dtype=target_dtype)
    if v is not None:
        v = v.to(device=target_device, dtype=target_dtype)

    if a.ndim == 1:
        a = a.unsqueeze(-1)
    if omega.ndim == 1:
        omega = omega.unsqueeze(-1)
    if u is not None and u.ndim == 1:
        u = u.unsqueeze(-1)
    if v is not None and v.ndim == 1:
        v = v.unsqueeze(-1)

    feats, _, _ = self._build_param_features(a=a, omega=omega, u=u, v=v)
    return self.amplitude_net(feats)
```

---

## 3. 修改 `physical_ansatz/asymptotic_loss.py`

当前 `compute_infinity_asymptotic_loss()` 只支持：

\[
B_{\rm used}=B_{\rm spec}dB
\]

需要扩展为三种模式。

### 3.1 修改函数签名

把函数签名改成：

```python
def compute_infinity_asymptotic_loss(
    model,
    cfg: dict,
    a_batch: torch.Tensor,
    omega_batch: torch.Tensor,
    lambda_batch: torch.Tensor,
    u_batch: torch.Tensor,
    v_batch: torch.Tensor,
    Binc_spec: torch.Tensor,
    Bref_spec: torch.Tensor,
    r_points,
    beta: float = 1.0,
    relative: bool = True,
    eps: float = 1.0e-12,
    amplitude_mode: str = "spectral_correction",
    hybrid_eta: float = 0.0,
):
```

### 3.2 新增 helper：log-polar teacher loss

在文件中加入：

```python
def compute_amplitude_teacher_loss(
    Binc_pred: torch.Tensor,
    Bref_pred: torch.Tensor,
    amp_raw: dict,
    Binc_spec: torch.Tensor,
    Bref_spec: torch.Tensor,
    eps: float = 1.0e-12,
):
    """
    Supervise absolute amplitude predictor with spectral coefficients.

    Magnitude uses log-amplitude loss.
    Phase uses unit-complex phase loss to avoid 2pi wrapping.
    """
    dtype = amp_raw["rho_inc"].dtype
    cdtype = Binc_pred.dtype

    Binc_spec = Binc_spec.to(dtype=cdtype, device=Binc_pred.device)
    Bref_spec = Bref_spec.to(dtype=cdtype, device=Bref_pred.device)

    abs_inc = torch.abs(Binc_spec).clamp_min(eps)
    abs_ref = torch.abs(Bref_spec).clamp_min(eps)

    rho_inc_t = torch.log(abs_inc).to(dtype=dtype)
    rho_ref_t = torch.log(abs_ref).to(dtype=dtype)

    mag_loss = torch.mean(
        (amp_raw["rho_inc"] - rho_inc_t) ** 2
        + (amp_raw["rho_ref"] - rho_ref_t) ** 2
    )

    phase_inc_t = Binc_spec / abs_inc.to(dtype=cdtype)
    phase_ref_t = Bref_spec / abs_ref.to(dtype=cdtype)

    phase_inc_p = torch.exp(1j * amp_raw["phi_inc"]).to(dtype=cdtype)
    phase_ref_p = torch.exp(1j * amp_raw["phi_ref"]).to(dtype=cdtype)

    phase_loss = torch.mean(
        torch.abs(phase_inc_p - phase_inc_t) ** 2
        + torch.abs(phase_ref_p - phase_ref_t) ** 2
    )

    rel_inc = torch.mean(torch.abs(Binc_pred - Binc_spec) / abs_inc.clamp_min(eps))
    rel_ref = torch.mean(torch.abs(Bref_pred - Bref_spec) / abs_ref.clamp_min(eps))

    loss_amp = mag_loss + phase_loss

    info = {
        "loss_amp": float(loss_amp.detach().cpu().item()),
        "loss_amp_mag": float(mag_loss.detach().cpu().item()),
        "loss_amp_phase": float(phase_loss.detach().cpu().item()),
        "amp_rel_Binc": float(rel_inc.detach().cpu().item()),
        "amp_rel_Bref": float(rel_ref.detach().cpu().item()),
    }
    return loss_amp, info
```

### 3.3 在 `compute_infinity_asymptotic_loss()` 中替换 B 的构造逻辑

找到当前逻辑：

```python
dB_inc, dB_ref = model.predict_asymptotic_delta(...)
Binc_net = Binc_spec * dB_inc
Bref_net = Bref_spec * dB_ref
...
loss_B = ...
```

替换成：

```python
amplitude_mode = str(amplitude_mode).lower()
if amplitude_mode not in ("spectral_correction", "learned_absolute", "hybrid_warmstart"):
    raise ValueError(f"Unsupported amplitude_mode={amplitude_mode}")

Binc_spec = Binc_spec.to(device=device, dtype=cdtype)
Bref_spec = Bref_spec.to(device=device, dtype=cdtype)

loss_B = torch.zeros((), device=device, dtype=dtype)
loss_amp = torch.zeros((), device=device, dtype=dtype)
amp_info = {}

if amplitude_mode == "spectral_correction":
    dB_inc, dB_ref = model.predict_asymptotic_delta(
        a_batch, omega_batch, u=u_batch, v=v_batch
    )
    Binc_net = Binc_spec * dB_inc
    Bref_net = Bref_spec * dB_ref

    a_1 = a_batch if a_batch.ndim == 2 else a_batch.unsqueeze(-1)
    omega_1 = omega_batch if omega_batch.ndim == 2 else omega_batch.unsqueeze(-1)
    u_1 = u_batch if u_batch.ndim == 2 else u_batch.unsqueeze(-1)
    v_1 = v_batch if v_batch.ndim == 2 else v_batch.unsqueeze(-1)
    feats_reg, _, _ = model._build_param_features(a=a_1, omega=omega_1, u=u_1, v=v_1)
    raw_out = model.amp_head(feats_reg)
    loss_B = torch.mean(raw_out[:, 0] ** 2 + raw_out[:, 1] ** 2 + raw_out[:, 2] ** 2 + raw_out[:, 3] ** 2)

    amp_info = {
        "mean_abs_dBinc": float(torch.mean(torch.abs(dB_inc.detach())).cpu().item()),
        "mean_abs_dBref": float(torch.mean(torch.abs(dB_ref.detach())).cpu().item()),
    }

else:
    if not hasattr(model, "predict_asymptotic_amplitudes"):
        raise AttributeError("Model must implement predict_asymptotic_amplitudes.")

    Binc_pred, Bref_pred, amp_raw = model.predict_asymptotic_amplitudes(
        a_batch, omega_batch, u=u_batch, v=v_batch
    )

    loss_amp, amp_info = compute_amplitude_teacher_loss(
        Binc_pred=Binc_pred,
        Bref_pred=Bref_pred,
        amp_raw=amp_raw,
        Binc_spec=Binc_spec,
        Bref_spec=Bref_spec,
        eps=eps,
    )

    if amplitude_mode == "learned_absolute":
        Binc_net = Binc_pred
        Bref_net = Bref_pred
    else:
        eta = float(hybrid_eta)
        eta = max(0.0, min(1.0, eta))
        Binc_net = (1.0 - eta) * Binc_spec + eta * Binc_pred
        Bref_net = (1.0 - eta) * Bref_spec + eta * Bref_pred
```

### 3.4 修改返回值

函数最后应返回：

```python
info.update(amp_info)
return loss_inf, loss_B, loss_amp, info
```

因此函数返回从原来的 3 个值变成 4 个值：

```python
loss_inf, loss_B, loss_amp, info
```

---

## 4. 修改 `trainer/atlas_patch_trainer.py`

### 4.1 读取新增 config

在读取 `infinity_asymptotic` 配置处，加入：

```python
self.inf_amplitude_mode = str(inf_cfg.get("amplitude_mode", "spectral_correction")).lower()

self.inf_hybrid_eta_init = float(inf_cfg.get("hybrid_eta_init", 0.0))
self.inf_hybrid_eta_final = float(inf_cfg.get("hybrid_eta_final", 1.0))
self.inf_hybrid_eta_start = int(inf_cfg.get("hybrid_eta_start", 5000))
self.inf_hybrid_eta_end = int(inf_cfg.get("hybrid_eta_end", 20000))

self.inf_weight_amp_init = float(inf_cfg.get("weight_amp_init", 1.0))
self.inf_weight_amp_final = float(inf_cfg.get("weight_amp_final", 0.1))
self.inf_weight_amp_decay_start = int(inf_cfg.get("weight_amp_decay_start", 3000))
self.inf_weight_amp_decay_end = int(inf_cfg.get("weight_amp_decay_end", 15000))
```

### 4.2 增加 schedule helper

在 `_infinity_loss_weights()` 后加入：

```python
def _infinity_hybrid_eta(self) -> float:
    return self._linear_schedule(
        self.global_step,
        self.inf_hybrid_eta_start,
        self.inf_hybrid_eta_end,
        self.inf_hybrid_eta_init,
        self.inf_hybrid_eta_final,
    )

def _infinity_amp_weight(self) -> float:
    return self._linear_schedule(
        self.global_step,
        self.inf_weight_amp_decay_start,
        self.inf_weight_amp_decay_end,
        self.inf_weight_amp_init,
        self.inf_weight_amp_final,
    )
```

### 4.3 修改训练 step 中对 `compute_infinity_asymptotic_loss` 的调用

当前大概率是：

```python
loss_inf, loss_B, inf_info = compute_infinity_asymptotic_loss(...)
total_loss = total_loss + w_inf * loss_inf + w_B * loss_B
```

替换为：

```python
loss_inf = torch.zeros((), device=self.device, dtype=self.dtype)
loss_B = torch.zeros((), device=self.device, dtype=self.dtype)
loss_amp = torch.zeros((), device=self.device, dtype=self.dtype)
w_inf = 0.0
w_B = 0.0
w_amp = 0.0
hybrid_eta = 0.0
inf_info = {}

if self.inf_enabled:
    if self.inf_coeff_cache is None:
        raise RuntimeError("self.inf_enabled=True but self.inf_coeff_cache is None.")

    Binc_spec, Bref_spec = self.inf_coeff_cache.get_batch(
        a_batch=a_batch,
        omega_batch=omega_batch,
        lambda_batch=lambda_batch,
    )

    hybrid_eta = self._infinity_hybrid_eta()

    loss_inf, loss_B, loss_amp, inf_info = compute_infinity_asymptotic_loss(
        model=self.model,
        cfg=self.physics_cfg,
        a_batch=a_batch,
        omega_batch=omega_batch,
        lambda_batch=lambda_batch,
        u_batch=u_batch,
        v_batch=v_batch,
        Binc_spec=Binc_spec,
        Bref_spec=Bref_spec,
        r_points=self.inf_r_points,
        beta=self.inf_beta,
        relative=self.inf_relative,
        eps=self.inf_eps,
        amplitude_mode=self.inf_amplitude_mode,
        hybrid_eta=hybrid_eta,
    )

    w_inf, w_B = self._infinity_loss_weights()
    w_amp = self._infinity_amp_weight()

    total_loss = total_loss + w_inf * loss_inf + w_B * loss_B + w_amp * loss_amp
```

### 4.4 修改日志

在 history dict 中加入：

```python
"loss_inf": float(loss_inf.detach().cpu().item()),
"loss_B": float(loss_B.detach().cpu().item()),
"loss_amp": float(loss_amp.detach().cpu().item()),
"weight_inf": float(w_inf),
"weight_B": float(w_B),
"weight_amp": float(w_amp),
"hybrid_eta": float(hybrid_eta),
"amplitude_mode": self.inf_amplitude_mode,
```

并把 `inf_info` 展开写入：

```python
for k, v in inf_info.items():
    record[f"inf_{k}"] = v
```

---

## 5. 修改 `config/pinn_config_phase4_sgdr.yaml`

在 `atlas_training.infinity_asymptotic` 下加入这些字段：

```yaml
  infinity_asymptotic:
    enabled: true

    # Current stable mode:
    #   spectral_correction : B_used = B_spec * dB
    # Final target mode:
    #   learned_absolute    : B_used = B_pred
    # Transition mode:
    #   hybrid_warmstart    : B_used = (1-eta) B_spec + eta B_pred
    amplitude_mode: hybrid_warmstart

    r_points: [300.0, 500.0, 800.0, 1000.0]
    beta: 1.0
    relative: true
    eps: 1.0e-12

    spectral_N: 64
    spectral_z_m: 0.3
    cache_file: outputs/domain/spectral_coeff_cache_l2_m2.json

    weight_inf_init: 0.05
    weight_inf_final: 0.2
    weight_inf_ramp_start: 1000
    weight_inf_ramp_end: 8000

    # Only used in spectral_correction mode.
    weight_B_init: 1.0
    weight_B_final: 0.1
    weight_B_decay_start: 3000
    weight_B_decay_end: 12000

    # Used in learned_absolute and hybrid_warmstart modes.
    weight_amp_init: 1.0
    weight_amp_final: 0.1
    weight_amp_decay_start: 3000
    weight_amp_decay_end: 15000

    # Hybrid teacher forcing schedule.
    # eta=0: use spectral amplitudes
    # eta=1: use learned amplitudes
    hybrid_eta_init: 0.0
    hybrid_eta_final: 1.0
    hybrid_eta_start: 5000
    hybrid_eta_end: 20000
```

推荐初期使用：

```yaml
amplitude_mode: hybrid_warmstart
```

等确认稳定后再改成：

```yaml
amplitude_mode: learned_absolute
```

---

## 6. 新增监测：用网络振幅构造 \(R_{\rm amp}\)，和 pybhpt \(R_{\rm in}\) 比较

### 6.1 新建 `utils/asymptotic_amplitude_monitor.py`

创建文件：

```bash
touch utils/asymptotic_amplitude_monitor.py
```

写入：

```python
from __future__ import annotations

from pathlib import Path
import math
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils.mode import KerrMode
from utils.amplitude import r_star
from pybhpt_usage.compute_solution import compute_pybhpt_solution


def _complex_rel_error(a: np.ndarray, b: np.ndarray, eps: float = 1.0e-30):
    return np.abs(a - b) / np.maximum(np.abs(b), eps)


def _best_complex_scale(y: np.ndarray, x: np.ndarray, eps: float = 1.0e-30) -> complex:
    """
    Find c minimizing ||c*x - y||_2.
    Useful because pybhpt and spectral normalization may differ by a constant.
    """
    denom = np.vdot(x, x)
    if abs(denom) < eps:
        return 1.0 + 0.0j
    return np.vdot(x, y) / denom


def compute_R_asym_from_B(
    r_grid: np.ndarray,
    Binc: complex,
    Bref: complex,
    M: float,
    a: float,
    omega: float,
    ell: int,
    m: int,
    s: int = -2,
    lam: complex | None = None,
):
    mode = KerrMode(M=M, a=a, omega=omega, ell=ell, m=m, lam=lam, s=s)
    r_grid = np.asarray(r_grid, dtype=np.float64)
    rs = r_star(r_grid, mode)
    return (
        Binc * r_grid ** (-1.0) * np.exp(-1j * omega * rs)
        + Bref * r_grid ** 3.0 * np.exp(+1j * omega * rs)
    )


@torch.no_grad()
def monitor_network_amplitudes_vs_pybhpt(
    model,
    physics_cfg: dict,
    a: float,
    omega: float,
    u: float,
    v: float,
    lam: complex | None,
    out_path: str | Path,
    r_points=None,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
    pybhpt_timeout: float = 20.0,
):
    """
    Compare network-predicted asymptotic R_amp with pybhpt R_in.

    R_amp(r) =
        B_inc_pred r^{-1} exp(-i omega r_*)
      + B_ref_pred r^3    exp(+i omega r_*)

    The function saves a diagnostic plot and returns a metrics dict.

    Note:
        Raw comparison is shown.
        A best-fit complex scaled comparison is also shown, because pybhpt and
        spectral conventions may differ by an overall constant.
    """
    problem_cfg = physics_cfg["problem"]
    M = float(problem_cfg.get("M", 1.0))
    s = int(problem_cfg.get("s", -2))
    ell = int(problem_cfg.get("l", problem_cfg.get("ell", 2)))
    m = int(problem_cfg.get("m", 2))

    if r_points is None:
        r_points = np.array([200.0, 300.0, 500.0, 800.0, 1000.0], dtype=np.float64)
    else:
        r_points = np.asarray(r_points, dtype=np.float64)

    dev = torch.device(device)

    a_t = torch.tensor([a], device=dev, dtype=dtype)
    omega_t = torch.tensor([omega], device=dev, dtype=dtype)
    u_t = torch.tensor([u], device=dev, dtype=dtype)
    v_t = torch.tensor([v], device=dev, dtype=dtype)

    if not hasattr(model, "predict_asymptotic_amplitudes"):
        raise AttributeError("model has no predict_asymptotic_amplitudes method.")

    Binc_t, Bref_t, amp_raw = model.predict_asymptotic_amplitudes(
        a_t, omega_t, u=u_t, v=v_t
    )

    Binc = complex(Binc_t.detach().cpu().numpy()[0])
    Bref = complex(Bref_t.detach().cpu().numpy()[0])

    R_amp = compute_R_asym_from_B(
        r_grid=r_points,
        Binc=Binc,
        Bref=Bref,
        M=M,
        a=a,
        omega=omega,
        ell=ell,
        m=m,
        s=s,
        lam=lam,
    )

    _, R_py = compute_pybhpt_solution(
        a=a,
        omega=omega,
        ell=ell,
        m=m,
        r_grid=r_points,
        timeout=pybhpt_timeout,
    )

    raw_rel = _complex_rel_error(R_amp, R_py)

    c_fit = _best_complex_scale(R_py, R_amp)
    R_amp_scaled = c_fit * R_amp
    scaled_rel = _complex_rel_error(R_amp_scaled, R_py)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(4, 1, figsize=(10, 13), sharex=True)

    axes[0].plot(r_points, np.abs(R_py), "o-", label="|R_in| pybhpt")
    axes[0].plot(r_points, np.abs(R_amp), "s--", label="|R_amp| raw")
    axes[0].plot(r_points, np.abs(R_amp_scaled), "d--", label="|c R_amp| scaled")
    axes[0].set_ylabel("abs")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(r_points, R_py.real, "o-", label="Re pybhpt")
    axes[1].plot(r_points, R_amp.real, "s--", label="Re R_amp raw")
    axes[1].plot(r_points, R_amp_scaled.real, "d--", label="Re c R_amp")
    axes[1].set_ylabel("real")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    axes[2].plot(r_points, R_py.imag, "o-", label="Im pybhpt")
    axes[2].plot(r_points, R_amp.imag, "s--", label="Im R_amp raw")
    axes[2].plot(r_points, R_amp_scaled.imag, "d--", label="Im c R_amp")
    axes[2].set_ylabel("imag")
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    axes[3].semilogy(r_points, raw_rel, "s--", label="raw rel error")
    axes[3].semilogy(r_points, scaled_rel, "d--", label="scaled rel error")
    axes[3].set_xlabel("r")
    axes[3].set_ylabel("relative error")
    axes[3].legend()
    axes[3].grid(alpha=0.3)

    fig.suptitle(
        f"Network amplitude asymptotic monitor vs pybhpt\\n"
        f"a={a:.8g}, omega={omega:.8g}, ell={ell}, m={m}\\n"
        f"Binc={Binc:.4e}, Bref={Bref:.4e}, c_fit={c_fit:.4e}"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)

    metrics = {
        "a": float(a),
        "omega": float(omega),
        "u": float(u),
        "v": float(v),
        "Binc_re": float(Binc.real),
        "Binc_im": float(Binc.imag),
        "Bref_re": float(Bref.real),
        "Bref_im": float(Bref.imag),
        "scale_fit_re": float(c_fit.real),
        "scale_fit_im": float(c_fit.imag),
        "raw_rel_mean": float(np.mean(raw_rel)),
        "raw_rel_max": float(np.max(raw_rel)),
        "scaled_rel_mean": float(np.mean(scaled_rel)),
        "scaled_rel_max": float(np.max(scaled_rel)),
        "plot_path": str(out_path),
    }
    return metrics
```

### 6.2 在 `trainer/atlas_patch_trainer.py` 中导入 monitor

顶部加入：

```python
from utils.asymptotic_amplitude_monitor import monitor_network_amplitudes_vs_pybhpt
```

### 6.3 读取 monitor config

在 `__init__` 的 `infinity_asymptotic` 配置读取部分加入：

```python
mon_cfg = inf_cfg.get("monitor", {})
self.inf_monitor_enabled = bool(mon_cfg.get("enabled", False))
self.inf_monitor_every = int(mon_cfg.get("every", self.viz_every))
self.inf_monitor_r_points = list(mon_cfg.get("r_points", [200.0, 300.0, 500.0, 800.0, 1000.0]))
self.inf_monitor_pybhpt_timeout = float(mon_cfg.get("pybhpt_timeout", self.viz_pybhpt_timeout))
```

### 6.4 新增 trainer 方法 `_run_asymptotic_amplitude_monitor`

在 `AtlasPatchTrainer` 类中加入：

```python
def _run_asymptotic_amplitude_monitor(self):
    if not self.inf_monitor_enabled:
        return None
    if not hasattr(self.model, "predict_asymptotic_amplitudes"):
        return None

    sample = self.ref_sample
    a = float(sample["a"].detach().cpu().item())
    omega = float(sample["omega"].detach().cpu().item())
    u = float(sample["u"].detach().cpu().item())
    v = float(sample["v"].detach().cpu().item())

    lam_t = get_lambda_from_cfg(
        self.physics_cfg,
        self.cache,
        sample["a"],
        sample["omega"],
    )
    lam = complex(lam_t.detach().cpu().item())

    out_path = self.fig_dir / f"asymptotic_amp_vs_pybhpt_step_{self.global_step:07d}.png"

    try:
        metrics = monitor_network_amplitudes_vs_pybhpt(
            model=self.model,
            physics_cfg=self.physics_cfg,
            a=a,
            omega=omega,
            u=u,
            v=v,
            lam=lam,
            out_path=out_path,
            r_points=self.inf_monitor_r_points,
            device=self.device,
            dtype=self.dtype,
            pybhpt_timeout=self.inf_monitor_pybhpt_timeout,
        )

        log_path = self.log_dir / "asymptotic_amp_monitor.jsonl"
        with open(log_path, "a", encoding="utf-8") as f:
            row = {"step": int(self.global_step), **metrics}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

        self._vprint(
            f"[amp-monitor] step={self.global_step} "
            f"raw_rel_mean={metrics['raw_rel_mean']:.3e}, "
            f"scaled_rel_mean={metrics['scaled_rel_mean']:.3e}, "
            f"plot={out_path}"
        )
        return metrics
    except Exception as e:
        self._vprint(f"[amp-monitor] failed at step={self.global_step}: {e}")
        return None
```

### 6.5 在训练 loop 中调用 monitor

在已有 visualization 调用附近加入：

```python
if (
    self.inf_monitor_enabled
    and self.global_step > 0
    and self.global_step % self.inf_monitor_every == 0
):
    self._run_asymptotic_amplitude_monitor()
```

要求：不要放在 `total_loss.backward()` 前，不参与梯度，只作为监测。

### 6.6 在 config 中加入 monitor 配置

在 `infinity_asymptotic` 下加入：

```yaml
    monitor:
      enabled: true
      every: 200
      r_points: [200.0, 300.0, 500.0, 800.0, 1000.0]
      pybhpt_timeout: 20.0
```

---

## 7. 建议训练流程

### 阶段 A：训练振幅 surrogate

先设置：

```yaml
amplitude_mode: learned_absolute
weight_amp_init: 1.0
weight_amp_final: 1.0
weight_inf_init: 0.0
weight_inf_final: 0.0
```

跑短训练，确认 `inf_amp_rel_Binc` 和 `inf_amp_rel_Bref` 下降。

### 阶段 B：hybrid warmstart

设置：

```yaml
amplitude_mode: hybrid_warmstart
hybrid_eta_init: 0.0
hybrid_eta_final: 1.0
hybrid_eta_start: 5000
hybrid_eta_end: 20000
```

此时：

\[
B_{\rm used}=(1-\eta)B_{\rm spec}+\eta B_{\rm pred}
\]

早期靠谱方法稳定边界，后期逐渐切换为网络振幅。

### 阶段 C：最终 inference 模式

训练稳定后设置：

```yaml
amplitude_mode: learned_absolute
```

此时无穷远边界完全使用网络输出振幅，不再依赖谱方法振幅作为 runtime 输入。

---

## 8. 测试命令

### 8.1 语法检查

```bash
python -m py_compile model/pinn_mlp.py
python -m py_compile physical_ansatz/asymptotic_loss.py
python -m py_compile utils/asymptotic_amplitude_monitor.py
python -m py_compile trainer/atlas_patch_trainer.py
```

### 8.2 短训练测试

```bash
python scripts/train_multipatch_parallel.py \
  --cfg config/pinn_config_phase4_sgdr.yaml \
  --patch_ids 1 \
  --max_parallel 1 \
  --steps 20
```

检查：

1. 不报错；
2. `history.jsonl` 中出现：
   - `loss_inf`
   - `loss_B`
   - `loss_amp`
   - `weight_inf`
   - `weight_B`
   - `weight_amp`
   - `hybrid_eta`
   - `inf_amp_rel_Binc`
   - `inf_amp_rel_Bref`
3. `logs/asymptotic_amp_monitor.jsonl` 能在 monitor step 后生成；
4. `figures/asymptotic_amp_vs_pybhpt_step_*.png` 能生成。

---

## 9. 验收标准

完成后必须满足：

1. 原始 `PINN_MLP.forward(a, omega, y, u, v)` 不变；
2. 当前 `amp_head` 可以保留，但最终独立振幅由 `AmplitudeNet` 输出；
3. 新增 `predict_asymptotic_amplitudes()`，不依赖 `B_spec`；
4. `learned_absolute` 模式下，\(\mathcal L_\infty\) 使用 \(B_{\rm pred}\)，不是 \(B_{\rm spec}dB\)；
5. `hybrid_warmstart` 模式下，使用：
   \[
   B_{\rm used}=(1-\eta)B_{\rm spec}+\eta B_{\rm pred}
   \]
6. 谱方法只作为 training teacher 和 cache 使用；
7. 新增 monitor 可以把：
   \[
   B^{\rm inc}_{\rm pred}r^{-1}e^{-i\omega r_*}
   +
   B^{\rm ref}_{\rm pred}r^3e^{i\omega r_*}
   \]
   与 pybhpt 的 \(R_{\rm in}\) 比较，并保存图像与误差日志；
8. 推理阶段可以直接调用：
   ```python
   Binc_pred, Bref_pred, raw = model.predict_asymptotic_amplitudes(a, omega, u, v)
   ```
   不再调用 `SpectralCoeffCache`。
