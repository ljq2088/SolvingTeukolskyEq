# Agent 任务说明：将谱方法振幅系数改为 per-parameter baseline，而不是一次性初始化

## 0. 背景与目标

当前计划是在 PINN 训练 Teukolsky 径向解时，对 \(y=-1\) 无穷远端加入渐近约束。上一版方案中提到添加 amplitude head 输出 \(B^{\rm inc}\) 与 \(B^{\rm ref}\)。这里需要进一步修正一个关键点：

**谱方法计算得到的 \(B^{\rm inc}_{\rm spec}\)、\(B^{\rm ref}_{\rm spec}\) 不能只在训练开始时使用一次。**

正确做法是：

\[
B^{\rm inc}_{\rm net}(a,\omega)
=
B^{\rm inc}_{\rm spec}(a,\omega)
+
s_{\rm inc}(a,\omega)\,\delta B^{\rm inc}_{\theta}(a,\omega)
\]

\[
B^{\rm ref}_{\rm net}(a,\omega)
=
B^{\rm ref}_{\rm spec}(a,\omega)
+
s_{\rm ref}(a,\omega)\,\delta B^{\rm ref}_{\theta}(a,\omega)
\]

其中：

\[
s_{\rm inc}(a,\omega)=|B^{\rm inc}_{\rm spec}(a,\omega)|+\epsilon
\]

\[
s_{\rm ref}(a,\omega)=|B^{\rm ref}_{\rm spec}(a,\omega)|+\epsilon
\]

也就是说，**谱方法系数是每个参数点 \((a,\omega)\) 的 teacher/baseline，网络只学习 correction**。

因此，agent 必须按本文档修改上一版实现方案。

---

## 1. 必须避免的错误做法

不要只在 patch 中心或训练开始时算一次谱方法系数，然后将 amplitude head 的 bias 初始化为这个固定值。

错误逻辑示例：

```python
# 不要这样做
Binc0, Bref0 = compute_spectral_coeff_at_patch_center()
model.amp_head[-1].bias[:] = [Binc0.real, Binc0.imag, Bref0.real, Bref0.imag]
```

这个做法只对一个参数点有效。patch 内的其他 \((a,\omega)\) 会被错误热启动。

---

## 2. 正确总体逻辑

每次训练 step 中，对于当前 batch：

```python
a_batch, omega_batch, u_batch, v_batch = self.sample_param_batch()
lambda_batch = self.resolve_aux_batch(a_batch, omega_batch)

Binc_spec, Bref_spec = self.get_spectral_coeff_batch(
    a_batch=a_batch,
    omega_batch=omega_batch,
    lambda_batch=lambda_batch,
)

loss_inf, loss_B, inf_info = compute_infinity_asymptotic_loss(
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
)
```

关键点：

1. `Binc_spec` 和 `Bref_spec` 是当前 batch 每个参数点自己的谱方法系数；
2. amplitude head 输出的是 `dB_inc` 和 `dB_ref`；
3. 真正进入无穷远渐近公式的是：

```python
Binc_net = Binc_spec + scale_inc * dB_inc
Bref_net = Bref_spec + scale_ref * dB_ref
```

其中：

```python
scale_inc = torch.abs(Binc_spec.detach()).clamp_min(eps)
scale_ref = torch.abs(Bref_spec.detach()).clamp_min(eps)
```

---

## 3. 修改 `model/pinn_mlp.py`

### 3.1 在 `PINN_MLP.__init__` 中添加 amplitude correction head

位置：在已有输出头附近。当前大致有：

```python
self.base_head = nn.Linear(self.fusion_dim, 2)
self.a_head = nn.Linear(self.fusion_dim, 2)
self.omega_head = nn.Linear(self.fusion_dim, 2)
self.nl_head = nn.Linear(self.fusion_dim, 2)
```

紧接着添加：

```python
# =========================================================
# 5) Infinity amplitude correction head
# =========================================================
# This head does NOT predict absolute B_inc/B_ref.
# It predicts dimensionless corrections dB_inc/dB_ref around
# the per-parameter spectral baseline:
#   B_net = B_spec + |B_spec| * dB
#
# Input: parameter features only, no y-coordinate.
# Output: [Re dB_inc, Im dB_inc, Re dB_ref, Im dB_ref]
self.amp_head = nn.Sequential(
    nn.Linear(self.param_in_dim, 32),
    _make_activation(activation),
    nn.Linear(32, 4),
)
```

### 3.2 在 `__init__` 末尾零初始化 amplitude head 最后一层

找到：

```python
self._init_weights()
```

紧接着添加：

```python
self._init_amp_head()
```

然后在 `PINN_MLP` 类中添加方法：

```python
def _init_amp_head(self):
    """
    Initialize amplitude correction head so that dB_inc=dB_ref=0 at start.
    Therefore B_net(a,omega)=B_spec(a,omega) for every parameter point.
    """
    if not hasattr(self, "amp_head"):
        return
    last = self.amp_head[-1]
    if isinstance(last, nn.Linear):
        nn.init.zeros_(last.weight)
        if last.bias is not None:
            nn.init.zeros_(last.bias)
```

### 3.3 添加 `predict_asymptotic_delta`

在 `PINN_MLP` 类中，放在 `forward` 之前或之后都可以。建议放在 `forward` 之前：

```python
def predict_asymptotic_delta(self, a, omega, u=None, v=None):
    """
    Predict dimensionless amplitude corrections.

    Args:
        a:     (B,) or (B,1)
        omega: (B,) or (B,1)
        u:     (B,) or (B,1), required when local_coord_mode='chart_uv'
        v:     (B,) or (B,1), required when local_coord_mode='chart_uv'

    Returns:
        dB_inc: (B,) complex tensor
        dB_ref: (B,) complex tensor

    Important:
        These are NOT absolute amplitudes.
        They are corrections used as:
            B_inc_net = B_inc_spec + |B_inc_spec| * dB_inc
            B_ref_net = B_ref_spec + |B_ref_spec| * dB_ref
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

    param_feats, _, _ = self._build_param_features(
        a=a,
        omega=omega,
        u=u,
        v=v,
    )
    out = self.amp_head(param_feats)

    dB_inc = torch.complex(out[:, 0], out[:, 1])
    dB_ref = torch.complex(out[:, 2], out[:, 3])
    return dB_inc, dB_ref
```

---

## 4. 新增 `physical_ansatz/infinity_asymptotic.py`

创建新文件：

```text
physical_ansatz/infinity_asymptotic.py
```

写入以下完整代码：

```python
from __future__ import annotations

import torch

from physical_ansatz.mapping import r_plus
from physical_ansatz.prefactor import Leaver_prefactors, build_prefactor_primitives, r_star
from physical_ansatz.transform_y import (
    h_factor,
    horizon_regularity_slope,
    compose_reduced_shape_from_f,
)


def _as_complex_dtype(real_dtype: torch.dtype) -> torch.dtype:
    if real_dtype == torch.float32:
        return torch.complex64
    return torch.complex128


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
):
    """
    Near-infinity asymptotic loss.

    This function uses per-parameter spectral coefficients as baselines:
        Binc_net = Binc_spec + |Binc_spec| * dBinc
        Bref_net = Bref_spec + |Bref_spec| * dBref

    The model only learns dBinc and dBref. The spectral coefficients are used
    for every parameter point in every batch.

    Returns:
        loss_inf: asymptotic S-matching loss near y=-1
        loss_B: regularization of amplitude correction
        info: python-float diagnostics
    """
    device = a_batch.device
    dtype = a_batch.dtype
    cdtype = _as_complex_dtype(dtype)

    problem_cfg = cfg["problem"]
    M = float(problem_cfg.get("M", 1.0))
    s = int(problem_cfg.get("s", -2))
    m = int(problem_cfg.get("m", 2))

    Binc_spec = Binc_spec.to(device=device, dtype=cdtype)
    Bref_spec = Bref_spec.to(device=device, dtype=cdtype)

    r_grid = torch.as_tensor(r_points, device=device, dtype=dtype)
    if r_grid.ndim != 1:
        raise ValueError("r_points must be a 1D list/array/tensor.")
    if torch.any(r_grid <= 0):
        raise ValueError("All r_points must be positive.")

    B = a_batch.shape[0]
    Nr = r_grid.numel()

    # Build batchwise r, x, y.
    rp = r_plus(a_batch, M=M)                       # (B,)
    r = r_grid.unsqueeze(0).expand(B, Nr)           # (B,Nr)
    x = rp.unsqueeze(-1) / r                        # (B,Nr)
    y = 2.0 * x - 1.0                               # (B,Nr)

    # Network f(y), then current ansatz S(y).
    f_pred = model(a_batch, omega_batch, y, u=u_batch, v=v_batch)
    slope = horizon_regularity_slope(
        a=a_batch,
        omega=omega_batch,
        lambda_=lambda_batch,
        m=m,
        M=M,
        s=s,
    )
    S_pred = compose_reduced_shape_from_f(
        f=f_pred,
        y=y,
        slope=slope,
    )

    # Amplitude correction head.
    if not hasattr(model, "predict_asymptotic_delta"):
        raise AttributeError(
            "model must implement predict_asymptotic_delta(a, omega, u, v)."
        )
    dBinc, dBref = model.predict_asymptotic_delta(
        a_batch,
        omega_batch,
        u=u_batch,
        v=v_batch,
    )

    scale_inc = torch.abs(Binc_spec.detach()).clamp_min(eps)
    scale_ref = torch.abs(Bref_spec.detach()).clamp_min(eps)

    Binc_net = Binc_spec + scale_inc * dBinc
    Bref_net = Bref_spec + scale_ref * dBref

    # Asymptotic R at infinity:
    # R_inf = B_inc r^{-1} exp(-i omega r_*) + B_ref r^3 exp(+i omega r_*).
    rs = r_star(r, a_batch.unsqueeze(-1), M=M)
    omega_b = omega_batch.unsqueeze(-1)

    R_inf = (
        Binc_net.unsqueeze(-1) * r.pow(-1.0) * torch.exp(-1j * omega_b * rs)
        + Bref_net.unsqueeze(-1) * r.pow(3.0) * torch.exp(+1j * omega_b * rs)
    )

    # Current convention: R = P * h2 * S.
    # Therefore S_inf = R_inf / (P*h2).
    rp_pref, rm_pref, _, _, _ = build_prefactor_primitives(
        r,
        a_batch.unsqueeze(-1),
        M=M,
        need_rs=False,
    )
    P, _, _ = Leaver_prefactors(
        r,
        a_batch.unsqueeze(-1),
        omega_batch.unsqueeze(-1),
        m=m,
        M=M,
        s=s,
        rp=rp_pref,
        rm=rm_pref,
    )
    h2 = h_factor(a_batch, omega_batch, m=m, M=M, s=s).unsqueeze(-1)
    S_inf = R_inf / (P * h2)

    # Weight points closer to x=0, i.e. y=-1.
    x_detached = x.detach().clamp_min(eps)
    w = x_detached.pow(-float(beta))
    w = w / w.sum(dim=1, keepdim=True).clamp_min(eps)

    err2 = torch.abs(S_pred - S_inf.detach()) ** 2

    if relative:
        scale = torch.sum(w * torch.abs(S_inf.detach()) ** 2, dim=1).clamp_min(eps)
        loss_inf_case = torch.sum(w * err2, dim=1) / scale
    else:
        loss_inf_case = torch.sum(w * err2, dim=1)

    loss_inf = loss_inf_case.mean()

    # Correction regularization.
    # Since dB is dimensionless, this directly penalizes deviation from spectral baseline.
    loss_B = torch.mean(torch.abs(dBinc) ** 2 + torch.abs(dBref) ** 2)

    info = {
        "loss_inf": float(loss_inf.detach().cpu().item()),
        "loss_B": float(loss_B.detach().cpu().item()),
        "mean_abs_dBinc": float(torch.mean(torch.abs(dBinc.detach())).cpu().item()),
        "mean_abs_dBref": float(torch.mean(torch.abs(dBref.detach())).cpu().item()),
        "mean_abs_Binc_spec": float(torch.mean(torch.abs(Binc_spec.detach())).cpu().item()),
        "mean_abs_Bref_spec": float(torch.mean(torch.abs(Bref_spec.detach())).cpu().item()),
    }
    return loss_inf, loss_B, info
```

---

## 5. 修改 `trainer/atlas_patch_trainer.py`

### 5.1 修改 import

找到：

```python
from physical_ansatz.residual_pinn import pinn_residual_loss, compute_data_anchor_loss, compute_integral_consistency_loss
```

替换为：

```python
from physical_ansatz.residual_pinn import (
    pinn_residual_loss,
    compute_data_anchor_loss,
    compute_integral_consistency_loss,
)
from physical_ansatz.infinity_asymptotic import compute_infinity_asymptotic_loss
```

文件顶部已经 import 了 `json`、`numpy as np`、`torch`、`Path`，后面 cache 会直接用这些。

### 5.2 在 `__init__` 中读取 infinity config

在读取 `integral_annealing` 后面，或者在 `self.cache = AuxCache()` 之前，加入：

```python
# ---- Infinity asymptotic loss config ----
inf_cfg = self.atlas_train_cfg.get("infinity_asymptotic", {})
self.inf_enabled = bool(inf_cfg.get("enabled", False))
self.inf_r_points = list(inf_cfg.get("r_points", [300.0, 500.0, 800.0, 1000.0]))
self.inf_beta = float(inf_cfg.get("beta", 1.0))
self.inf_relative = bool(inf_cfg.get("relative", True))
self.inf_eps = float(inf_cfg.get("eps", 1.0e-12))

self.inf_spectral_N = int(inf_cfg.get("spectral_N", self.viz_spectral_N))
self.inf_spectral_z_m = float(inf_cfg.get("spectral_z_m", self.viz_spectral_z_m))
self.inf_cache_file = str(inf_cfg.get(
    "cache_file",
    "outputs/domain/spectral_coeff_cache_l2_m2.json",
))

self.inf_weight_init = float(inf_cfg.get("weight_inf_init", 0.05))
self.inf_weight_final = float(inf_cfg.get("weight_inf_final", 0.2))
self.inf_weight_ramp_start = int(inf_cfg.get("weight_inf_ramp_start", 1000))
self.inf_weight_ramp_end = int(inf_cfg.get("weight_inf_ramp_end", 8000))

self.inf_weight_B_init = float(inf_cfg.get("weight_B_init", 1.0))
self.inf_weight_B_final = float(inf_cfg.get("weight_B_final", 0.1))
self.inf_weight_B_decay_start = int(inf_cfg.get("weight_B_decay_start", 3000))
self.inf_weight_B_decay_end = int(inf_cfg.get("weight_B_decay_end", 12000))

self.spectral_coeff_cache = {}
if self.inf_enabled:
    self._load_spectral_coeff_cache()
```

### 5.3 在 `AtlasPatchTrainer` 类中添加 helper 方法

把下面方法加入 `AtlasPatchTrainer` 类中。建议放在 `resolve_aux_batch` 后面。

```python
def _linear_schedule(self, step: int, start: int, end: int, v0: float, v1: float) -> float:
    if end <= start:
        return float(v1)
    if step <= start:
        return float(v0)
    if step >= end:
        return float(v1)
    t = (float(step) - float(start)) / (float(end) - float(start))
    return float((1.0 - t) * v0 + t * v1)


def _spectral_key(self, a: float, omega: float) -> str:
    return f"a={float(a):.12e}|omega={float(omega):.12e}"


def _load_spectral_coeff_cache(self):
    path = Path(self.inf_cache_file)
    self.spectral_coeff_cache = {}
    if not path.exists():
        return
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.spectral_coeff_cache = dict(data.get("records", {}))
        self._vprint(f"[inf-cache] loaded {len(self.spectral_coeff_cache)} entries from {path}")
    except Exception as e:
        self._vprint(f"[inf-cache] failed to load {path}: {e}")
        self.spectral_coeff_cache = {}


def _save_spectral_coeff_cache(self):
    path = Path(self.inf_cache_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": {
            "spectral_N": int(self.inf_spectral_N),
            "spectral_z_m": float(self.inf_spectral_z_m),
        },
        "records": self.spectral_coeff_cache,
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def get_spectral_coeff_batch(self, a_batch, omega_batch, lambda_batch):
    """
    Return per-parameter spectral coefficients for the current batch.

    This is the key part:
    - The coefficients are NOT computed once at training start.
    - For every parameter point (a, omega), use its own B_inc/B_ref.
    - The first time a point appears, compute_smatrix is called.
    - Later calls use the JSON/in-memory cache.
    """
    from utils.mode import KerrMode
    from utils.amplitude import compute_smatrix

    problem_cfg = self.physics_cfg["problem"]
    M = float(problem_cfg.get("M", 1.0))
    s = int(problem_cfg.get("s", -2))
    ell = int(problem_cfg.get("l", 2))
    m = int(problem_cfg.get("m", 2))

    Binc_list = []
    Bref_list = []
    updated = False

    for i in range(a_batch.shape[0]):
        a_i = float(a_batch[i].detach().cpu().item())
        omega_i = float(omega_batch[i].detach().cpu().item())
        lam_i = complex(lambda_batch[i].detach().cpu().item())

        key = self._spectral_key(a_i, omega_i)

        if key not in self.spectral_coeff_cache:
            mode = KerrMode(
                M=M,
                a=a_i,
                omega=omega_i,
                ell=ell,
                m=m,
                lam=lam_i,
                s=s,
            )
            smat = compute_smatrix(
                mode,
                N_in=self.inf_spectral_N,
                N_out=self.inf_spectral_N,
                z_m=self.inf_spectral_z_m,
                return_profile=False,
            )

            Binc = complex(smat["B_inc"])
            Bref = complex(smat["B_ref"])

            self.spectral_coeff_cache[key] = {
                "a": a_i,
                "omega": omega_i,
                "lambda_re": float(lam_i.real),
                "lambda_im": float(lam_i.imag),
                "B_inc_re": float(Binc.real),
                "B_inc_im": float(Binc.imag),
                "B_ref_re": float(Bref.real),
                "B_ref_im": float(Bref.imag),
            }
            updated = True

        rec = self.spectral_coeff_cache[key]
        Binc_list.append(complex(float(rec["B_inc_re"]), float(rec["B_inc_im"])))
        Bref_list.append(complex(float(rec["B_ref_re"]), float(rec["B_ref_im"])))

    if updated:
        self._save_spectral_coeff_cache()

    Binc_np = np.asarray(Binc_list, dtype=np.complex128)
    Bref_np = np.asarray(Bref_list, dtype=np.complex128)

    if self.dtype == torch.float32:
        cdtype = torch.complex64
    else:
        cdtype = torch.complex128

    Binc_t = torch.tensor(Binc_np, device=self.device, dtype=cdtype)
    Bref_t = torch.tensor(Bref_np, device=self.device, dtype=cdtype)

    return Binc_t, Bref_t
```

---

## 6. 修改训练 loop：把 `loss_inf` 和 `loss_B` 加入 total loss

由于当前 `trainer/atlas_patch_trainer.py` 较长，agent 需要在训练 step 中找到如下典型逻辑：

```python
loss, info = pinn_residual_loss(...)
...
loss_int = compute_integral_consistency_loss(...)
...
total_loss = ...
total_loss.backward()
```

在 `total_loss.backward()` 之前加入以下代码。

如果当前变量名不是 `total_loss`，请以实际变量为准，但必须保证在反向传播之前加入：

```python
loss_inf = torch.zeros((), device=self.device, dtype=self.dtype)
loss_B = torch.zeros((), device=self.device, dtype=self.dtype)
w_inf = 0.0
w_B = 0.0
inf_info = {}

if self.inf_enabled:
    Binc_spec, Bref_spec = self.get_spectral_coeff_batch(
        a_batch=a_batch,
        omega_batch=omega_batch,
        lambda_batch=lambda_batch,
    )

    loss_inf, loss_B, inf_info = compute_infinity_asymptotic_loss(
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
    )

    w_inf = self._linear_schedule(
        step=self.global_step,
        start=self.inf_weight_ramp_start,
        end=self.inf_weight_ramp_end,
        v0=self.inf_weight_init,
        v1=self.inf_weight_final,
    )
    w_B = self._linear_schedule(
        step=self.global_step,
        start=self.inf_weight_B_decay_start,
        end=self.inf_weight_B_decay_end,
        v0=self.inf_weight_B_init,
        v1=self.inf_weight_B_final,
    )

    total_loss = total_loss + w_inf * loss_inf + w_B * loss_B
```

同时把日志 `info` 或 `record` 中加入：

```python
info.update({
    "loss_inf": float(loss_inf.detach().cpu().item()),
    "loss_B": float(loss_B.detach().cpu().item()),
    "weight_inf": float(w_inf),
    "weight_B": float(w_B),
})
for k, v in inf_info.items():
    info[f"inf_{k}"] = v
```

如果训练日志不是叫 `info`，则加入到当前每 step 写入 `history.jsonl` 的字典中。

---

## 7. 修改 validation loop

在 validation 中是否加入 `loss_inf`，建议分两步：

第一步：**不把 `loss_inf` 加进 validation selection**，避免 early stopping 逻辑被新项扰动。

第二步：只记录 validation near-infinity 误差。也就是说，在 validation 中可以计算：

```python
val_loss_inf, val_loss_B, val_inf_info = compute_infinity_asymptotic_loss(...)
```

但不要把它加入 `val_mean`。先仅日志记录。等确认稳定后再考虑加入。

agent 如果没有把握 validation 位置，可以暂时只改 train loop，不改 validation。

---

## 8. 修改 `config/pinn_config_phase4_sgdr.yaml`

在 `atlas_training:` 段落下添加：

```yaml
  infinity_asymptotic:
    enabled: true

    # Use several finite but large radii.
    # Do not use y=-1 exactly.
    r_points: [300.0, 500.0, 800.0, 1000.0]

    # Larger beta gives larger weight closer to infinity.
    beta: 1.0

    # Use relative loss normalized by |S_inf|^2.
    relative: true
    eps: 1.0e-12

    # Spectral coefficient baseline.
    # These coefficients are used per parameter point, not once.
    spectral_N: 64
    spectral_z_m: 0.3
    cache_file: outputs/domain/spectral_coeff_cache_l2_m2.json

    # Weight for near-infinity S matching.
    weight_inf_init: 0.05
    weight_inf_final: 0.2
    weight_inf_ramp_start: 1000
    weight_inf_ramp_end: 8000

    # Weight for amplitude correction regularization.
    # Starts large, then decays: initially force B_net≈B_spec.
    weight_B_init: 1.0
    weight_B_final: 0.1
    weight_B_decay_start: 3000
    weight_B_decay_end: 12000
```

如果训练稳定，再把：

```yaml
    weight_inf_final: 0.5
```

但第一轮不要超过 `0.2`。

---

## 9. 推荐先做无 correction 的更稳版本

为了排除 amplitude head 带来的额外不稳定，建议第一轮可以固定：

```python
Binc_net = Binc_spec
Bref_net = Bref_spec
loss_B = 0
```

也就是先只用谱方法系数构造 \(S_\infty\) 监督网络的 \(S(y)\)。

具体改法是在 `compute_infinity_asymptotic_loss` 里临时替换：

```python
dBinc, dBref = model.predict_asymptotic_delta(...)
...
Binc_net = Binc_spec + scale_inc * dBinc
Bref_net = Bref_spec + scale_ref * dBref
```

为：

```python
dBinc = torch.zeros_like(Binc_spec)
dBref = torch.zeros_like(Bref_spec)
Binc_net = Binc_spec
Bref_net = Bref_spec
```

验证 \(y=-1\) 端改善后，再恢复 correction head。

---

## 10. 测试命令

### 10.1 语法检查

```bash
python -m py_compile model/pinn_mlp.py
python -m py_compile physical_ansatz/infinity_asymptotic.py
python -m py_compile trainer/atlas_patch_trainer.py
```

### 10.2 短训练测试

```bash
python scripts/train_multipatch_parallel.py \
  --cfg config/pinn_config_phase4_sgdr.yaml \
  --patch_ids 1 \
  --max_parallel 1 \
  --steps 20
```

要求：

1. 不报错；
2. `history.jsonl` 中出现 `loss_inf`、`loss_B`、`weight_inf`、`weight_B`；
3. `outputs/domain/spectral_coeff_cache_l2_m2.json` 被创建或更新；
4. cache 中每个不同 \((a,\omega)\) 都有自己的 `B_inc_re/im` 和 `B_ref_re/im`。

### 10.3 正式测试

```bash
python scripts/train_multipatch_parallel.py \
  --cfg config/pinn_config_phase4_sgdr.yaml \
  --patch_ids 1 \
  --max_parallel 1
```

观察图像中 \(y=-1\) 端的 `Re(S)`、`Im(S)`、`|S|` 是否改善。

---

## 11. 验收标准

完成后必须满足：

1. `PINN_MLP` 的原始 `forward(a, omega, y, u, v)` 接口不变；
2. amplitude head 不输出绝对 \(B\)，只输出 dimensionless correction；
3. 谱方法系数不是训练开始时只用一次，而是每个 batch 的每个参数点都调用 `get_spectral_coeff_batch` 获取；
4. `get_spectral_coeff_batch` 必须有 cache，避免重复计算；
5. `compute_infinity_asymptotic_loss` 中使用：

```python
Binc_net = Binc_spec + scale_inc * dBinc
Bref_net = Bref_spec + scale_ref * dBref
```

6. `S_inf` 必须使用当前项目约定：

```python
S_inf = R_inf / (P * h2)
```

不要写成单纯 `R_inf / P`，除非明确确认 `P` 已经包含 `h2`。

---

## 12. 提交说明

完成并测试后提交：

```bash
git add model/pinn_mlp.py \
        physical_ansatz/infinity_asymptotic.py \
        trainer/atlas_patch_trainer.py \
        config/pinn_config_phase4_sgdr.yaml

git commit -m "Add per-parameter spectral infinity asymptotic loss"
```
