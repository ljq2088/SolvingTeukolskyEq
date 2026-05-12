# Agent Implementation Guide: add near-infinity asymptotic amplitude loss for `test3`

目标：在 `ljq2088/SolvingTeukolskyEq` 的 `test3` 分支中，为当前 Phase-4 Atlas-PINN 增加一个**只依赖参数特征、不依赖坐标 \(y\)** 的 amplitude head，用谱方法给出的 \(B^{\rm inc}\)、\(B^{\rm ref}\) 作为初始化/锚点，并新增靠近 \(y=-1\) 的无穷远渐近约束。

当前训练入口保持不变：

```bash
python scripts/train_multipatch_parallel.py \
  --cfg config/pinn_config_phase4_sgdr.yaml \
  --patch_ids 1 \
  --max_parallel 1
```

当前项目中：
- `scripts/train_multipatch_parallel.py` 调用 `AtlasPatchTrainer` 训练单个 patch。
- `config/pinn_config_phase4_sgdr.yaml` 已经使用 `viz_benchmark_backend: spectral`，并设置了 `viz_spectral_N: 64`、`viz_spectral_z_m: 0.3`。
- `model/pinn_mlp.py` 中 `PINN_MLP.forward(a, omega, y, u, v)` 输出复数 \(f(y)\)。
- `physical_ansatz/transform_y.py` 中当前 reduced shape 约定为 \(S(x)=g(x)[h_1(x)f(y)+1]+1\)，其中 \(x=(y+1)/2\)。
- `physical_ansatz/prefactor.py` 中已有 `r_star`、`Leaver_prefactors`。
- `utils/amplitude.py` 中已有谱方法函数 `compute_smatrix`，返回 `B_inc`、`B_ref`、`B_trans`。

---

## 0. 建议先创建工作分支

```bash
git checkout test3
git pull
git checkout -b feature/infinity-asymptotic-loss
```

---

## 1. 修改 `model/pinn_mlp.py`

### 1.1 在 `PINN_MLP.__init__` 中新增 amplitude head

找到下面这段：

```python
self.base_head = nn.Linear(self.fusion_dim, 2)
self.a_head = nn.Linear(self.fusion_dim, 2)
self.omega_head = nn.Linear(self.fusion_dim, 2)
self.nl_head = nn.Linear(self.fusion_dim, 2)

if self.output_activation == "tanh":
    self.out_act = nn.Tanh()
else:
    self.out_act = None

self._init_weights()
```

替换为：

```python
self.base_head = nn.Linear(self.fusion_dim, 2)
self.a_head = nn.Linear(self.fusion_dim, 2)
self.omega_head = nn.Linear(self.fusion_dim, 2)
self.nl_head = nn.Linear(self.fusion_dim, 2)

# =========================================================
# 5) Near-infinity amplitude correction head
# ---------------------------------------------------------
# This head uses only parameter features p(a, omega, u, v),
# not the coordinate y. It predicts normalized corrections
# to spectral coefficients:
#
#   B_inc_net = B_inc_spec + scale_inc * dB_inc
#   B_ref_net = B_ref_spec + scale_ref * dB_ref
#
# The final layer is zero-initialized after _init_weights(),
# so initially dB_inc = dB_ref = 0.
# =========================================================
self.amp_head = nn.Sequential(
    nn.Linear(self.param_in_dim, 32),
    _make_activation(activation),
    nn.Linear(32, 4),
)

if self.output_activation == "tanh":
    self.out_act = nn.Tanh()
else:
    self.out_act = None

self._init_weights()
self._init_amp_head()
```

### 1.2 在 `PINN_MLP` 类中新增 `_init_amp_head`

在 `_init_weights(self)` 方法结束之后、`forward(self, ...)` 方法之前，插入：

```python
def _init_amp_head(self):
    """
    Zero-initialize only the final layer of amp_head.

    This guarantees:
        dB_inc = 0
        dB_ref = 0

    Therefore the full coefficients used in the asymptotic loss
    initially equal the spectral coefficients:
        B_inc_net = B_inc_spec
        B_ref_net = B_ref_spec
    """
    last = self.amp_head[-1]
    nn.init.zeros_(last.weight)
    if last.bias is not None:
        nn.init.zeros_(last.bias)
```

### 1.3 在 `PINN_MLP` 类中新增 `predict_asymptotic_delta`

仍然放在 `forward(self, ...)` 方法之前，紧接 `_init_amp_head` 后插入：

```python
def predict_asymptotic_delta(self, a, omega, u=None, v=None):
    """
    Predict normalized complex corrections to spectral asymptotic amplitudes.

    Args:
        a:     (B,) or (B,1)
        omega: (B,) or (B,1)
        u:     (B,) or (B,1), required when local_coord_mode='chart_uv'
        v:     (B,) or (B,1), required when local_coord_mode='chart_uv'

    Returns:
        dB_inc: (B,) complex tensor
        dB_ref: (B,) complex tensor
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
    out = self.amp_head(feats)

    dB_inc = torch.complex(out[:, 0], out[:, 1])
    dB_ref = torch.complex(out[:, 2], out[:, 3])
    return dB_inc, dB_ref
```

---

## 2. 新增 `utils/spectral_coeff_cache.py`

创建新文件：

```bash
touch utils/spectral_coeff_cache.py
```

写入以下完整代码：

```python
from __future__ import annotations

import json
from pathlib import Path

import torch

from utils.mode import KerrMode
from utils.amplitude import compute_smatrix


def _complex_to_json(z: complex) -> dict:
    z = complex(z)
    return {"re": float(z.real), "im": float(z.imag)}


def _complex_from_json(obj: dict) -> complex:
    return complex(float(obj["re"]), float(obj["im"]))


def _complex_dtype_from_float_dtype(dtype: torch.dtype) -> torch.dtype:
    if dtype == torch.float32:
        return torch.complex64
    return torch.complex128


class SpectralCoeffCache:
    """
    Cache spectral coefficients B_inc and B_ref for the in-mode solution.

    The cache key is rounded in (a, omega, lambda) to avoid duplicate entries
    caused by tiny floating-point formatting differences.
    """

    def __init__(
        self,
        cache_file: str | Path,
        physics_cfg: dict,
        device: torch.device,
        dtype: torch.dtype,
        N: int = 64,
        z_m: float = 0.3,
    ):
        self.cache_file = Path(cache_file)
        self.physics_cfg = physics_cfg
        self.device = device
        self.dtype = dtype
        self.complex_dtype = _complex_dtype_from_float_dtype(dtype)
        self.N = int(N)
        self.z_m = float(z_m)

        problem_cfg = physics_cfg["problem"]
        self.M = float(problem_cfg.get("M", 1.0))
        self.s = int(problem_cfg.get("s", -2))
        self.ell = int(problem_cfg.get("l", problem_cfg.get("ell", 2)))
        self.m = int(problem_cfg.get("m", 2))

        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        self._cache = self._load()

    def _make_key(self, a: float, omega: float, lam: complex) -> str:
        lam = complex(lam)
        return (
            f"s={self.s}|l={self.ell}|m={self.m}|"
            f"a={a:.12e}|omega={omega:.12e}|"
            f"lam_re={lam.real:.12e}|lam_im={lam.imag:.12e}|"
            f"N={self.N}|z_m={self.z_m:.12e}"
        )

    def _load(self) -> dict:
        if not self.cache_file.exists():
            return {}
        with open(self.cache_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def _save(self):
        tmp = self.cache_file.with_suffix(self.cache_file.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self._cache, f, ensure_ascii=False, indent=2)
        tmp.replace(self.cache_file)

    def _compute_one(self, a: float, omega: float, lam: complex) -> tuple[complex, complex]:
        mode = KerrMode(
            M=self.M,
            a=float(a),
            omega=float(omega),
            ell=self.ell,
            m=self.m,
            lam=complex(lam),
            s=self.s,
        )
        smat = compute_smatrix(
            mode=mode,
            N_in=self.N,
            N_out=self.N,
            z_m=self.z_m,
            return_profile=False,
        )
        return complex(smat["B_inc"]), complex(smat["B_ref"])

    def get_batch(
        self,
        a_batch: torch.Tensor,
        omega_batch: torch.Tensor,
        lambda_batch: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        Binc_list = []
        Bref_list = []
        changed = False

        B = int(a_batch.shape[0])
        for i in range(B):
            a_i = float(a_batch[i].detach().cpu().item())
            omega_i = float(omega_batch[i].detach().cpu().item())
            lam_i = lambda_batch[i].detach().cpu().item()
            lam_i = complex(lam_i)

            key = self._make_key(a_i, omega_i, lam_i)
            if key not in self._cache:
                Binc, Bref = self._compute_one(a_i, omega_i, lam_i)
                self._cache[key] = {
                    "a": a_i,
                    "omega": omega_i,
                    "lambda": _complex_to_json(lam_i),
                    "B_inc": _complex_to_json(Binc),
                    "B_ref": _complex_to_json(Bref),
                    "N": self.N,
                    "z_m": self.z_m,
                }
                changed = True

            row = self._cache[key]
            Binc_list.append(_complex_from_json(row["B_inc"]))
            Bref_list.append(_complex_from_json(row["B_ref"]))

        if changed:
            self._save()

        Binc_tensor = torch.tensor(
            Binc_list,
            device=self.device,
            dtype=self.complex_dtype,
        )
        Bref_tensor = torch.tensor(
            Bref_list,
            device=self.device,
            dtype=self.complex_dtype,
        )
        return Binc_tensor, Bref_tensor
```

---

## 3. 新增 `physical_ansatz/asymptotic_loss.py`

创建新文件：

```bash
touch physical_ansatz/asymptotic_loss.py
```

写入以下完整代码：

```python
from __future__ import annotations

import torch

from .prefactor import Leaver_prefactors, build_prefactor_primitives, r_star
from .transform_y import (
    h_factor,
    horizon_regularity_slope,
    compose_reduced_shape_from_f,
)


def _complex_dtype_from_float_dtype(dtype: torch.dtype) -> torch.dtype:
    if dtype == torch.float32:
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

    It compares the network reduced shape S_pred with the scattering-form
    asymptotic reduced shape S_inf near y=-1.

    Current project convention:
        R_in(r) = P(r) * h2(a, omega) * S(x)

    Therefore:
        S_inf = R_inf / [P(r) * h2(a, omega)]

    where:
        R_inf =
            B_inc r^{-1} exp(-i omega r_*)
          + B_ref r^3    exp(+i omega r_*)

    The amplitude head predicts normalized corrections:
        B_inc_net = B_inc_spec + (|B_inc_spec| + eps) * dB_inc
        B_ref_net = B_ref_spec + (|B_ref_spec| + eps) * dB_ref

    Returns:
        loss_inf: scalar tensor
        loss_B:   scalar tensor
        info:     dict of Python floats
    """
    device = a_batch.device
    dtype = a_batch.dtype
    cdtype = _complex_dtype_from_float_dtype(dtype)

    M = float(cfg["problem"].get("M", 1.0))
    s = int(cfg["problem"].get("s", -2))
    m = int(cfg["problem"].get("m", 2))

    B = int(a_batch.shape[0])

    # r_points are physical radii near infinity.
    r_base = torch.as_tensor(r_points, device=device, dtype=dtype)
    if r_base.ndim != 1:
        raise ValueError("r_points must be a 1D list/tensor of physical radii.")

    N = int(r_base.numel())

    # Build batch-dependent compact coordinates:
    #   x = r_+ / r
    #   y = 2x - 1
    r_grid = r_base.unsqueeze(0).expand(B, N)
    rp = torch.sqrt(torch.clamp(M * M - a_batch * a_batch, min=0.0)) + M
    x_grid = rp.unsqueeze(-1) / r_grid

    if torch.any(x_grid <= 0.0) or torch.any(x_grid >= 1.0):
        raise ValueError(
            "Invalid infinity r_points: require r_points > r_+ for every batch element."
        )

    y_grid = 2.0 * x_grid - 1.0

    # Network prediction f(y), then compose reduced shape S(x).
    f_pred = model(a_batch, omega_batch, y_grid, u=u_batch, v=v_batch)

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
        y=y_grid,
        slope=slope,
    )

    # Amplitude correction head.
    if not hasattr(model, "predict_asymptotic_delta"):
        raise AttributeError(
            "Model has no method predict_asymptotic_delta. "
            "Add the amplitude head to model/pinn_mlp.py first."
        )

    dB_inc, dB_ref = model.predict_asymptotic_delta(
        a_batch,
        omega_batch,
        u=u_batch,
        v=v_batch,
    )

    Binc_spec = Binc_spec.to(device=device, dtype=cdtype)
    Bref_spec = Bref_spec.to(device=device, dtype=cdtype)

    scale_inc = torch.abs(Binc_spec).to(dtype=dtype).clamp_min(eps)
    scale_ref = torch.abs(Bref_spec).to(dtype=dtype).clamp_min(eps)

    Binc_net = Binc_spec + scale_inc.to(dtype=cdtype) * dB_inc
    Bref_net = Bref_spec + scale_ref.to(dtype=cdtype) * dB_ref

    # Build S_inf batch-by-batch because Leaver_prefactors expects scalar a.
    S_inf_rows = []
    for i in range(B):
        r_i = r_grid[i]
        a_i = a_batch[i]
        omega_i = omega_batch[i]

        rp_pref, rm_pref, rs_pref, _, _ = build_prefactor_primitives(
            r_i,
            a_i,
            M=M,
            need_rs=True,
        )
        P_i, _, _ = Leaver_prefactors(
            r=r_i,
            a=a_i,
            omega=omega_i,
            m=m,
            M=M,
            s=s,
            rp=rp_pref,
            rm=rm_pref,
        )

        h2_i = h_factor(
            a_i.reshape(()),
            omega_i.reshape(()),
            m=m,
            M=M,
            s=s,
        )

        rs_i = r_star(r_i, a_i, M=M)

        R_inf_i = (
            Binc_net[i] * (r_i.to(cdtype) ** (-1.0)) * torch.exp(-1j * omega_i * rs_i)
            + Bref_net[i] * (r_i.to(cdtype) ** 3.0) * torch.exp(+1j * omega_i * rs_i)
        )

        S_inf_i = R_inf_i / (P_i * h2_i)
        S_inf_rows.append(S_inf_i)

    S_inf = torch.stack(S_inf_rows, dim=0)

    # Weighted near-infinity error.
    # Larger weight closer to y=-1, i.e. smaller x.
    weights = x_grid.clamp_min(eps) ** (-float(beta))
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(eps)

    err2 = torch.abs(S_pred - S_inf) ** 2

    if relative:
        scale = torch.mean(torch.abs(S_inf.detach()) ** 2, dim=-1, keepdim=True)
        err2 = err2 / scale.clamp_min(eps)

    loss_inf = torch.mean(torch.sum(weights * err2, dim=-1))

    # Since B_net - B_spec = scale * dB, this is the normalized B-regularizer.
    loss_B = torch.mean(torch.abs(dB_inc) ** 2 + torch.abs(dB_ref) ** 2)

    info = {
        "loss_inf": float(loss_inf.detach().cpu().item()),
        "loss_B": float(loss_B.detach().cpu().item()),
        "mean_abs_dB_inc": float(torch.mean(torch.abs(dB_inc)).detach().cpu().item()),
        "mean_abs_dB_ref": float(torch.mean(torch.abs(dB_ref)).detach().cpu().item()),
        "mean_abs_Sinf": float(torch.mean(torch.abs(S_inf)).detach().cpu().item()),
        "mean_abs_Spred_inf": float(torch.mean(torch.abs(S_pred)).detach().cpu().item()),
    }
    return loss_inf, loss_B, info
```

---

## 4. 修改 `trainer/atlas_patch_trainer.py`

### 4.1 修改 import

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
from physical_ansatz.asymptotic_loss import compute_infinity_asymptotic_loss
from utils.spectral_coeff_cache import SpectralCoeffCache
```

### 4.2 在 `__init__` 里读取 `infinity_asymptotic` 配置

找到：

```python
self.viz_spectral_N = int(atlas_train_cfg.get("viz_spectral_N", 64))
self.viz_spectral_z_m = float(atlas_train_cfg.get("viz_spectral_z_m", 0.3))
self.viz_mma_enabled = bool(atlas_train_cfg.get("viz_mma_enabled", False))
```

替换为：

```python
self.viz_spectral_N = int(atlas_train_cfg.get("viz_spectral_N", 64))
self.viz_spectral_z_m = float(atlas_train_cfg.get("viz_spectral_z_m", 0.3))
self.viz_mma_enabled = bool(atlas_train_cfg.get("viz_mma_enabled", False))

# ---- Near-infinity asymptotic loss ----
inf_cfg = atlas_train_cfg.get("infinity_asymptotic", {})
self.inf_enabled = bool(inf_cfg.get("enabled", False))
self.inf_r_points = list(inf_cfg.get("r_points", [300.0, 500.0, 800.0, 1000.0]))
self.inf_beta = float(inf_cfg.get("beta", 1.0))
self.inf_relative = bool(inf_cfg.get("relative", True))
self.inf_eps = float(inf_cfg.get("eps", 1.0e-12))

self.inf_weight_init = float(inf_cfg.get("weight_inf_init", 0.05))
self.inf_weight_final = float(inf_cfg.get("weight_inf_final", 0.2))
self.inf_weight_ramp_start = int(inf_cfg.get("weight_inf_ramp_start", 1000))
self.inf_weight_ramp_end = int(inf_cfg.get("weight_inf_ramp_end", 8000))

self.inf_weight_B_init = float(inf_cfg.get("weight_B_init", 1.0))
self.inf_weight_B_final = float(inf_cfg.get("weight_B_final", 0.1))
self.inf_weight_B_decay_start = int(inf_cfg.get("weight_B_decay_start", 3000))
self.inf_weight_B_decay_end = int(inf_cfg.get("weight_B_decay_end", 12000))

self.inf_spectral_N = int(inf_cfg.get("spectral_N", self.viz_spectral_N))
self.inf_spectral_z_m = float(inf_cfg.get("spectral_z_m", self.viz_spectral_z_m))
self.inf_cache_file = str(
    inf_cfg.get("cache_file", "outputs/domain/spectral_coeff_cache_l2_m2.json")
)
```

### 4.3 在 `__init__` 中创建 `SpectralCoeffCache`

找到：

```python
self.cache = AuxCache()
self._preload_lambda_cache_from_probe(probe_json)
```

替换为：

```python
self.cache = AuxCache()
self._preload_lambda_cache_from_probe(probe_json)

self.inf_coeff_cache = None
if self.inf_enabled:
    self.inf_coeff_cache = SpectralCoeffCache(
        cache_file=self.inf_cache_file,
        physics_cfg=self.physics_cfg,
        device=self.device,
        dtype=self.dtype,
        N=self.inf_spectral_N,
        z_m=self.inf_spectral_z_m,
    )
```

### 4.4 在类中新增线性 schedule 函数

在 `_vprint(self, *args, **kwargs)` 方法之后插入：

```python
def _linear_schedule(self, step: int, start: int, end: int, w0: float, w1: float) -> float:
    if end <= start:
        return float(w1)
    if step <= start:
        return float(w0)
    if step >= end:
        return float(w1)
    t = float(step - start) / float(end - start)
    return float((1.0 - t) * w0 + t * w1)

def _infinity_loss_weights(self) -> tuple[float, float]:
    w_inf = self._linear_schedule(
        self.global_step,
        self.inf_weight_ramp_start,
        self.inf_weight_ramp_end,
        self.inf_weight_init,
        self.inf_weight_final,
    )
    w_B = self._linear_schedule(
        self.global_step,
        self.inf_weight_B_decay_start,
        self.inf_weight_B_decay_end,
        self.inf_weight_B_init,
        self.inf_weight_B_final,
    )
    return w_inf, w_B
```

### 4.5 在训练 step 中加入无穷远 loss

在 `train()` 方法内部，找到当前计算总 loss 的位置。通常会有类似结构：

```python
loss_pde, pde_info = pinn_residual_loss(...)
...
loss_int = compute_integral_consistency_loss(...)
...
total_loss = loss_pde + int_weight * loss_int
```

在 `total_loss` 已经包含 PDE loss 和 integral consistency loss 之后、`total_loss.backward()` 之前，插入下面完整代码：

```python
# ---------------------------------------------------------
# Near-infinity asymptotic loss
# ---------------------------------------------------------
loss_inf = torch.zeros((), device=self.device, dtype=self.dtype)
loss_B = torch.zeros((), device=self.device, dtype=self.dtype)
w_inf = 0.0
w_B = 0.0
inf_info = {}

if self.inf_enabled:
    if self.inf_coeff_cache is None:
        raise RuntimeError("self.inf_enabled=True but self.inf_coeff_cache is None.")

    Binc_spec, Bref_spec = self.inf_coeff_cache.get_batch(
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

    w_inf, w_B = self._infinity_loss_weights()
    total_loss = total_loss + w_inf * loss_inf + w_B * loss_B
```

### 4.6 把新 loss 写入日志

找到 `record`、`history_row` 或 `info` 字典构造的位置。里面一般已有：

```python
"loss": ...
"loss_pde": ...
"loss_int": ...
```

在同一个字典中加入：

```python
"loss_inf": float(loss_inf.detach().cpu().item()),
"loss_B": float(loss_B.detach().cpu().item()),
"weight_inf": float(w_inf),
"weight_B": float(w_B),
"mean_abs_dB_inc": float(inf_info.get("mean_abs_dB_inc", 0.0)),
"mean_abs_dB_ref": float(inf_info.get("mean_abs_dB_ref", 0.0)),
"mean_abs_Sinf": float(inf_info.get("mean_abs_Sinf", 0.0)),
"mean_abs_Spred_inf": float(inf_info.get("mean_abs_Spred_inf", 0.0)),
```

不要删除原来的日志字段。

---

## 5. 修改 `config/pinn_config_phase4_sgdr.yaml`

在 `atlas_training:` 下，建议放在：

```yaml
  integral_annealing:
    enabled: true
    weight_init: 2.0
    weight_final: 0.1
    decay_start: 5000
    decay_end: 15000
```

之后，加入：

```yaml
  infinity_asymptotic:
    enabled: true

    # Physical radii close to infinity. Do not use y=-1 exactly.
    r_points: [300.0, 500.0, 800.0, 1000.0]

    # Weight points closer to y=-1 more strongly:
    # w_j ∝ x_j^{-beta}, x_j = r_+ / r_j.
    beta: 1.0

    relative: true
    eps: 1.0e-12

    # Spectral coefficient source.
    spectral_N: 64
    spectral_z_m: 0.3
    cache_file: outputs/domain/spectral_coeff_cache_l2_m2.json

    # Conservative first run for patch 1.
    weight_inf_init: 0.05
    weight_inf_final: 0.2
    weight_inf_ramp_start: 1000
    weight_inf_ramp_end: 8000

    # Keep B close to spectral values early,
    # then allow small correction later.
    weight_B_init: 1.0
    weight_B_final: 0.1
    weight_B_decay_start: 3000
    weight_B_decay_end: 12000
```

第一轮只训练 patch 1，不要一开始跑全部 patch。

---

## 6. 训练前必须做的 sanity check

创建新文件：

```bash
touch scripts/check_infinity_asymptotic_loss.py
```

写入：

```python
#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from config.config_loader import load_pinn_full_config
from physical_ansatz.residual import AuxCache, get_lambda_from_cfg
from utils.spectral_coeff_cache import SpectralCoeffCache


def main():
    cfg_path = "config/pinn_config_phase4_sgdr.yaml"
    full_cfg = load_pinn_full_config(cfg_path)
    physics_cfg = full_cfg["physics"]
    train_cfg = full_cfg["train"]
    atlas_cfg = train_cfg["atlas_training"]
    inf_cfg = atlas_cfg["infinity_asymptotic"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64

    # Use the same point shown in the user's latest diagnostic plot.
    a = torch.tensor([0.050833], device=device, dtype=dtype)
    omega = torch.tensor([0.051377], device=device, dtype=dtype)

    cache = AuxCache()
    lam = get_lambda_from_cfg(physics_cfg, cache, a[0], omega[0]).reshape(1)

    coeff_cache = SpectralCoeffCache(
        cache_file=inf_cfg["cache_file"],
        physics_cfg=physics_cfg,
        device=device,
        dtype=dtype,
        N=int(inf_cfg.get("spectral_N", 64)),
        z_m=float(inf_cfg.get("spectral_z_m", 0.3)),
    )

    Binc, Bref = coeff_cache.get_batch(a, omega, lam)

    print("a       =", float(a[0].detach().cpu().item()))
    print("omega   =", float(omega[0].detach().cpu().item()))
    print("lambda  =", complex(lam[0].detach().cpu().item()))
    print("B_inc   =", complex(Binc[0].detach().cpu().item()))
    print("B_ref   =", complex(Bref[0].detach().cpu().item()))
    print("cache   =", inf_cfg["cache_file"])
    print("[OK] spectral coefficients were computed or loaded.")


if __name__ == "__main__":
    main()
```

运行：

```bash
python scripts/check_infinity_asymptotic_loss.py
```

预期结果：
- 能打印 `B_inc` 和 `B_ref`。
- `outputs/domain/spectral_coeff_cache_l2_m2.json` 被创建或更新。
- 没有 CUDA dtype 错误。
- 没有 complex-to-float 错误。

---

## 7. 最小运行测试

先不要训练 30000 步，跑 200 步检查是否崩溃：

```bash
python scripts/train_multipatch_parallel.py \
  --cfg config/pinn_config_phase4_sgdr.yaml \
  --patch_ids 1 \
  --max_parallel 1 \
  --steps 200
```

检查日志：

```bash
grep -R "\"loss_inf\"" outputs/atlas_multipatch_phase4_sgdr -n | tail -5
grep -R "\"loss_B\"" outputs/atlas_multipatch_phase4_sgdr -n | tail -5
```

如果报错，优先检查：
1. `predict_asymptotic_delta` 是否只加在 `PINN_MLP`，而当前训练是否确实使用 `--model_type pinn_mlp`。
2. `Leaver_prefactors` 是否收到 batch 维度的 `a`。本方案在 asymptotic loss 内已经逐 batch 调用，避免了广播错误。
3. `Binc_spec` 和 `Bref_spec` 是否为 complex tensor。
4. `r_points` 是否都大于当前 batch 的 \(r_+\)。

---

## 8. 正式跑 patch 1

```bash
python scripts/train_multipatch_parallel.py \
  --cfg config/pinn_config_phase4_sgdr.yaml \
  --patch_ids 1 \
  --max_parallel 1
```

观察：
- `loss_inf` 应该在前几千步逐渐下降或保持稳定。
- `loss_B` 初始应接近 0，因为 amplitude head 最后一层是 0 初始化。
- 如果 `loss_inf` 急剧增大并拖坏中间区域，把配置改成：

```yaml
weight_inf_final: 0.1
weight_B_final: 0.3
```

如果 \(y=-1\) 端改善明显但仍有系统偏差，把配置改成：

```yaml
weight_inf_final: 0.5
weight_B_final: 0.05
```

---

## 9. 不要做的修改

不要改：
- `scripts/train_multipatch_parallel.py` 的并行调度逻辑；
- `PINN_MLP.forward()` 的返回值；
- 现有 \(S(x)=g(x)[h_1(x)f(y)+1]+1\) ansatz；
- PDE residual 和 integral consistency 的现有实现；
- `r_from_x` 的定义。

这个改动只新增：
1. 一个参数-only amplitude correction head；
2. 一个谱方法 \(B^{\rm inc}\)、\(B^{\rm ref}\) cache；
3. 一个 near-infinity asymptotic loss；
4. config 中的 loss 开关与权重 schedule。

---

## 10. 完成后提交

```bash
git status
git add model/pinn_mlp.py \
        physical_ansatz/asymptotic_loss.py \
        utils/spectral_coeff_cache.py \
        trainer/atlas_patch_trainer.py \
        config/pinn_config_phase4_sgdr.yaml \
        scripts/check_infinity_asymptotic_loss.py

git commit -m "Add near-infinity asymptotic loss for Teukolsky PINN"
```
