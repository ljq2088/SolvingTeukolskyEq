from __future__ import annotations
import sys

sys.path.append("/home/ljq/code/PINN/SolvingTeukolsky")
sys.path.append("/home/ljq/code/PINN/SolvingTeukolsky/pybhpt")
import json
import math
import random
import shutil
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from tqdm.auto import trange

from config.config_loader import load_pinn_full_config
from model.pinn_mlp import PINN_MLP
from model.cheb_coeff_net import ChebCoeffNet
from model.cheb_deeponet import ChebDeepONet
from utils.mode import KerrMode
from utils.amplitude import TeukRadAmplitudeInWithInterpolant
from dataset.sampling import sample_points_chebyshev_grid, sample_points_uniform_grid
from physical_ansatz.residual import AuxCache, get_lambda_from_cfg
from physical_ansatz.residual_pinn import (
    pinn_residual_loss,
    compute_data_anchor_loss,
    compute_integral_consistency_loss,
)
from physical_ansatz.asymptotic_loss import compute_infinity_asymptotic_loss
from utils.spectral_coeff_cache import SpectralCoeffCache
from utils.asymptotic_amplitude_monitor import monitor_network_amplitudes_vs_pybhpt
from domain.patch_cover import load_patch_cover, load_valid_chart_points
from physical_ansatz.mapping import r_plus, r_from_x
from physical_ansatz.transform_y import (
    h_factor,
    horizon_regularity_slope,
    compose_reduced_shape_from_f,
)
from physical_ansatz.prefactor import Leaver_prefactors, build_prefactor_primitives
from mma.rin_sampler import MathematicaRinSampler
from pybhpt_usage.compute_solution import compute_pybhpt_solution


def _get_dtype(dtype_name: str):
    if dtype_name == "float32":
        return torch.float32
    return torch.float64
class AtlasPatchTrainer:
    """
    真正训练版的 atlas patch trainer。
    仍然是“单 patch 训练单元”，但已经具备：
      - train/val split
      - 长训练 loop
      - 在线 anchor（失败自动跳过）
      - 动态 anchor 权重
      - 可视化
      - best/latest/periodic checkpoint
      - metrics / summary / failure log
    """

    def __init__(
        self,
        cfg_path: str,
        probe_json: str,
        atlas_json: str,
        patch_json: str,
        patch_id: int,
        device: str = "cpu",
        anchor_enabled: bool = False,
        n_anchor_y: int = 4,
        verbose: bool = False,
        output_root: str | None = None,
        init_checkpoint: str | None = None,
        init_load_optimizer: bool = False,
        resume_checkpoint: str | None = None,
        resume_run_dir: str | None = None,
        model_type: str = "pinn_mlp",
        cheb_N: int = 64,
    ):
        self.device = torch.device(device)

        # ---------------------------------------------------------
        # config
        # ---------------------------------------------------------
        self.cfg_path = Path(cfg_path).resolve()
        self.project_root = self.cfg_path.parent.parent
        full_cfg = load_pinn_full_config(cfg_path)
        self.full_cfg = full_cfg
        self.physics_cfg = full_cfg["physics"]
        self.cfg = full_cfg["train"]
        self.train_cfg = self.cfg

        runtime_cfg = self.cfg.get("runtime", {})
        self.dtype = _get_dtype(runtime_cfg.get("dtype", "float64"))

        model_cfg = self.cfg.get("model", {})
        self.model_cfg = model_cfg

        # atlas-specific training config
        atlas_train_cfg = self.cfg.get("atlas_training", {})
        self.atlas_train_cfg = atlas_train_cfg
        cfg_output_root = atlas_train_cfg.get("output_root", "outputs/atlas_patch_train")
        sampling_cfg = self.cfg.get("sampling", {})

        early_cfg = atlas_train_cfg.get("early_stopping", {})
        self.es_enabled = bool(early_cfg.get("enabled", False))
        self.es_patience = int(early_cfg.get("patience", 20))
        self.es_min_rel_improve = float(early_cfg.get("min_rel_improve", 1.0e-4))
        self.es_min_steps = int(early_cfg.get("min_steps", 2000))
        self.es_bad_count = 0

        self.batch_size = int(
            atlas_train_cfg.get(
                "batch_size",
                sampling_cfg.get("parameter_batch", {}).get("batch_size", 4),
            )
        )
        self.n_interior = int(atlas_train_cfg.get("n_interior", 64))
        self.n_interior_outer_extra = int(atlas_train_cfg.get("n_interior_outer_extra", 8))
        self.n_anchor_y = int(n_anchor_y if n_anchor_y is not None else atlas_train_cfg.get("n_anchor_y", 4))
        self.steps_default = int(atlas_train_cfg.get("steps", 2000))
        self.grad_clip = float(atlas_train_cfg.get("grad_clip", 1.0))
        self.normalize_residual = bool(
            atlas_train_cfg.get(
                "normalize_residual",
                sampling_cfg.get("collocation", {}).get("adaptive", {}).get("normalize_residual", False),
            )
        )

        self.val_fraction = float(atlas_train_cfg.get("val_fraction", 0.2))
        self.val_param_samples = int(atlas_train_cfg.get("val_param_samples", 12))
        self.val_every = int(atlas_train_cfg.get("val_every", 50))
        self.val_n_points = int(atlas_train_cfg.get("val_n_points", 64))

        self.save_every = int(atlas_train_cfg.get("save_every", 200))
        self.ckpt_every = int(atlas_train_cfg.get("ckpt_every", 200))
        self.viz_every = int(atlas_train_cfg.get("viz_every", 200))
        self.viz_num_points = int(atlas_train_cfg.get("viz_num_points", 128))
        self.viz_r_min = float(atlas_train_cfg.get("viz_r_min", 2.0))
        self.viz_r_max = float(atlas_train_cfg.get("viz_r_max", 80.0))
        self.anchor_backend = str(atlas_train_cfg.get("anchor_backend", "pybhpt")).lower()
        if self.anchor_backend not in ("pybhpt", "mma"):
            raise ValueError(
                f"Unsupported atlas_training.anchor_backend={self.anchor_backend}; "
                "must be one of {'pybhpt','mma'}."
            )

        self.viz_benchmark_backend = str(atlas_train_cfg.get("viz_benchmark_backend", "none")).lower()
        if self.viz_benchmark_backend not in ("none", "pybhpt", "mma", "spectral"):
            raise ValueError(
                f"Unsupported atlas_training.viz_benchmark_backend={self.viz_benchmark_backend}; "
                "must be one of {'none','pybhpt','mma','spectral'}."
            )
        self.viz_pybhpt_timeout = float(atlas_train_cfg.get("viz_pybhpt_timeout", 10.0))
        self.anchor_pybhpt_timeout = float(atlas_train_cfg.get("anchor_pybhpt_timeout", self.viz_pybhpt_timeout))
        self.viz_spectral_N = int(atlas_train_cfg.get("viz_spectral_N", 64))
        self.viz_spectral_z_m = float(atlas_train_cfg.get("viz_spectral_z_m", 0.3))
        self.viz_mma_enabled = bool(atlas_train_cfg.get("viz_mma_enabled", False))

        # Near-infinity asymptotic loss config
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
        self.inf_cache_file = str(inf_cfg.get("cache_file", "outputs/domain/spectral_coeff_cache_l2_m2.json"))

        self.inf_stage = str(inf_cfg.get("stage", "joint")).lower()
        self.inf_amplitude_mode = str(inf_cfg.get("amplitude_mode", "spectral_correction")).lower()
        self.inf_hybrid_eta_init = float(inf_cfg.get("hybrid_eta_init", 0.0))
        self.inf_hybrid_eta_final = float(inf_cfg.get("hybrid_eta_final", 1.0))
        self.inf_hybrid_eta_start = int(inf_cfg.get("hybrid_eta_start", 5000))
        self.inf_hybrid_eta_end = int(inf_cfg.get("hybrid_eta_end", 20000))

        self.inf_weight_amp_init = float(inf_cfg.get("weight_amp_init", 1.0))
        self.inf_weight_amp_final = float(inf_cfg.get("weight_amp_final", 0.1))
        self.inf_weight_amp_decay_start = int(inf_cfg.get("weight_amp_decay_start", 3000))
        self.inf_weight_amp_decay_end = int(inf_cfg.get("weight_amp_decay_end", 15000))

        # Monitor config
        mon_cfg = inf_cfg.get("monitor", {})
        self.inf_monitor_enabled = bool(mon_cfg.get("enabled", False))
        self.inf_monitor_every = int(mon_cfg.get("every", self.viz_every))
        self.inf_monitor_r_points = list(mon_cfg.get("r_points", [200.0, 300.0, 500.0, 800.0, 1000.0]))
        self.inf_monitor_pybhpt_timeout = float(mon_cfg.get("pybhpt_timeout", self.viz_pybhpt_timeout))

        self.anchor_enabled = bool(anchor_enabled)
        self.anchor_target_ratio = float(atlas_train_cfg.get("anchor_target_ratio", 1.0e-2))
        self.lr = float(self.cfg.get("training", {}).get("optimizer", {}).get("lr", 1.0e-3))
        self.verbose = bool(verbose)
        self.model_type = str(model_type)
        self.cheb_N = int(cheb_N)
        if "viz_mma_enabled" in atlas_train_cfg and self.verbose:
            self._vprint(
                "[viz] atlas_training.viz_mma_enabled is deprecated; "
                "viz_benchmark_backend takes precedence."
            )

        self.train_seed = int(atlas_train_cfg.get("train_seed", 1234))
        self.output_root = Path(output_root) if output_root is not None else Path(cfg_output_root)
        self.anchor_fail_log_name = str(atlas_train_cfg.get("anchor_fail_log_name", "anchor_failures.jsonl"))

        self.anchor_mma_sampler = None
        self.viz_mma_sampler = None
        if self.anchor_enabled or self.viz_benchmark_backend == "mma":
            mathematica_cfg = full_cfg.get("mathematica", {})
            self._mathematica_cfg = mathematica_cfg
            if self.anchor_enabled and self.anchor_backend == "mma":
                self.anchor_mma_sampler = MathematicaRinSampler(
                    mathematica_cfg,
                    timeout_sec=float(atlas_train_cfg.get("anchor_mma_timeout", 20.0)),
                )
            if self.viz_benchmark_backend == "mma":
                self.viz_mma_sampler = MathematicaRinSampler(
                    mathematica_cfg,
                    timeout_sec=float(atlas_train_cfg.get("viz_mma_timeout", 20.0)),
                )

        # ---------------------------------------------------------
        # patch cover + valid chart points
        # ---------------------------------------------------------
        self.patch_cover = load_patch_cover(patch_json)
        patches = [p for p in self.patch_cover.patches if p.patch_id == patch_id]
        if len(patches) != 1:
            raise ValueError(f"patch_id={patch_id} not found in {patch_json}")
        self.patch = patches[0]

        uv_points, aw_points = load_valid_chart_points(
            probe_json=probe_json,
            atlas_json=atlas_json,
            component_id=self.patch.component_id,
        )

        mask = (
            (np.abs(uv_points[:, 0] - self.patch.u_center) <= self.patch.h_u)
            & (np.abs(uv_points[:, 1] - self.patch.v_center) <= self.patch.h_v)
        )
        self.patch_uv = uv_points[mask]
        self.patch_aw = aw_points[mask]

        if len(self.patch_uv) == 0:
            raise RuntimeError(f"Patch {patch_id} contains no safe points.")

        # ---------------------------------------------------------
        # train / val split
        # ---------------------------------------------------------
        self._seed_everything(self.train_seed)
        self._split_train_val()

        # ---------------------------------------------------------
        # model
        # ---------------------------------------------------------
        problem_cfg = self.physics_cfg["problem"]
        M = float(problem_cfg.get("M", 1.0))
        m_mode = int(problem_cfg.get("m", 2))

        if self.model_type == "cheb":
            self.model = ChebCoeffNet(
                N=self.cheb_N,
                hidden_dims=model_cfg.get("hidden_dims", [128, 256, 256, 128]),
                activation=model_cfg.get("activation", "silu"),
                param_embed_dim=model_cfg.get("param_embed_dim", 64),
                local_coord_mode="chart_uv",
                u_center_local=self.patch.u_center,
                v_center_local=self.patch.v_center,
                u_half_range_local=self.patch.h_u,
                v_half_range_local=self.patch.h_v,
                M=M,
                m_mode=m_mode,
            ).to(device=self.device, dtype=self.dtype)
        elif self.model_type == "cheb_deeponet":
            self.model = ChebDeepONet(
                N=self.cheb_N,
                hidden_dims=model_cfg.get("hidden_dims", [128, 256, 256, 128]),
                activation=model_cfg.get("activation", "silu"),
                param_embed_dim=model_cfg.get("param_embed_dim", 128),
                local_coord_mode="chart_uv",
                use_taylor=model_cfg.get("use_taylor", True),
                u_center_local=self.patch.u_center,
                v_center_local=self.patch.v_center,
                u_half_range_local=self.patch.h_u,
                v_half_range_local=self.patch.h_v,
                M=M,
                m_mode=m_mode,
            ).to(device=self.device, dtype=self.dtype)
        else:
            self.model = PINN_MLP(
                hidden_dims=model_cfg.get("hidden_dims", [128, 128, 128, 128]),
                activation=model_cfg.get("activation", "silu"),
                fourier_num_freqs=model_cfg.get("fourier_num_freqs", 2),
                fourier_base_scale=model_cfg.get("fourier_base_scale", 1.0),
                fourier_scales=model_cfg.get("fourier_scales", None),
                fourier_scale=model_cfg.get("fourier_scale", None),
                param_embed_dim=model_cfg.get("param_embed_dim", 64),
                use_film=model_cfg.get("use_film", True),
                use_residual=model_cfg.get("use_residual", True),
                local_coord_mode="chart_uv",

                # backward compatibility fields
                a_center_local=model_cfg.get("a_center_local", 0.125),
                a_half_range_local=model_cfg.get("a_half_range_local", 0.075),
                omega_min_local=model_cfg.get("omega_min_local", 1.0e-4),
                omega_max_local=model_cfg.get("omega_max_local", 10.0),

                # actual patch-local chart coords
                u_center_local=self.patch.u_center,
                v_center_local=self.patch.v_center,
                u_half_range_local=self.patch.h_u,
                v_half_range_local=self.patch.h_v,

                M=M,
                m_mode=m_mode,
            ).to(device=self.device, dtype=self.dtype)

        # ---- Freeze policy ----
        self._apply_freeze_policy()
        self._print_trainable_parameters(tag="after freeze policy")

        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.Adam(trainable_params, lr=self.lr)

        # ---- SGDR learning rate scheduler ----
        sched_cfg = self.cfg.get("training", {}).get("scheduler", {})
        self.use_sgdr = sched_cfg.get("type", "") == "sgdr"
        if self.use_sgdr:
            self.sgdr_T0 = int(sched_cfg.get("sgdr_T0", 3000))
            self.sgdr_T_mult = int(sched_cfg.get("sgdr_T_mult", 2))
            self.sgdr_eta_min = float(sched_cfg.get("sgdr_eta_min", 0.1))
            self.sgdr_T_cur = self.sgdr_T0
            self.sgdr_cycle_start = 0

        # ---- Integral consistency weight annealing ----
        int_anneal_cfg = self.atlas_train_cfg.get("integral_annealing", {})
        self.int_anneal_enabled = bool(int_anneal_cfg.get("enabled", False))
        if self.int_anneal_enabled:
            self.int_weight_init = float(int_anneal_cfg.get("weight_init", 2.0))
            self.int_weight_final = float(int_anneal_cfg.get("weight_final", 0.1))
            self.int_weight_decay_start = int(int_anneal_cfg.get("decay_start", 5000))
            self.int_weight_decay_end = int(int_anneal_cfg.get("decay_end", 15000))
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

        self.init_checkpoint = init_checkpoint
        self.init_load_optimizer = bool(init_load_optimizer)
        self.resume_checkpoint = resume_checkpoint
        self.resume_run_dir = Path(resume_run_dir) if resume_run_dir is not None else None

        self.global_step = 0
        self.best_val_mean = float("inf")
        self.best_val_case_means = None
        self.best_val_case_steps = None
        self.history = []
        self.anchor_fail_history = []

        # ---------------------------------------------------------
        # validation metadata
        # ---------------------------------------------------------
        self._build_validation_metadata()
        self._init_best_val_case_tracking()

        # ---------------------------------------------------------
        # output dirs
        # ---------------------------------------------------------
        self._init_run_dirs()
        self._save_config_snapshot()

        if self.resume_checkpoint is not None:
            self._load_resume_checkpoint()
        elif self.init_checkpoint is not None:
            self._load_init_checkpoint()

        # reference sample for visualization
        self.ref_sample = self._choose_reference_sample()

    def _vprint(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

    def _linear_schedule(self, step: int, start: int, end: int, w0: float, w1: float) -> float:
        if end <= start:
            return float(w1)
        if step <= start:
            return float(w0)
        if step >= end:
            return float(w1)
        t = float(step - start) / float(end - start)
        return float((1.0 - t) * w0 + t * w1)

    def _apply_freeze_policy(self):
        """Apply freeze policy based on infinity_asymptotic config."""
        inf_cfg = self.atlas_train_cfg.get("infinity_asymptotic", {})
        freeze_pinn = bool(inf_cfg.get("freeze_pinn", False))
        train_amplitude_net = bool(inf_cfg.get("train_amplitude_net", True))
        train_old_amp_head = bool(inf_cfg.get("train_old_amp_head", False))

        if not freeze_pinn:
            for _, p in self.model.named_parameters():
                p.requires_grad = True
            self._vprint("[freeze] freeze_pinn=false, all parameters trainable")
            return

        # Freeze everything first
        for _, p in self.model.named_parameters():
            p.requires_grad = False

        # Unfreeze amplitude_net
        if train_amplitude_net:
            if not hasattr(self.model, "amplitude_net"):
                raise RuntimeError(
                    "freeze_pinn=True and train_amplitude_net=True, "
                    "but model has no amplitude_net."
                )
            for name, p in self.model.named_parameters():
                if name.startswith("amplitude_net."):
                    p.requires_grad = True

        # Optionally unfreeze old amp_head
        if train_old_amp_head:
            for name, p in self.model.named_parameters():
                if name.startswith("amp_head."):
                    p.requires_grad = True

        # Safety: amplitude_net must have sufficient trainable params
        amp_trainable = sum(
            p.numel()
            for name, p in self.model.named_parameters()
            if name.startswith("amplitude_net.") and p.requires_grad
        )
        if amp_trainable > 0 and amp_trainable < 10000:
            raise RuntimeError(
                f"amplitude_net trainable params = {amp_trainable}, too small. "
                "Expected at least O(1e4). Check ModuleList registration and freeze policy."
            )
        self._vprint(
            f"[freeze] freeze_pinn=true, train_amplitude_net={train_amplitude_net}, "
            f"train_old_amp_head={train_old_amp_head}, "
            f"amplitude_net_trainable_params={amp_trainable}"
        )

    def _print_trainable_parameters(self, tag: str = ""):
        total = 0
        rows = []
        for name, p in self.model.named_parameters():
            if p.requires_grad:
                n = p.numel()
                total += n
                rows.append((name, tuple(p.shape), n))

        self._vprint("=" * 100)
        self._vprint(f"[trainable-params] {tag} total = {total}")
        for name, shape, n in rows:
            self._vprint(f"[trainable-params] {name:90s} {str(shape):30s} {n}")
        self._vprint("=" * 100)
        return total

    def _infinity_loss_weights(self) -> tuple[float, float]:
        w_inf = self._linear_schedule(
            self.global_step, self.inf_weight_ramp_start, self.inf_weight_ramp_end,
            self.inf_weight_init, self.inf_weight_final,
        )
        w_B = self._linear_schedule(
            self.global_step, self.inf_weight_B_decay_start, self.inf_weight_B_decay_end,
            self.inf_weight_B_init, self.inf_weight_B_final,
        )
        return w_inf, w_B

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

    def _tensor_scalar_for_json(self, x: torch.Tensor, name: str = "value", tol: float = 1.0e-10):
        """
        把 0-d tensor 转成可写入 json 的 Python 标量。

        若是 complex：
        - imag 很小，则自动取 real
        - imag 不小，则返回 {real, imag}，避免直接 float(complex) 报错
        """
        v = x.detach().cpu().item()

        if isinstance(v, complex):
            if abs(v.imag) < tol:
                return float(v.real)
            # 对 lambda 这种理论上应为实数的量，这里也可以给出 warning
            self._vprint(f"[warn] {name} has non-negligible imaginary part: {v}")
            return {
                "real": float(v.real),
                "imag": float(v.imag),
            }

        return float(v)
    def _preload_lambda_cache_from_probe(self, probe_json: str | Path):
        path = Path(probe_json)
        if not path.exists():
            return

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            self._vprint(f"[lambda-cache] skip loading {path}: {e}")
            return

        problem_cfg = self.physics_cfg["problem"]
        default_l = int(problem_cfg["l"])
        default_m = int(problem_cfg["m"])
        default_s = int(problem_cfg.get("s", -2))
        loaded = 0

        for item in data.get("lambda_cache", {}).values():
            lam_re = item.get("lambda_re")
            lam_im = item.get("lambda_im")
            if lam_re is None or lam_im is None:
                continue
            key = (
                "lambda",
                round(float(item["a"]), 12),
                round(float(item["omega"]), 12),
                int(item.get("l", default_l)),
                int(item.get("m", default_m)),
                int(item.get("s", default_s)),
            )
            self.cache.lambda_cache[key] = complex(float(lam_re), float(lam_im))
            loaded += 1

        if loaded == 0:
            meta = data.get("meta", {})
            rec_l = int(meta.get("l", default_l))
            rec_m = int(meta.get("m", default_m))
            rec_s = int(meta.get("s", default_s))
            for item in data.get("records", []):
                lam_re = item.get("lambda_re")
                lam_im = item.get("lambda_im")
                if lam_re is None or lam_im is None:
                    continue
                key = (
                    "lambda",
                    round(float(item["a"]), 12),
                    round(float(item["omega"]), 12),
                    rec_l,
                    rec_m,
                    rec_s,
                )
                self.cache.lambda_cache[key] = complex(float(lam_re), float(lam_im))
                loaded += 1

        if loaded > 0:
            self._vprint(f"[lambda-cache] preloaded {loaded} entries from {path}")

    def _seed_everything(self, seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def close(self):
        if self.anchor_mma_sampler is not None:
            self.anchor_mma_sampler.close()
        if self.viz_mma_sampler is not None and self.viz_mma_sampler is not self.anchor_mma_sampler:
            self.viz_mma_sampler.close()

    # =========================================================
    # setup
    # =========================================================
    def _split_train_val(self):
        n = len(self.patch_aw)
        n_val = max(1, min(self.val_param_samples, int(round(self.val_fraction * n))))
        perm = np.random.permutation(n)
        val_idx = perm[:n_val]
        train_idx = perm[n_val:]

        if len(train_idx) == 0:
            train_idx = val_idx[:1]
            val_idx = val_idx[1:]

        self.train_aw = self.patch_aw[train_idx]
        self.train_uv = self.patch_uv[train_idx]

        self.val_aw = self.patch_aw[val_idx] if len(val_idx) > 0 else self.patch_aw[:1]
        self.val_uv = self.patch_uv[val_idx] if len(val_idx) > 0 else self.patch_uv[:1]

    def _build_validation_metadata(self):
        val_a = []
        val_omega = []
        val_u = []
        val_v = []
        val_lambda = []

        for i in range(len(self.val_aw)):
            a_i = torch.tensor(float(self.val_aw[i, 0]), device=self.device, dtype=self.dtype)
            omega_i = torch.tensor(float(self.val_aw[i, 1]), device=self.device, dtype=self.dtype)

            lam_i = get_lambda_from_cfg(self.physics_cfg, self.cache, a_i, omega_i)

            val_a.append(a_i)
            val_omega.append(omega_i)
            val_u.append(torch.tensor(float(self.val_uv[i, 0]), device=self.device, dtype=self.dtype))
            val_v.append(torch.tensor(float(self.val_uv[i, 1]), device=self.device, dtype=self.dtype))
            val_lambda.append(lam_i)

        self.val_meta = {
            "a": torch.stack(val_a, dim=0),
            "omega": torch.stack(val_omega, dim=0),
            "u": torch.stack(val_u, dim=0),
            "v": torch.stack(val_v, dim=0),
            "lambda": torch.stack(val_lambda, dim=0),
        }

    def _init_best_val_case_tracking(self):
        n_val = int(self.val_meta["a"].shape[0])
        self.best_val_case_means = np.full(n_val, float("inf"), dtype=np.float64)
        self.best_val_case_steps = np.full(n_val, -1, dtype=np.int64)

    def _serialize_best_val_cases(self):
        rows = []
        if self.best_val_case_means is None or self.best_val_case_steps is None:
            return rows
        n_val = int(self.val_meta["a"].shape[0])

        for i in range(n_val):
            #检查λ虚部是否显著非零，若是则在日志中标记该案例可能具有更复杂的行为
            lam_i = self.val_meta["lambda"][i]
            lam_imag = lam_i.imag if isinstance(lam_i, complex) else 0
            if abs(lam_imag) > 1.0e-6:
                self._vprint(
                    f"[val-case-{i}] warning: λ has significant imaginary part ({lam_i}), "
                    "which may indicate more complex behavior and affect training dynamics."
                )


            rows.append(
                {
                    "case_index": int(i),
                    "a": float(self.val_meta["a"][i].detach().cpu().item()),
                    "omega": float(self.val_meta["omega"][i].detach().cpu().item()),
                    "u": float(self.val_meta["u"][i].detach().cpu().item()),
                    "v": float(self.val_meta["v"][i].detach().cpu().item()),
                    "lambda": float(self.val_meta["lambda"][i].real.detach().cpu().item()),
                    "best_y_mean": float(self.best_val_case_means[i]),
                    "best_step": int(self.best_val_case_steps[i]),
                }
            )
        return rows

    def _update_best_val_cases(self, val_metrics: dict):
        case_losses = val_metrics.get("case_losses", None)
        if case_losses is None:
            return 0
        if self.best_val_case_means is None or self.best_val_case_steps is None:
            self._init_best_val_case_tracking()
        improved_count = 0
        for i, loss_i in enumerate(case_losses):
            loss_i = float(loss_i)
            best_i = float(self.best_val_case_means[i])
            if np.isfinite(best_i):
                rel_improve_i = (best_i - loss_i) / max(abs(best_i), 1.0e-12)
                improved_i = rel_improve_i > self.es_min_rel_improve
            else:
                improved_i = True
            if improved_i:
                self.best_val_case_means[i] = loss_i
                self.best_val_case_steps[i] = int(self.global_step)
                improved_count += 1
        if self.best_val_case_means is not None and len(self.best_val_case_means) > 0:
            self.best_val_mean = float(np.mean(self.best_val_case_means))
        return improved_count

    def _init_run_dirs(self):
        if self.resume_run_dir is not None:
            self.run_dir = self.resume_run_dir
            self.ckpt_dir = self.run_dir / "checkpoints"
            self.fig_dir = self.run_dir / "figures"
            self.log_dir = self.run_dir / "logs"
            self.ckpt_dir.mkdir(parents=True, exist_ok=True)
            self.fig_dir.mkdir(parents=True, exist_ok=True)
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.anchor_fail_log = self.log_dir / self.anchor_fail_log_name
            self.history_jsonl = self.log_dir / "history.jsonl"
            self.summary_json = self.log_dir / "summary.json"
            return

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = (
            f"{ts}"
            f"_patch_{self.patch.patch_id:03d}"
            f"_comp_{self.patch.component_id}"
            f"_u_{self.patch.u_center:.3f}"
            f"_v_{self.patch.v_center:.3f}"
        )
        self.run_dir = self.output_root / run_name
        self.ckpt_dir = self.run_dir / "checkpoints"
        self.fig_dir = self.run_dir / "figures"
        self.log_dir = self.run_dir / "logs"

        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.fig_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.anchor_fail_log = self.log_dir / self.anchor_fail_log_name
        self.history_jsonl = self.log_dir / "history.jsonl"
        self.summary_json = self.log_dir / "summary.json"

    def _load_state_dict_strict_amp_head(self, state_dict, tag: str = ""):
        """Load state_dict allowing only amp_head.* and amplitude_net.* keys to be missing."""
        incompat = self.model.load_state_dict(state_dict, strict=False)
        allowed_missing_prefixes = ("amp_head", "amplitude_net")
        amp_missing = [k for k in incompat.missing_keys if any(k.startswith(p) for p in allowed_missing_prefixes)]
        other_missing = [k for k in incompat.missing_keys if not any(k.startswith(p) for p in allowed_missing_prefixes)]
        if other_missing:
            raise RuntimeError(
                f"[{tag}] Unexpected missing keys (model has parameters not in checkpoint): {other_missing}"
            )
        if incompat.unexpected_keys:
            raise RuntimeError(
                f"[{tag}] Unexpected keys in checkpoint (checkpoint has parameters not in model): {incompat.unexpected_keys}"
            )
        if amp_missing:
            self._vprint(f"[{tag}] amplitude keys missing (will be init-trained): {len(amp_missing)} keys")
        return incompat

    def _load_init_checkpoint(self):
        ckpt = torch.load(self.init_checkpoint, map_location=self.device)
        state_dict = ckpt.get("model_state_dict", ckpt)
        self._load_state_dict_strict_amp_head(state_dict, tag="init")

        if self.init_load_optimizer and "optimizer_state_dict" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])

        self._vprint(f"[init] loaded checkpoint: {self.init_checkpoint}")

    def _load_resume_checkpoint(self):
        ckpt = torch.load(self.resume_checkpoint, map_location=self.device)
        state_dict = ckpt.get("model_state_dict", ckpt)
        self._load_state_dict_strict_amp_head(state_dict, tag="resume")

        if "optimizer_state_dict" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])

        self.global_step = int(ckpt.get("step", 0))
        self.best_val_mean = float(ckpt.get("best_val_mean", float("inf")))
        best_case_means = ckpt.get("best_val_case_means", None)
        best_case_steps = ckpt.get("best_val_case_steps", None)
        if best_case_means is not None and best_case_steps is not None:
            self.best_val_case_means = np.asarray(best_case_means, dtype=np.float64)
            self.best_val_case_steps = np.asarray(best_case_steps, dtype=np.int64)
        self._vprint(
            f"[resume] loaded checkpoint: {self.resume_checkpoint} "
            f"(step={self.global_step}, best_val_mean={self.best_val_mean:.6e})"
        )

    def _save_config_snapshot(self):
        snapshot_path = self.run_dir / "config_snapshot.yaml"
        with open(snapshot_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.full_cfg, f, allow_unicode=True, sort_keys=False)

    def _choose_reference_sample(self):
        # choose patch-center nearest point from validation pool, fallback to train pool
        candidates_aw = self.val_aw if len(self.val_aw) > 0 else self.train_aw
        candidates_uv = self.val_uv if len(self.val_uv) > 0 else self.train_uv

        du = candidates_uv[:, 0] - self.patch.u_center
        dv = candidates_uv[:, 1] - self.patch.v_center
        idx = int(np.argmin(du * du + dv * dv))

        return {
            "a": torch.tensor(float(candidates_aw[idx, 0]), device=self.device, dtype=self.dtype),
            "omega": torch.tensor(float(candidates_aw[idx, 1]), device=self.device, dtype=self.dtype),
            "u": torch.tensor(float(candidates_uv[idx, 0]), device=self.device, dtype=self.dtype),
            "v": torch.tensor(float(candidates_uv[idx, 1]), device=self.device, dtype=self.dtype),
        }

    # =========================================================
    # sampling
    # =========================================================
    def sample_param_batch(self):
        n_pool = len(self.train_aw)
        replace = n_pool < self.batch_size
        idx = np.random.choice(n_pool, size=self.batch_size, replace=replace)

        aw = self.train_aw[idx]
        uv = self.train_uv[idx]

        a_batch = torch.tensor(aw[:, 0], device=self.device, dtype=self.dtype)
        omega_batch = torch.tensor(aw[:, 1], device=self.device, dtype=self.dtype)
        u_batch = torch.tensor(uv[:, 0], device=self.device, dtype=self.dtype)
        v_batch = torch.tensor(uv[:, 1], device=self.device, dtype=self.dtype)
        return a_batch, omega_batch, u_batch, v_batch

    def sample_y_interior(self):
        y_base = sample_points_chebyshev_grid(
            n_points=self.n_interior,
            y_min=-0.99,
            y_max=0.99,
            device=self.device,
            dtype=self.dtype,
        )
        if self.n_interior_outer_extra > 0:
            y_extra = sample_points_uniform_grid(
                n_points=self.n_interior_outer_extra,
                y_min=-0.99,
                y_max=0.0,
                device=self.device,
                dtype=self.dtype,
                shuffle=False,
            )
            y = torch.cat([y_base, y_extra], dim=0)
        else:
            y = y_base
        return y.clone().requires_grad_(True)

    def sample_y_anchor(self):
        y = sample_points_chebyshev_grid(
            n_points=self.n_anchor_y,
            y_min=-0.99,
            y_max=0.99,
            device=self.device,
            dtype=self.dtype,
        )
        return y

    def sample_y_validation(self):
        y = sample_points_chebyshev_grid(
            n_points=self.val_n_points,
            y_min=-0.99,
            y_max=0.99,
            device=self.device,
            dtype=self.dtype,
        )
        return y.clone().requires_grad_(True)

    # =========================================================
    # aux
    # =========================================================
    def resolve_aux_batch(self, a_batch, omega_batch):
        lambda_list = []
        for i in range(a_batch.shape[0]):
            lam_i = get_lambda_from_cfg(self.physics_cfg, self.cache, a_batch[i], omega_batch[i])
            lambda_list.append(lam_i)
        lambda_batch = torch.stack(lambda_list, dim=0)
        return lambda_batch

    # =========================================================
    # MMA anchor
    # =========================================================
    def query_mma_Rin_batch(self, a_batch, omega_batch, y_anchors):
        """
        返回:
            R_mma_ok: (B_ok, N_anchor) or None
            ok_mask:  (B_ok,) long or None
            failed:   list[dict]
        """
        problem_cfg = self.physics_cfg["problem"]
        M = float(problem_cfg.get("M", 1.0))
        s = int(problem_cfg.get("s", -2))
        l = int(problem_cfg.get("l", 2))
        m = int(problem_cfg.get("m", 2))

        x_anchors = 0.5 * (y_anchors + 1.0)

        rows = []
        ok_idx = []
        failed = []

        for i in range(a_batch.shape[0]):
            a_i = float(a_batch[i].detach().cpu().item())
            omega_i = float(omega_batch[i].detach().cpu().item())

            try:
                rp_i = r_plus(a_batch[i], M)
                r_i = r_from_x(x_anchors, rp_i).detach().cpu().numpy()

                if self.anchor_backend == "pybhpt":
                    _, Rin_i = compute_pybhpt_solution(
                        a=a_i,
                        omega=omega_i,
                        ell=l,
                        m=m,
                        r_grid=r_i,
                        timeout=self.anchor_pybhpt_timeout,
                    )
                else:
                    Rin_i = self.anchor_mma_sampler.evaluate_rin_at_points_direct(
                        s=s,
                        l=l,
                        m=m,
                        a=a_i,
                        omega=omega_i,
                        r_query=r_i,
                    )
                rows.append(Rin_i)
                ok_idx.append(i)

            except Exception as e:
                failed.append(
                    {
                        "batch_index": int(i),
                        "a": a_i,
                        "omega": omega_i,
                        "err": str(e),
                    }
                )

        if len(ok_idx) == 0:
            return None, None, failed

        arr = np.stack(rows, axis=0)
        ok_mask = torch.tensor(ok_idx, device=self.device, dtype=torch.long)
        R_ok = torch.as_tensor(arr, device=self.device, dtype=torch.complex128)
        return R_ok, ok_mask, failed

    def _append_anchor_failures(self, failed: list[dict], step: int):
        if len(failed) == 0:
            return
        with open(self.anchor_fail_log, "a", encoding="utf-8") as f:
            for item in failed:
                row = dict(item)
                row["step"] = int(step)
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _compute_anchor_weight_eff(self, loss_pde: torch.Tensor, loss_anchor: torch.Tensor) -> float:
        pde_val = float(loss_pde.detach().cpu().item())
        anc_val = float(loss_anchor.detach().cpu().item())
        if anc_val <= 0.0:
            return 0.0
        return self.anchor_target_ratio * pde_val / (anc_val + 1.0e-12)

    # =========================================================
    # one step
    # =========================================================
    def train_one_step(self):
        self.model.train()
        self.optimizer.zero_grad()

        a_batch, omega_batch, u_batch, v_batch = self.sample_param_batch()
        lambda_batch = self.resolve_aux_batch(a_batch, omega_batch)

        # ---- amplitude_pretrain stage: only amplitude teacher loss ----
        if self.inf_enabled and self.inf_stage == "amplitude_pretrain":
            if self.inf_coeff_cache is None:
                raise RuntimeError("amplitude_pretrain requires inf_coeff_cache")

            Binc_spec, Bref_spec = self.inf_coeff_cache.get_batch(
                a_batch=a_batch, omega_batch=omega_batch, lambda_batch=lambda_batch,
            )
            loss_inf, loss_B, loss_amp, inf_info = compute_infinity_asymptotic_loss(
                model=self.model, cfg=self.physics_cfg,
                a_batch=a_batch, omega_batch=omega_batch, lambda_batch=lambda_batch,
                u_batch=u_batch, v_batch=v_batch,
                Binc_spec=Binc_spec, Bref_spec=Bref_spec,
                r_points=self.inf_r_points,
                beta=self.inf_beta, relative=self.inf_relative, eps=self.inf_eps,
                amplitude_mode=self.inf_amplitude_mode,
                hybrid_eta=0.0,
            )
            w_amp = self._infinity_amp_weight()
            total_loss = w_amp * loss_amp

            total_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()

            info = {
                "step": int(self.global_step),
                "stage": self.inf_stage,
                "amplitude_mode": self.inf_amplitude_mode,
                "total_loss": float(total_loss.detach().cpu().item()),
                "loss_amp": float(loss_amp.detach().cpu().item()),
                "loss_inf": float(loss_inf.detach().cpu().item()),
                "loss_B": float(loss_B.detach().cpu().item()),
                "weight_amp": float(w_amp),
                "weight_inf": 0.0,
                "weight_B": 0.0,
                "hybrid_eta": 0.0,
                "grad_norm": float(grad_norm.detach().cpu().item()),
                "loss_interior": 0.0,
                "loss_boundary": 0.0,
                "loss_pde": 0.0,
                "loss_anchor": 0.0,
                "loss_coeff_reg": 0.0,
                "loss_integral": 0.0,
                "anchor_weight_eff": 0.0,
                "anchor_success_count": 0,
                "anchor_failed_count": 0,
            }
            for k, v in inf_info.items():
                info[f"inf_{k}"] = v
            return info

        # ---- joint / hybrid stages: full training ----
        y_interior = self.sample_y_interior()
        y_boundary = torch.empty(0, device=self.device, dtype=self.dtype)

        # Compute model output and ODE coefficients once (shared by PDE + integral losses)
        from physical_ansatz.residual_pinn import compute_f_derivatives_autograd
        from physical_ansatz.teukolsky_coeffs import coeffs_x
        from physical_ansatz.transform_y import (
            transform_coeffs_x_to_y, transform_coeffs_x_to_y_S,
            horizon_regularity_slope as _horizon_slope,
        )

        M_phys = float(self.physics_cfg["problem"].get("M", 1.0))
        s_phys = int(self.physics_cfg["problem"].get("s", -2))
        m_phys = int(self.physics_cfg["problem"].get("m", 2))

        output_type = getattr(self.model, "output_type", "f")

        if output_type == "S":
            # S(y) output: smooth shape function, S-PDE residual
            S_int, S_y_int, S_yy_int = compute_f_derivatives_autograd(
                self.model, a_batch, omega_batch, y_interior,
                u_batch=u_batch, v_batch=v_batch,
            )

            x_int = (y_interior + 1.0) / 2.0
            # r = r+/x: each batch element has different r+
            rp_vals = torch.stack([r_plus(a_batch[i], M_phys) for i in range(a_batch.shape[0])])
            r_int = rp_vals.unsqueeze(-1) / x_int.unsqueeze(0)  # (B, Ny)

            A2_l, A1_l, A0_l = [], [], []
            for i in range(a_batch.shape[0]):
                A2i, A1i, A0i = coeffs_x(
                    x=x_int, a=a_batch[i], omega=omega_batch[i],
                    m=m_phys, lambda_=lambda_batch[i], s=s_phys, M=M_phys,
                )
                A2_l.append(A2i); A1_l.append(A1i); A0_l.append(A0i)
            A2_int = torch.stack(A2_l, dim=0)
            A1_int = torch.stack(A1_l, dim=0)
            A0_int = torch.stack(A0_l, dim=0)

            slope_int = _horizon_slope(
                a=a_batch, omega=omega_batch, lambda_=lambda_batch,
                m=m_phys, M=M_phys, s=s_phys,
            )

            # S-PDE: D2*S_yy + D1*S_y + D0*S = 0
            D2_int, D1_int, D0_int = transform_coeffs_x_to_y_S(
                A2_int, A1_int, A0_int, r_int, a_batch, omega_batch,
                m=m_phys, M=M_phys, s=s_phys,
            )
            residual_int = D2_int * S_yy_int + D1_int * S_y_int + D0_int * S_int
            pointwise_interior = torch.abs(residual_int) ** 2
            if self.normalize_residual:
                scale = (1.0 + torch.abs(D2_int.detach() * S_yy_int) ** 2
                         + torch.abs(D1_int.detach() * S_y_int) ** 2
                         + torch.abs(D0_int.detach() * S_int) ** 2)
                pointwise_interior = pointwise_interior / scale.clamp_min(1e-12)
            loss_pde = torch.mean(pointwise_interior)

            # For integral consistency: pass S directly
            _f_arg, _f_y_arg = None, None
            _S_arg, _S_y_arg = S_int, S_y_int
        else:
            # Original f(y) path
            f_int, f_y_int, f_yy_int = compute_f_derivatives_autograd(
                self.model, a_batch, omega_batch, y_interior,
                u_batch=u_batch, v_batch=v_batch,
            )

            x_int = (y_interior + 1.0) / 2.0
            A2_l, A1_l, A0_l = [], [], []
            for i in range(a_batch.shape[0]):
                A2i, A1i, A0i = coeffs_x(
                    x=x_int, a=a_batch[i], omega=omega_batch[i],
                    m=m_phys, lambda_=lambda_batch[i], s=s_phys, M=M_phys,
                )
                A2_l.append(A2i); A1_l.append(A1i); A0_l.append(A0i)
            A2_int = torch.stack(A2_l, dim=0)
            A1_int = torch.stack(A1_l, dim=0)
            A0_int = torch.stack(A0_l, dim=0)

            slope_int = _horizon_slope(
                a=a_batch, omega=omega_batch, lambda_=lambda_batch,
                m=m_phys, M=M_phys, s=s_phys,
            )

            # PDE residual loss
            B2_int, B1_int, B0_int, rhs = transform_coeffs_x_to_y(
                A2_int, A1_int, A0_int, y_interior, slope=slope_int,
            )
            residual_int = B2_int * f_yy_int + B1_int * f_y_int + B0_int * f_int - rhs
            pointwise_interior = torch.abs(residual_int) ** 2
            if self.normalize_residual:
                scale = (1.0 + torch.abs(B2_int.detach() * f_yy_int) ** 2
                         + torch.abs(B1_int.detach() * f_y_int) ** 2
                         + torch.abs(B0_int.detach() * f_int) ** 2
                         + torch.abs(rhs.detach()) ** 2)
                pointwise_interior = pointwise_interior / scale.clamp_min(1e-12)
            loss_pde = torch.mean(pointwise_interior)

            _f_arg, _f_y_arg = f_int, f_y_int
            _S_arg, _S_y_arg = None, None

        total_loss = loss_pde
        info = {
            "loss_interior": float(loss_pde.detach().cpu().item()),
            "loss_boundary": 0.0,
            "total_loss": float(loss_pde.detach().cpu().item()),
        }
        info["loss_pde"] = float(loss_pde.detach().cpu().item())
        info["loss_anchor"] = 0.0
        info["loss_coeff_reg"] = 0.0
        info["loss_integral"] = 0.0
        info["anchor_weight_eff"] = 0.0
        info["anchor_success_count"] = 0
        info["anchor_failed_count"] = 0

        # Integral consistency loss (uses pre-computed outputs + ODE coefficients)
        integral_weight = float(self.atlas_train_cfg.get("integral_consistency_weight", 0.0))
        if integral_weight > 0 and self.int_anneal_enabled:
            step = self.global_step
            if step < self.int_weight_decay_start:
                integral_weight = self.int_weight_init
            elif step > self.int_weight_decay_end:
                integral_weight = self.int_weight_final
            else:
                frac = (step - self.int_weight_decay_start) / (self.int_weight_decay_end - self.int_weight_decay_start)
                integral_weight = self.int_weight_init + frac * (self.int_weight_final - self.int_weight_init)
        if integral_weight > 0:
            loss_integral = compute_integral_consistency_loss(
                model=self.model,
                cfg=self.physics_cfg,
                a_batch=a_batch,
                omega_batch=omega_batch,
                lambda_batch=lambda_batch,
                y_interior=y_interior,
                u_batch=u_batch,
                v_batch=v_batch,
                n_anchors=int(self.atlas_train_cfg.get("integral_n_anchors", 32)),
                _f=_f_arg,
                _f_y=_f_y_arg,
                _S=_S_arg,
                _S_y=_S_y_arg,
                _A2=A2_int,
                _A1=A1_int,
                _A0=A0_int,
                _slope=slope_int,
            )
            if torch.isfinite(loss_integral):
                total_loss = total_loss + integral_weight * loss_integral
                info["loss_integral"] = float(loss_integral.detach().cpu().item())

        # Chebyshev coefficient regularization (spectral decay prior)
        if self.model_type in ("cheb", "cheb_deeponet"):
            coeff_reg_weight = float(self.atlas_train_cfg.get("cheb_coeff_reg_weight", 0.0))
            if coeff_reg_weight > 0:
                coeff = self.model.compute_coefficients(a_batch, omega_batch, u=u_batch, v=v_batch)
                n_weights = torch.arange(self.cheb_N, device=coeff.device, dtype=coeff.real.dtype)
                # Penalize |c_n|^2 * n (higher modes penalized more)
                coeff_reg = coeff_reg_weight * torch.mean(
                    (1.0 + n_weights) * (coeff.real ** 2 + coeff.imag ** 2)
                )
                total_loss = total_loss + coeff_reg
                info["loss_coeff_reg"] = float(coeff_reg.detach().cpu().item())

        if self.anchor_enabled:
            y_anchor = self.sample_y_anchor()
            R_mma_ok, ok_mask, failed = self.query_mma_Rin_batch(
                a_batch=a_batch,
                omega_batch=omega_batch,
                y_anchors=y_anchor,
            )
            self._append_anchor_failures(failed, step=self.global_step)
            info["anchor_failed_count"] = len(failed)

            if R_mma_ok is not None and ok_mask is not None and ok_mask.numel() > 0:
                loss_anchor = compute_data_anchor_loss(
                    model=self.model,
                    cfg=self.physics_cfg,
                    a_batch=a_batch[ok_mask],
                    omega_batch=omega_batch[ok_mask],
                    lambda_batch=lambda_batch[ok_mask],
                    y_anchors=y_anchor,
                    R_mma_anchors=R_mma_ok,
                    relative=False,
                    eps=1.0e-12,
                    u_batch=u_batch[ok_mask],
                    v_batch=v_batch[ok_mask],
                )

                if torch.isfinite(loss_anchor):
                    anchor_weight_eff = self._compute_anchor_weight_eff(loss_pde, loss_anchor)
                    total_loss = loss_pde + anchor_weight_eff * loss_anchor

                    info["loss_anchor"] = float(loss_anchor.detach().cpu().item())
                    info["anchor_weight_eff"] = float(anchor_weight_eff)
                    info["anchor_success_count"] = int(ok_mask.numel())
                else:
                    info["anchor_failed_count"] += ok_mask.numel()

        # Near-infinity asymptotic loss
        loss_inf = torch.zeros((), device=self.device, dtype=self.dtype)
        loss_B = torch.zeros((), device=self.device, dtype=self.dtype)
        loss_amp = torch.zeros((), device=self.device, dtype=self.dtype)
        w_inf = 0.0
        w_B = 0.0
        w_amp = 0.0
        hybrid_eta = 0.0
        inf_info = {}

        if self.inf_enabled and self.inf_coeff_cache is not None:
            Binc_spec, Bref_spec = self.inf_coeff_cache.get_batch(
                a_batch=a_batch, omega_batch=omega_batch, lambda_batch=lambda_batch,
            )
            hybrid_eta = self._infinity_hybrid_eta()
            loss_inf, loss_B, loss_amp, inf_info = compute_infinity_asymptotic_loss(
                model=self.model, cfg=self.physics_cfg,
                a_batch=a_batch, omega_batch=omega_batch, lambda_batch=lambda_batch,
                u_batch=u_batch, v_batch=v_batch,
                Binc_spec=Binc_spec, Bref_spec=Bref_spec,
                r_points=self.inf_r_points,
                beta=self.inf_beta, relative=self.inf_relative, eps=self.inf_eps,
                amplitude_mode=self.inf_amplitude_mode,
                hybrid_eta=hybrid_eta,
            )
            w_inf, w_B = self._infinity_loss_weights()
            w_amp = self._infinity_amp_weight()
            total_loss = total_loss + w_inf * loss_inf + w_B * loss_B + w_amp * loss_amp

        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optimizer.step()

        # ---- SGDR LR update ----
        if self.use_sgdr:
            steps_in_cycle = self.global_step - self.sgdr_cycle_start
            if steps_in_cycle >= self.sgdr_T_cur:
                self.sgdr_cycle_start = self.global_step
                self.sgdr_T_cur = self.sgdr_T_cur * self.sgdr_T_mult
                steps_in_cycle = 0
            t_frac = steps_in_cycle / max(self.sgdr_T_cur, 1)
            lr = self.lr * (self.sgdr_eta_min + 0.5 * (1.0 - self.sgdr_eta_min) * (1.0 + math.cos(math.pi * t_frac)))
            for pg in self.optimizer.param_groups:
                pg['lr'] = lr

        info["total_loss"] = float(total_loss.detach().cpu().item())
        info["grad_norm"] = float(grad_norm.detach().cpu().item())
        info["loss_inf"] = float(loss_inf.detach().cpu().item())
        info["loss_B"] = float(loss_B.detach().cpu().item())
        info["loss_amp"] = float(loss_amp.detach().cpu().item())
        info["weight_inf"] = float(w_inf)
        info["weight_B"] = float(w_B)
        info["weight_amp"] = float(w_amp)
        info["hybrid_eta"] = float(hybrid_eta)
        info["amplitude_mode"] = self.inf_amplitude_mode
        for k, v in inf_info.items():
            info[f"inf_{k}"] = v
        return info

    # =========================================================
    # L-BFGS refinement (for Chebyshev spectral methods)
    # =========================================================
    def _train_lbfgs(self, steps: int, val_every: int = 10):
        """
        L-BFGS refinement after Adam warmup. Uses fixed parameter/collocation
        batches for deterministic optimization.
        """
        self._vprint(f"[lbfgs] starting {steps} L-BFGS steps ...")
        optimizer = torch.optim.LBFGS(
            self.model.parameters(),
            lr=0.5,
            max_iter=20,
            max_eval=25,
            tolerance_grad=1e-12,
            tolerance_change=1e-14,
            history_size=100,
            line_search_fn="strong_wolfe",
        )

        # Fixed batches for deterministic L-BFGS
        n_fixed = 4
        fixed_batches = []
        for _ in range(n_fixed):
            fixed_batches.append((
                *self.sample_param_batch(),
                self.sample_y_interior(),
            ))

        final_val = None
        step_count = [0]
        pbar = trange(steps, desc=f"lbfgs patch {self.patch.patch_id}", dynamic_ncols=True)

        for _ in pbar:
            batch_idx = step_count[0] % n_fixed
            a_b, omega_b, u_b, v_b = fixed_batches[batch_idx][:4]
            y_int = fixed_batches[batch_idx][4]

            def closure():
                optimizer.zero_grad()
                lambda_b = self.resolve_aux_batch(a_b, omega_b)
                y_boundary = torch.empty(0, device=self.device, dtype=self.dtype)

                loss_pde, _ = pinn_residual_loss(
                    model=self.model,
                    cfg=self.physics_cfg,
                    a_batch=a_b,
                    omega_batch=omega_b,
                    lambda_batch=lambda_b,
                    y_interior=y_int,
                    y_boundary=y_boundary,
                    weight_interior=1.0,
                    weight_boundary=0.0,
                    normalize_residual=self.normalize_residual,
                    residual_scale_eps=1.0e-12,
                    return_pointwise=False,
                    u_batch=u_b,
                    v_batch=v_b,
                )
                loss_pde.backward()
                return loss_pde

            loss_val = optimizer.step(closure)
            step_count[0] += 1

            if step_count[0] % val_every == 0 or step_count[0] == 1:
                final_val = self.validate()
                improved = self._update_best_val_cases(final_val)
                final_val["case_improved_count"] = int(improved)
                if improved:
                    self._save_checkpoint(self.ckpt_dir / "best_model.pt", val_metrics=final_val)

            pbar.set_postfix({
                "loss": f"{loss_val.item():.2e}",
                "val": f"{final_val['val_mean']:.2e}" if final_val is not None else "-",
                "best": f"{self.best_val_mean:.2e}" if np.isfinite(self.best_val_mean) else "-",
            })

        self._vprint(f"[lbfgs] done. best_val_mean={self.best_val_mean:.6e}")
        return final_val

    # =========================================================
    # validation
    # =========================================================
    def _validate_single_case(self, a_val, omega_val, u_val, v_val, lambda_val):
        was_training = self.model.training
        self.model.eval()
        try:
            y_val = self.sample_y_validation()
            y_boundary = torch.empty(0, device=self.device, dtype=self.dtype)

            loss, _ = pinn_residual_loss(
                model=self.model,
                cfg=self.physics_cfg,
                a_batch=a_val.unsqueeze(0),
                omega_batch=omega_val.unsqueeze(0),
                lambda_batch=lambda_val.unsqueeze(0),
                y_interior=y_val,
                y_boundary=y_boundary,
                weight_interior=1.0,
                weight_boundary=0.0,
                normalize_residual=self.normalize_residual,
                residual_scale_eps=1.0e-12,
                return_pointwise=False,
                u_batch=u_val.unsqueeze(0),
                v_batch=v_val.unsqueeze(0),
            )
            return float(loss.detach().cpu().item())
        finally:
            if was_training:
                self.model.train()

    def validate(self):
        losses = []
        worst_loss = -float("inf")
        worst_idx = -1
        case_metrics = []

        for i in range(self.val_meta["a"].shape[0]):
            loss_i = self._validate_single_case(
                a_val=self.val_meta["a"][i],
                omega_val=self.val_meta["omega"][i],
                u_val=self.val_meta["u"][i],
                v_val=self.val_meta["v"][i],
                lambda_val=self.val_meta["lambda"][i],
            )
            losses.append(loss_i)
            case_metrics.append(
                {
                    "case_index": int(i),
                    "a": float(self.val_meta["a"][i].detach().cpu().item()),
                    "omega": float(self.val_meta["omega"][i].detach().cpu().item()),
                    "u": float(self.val_meta["u"][i].detach().cpu().item()),
                    "v": float(self.val_meta["v"][i].detach().cpu().item()),
                    "lambda": self._tensor_scalar_for_json(self.val_meta["lambda"][i], name="lambda"),
                    "y_mean_loss": float(loss_i),
                }
            )
            if loss_i > worst_loss:
                worst_loss = loss_i
                worst_idx = i

        val_mean = float(np.mean(losses)) if losses else float("inf")
        metrics = {
            "val_mean": val_mean,
            "val_worst": float(worst_loss if losses else float("inf")),
            "n_val_samples": int(len(losses)),
            "case_losses": [float(x) for x in losses],
            "case_metrics": case_metrics,
        }
        if worst_idx >= 0:
            metrics["worst_a"] = float(self.val_meta["a"][worst_idx].detach().cpu().item())
            metrics["worst_omega"] = float(self.val_meta["omega"][worst_idx].detach().cpu().item())
            metrics["worst_u"] = float(self.val_meta["u"][worst_idx].detach().cpu().item())
            metrics["worst_v"] = float(self.val_meta["v"][worst_idx].detach().cpu().item())
        else:
            metrics["worst_a"] = None
            metrics["worst_omega"] = None
            metrics["worst_u"] = None
            metrics["worst_v"] = None
        return metrics

    # =========================================================
    # visualization
    # =========================================================
    def _predict_shape(self, a_t, omega_t, lambda_t, u_t, v_t, y_grid):
        M = float(self.physics_cfg["problem"].get("M", 1.0))
        m = int(self.physics_cfg["problem"].get("m", 2))
        s = int(self.physics_cfg["problem"].get("s", -2))

        f_pred = self.model(
            a_t.unsqueeze(0),
            omega_t.unsqueeze(0),
            y_grid,
            u=u_t.unsqueeze(0),
            v=v_t.unsqueeze(0),
        ).squeeze(0)

        slope = horizon_regularity_slope(
            a=a_t.unsqueeze(0),
            omega=omega_t.unsqueeze(0),
            lambda_=lambda_t.unsqueeze(0),
            m=m,
            M=M,
            s=s,
        ).squeeze(0)

        return compose_reduced_shape_from_f(
            f=f_pred,
            y=y_grid,
            slope=slope,
        )
    def _eval_pybhpt_preserve_order(self, a_scalar, omega_scalar, ell, m, r_query, timeout):
        r_query = np.asarray(r_query, dtype=float)
        order = np.argsort(r_query)
        inv_order = np.empty_like(order)
        inv_order[order] = np.arange(order.size)

        r_sorted = r_query[order]
        _, R_sorted = compute_pybhpt_solution(
            a=a_scalar,
            omega=omega_scalar,
            ell=ell,
            m=m,
            r_grid=r_sorted,
            timeout=timeout,
        )

        R_sorted = np.asarray(R_sorted, dtype=np.complex128)
        R_back = R_sorted[inv_order]
        return R_back
    def visualize_reference(self, step: int):
        problem_cfg = self.physics_cfg["problem"]
        M = float(problem_cfg.get("M", 1.0))
        l = int(problem_cfg.get("l", 2))
        m = int(problem_cfg.get("m", 2))
        s = int(problem_cfg.get("s", -2))

        a_t = self.ref_sample["a"]
        omega_t = self.ref_sample["omega"]
        u_t = self.ref_sample["u"]
        v_t = self.ref_sample["v"]

        lam = get_lambda_from_cfg(self.physics_cfg, self.cache, a_t, omega_t)
        h2 = h_factor(a_t, omega_t, m=m, M=M, s=s)

        rp = r_plus(a_t, M)
        r_min = max(self.viz_r_min, float(rp.detach().cpu().item()) + 1.0e-4)
        r_max = self.viz_r_max

        # left: uniform r
        r_grid_uniform = torch.linspace(
            r_min, r_max, self.viz_num_points, device=self.device, dtype=self.dtype
        )
        x_grid_from_r = rp / r_grid_uniform
        y_grid_from_r = 2.0 * x_grid_from_r - 1.0

        # right: cheb y
        y_min = 2.0 * float(rp.detach().cpu().item()) / r_max - 1.0
        y_max = 2.0 * float(rp.detach().cpu().item()) / r_min - 1.0
        y_grid_cheb = sample_points_chebyshev_grid(
            n_points=self.viz_num_points,
            y_min=y_min,
            y_max=y_max,
            device=self.device,
            dtype=self.dtype,
        )
        x_grid_from_y = 0.5 * (y_grid_cheb + 1.0)
        r_grid_from_y = rp / x_grid_from_y

        self.model.eval()
        with torch.no_grad():
            # prediction on r-grid
            shape_pred_r = self._predict_shape(a_t, omega_t, lam, u_t, v_t, y_grid_from_r)
            rp_r, rm_r, _, _, _ = build_prefactor_primitives(r_grid_uniform, a_t, M=M, need_rs=False)
            P_r, _, _ = Leaver_prefactors(
                r_grid_uniform, a_t, omega_t, m=m, M=M, s=s, rp=rp_r, rm=rm_r
            )
            R_pred_r = P_r * h2 * shape_pred_r

            # prediction on y-grid
            shape_pred_y = self._predict_shape(a_t, omega_t, lam, u_t, v_t, y_grid_cheb)
            rp_y, rm_y, _, _, _ = build_prefactor_primitives(r_grid_from_y, a_t, M=M, need_rs=False)
            P_y, _, _ = Leaver_prefactors(
                r_grid_from_y, a_t, omega_t, m=m, M=M, s=s, rp=rp_y, rm=rm_y
            )

        benchmark_available = False
        benchmark_status = "benchmark=off"
        a_scalar = float(a_t.detach().cpu().item())
        omega_scalar = float(omega_t.detach().cpu().item())
        r_uniform_np = r_grid_uniform.detach().cpu().numpy()
        y_cheb_np = y_grid_cheb.detach().cpu().numpy()
        R_pred_r_np = R_pred_r.detach().cpu().numpy()
        shape_pred_y_np = shape_pred_y.detach().cpu().numpy()
        R_ref_r_np = None
        shape_ref_y_np = None

        if self.viz_benchmark_backend == "spectral":
            try:
                mode = KerrMode(
                    M=M,
                    a=a_scalar,
                    omega=omega_scalar,
                    ell=l,
                    m=m,
                    lam=complex(lam.detach().cpu().item()),
                    s=s,
                )
                spectral = TeukRadAmplitudeInWithInterpolant(
                    mode=mode,
                    N_in=self.viz_spectral_N,
                    N_out=self.viz_spectral_N,
                    z_m=self.viz_spectral_z_m,
                )
                profile = spectral.profile
                R_ref_r_np = np.asarray(profile.R_of_r(r_uniform_np), dtype=np.complex128)
                R_ref_y_np = np.asarray(profile.R_of_r(r_grid_from_y.detach().cpu().numpy()), dtype=np.complex128)
                shape_ref_y_np = R_ref_y_np / (P_y.detach().cpu().numpy() * complex(h2.detach().cpu().item()))
                benchmark_available = True
                benchmark_status = f"benchmark=spectral(N={self.viz_spectral_N})"
            except Exception as e:
                benchmark_status = "benchmark=spectral-failed"
                with open(self.log_dir / "viz_failures.jsonl", "a", encoding="utf-8") as f:
                    f.write(json.dumps({
                        "step": int(step),
                        "backend": "spectral",
                        "a": a_scalar,
                        "omega": omega_scalar,
                        "err": str(e),
                    }, ensure_ascii=False) + "\n")

        fig, axes = plt.subplots(3, 2, figsize=(10, 10), sharex=False)

        axes[0, 0].plot(r_uniform_np, np.real(R_pred_r_np), label="Pred Re(R)", lw=1.6)
        if benchmark_available:
            axes[0, 0].plot(r_uniform_np, np.real(R_ref_r_np), "--", label="ref Re(R)", lw=1.0)
        axes[0, 0].set_ylabel("Re(R)")
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)

        axes[1, 0].plot(r_uniform_np, np.imag(R_pred_r_np), label="Pred Im(R)", lw=1.6)
        if benchmark_available:
            axes[1, 0].plot(r_uniform_np, np.imag(R_ref_r_np), "--", label="ref Im(R)", lw=1.0)
        axes[1, 0].set_ylabel("Im(R)")
        axes[1, 0].legend()
        axes[1, 0].grid(alpha=0.3)

        axes[2, 0].plot(r_uniform_np, np.abs(R_pred_r_np), label="Pred |R|", lw=1.6)
        if benchmark_available:
            axes[2, 0].plot(r_uniform_np, np.abs(R_ref_r_np), "--", label="ref |R|", lw=1.0)
        axes[2, 0].set_ylabel("|R|")
        axes[2, 0].set_xlabel("r")
        axes[2, 0].legend()
        axes[2, 0].grid(alpha=0.3)

        axes[0, 1].plot(y_cheb_np, np.real(shape_pred_y_np), label="Pred Re(S)", lw=1.6)
        if benchmark_available:
            axes[0, 1].plot(y_cheb_np, np.real(shape_ref_y_np), "--", label="ref Re(S)", lw=1.0)
        axes[0, 1].set_ylabel("Re(S)")
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)

        axes[1, 1].plot(y_cheb_np, np.imag(shape_pred_y_np), label="Pred Im(S)", lw=1.6)
        if benchmark_available:
            axes[1, 1].plot(y_cheb_np, np.imag(shape_ref_y_np), "--", label="ref Im(S)", lw=1.0)
        axes[1, 1].set_ylabel("Im(S)")
        axes[1, 1].legend()
        axes[1, 1].grid(alpha=0.3)

        axes[2, 1].plot(y_cheb_np, np.abs(shape_pred_y_np), label="Pred |S|", lw=1.6)
        if benchmark_available:
            axes[2, 1].plot(y_cheb_np, np.abs(shape_ref_y_np), "--", label="ref |S|", lw=1.0)
        axes[2, 1].set_ylabel("|S|")
        axes[2, 1].set_xlabel("y")
        axes[2, 1].legend()
        axes[2, 1].grid(alpha=0.3)

        fig.suptitle(
            f"patch={self.patch.patch_id}, step={step}, a={float(a_t):.6f}, "
            f"omega={float(omega_t):.6f}, {benchmark_status}",
            fontsize=12
        )
        fig.tight_layout()

        save_path = self.fig_dir / f"step_{step:06d}_ref.png"
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        self.model.train()

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

    # =========================================================
    # save / log
    # =========================================================
    def _save_checkpoint(self, path: Path, val_metrics: dict | None = None):
        payload = {
            "step": int(self.global_step),
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "full_cfg": self.full_cfg,
            "patch_id": int(self.patch.patch_id),
            "component_id": int(self.patch.component_id),
            "patch_center": {"u": float(self.patch.u_center), "v": float(self.patch.v_center)},
            "history_tail": self.history[-50:],
            "best_val_mean": float(self.best_val_mean),
            "best_val_case_means": self.best_val_case_means.tolist() if self.best_val_case_means is not None else None,
            "best_val_case_steps": self.best_val_case_steps.tolist() if self.best_val_case_steps is not None else None,
            "best_val_cases": self._serialize_best_val_cases(),
            "latest_val_metrics": val_metrics,
            "model_type": self.model_type,
            "cheb_N": self.cheb_N if self.model_type in ("cheb", "cheb_deeponet") else None,
        }
        torch.save(payload, path)

    def _append_history(self, info: dict):
        self.history.append(info)
        with open(self.history_jsonl, "a", encoding="utf-8") as f:
            f.write(json.dumps(info, ensure_ascii=False) + "\n")

    def _write_summary(self, final_val: dict | None):
        summary = {
            "patch_id": int(self.patch.patch_id),
            "component_id": int(self.patch.component_id),
            "patch_center": {"u": float(self.patch.u_center), "v": float(self.patch.v_center)},
            "patch_pool_size": int(len(self.patch_aw)),
            "train_pool_size": int(len(self.train_aw)),
            "val_pool_size": int(len(self.val_aw)),
            "global_step": int(self.global_step),
            "best_val_mean": float(self.best_val_mean),
            "best_val_cases": self._serialize_best_val_cases(),
            "latest_val_metrics": final_val,
            "run_dir": str(self.run_dir),
            "best_model": str(self.ckpt_dir / "best_model.pt"),
            "latest_model": str(self.ckpt_dir / "latest_model.pt"),
            "fig_dir": str(self.fig_dir),
            "history_jsonl": str(self.history_jsonl),
            "anchor_fail_log": str(self.anchor_fail_log),
        }
        with open(self.summary_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    # =========================================================
    # main train loop
    # =========================================================
    def train(self, steps: int | None = None):
        steps = int(self.steps_default if steps is None else steps)

        final_val = None
        start_step = int(self.global_step) + 1
        if start_step > steps:
            self._vprint(
                f"[patch {self.patch.patch_id}] checkpoint step={self.global_step} "
                f"already reached target steps={steps}; skipping train loop."
            )
            final_val = self.validate()
            improved_count = self._update_best_val_cases(final_val)
            final_val["case_improved_count"] = int(improved_count)
            self._save_checkpoint(self.ckpt_dir / "latest_model.pt", val_metrics=final_val)
            self._write_summary(final_val)
            return {
                "run_dir": str(self.run_dir),
                "best_val_mean": self.best_val_mean,
                "final_val": final_val,
            }

        pbar = trange(start_step, steps + 1, desc=f"patch {self.patch.patch_id}", dynamic_ncols=True)

        try:
            for step in pbar:
                self.global_step = step
                info = self.train_one_step()
                info["step"] = int(step)
                self._append_history(info)

                # validation
                if step % self.val_every == 0 or step == 1:
                    final_val = self.validate()
                    improved_count = self._update_best_val_cases(final_val)
                    final_val["case_improved_count"] = int(improved_count)
                    info["val_mean"] = final_val["val_mean"]
                    info["val_worst"] = final_val["val_worst"]
                    info["val_case_improved_count"] = int(improved_count)
                    improved = improved_count > 0
                    info["val_rel_improve"] = None

                    if improved:
                        self.es_bad_count = 0
                        self._save_checkpoint(self.ckpt_dir / "best_model.pt", val_metrics=final_val)
                    else:
                        if self.best_val_case_means is not None and np.all(np.isfinite(self.best_val_case_means)):
                            self.es_bad_count += 1

                    if self.es_enabled and step >= self.es_min_steps and self.es_bad_count >= self.es_patience:
                        self._vprint(
                            f"[early-stop] patch {self.patch.patch_id}: "
                            f"no val improvement for {self.es_bad_count} validations, stop at step={step}"
                        )
                        break

                # periodic visualization
                if step % self.viz_every == 0 or step == 1:
                    self.visualize_reference(step)

                # amplitude monitor (no gradient)
                if (
                    self.inf_monitor_enabled
                    and self.global_step > 0
                    and self.global_step % self.inf_monitor_every == 0
                ):
                    self._run_asymptotic_amplitude_monitor()

                # periodic latest / step ckpt
                if step % self.save_every == 0 or step == steps:
                    self._save_checkpoint(self.ckpt_dir / "latest_model.pt", val_metrics=final_val)

                if step % self.ckpt_every == 0:
                    self._save_checkpoint(self.ckpt_dir / f"step_{step:06d}.pt", val_metrics=final_val)

                pbar.set_postfix({
                    "tot": f"{info['total_loss']:.2e}",
                    "pde": f"{info['loss_pde']:.2e}",
                    "int": f"{info.get('loss_integral', 0.0):.2e}",
                    "anc": f"{info.get('loss_anchor', 0.0):.2e}",
                    "aw": f"{info.get('anchor_weight_eff', 0.0):.2e}",
                    "ok": f"{info.get('anchor_success_count', 0)}",
                    "fail": f"{info.get('anchor_failed_count', 0)}",
                    "g": f"{info['grad_norm']:.2e}",
                    "val": f"{final_val['val_mean']:.2e}" if final_val is not None else "-",
                    "best": f"{self.best_val_mean:.2e}" if np.isfinite(self.best_val_mean) else "-",
                })

            # L-BFGS refinement
            lbfgs_steps = self.atlas_train_cfg.get("lbfgs_steps", 0)
            if lbfgs_steps > 0:
                final_val = self._train_lbfgs(lbfgs_steps)

        finally:
            self._save_checkpoint(self.ckpt_dir / "latest_model.pt", val_metrics=final_val)
            self._write_summary(final_val)
            self.close()

        return {
            "run_dir": str(self.run_dir),
            "best_val_mean": self.best_val_mean,
            "final_val": final_val,
        }
