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
from dataset.sampling import sample_points_chebyshev_grid
from physical_ansatz.residual import AuxCache, get_lambda_from_cfg
from physical_ansatz.residual_pinn import pinn_residual_loss, compute_data_anchor_loss
from domain.patch_cover import load_patch_cover, load_valid_chart_points
from physical_ansatz.mapping import r_plus, r_from_x
from physical_ansatz.transform_y import (
    h_factor,
    horizon_regularity_slope,
    compose_reduced_shape_from_f,
)
from physical_ansatz.prefactor import Leaver_prefactors, build_prefactor_primitives
from mma.rin_sampler import MathematicaRinSampler
from compute_solution import compute_pybhpt_solution


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
        if self.viz_benchmark_backend not in ("none", "pybhpt", "mma"):
            raise ValueError(
                f"Unsupported atlas_training.viz_benchmark_backend={self.viz_benchmark_backend}; "
                "must be one of {'none','pybhpt','mma'}."
            )
        self.viz_pybhpt_timeout = float(atlas_train_cfg.get("viz_pybhpt_timeout", 10.0))
        self.anchor_pybhpt_timeout = float(atlas_train_cfg.get("anchor_pybhpt_timeout", self.viz_pybhpt_timeout))
        self.viz_mma_enabled = bool(atlas_train_cfg.get("viz_mma_enabled", False))

        self.anchor_enabled = bool(anchor_enabled)
        self.anchor_target_ratio = float(atlas_train_cfg.get("anchor_target_ratio", 1.0e-2))
        self.lr = float(self.cfg.get("training", {}).get("optimizer", {}).get("lr", 1.0e-3))
        self.verbose = bool(verbose)
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

        self.model = PINN_MLP(
            hidden_dims=model_cfg.get("hidden_dims", [128, 128, 128, 128]),
            activation=model_cfg.get("activation", "silu"),
            fourier_num_freqs=model_cfg.get("fourier_num_freqs", 2),
            fourier_scale=model_cfg.get("fourier_scale", 1.0),
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

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.cache = AuxCache()
        self._preload_lambda_cache_from_probe(probe_json)
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
            rows.append(
                {
                    "case_index": int(i),
                    "a": float(self.val_meta["a"][i].detach().cpu().item()),
                    "omega": float(self.val_meta["omega"][i].detach().cpu().item()),
                    "u": float(self.val_meta["u"][i].detach().cpu().item()),
                    "v": float(self.val_meta["v"][i].detach().cpu().item()),
                    "lambda": float(self.val_meta["lambda"][i].detach().cpu().item()),
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

    def _load_init_checkpoint(self):
        ckpt = torch.load(self.init_checkpoint, map_location=self.device)
        state_dict = ckpt.get("model_state_dict", ckpt)
        self.model.load_state_dict(state_dict, strict=True)

        if self.init_load_optimizer and "optimizer_state_dict" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])

        self._vprint(f"[init] loaded checkpoint: {self.init_checkpoint}")

    def _load_resume_checkpoint(self):
        ckpt = torch.load(self.resume_checkpoint, map_location=self.device)
        state_dict = ckpt.get("model_state_dict", ckpt)
        self.model.load_state_dict(state_dict, strict=True)

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
        y = sample_points_chebyshev_grid(
            n_points=self.n_interior,
            y_min=-0.99,
            y_max=0.99,
            device=self.device,
            dtype=self.dtype,
        )
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

        y_interior = self.sample_y_interior()
        y_boundary = torch.empty(0, device=self.device, dtype=self.dtype)

        loss_pde, info_pde = pinn_residual_loss(
            model=self.model,
            cfg=self.physics_cfg,
            a_batch=a_batch,
            omega_batch=omega_batch,
            lambda_batch=lambda_batch,
            y_interior=y_interior,
            y_boundary=y_boundary,
            weight_interior=1.0,
            weight_boundary=0.0,
            normalize_residual=self.normalize_residual,
            residual_scale_eps=1.0e-12,
            return_pointwise=False,
            u_batch=u_batch,
            v_batch=v_batch,
        )

        total_loss = loss_pde
        info = dict(info_pde)
        info["loss_pde"] = float(loss_pde.detach().cpu().item())
        info["loss_anchor"] = 0.0
        info["anchor_weight_eff"] = 0.0
        info["anchor_success_count"] = 0
        info["anchor_failed_count"] = 0

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

                anchor_weight_eff = self._compute_anchor_weight_eff(loss_pde, loss_anchor)
                total_loss = loss_pde + anchor_weight_eff * loss_anchor

                info["loss_anchor"] = float(loss_anchor.detach().cpu().item())
                info["anchor_weight_eff"] = float(anchor_weight_eff)
                info["anchor_success_count"] = int(ok_mask.numel())

        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optimizer.step()

        info["total_loss"] = float(total_loss.detach().cpu().item())
        info["grad_norm"] = float(grad_norm.detach().cpu().item())
        return info

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
                    "lambda": float(self.val_meta["lambda"][i].detach().cpu().item()),
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

        benchmark_status = "benchmark=off"
        a_scalar = float(a_t.detach().cpu().item())
        omega_scalar = float(omega_t.detach().cpu().item())
        r_uniform_np = r_grid_uniform.detach().cpu().numpy()
        y_cheb_np = y_grid_cheb.detach().cpu().numpy()
        R_pred_r_np = R_pred_r.detach().cpu().numpy()
        shape_pred_y_np = shape_pred_y.detach().cpu().numpy()

        fig, axes = plt.subplots(3, 2, figsize=(10, 10), sharex=False)

        axes[0, 0].plot(r_uniform_np, np.real(R_pred_r_np), label="Pred Re(R)", lw=1.6)
        axes[0, 0].set_ylabel("Re(R)")
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)

        axes[1, 0].plot(r_uniform_np, np.imag(R_pred_r_np), label="Pred Im(R)", lw=1.6)
        axes[1, 0].set_ylabel("Im(R)")
        axes[1, 0].legend()
        axes[1, 0].grid(alpha=0.3)

        axes[2, 0].plot(r_uniform_np, np.abs(R_pred_r_np), label="Pred |R|", lw=1.6)
        axes[2, 0].set_ylabel("|R|")
        axes[2, 0].set_xlabel("r")
        axes[2, 0].legend()
        axes[2, 0].grid(alpha=0.3)

        axes[0, 1].plot(y_cheb_np, np.real(shape_pred_y_np), label="Pred Re(S)", lw=1.6)
        axes[0, 1].set_ylabel("Re(S)")
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)

        axes[1, 1].plot(y_cheb_np, np.imag(shape_pred_y_np), label="Pred Im(S)", lw=1.6)
        axes[1, 1].set_ylabel("Im(S)")
        axes[1, 1].legend()
        axes[1, 1].grid(alpha=0.3)

        axes[2, 1].plot(y_cheb_np, np.abs(shape_pred_y_np), label="Pred |S|", lw=1.6)
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

                # periodic latest / step ckpt
                if step % self.save_every == 0 or step == steps:
                    self._save_checkpoint(self.ckpt_dir / "latest_model.pt", val_metrics=final_val)

                if step % self.ckpt_every == 0:
                    self._save_checkpoint(self.ckpt_dir / f"step_{step:06d}.pt", val_metrics=final_val)

                pbar.set_postfix({
                    "tot": f"{info['total_loss']:.2e}",
                    "pde": f"{info['loss_pde']:.2e}",
                    "anc": f"{info.get('loss_anchor', 0.0):.2e}",
                    "aw": f"{info.get('anchor_weight_eff', 0.0):.2e}",
                    "ok": f"{info.get('anchor_success_count', 0)}",
                    "fail": f"{info.get('anchor_failed_count', 0)}",
                    "g": f"{info['grad_norm']:.2e}",
                    "val": f"{final_val['val_mean']:.2e}" if final_val is not None else "-",
                    "best": f"{self.best_val_mean:.2e}" if np.isfinite(self.best_val_mean) else "-",
                })

        finally:
            self._save_checkpoint(self.ckpt_dir / "latest_model.pt", val_metrics=final_val)
            self._write_summary(final_val)
            self.close()

        return {
            "run_dir": str(self.run_dir),
            "best_val_mean": self.best_val_mean,
            "final_val": final_val,
        }
