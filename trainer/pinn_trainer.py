"""
PINN Trainer
"""
import os
import torch
import yaml
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from pathlib import Path

from config.config_loader import load_pinn_full_config
from model.pinn_mlp import PINN_MLP
from physical_ansatz.residual_pinn import pinn_residual_loss, compute_data_anchor_loss,compute_variance_regularizer
from physical_ansatz.residual import AuxCache, get_lambda_from_cfg, get_ramp_and_p_from_cfg
from physical_ansatz.transform_y import g_factor, transform_coeffs_x_to_y,h_factor
from physical_ansatz.mapping import r_plus, r_from_x
from physical_ansatz.prefactor import Leaver_prefactors, prefactor_Q, U_prefactor
from dataset.sampling import sample_points_luna_style, sample_parameters
from dataset.sampling import (
    sample_points_luna_style,
    sample_parameters,
    sample_parameters_sobol,
    build_candidate_pool_1d,
    sample_points_rard,
    get_sentinel_anchor_points,
    sample_interior_points,
)

from dataset.mathematica_anchor import get_mathematica_Rin




class PINNTrainer:
    def __init__(self, cfg_path, device='cpu'):
        """
        Args:
            cfg_path: 配置文件路径
            device: 'cpu' 或 'cuda'
        """
        self.device = device
        # 加载总配置，并拆分 train / physics
        self.full_cfg = load_pinn_full_config(cfg_path)
        self.physics_cfg = self.full_cfg["physics"]
        self.train_cfg = self.full_cfg["train"]


        # 训练配置分组  
        param_cfg = self.train_cfg.get("param_sampling", {})
        model_cfg = self.train_cfg.get("model", {})
        sampling_cfg = self.train_cfg.get("sampling", {})
        self.param_sampling_mode = sampling_cfg.get("mode", "random")
        train_cfg = self.train_cfg.get("train", {})
        loss_cfg = self.train_cfg.get("loss", {})
        val_cfg = self.train_cfg.get("validation", {})
        early_cfg = self.train_cfg.get("early_stop", {})
        runtime_cfg = self.train_cfg.get("runtime", {})
        flat_var_cfg = self.train_cfg.get("flat_variance", {})

        sched_cfg = self.train_cfg.get("scheduler", {})
        self.scheduler_enabled = sched_cfg.get("enabled", False)
        self.scheduler_type = sched_cfg.get("type", "none")
        self.lr_warmup_steps = sched_cfg.get("warmup_steps", 0)
        self.lr_decay_start = sched_cfg.get("decay_start", 0)
        self.lr_decay_rate = sched_cfg.get("decay_rate", 1.0)
        self.min_lr = sched_cfg.get("min_lr", 1.0e-4)
        self.base_lr = train_cfg.get("lr", 1.0e-3)
        #平坦惩罚
        self.flat_var_enabled = flat_var_cfg.get("enabled", False)
        self.flat_var_target = flat_var_cfg.get("target", "Rprime")
        self.flat_var_kappa = flat_var_cfg.get("kappa", 20.0)
        self.flat_var_eps = flat_var_cfg.get("eps", 1.0e-12)
        self.flat_var_weight = flat_var_cfg.get("weight", 1.0e-3)
        self.flat_var_steps = flat_var_cfg.get("steps", 0)
        self.flat_var_decay_enabled = flat_var_cfg.get("decay_enabled", False)
        self.flat_var_warmup_steps = flat_var_cfg.get("warmup_steps", 0)
        self.flat_var_decay_start = flat_var_cfg.get("decay_start", 0)
        self.flat_var_decay_rate = flat_var_cfg.get("decay_rate", 1.0)
        self.flat_var_min_weight = flat_var_cfg.get("min_weight", 0.0)
        self.flat_var_base_weight = self.flat_var_weight


        self.flat_var_use_dedicated_points = flat_var_cfg.get("use_dedicated_points", True)
        self.flat_var_strategy = flat_var_cfg.get("strategy", "article_uniform")
        self.flat_var_n_points = flat_var_cfg.get("n_points", 64)
        # loss 权重
        self.weight_interior = loss_cfg.get("weight_interior", 1.0)
        self.weight_boundary = loss_cfg.get("weight_boundary", 10.0)
        self.weight_anchor = loss_cfg.get("weight_anchor", 1.0)
        # 采样配置
        self.n_interior = sampling_cfg.get("n_interior", 100)
        self.n_boundary = sampling_cfg.get("n_boundary", 20)
        self.batch_size = sampling_cfg.get("batch_size", 4)
        self.use_batch_gd = sampling_cfg.get("use_batch_gd", False)
        self.n_param_samples = sampling_cfg.get("n_param_samples", 20)
        self.n_epochs = sampling_cfg.get("n_epochs", 100)

        adaptive_cfg = self.train_cfg.get("adaptive_sampling", {})
        anchor_cfg = self.train_cfg.get("anchors", {})
        loss_balance_cfg = self.train_cfg.get("loss_balance", {})
        curriculum_cfg = self.train_cfg.get("curriculum", {})


        collocation_curr_cfg = self.train_cfg.get("collocation_curriculum", {})
        anti_trivial_cfg = self.train_cfg.get("anti_trivial", {})
        restart_cfg = self.train_cfg.get("restart", {})
        dynamic_balance_cfg = self.train_cfg.get("dynamic_balance", {})

        self.param_sampler = sampling_cfg.get("param_sampler", "sobol")

        self.interior_strategy = sampling_cfg.get("interior_strategy", "rard")
        self.boundary_strategy = sampling_cfg.get("boundary_strategy", "none")
        self.article_uniform_cfg = sampling_cfg.get("article_uniform", {})

        self.collocation_curriculum_enabled = collocation_curr_cfg.get("enabled", False)
        self.collocation_curriculum_stages = collocation_curr_cfg.get("stages", [])
        self.curr_interior_strategy = self.interior_strategy
        self.curr_n_interior = self.n_interior

        self.anti_trivial_enabled = anti_trivial_cfg.get("enabled", False)
        self.seed_steps = anti_trivial_cfg.get("seed_steps", 0)
        self.seed_every = anti_trivial_cfg.get("seed_every", 10)
        self.n_seed = anti_trivial_cfg.get("n_seed", 8)
        self.seed_strategy = anti_trivial_cfg.get("seed_strategy", "article_uniform")
        self.weight_seed = anti_trivial_cfg.get("weight_seed", 0.1)
        self.seed_relative = anti_trivial_cfg.get("relative", False)

        self.restart_enabled = restart_cfg.get("enabled", False)
        self.stall_window = restart_cfg.get("stall_window", 300)
        self.stall_tol = restart_cfg.get("stall_tol", 1e-4)
        self.lr_decay_on_restart = restart_cfg.get("lr_decay_on_restart", 0.5)
        self._stall_best = float("inf")
        self._stall_count = 0

        self.dynamic_balance_enabled = dynamic_balance_cfg.get("enabled", False)
        self.dynamic_balance_every = dynamic_balance_cfg.get("every_steps", 20)
        self.w_anchor_min = dynamic_balance_cfg.get("w_anchor_min", 1e-3)
        self.w_anchor_max = dynamic_balance_cfg.get("w_anchor_max", 10.0)
        self.dynamic_anchor_weight = self.weight_anchor



        # dtype
        dtype_name = runtime_cfg.get("dtype", "float64")
        self.dtype = torch.float64 if dtype_name == "float64" else torch.float32

        # 参数范围
        self.a_center = param_cfg.get("a_center", 0.1)
        self.a_range = param_cfg.get("a_range", 0.01)
        self.omega_center = param_cfg.get("omega_center", 0.1)
        self.omega_range = param_cfg.get("omega_range", 0.01)



        # 训练配置
        self.n_steps = train_cfg.get("n_steps", 50000)
        self.lr = train_cfg.get("lr", 1e-3)
        self.anchor_freq = train_cfg.get("anchor_freq", 100)
        self.anchor_start_step = train_cfg.get("anchor_start_step", 0)
        self.n_anchors = train_cfg.get("n_anchors", 10)
        self.viz_enabled = train_cfg.get("viz_enabled", False)
        self.viz_every_steps = train_cfg.get("viz_every_steps", 500)
        self.viz_auto_close_sec = train_cfg.get("viz_auto_close_sec", 3.0)
        self.viz_num_points = train_cfg.get("viz_num_points", 400)
        self.viz_r_min = train_cfg.get("viz_r_min", 2.0)
        self.viz_r_max = train_cfg.get("viz_r_max", 100.0)
        self.viz_save_enabled = train_cfg.get("viz_save_enabled", True)
        self.viz_subdir = train_cfg.get("viz_subdir", "viz_compare")
        self.viz_show_enabled = train_cfg.get("viz_show_enabled", True)
        # 如果 trainer.train(save_dir=...) 之后会再覆盖主输出目录，这里先给默认值
        self.output_dir = Path(getattr(self, "output_dir", "outputs/pinn"))
        self.viz_dir = self.output_dir / self.viz_subdir
        self.viz_dir.mkdir(parents=True, exist_ok=True)



        # 早停
        self.early_stop_patience = early_cfg.get("patience", 500)
        self.early_stop_threshold = early_cfg.get("threshold", 1e-6)

        # 验证
        self.val_freq = val_cfg.get("val_freq", 50)
        self.val_n_points = val_cfg.get("val_n_points", 200)
        self.save_best_freq = val_cfg.get("save_best_freq", 50)



        self.adaptive_sampling_enabled = adaptive_cfg.get("enabled", True)
        self.candidate_method = adaptive_cfg.get("candidate_method", "sobol")
        self.n_candidates = adaptive_cfg.get("n_candidates", 512)
        self.refresh_freq = adaptive_cfg.get("refresh_freq", 50)
        self.adaptive_frac = adaptive_cfg.get("adaptive_frac", 0.7)
        self.normalize_residual = adaptive_cfg.get("normalize_residual", True)

        self.anchor_mode = anchor_cfg.get("mode", "sentinel_plus_adaptive")
        self.n_anchor_adaptive = anchor_cfg.get("n_adaptive", 6)
        self.n_anchor_random = anchor_cfg.get("n_random", 2)

        self.loss_balance_mode = loss_balance_cfg.get("mode", "lbpinn")
        self.loss_balance_eps = loss_balance_cfg.get("eps", 1e-12)

        self.curriculum_enabled = curriculum_cfg.get("enabled", False)
        self.curriculum_stages = curriculum_cfg.get("stages", [])

        self.curr_a_range = self.a_range
        self.curr_omega_range = self.omega_range
        self._candidate_y = None


        # 创建模型
        hidden_dims = model_cfg.get("hidden_dims", [64, 64, 64, 64])
        activation = model_cfg.get("activation", "tanh")
        self.model = PINN_MLP(
            hidden_dims=hidden_dims,
            activation=activation,
        ).to(device=self.device, dtype=self.dtype)

        # self.model = PINN_MLP(
        #     hidden_dims=hidden_dims,
        #     activation='tanh',
        # ).to(device=self.device, dtype=self.dtype)

        # 优化器
        extra_params = []
        if self.loss_balance_mode == "lbpinn":
            self.log_sigma_interior = torch.nn.Parameter(
                torch.zeros((), device=self.device, dtype=self.dtype)
            )
            self.log_sigma_anchor = torch.nn.Parameter(
                torch.zeros((), device=self.device, dtype=self.dtype)
            )
            extra_params = [self.log_sigma_interior, self.log_sigma_anchor]

        self.optimizer = torch.optim.Adam(
            list(self.model.parameters()) + extra_params,
            lr=self.lr,
        )

        # 缓存
        self.cache = AuxCache()

        # 历史记录
        self.loss_history = []
        self.step_history = []
        self.val_loss_history = []
        self.best_val_loss = float('inf')

        print(f"Parameter sampling mode: {self.param_sampling_mode}")


    def _build_fixed_param_pool(self):
        """
        构造固定参数池，只在 fixed_pool 模式下使用。
        返回:
            {
                "a_all": ...,
                "omega_all": ...,
                "lambda_all": ...,
                "ramp_all": ...,
                "p": int
            }
        }
        """
        # a_all, omega_all = sample_parameters(
        #     batch_size=self.n_param_samples,
        #     a_center=self.a_center,
        #     a_range=self.a_range,
        #     omega_center=self.omega_center,
        #     omega_range=self.omega_range,
        #     device=self.device,
        #     dtype=self.dtype,
        # )

        a_all, omega_all = self._sample_param_batch(
            batch_size=self.n_param_samples,
            skip=0,
        )

        lambda_list = []
        ramp_list = []
        p_val = None
        cache = AuxCache()

        for i in range(self.n_param_samples):
            lam_i = get_lambda_from_cfg(
                self.physics_cfg, cache, a_all[i], omega_all[i]
            )
            p_i, ramp_i = get_ramp_and_p_from_cfg(
                self.physics_cfg, cache, a_all[i], omega_all[i]
            )

            if p_val is None:
                p_val = int(p_i or 5)
            else:
                if int(p_i or 5) != p_val:
                    raise ValueError("不同参数样本返回了不同的 p，当前实现不支持。")

            lambda_list.append(lam_i)
            ramp_list.append(ramp_i)

        lambda_all = torch.stack(lambda_list, dim=0)
        ramp_all = torch.stack(ramp_list, dim=0)

        return {
            "a_all": a_all,
            "omega_all": omega_all,
            "lambda_all": lambda_all,
            "ramp_all": ramp_all,
            "p": int(p_val or 5),
        }


    def _get_param_batches_for_epoch(self, fixed_pool=None):
        """
        返回当前 epoch 需要遍历的参数 batch 列表。
        每个元素是一个 dict:
        {
            "a_batch": ...,
            "omega_batch": ...,
            "lambda_batch": ...,
            "ramp_batch": ...,
            "p": int
        }
        """
        batches = []

        if self.param_sampling_mode == "random":
            # random 模式下，每个 epoch 只生成一个 batch
            # a_batch, omega_batch = sample_parameters(
            #     batch_size=self.batch_size,
            #     a_center=self.a_center,
            #     a_range=self.a_range,
            #     omega_center=self.omega_center,
            #     omega_range=self.omega_range,
            #     device=self.device,
            #     dtype=self.dtype,
            # )

            a_batch, omega_batch = self._sample_param_batch(
                batch_size=self.batch_size,
                skip=0,
            )

            lambda_list = []
            ramp_list = []
            p_val = None

            for i in range(self.batch_size):
                lam_i = get_lambda_from_cfg(
                    self.physics_cfg, self.cache, a_batch[i], omega_batch[i]
                )
                p_i, ramp_i = get_ramp_and_p_from_cfg(
                    self.physics_cfg, self.cache, a_batch[i], omega_batch[i]
                )

                if p_val is None:
                    p_val = int(p_i or 5)
                else:
                    if int(p_i or 5) != p_val:
                        raise ValueError("当前 batch 内不同样本返回了不同的 p。")

                lambda_list.append(lam_i)
                ramp_list.append(ramp_i)

            batches.append({
                "a_batch": a_batch,
                "omega_batch": omega_batch,
                "lambda_batch": torch.stack(lambda_list, dim=0),
                "ramp_batch": torch.stack(ramp_list, dim=0),
                "p": int(p_val or 5),
            })

        elif self.param_sampling_mode == "fixed_pool":
            if fixed_pool is None:
                raise ValueError("fixed_pool 模式下必须传入固定参数池")

            a_all = fixed_pool["a_all"]
            omega_all = fixed_pool["omega_all"]
            lambda_all = fixed_pool["lambda_all"]
            ramp_all = fixed_pool["ramp_all"]
            p = fixed_pool["p"]

            n_total = a_all.shape[0]
            for start in range(0, n_total, self.batch_size):
                end = min(start + self.batch_size, n_total)
                batches.append({
                    "a_batch": a_all[start:end],
                    "omega_batch": omega_all[start:end],
                    "lambda_batch": lambda_all[start:end],
                    "ramp_batch": ramp_all[start:end],
                    "p": p,
                })
        else:
            raise ValueError(f"Unknown param_sampling_mode: {self.param_sampling_mode}")

        return batches

    def _current_lr(self, global_step: int) -> float:
        if not self.scheduler_enabled:
            return self.base_lr

        if global_step < self.lr_decay_start:
            return self.base_lr

        lr = self.base_lr * (self.lr_decay_rate ** (global_step - self.lr_decay_start))
        return max(self.min_lr, lr)


    def _current_flat_var_weight(self, global_step: int) -> float:
        if not self.flat_var_enabled:
            return 0.0

        if not self.flat_var_decay_enabled:
            return self.flat_var_base_weight

        if global_step < self.flat_var_decay_start:
            return self.flat_var_base_weight

        w = self.flat_var_base_weight * (
            self.flat_var_decay_rate ** (global_step - self.flat_var_decay_start)
        )
        return max(self.flat_var_min_weight, w)
        
    def _apply_curriculum(self, epoch):
        if not self.curriculum_enabled or not self.curriculum_stages:
            return False

        acc = 0
        changed = False
        for stage in self.curriculum_stages:
            acc += int(stage["epochs"])
            if epoch < acc:
                new_a_range = float(stage.get("a_range", self.a_range))
                new_omega_range = float(stage.get("omega_range", self.omega_range))
                new_weight_anchor = float(stage.get("weight_anchor", self.weight_anchor))

                if (
                    new_a_range != self.curr_a_range
                    or new_omega_range != self.curr_omega_range
                    or new_weight_anchor != self.weight_anchor
                ):
                    self.curr_a_range = new_a_range
                    self.curr_omega_range = new_omega_range
                    self.weight_anchor = new_weight_anchor
                    changed = True
                break
        return changed
    def _apply_collocation_curriculum(self, epoch):
        if not self.collocation_curriculum_enabled or not self.collocation_curriculum_stages:
            return False

        acc = 0
        changed = False
        for stage in self.collocation_curriculum_stages:
            acc += int(stage["epochs"])
            if epoch < acc:
                new_strategy = stage.get("interior_strategy", self.curr_interior_strategy)
                new_n_interior = int(stage.get("n_interior", self.curr_n_interior))
                if new_strategy != self.curr_interior_strategy or new_n_interior != self.curr_n_interior:
                    self.curr_interior_strategy = new_strategy
                    self.curr_n_interior = new_n_interior
                    changed = True
                break
        return changed

    def _maybe_restart_on_stall(self, current_loss):
        if not self.restart_enabled:
            return

        if current_loss < self._stall_best - self.stall_tol:
            self._stall_best = current_loss
            self._stall_count = 0
            return

        self._stall_count += 1
        if self._stall_count < self.stall_window:
            return

        # 重置候选点池
        self._candidate_y = None

        # 清空 Adam 动量
        self.optimizer.state.clear()

        # 降低学习率
        for group in self.optimizer.param_groups:
            group["lr"] *= self.lr_decay_on_restart

        print(
            f"[restart] collocation resampled, optimizer state cleared, "
            f"new lr={self.optimizer.param_groups[0]['lr']:.3e}"
        )

        self._stall_count = 0
        self._stall_best = current_loss    

    def _sample_flat_var_points(self):
        y_var = sample_interior_points(
            strategy=self.flat_var_strategy,
            n_points=self.flat_var_n_points,
            device=self.device,
            dtype=self.dtype,
            article_cfg=self.article_uniform_cfg,
        )
        return y_var

    def _sample_param_batch(self, batch_size, skip=0):
        if self.param_sampler == "sobol":
            return sample_parameters_sobol(
                batch_size=batch_size,
                a_center=self.a_center,
                a_range=self.curr_a_range,
                omega_center=self.omega_center,
                omega_range=self.curr_omega_range,
                device=self.device,
                dtype=self.dtype,
                skip=skip,
            )
        return sample_parameters(
            batch_size=batch_size,
            a_center=self.a_center,
            a_range=self.curr_a_range,
            omega_center=self.omega_center,
            omega_range=self.curr_omega_range,
            device=self.device,
            dtype=self.dtype,
        )

    
    def _sample_seed_points(self):
        y_seed = sample_interior_points(
            strategy=self.seed_strategy,
            n_points=self.n_seed,
            device=self.device,
            dtype=self.dtype,
            article_cfg=self.article_uniform_cfg,
        )
        return y_seed.clone().requires_grad_(True)


    def _sample_training_points(self, batch, global_step):
        strategy = self.curr_interior_strategy

        # 2212 / chebyshev / random / sobol / luna：都走固定或随机规则采样
        if strategy != "rard":
            y_interior = sample_interior_points(
                strategy=strategy,
                n_points=self.curr_n_interior,
                device=self.device,
                dtype=self.dtype,
                article_cfg=self.article_uniform_cfg,
            )

            if self.boundary_strategy == "none" or self.n_boundary == 0:
                y_boundary = torch.empty(0, device=self.device, dtype=self.dtype)
            elif self.boundary_strategy == "luna_boundary":
                _, y_boundary = sample_points_luna_style(
                    n_interior=0,
                    n_boundary=self.n_boundary,
                    device=self.device,
                    dtype=self.dtype,
                )
            elif self.boundary_strategy == "exact_safe":
                y_boundary = torch.tensor(
                    [-0.99] * self.n_boundary + [0.99] * self.n_boundary,
                    device=self.device,
                    dtype=self.dtype,
                )
            else:
                raise ValueError(f"Unknown boundary strategy: {self.boundary_strategy}")

            return (
                y_interior.clone().requires_grad_(True),
                y_boundary.clone().requires_grad_(True) if y_boundary.numel() > 0 else y_boundary,
            ), None

        # 你当前的 RAR-D 自适应采样
        if not self.adaptive_sampling_enabled:
            y_interior = sample_interior_points(
                strategy="random_uniform",
                n_points=self.curr_n_interior,
                device=self.device,
                dtype=self.dtype,
                article_cfg=self.article_uniform_cfg,
            )
            y_boundary = torch.empty(0, device=self.device, dtype=self.dtype)
            return (y_interior.clone().requires_grad_(True), y_boundary), None

        if self._candidate_y is None or (global_step % self.refresh_freq == 1):
            self._candidate_y = build_candidate_pool_1d(
                n_points=self.n_candidates,
                method=self.candidate_method,
                device=self.device,
                dtype=self.dtype,
            )

        candidate_y = self._candidate_y.clone().requires_grad_(True)
        _, probe_info = pinn_residual_loss(
            model=self.model,
            cfg=self.physics_cfg,
            a_batch=batch["a_batch"],
            omega_batch=batch["omega_batch"],
            lambda_batch=batch["lambda_batch"],
            ramp_batch=batch["ramp_batch"],
            p=batch["p"],
            y_interior=candidate_y,
            y_boundary=torch.empty(0, device=self.device, dtype=self.dtype),
            weight_interior=1.0,
            weight_boundary=0.0,
            normalize_residual=self.normalize_residual,
            return_pointwise=True,
        )

        score = probe_info["pointwise_interior"].mean(dim=0)
        y_interior = sample_points_rard(
            candidate_y=self._candidate_y,
            residual_score=score,
            n_select=self.curr_n_interior,
            adaptive_frac=self.adaptive_frac,
        )
        y_boundary = torch.empty(0, device=self.device, dtype=self.dtype)
        return (y_interior, y_boundary), score


    def _sample_anchor_points(self, residual_score=None):
        y_parts = [get_sentinel_anchor_points(device=self.device, dtype=self.dtype)]

        if residual_score is not None and self.n_anchor_adaptive > 0 and self._candidate_y is not None:
            k = min(self.n_anchor_adaptive, self._candidate_y.numel())
            idx = torch.topk(residual_score, k=k).indices
            y_parts.append(self._candidate_y[idx].detach())

        y = torch.unique(torch.cat(y_parts, dim=0))
        return y.clone().requires_grad_(True)


    def _combine_losses(
        self,
        loss_pde,
        loss_seed=None,
        loss_anchor=None,
        loss_var=None,
        anchor_weight=None,
        var_weight=None,
    ):
        if self.loss_balance_mode == "lbpinn":
            total = torch.exp(-self.log_sigma_interior) * loss_pde + self.log_sigma_interior

            if loss_seed is not None:
                total = total + self.weight_seed * loss_seed

            if loss_anchor is not None:
                wa = self.weight_anchor if anchor_weight is None else anchor_weight
                total = total + wa * (
                    torch.exp(-self.log_sigma_anchor) * loss_anchor + self.log_sigma_anchor
                )

            if loss_var is not None:
                wv = self.flat_var_weight if var_weight is None else var_weight
                total = total + wv * loss_var

            return total

        total = self.weight_interior * loss_pde

        if loss_seed is not None:
            total = total + self.weight_seed * loss_seed

        if loss_anchor is not None:
            wa = self.weight_anchor if anchor_weight is None else anchor_weight
            total = total + wa * loss_anchor

        if loss_var is not None:
            wv = self.flat_var_weight if var_weight is None else var_weight
            total = total + wv * loss_var

        return total


    def _run_one_training_batch(self, batch, global_step):
        current_lr = self._current_lr(global_step)
        for group in self.optimizer.param_groups:
            group["lr"] = current_lr

        # info["lr"] = float(current_lr)
        a_batch = batch["a_batch"]
        omega_batch = batch["omega_batch"]
        lambda_batch = batch["lambda_batch"]
        ramp_batch = batch["ramp_batch"]
        p = batch["p"]

        (y_interior, y_boundary), residual_score = self._sample_training_points(
            batch, global_step
        )

        self.optimizer.zero_grad()

        loss_pde, info = pinn_residual_loss(
            model=self.model,
            cfg=self.physics_cfg,
            a_batch=a_batch,
            omega_batch=omega_batch,
            lambda_batch=lambda_batch,
            ramp_batch=ramp_batch,
            p=p,
            y_interior=y_interior,
            y_boundary=y_boundary,
            weight_interior=1.0,
            weight_boundary=0.0,
            normalize_residual=self.normalize_residual,
        )


        loss_var = None
        info["loss_var"] = 0.0
        info["sigma_var"] = 0.0

        use_flat_var = (
            self.flat_var_enabled
            and global_step <= self.flat_var_steps
        )

        if use_flat_var:
            if self.flat_var_use_dedicated_points:
                y_var = self._sample_flat_var_points()
            else:
                y_var = y_interior

            loss_var, var_info = compute_variance_regularizer(
                model=self.model,
                cfg=self.physics_cfg,
                a_batch=a_batch,
                omega_batch=omega_batch,
                y_points=y_var,
                target=self.flat_var_target,
                kappa=self.flat_var_kappa,
                eps=self.flat_var_eps,
            )

            info["loss_var"] = float(loss_var.item())
            info["sigma_var"] = float(var_info["sigma_var"])

        var_weight = self._current_flat_var_weight(global_step)
        info["weight_var"] = float(var_weight)

        loss_seed = None
        info["loss_seed"] = 0.0

        use_seed = (
            self.anti_trivial_enabled
            and global_step <= self.seed_steps
            and global_step % self.seed_every == 0
        )

        if use_seed:
            y_seed = self._sample_seed_points()

            M = float(self.physics_cfg["problem"].get("M", 1.0))
            l = int(self.physics_cfg["problem"].get("l", 2))
            m = int(self.physics_cfg["problem"].get("m", 2))
            s = int(self.physics_cfg["problem"].get("s", -2))

            x_seed = (y_seed + 1.0) / 2.0
            R_mma_list = []

            for i in range(a_batch.shape[0]):
                rp = r_plus(a_batch[i], M)
                r_seed = r_from_x(x_seed, rp).detach().cpu().numpy()

                R_mma = get_mathematica_Rin(
                    float(a_batch[i]),
                    float(omega_batch[i]),
                    l, m, s,
                    r_seed,
                )
                R_mma_list.append(
                    torch.tensor(R_mma, dtype=torch.complex128, device=self.device)
                )

            R_mma_seed = torch.stack(R_mma_list, dim=0)

            loss_seed = compute_data_anchor_loss(
                self.model,
                self.physics_cfg,
                a_batch,
                omega_batch,
                y_seed,
                R_mma_seed,
                relative=self.seed_relative,
            )
            info["loss_seed"] = float(loss_seed.item())

        loss_anchor = None
        info["loss_anchor"] = 0.0

        use_anchor = (
            global_step >= self.anchor_start_step
            and global_step % self.anchor_freq == 0
        )

        if use_anchor:
            y_anchors = self._sample_anchor_points(residual_score)

            M = float(self.physics_cfg["problem"].get("M", 1.0))
            l = int(self.physics_cfg["problem"].get("l", 2))
            m = int(self.physics_cfg["problem"].get("m", 2))
            s = int(self.physics_cfg["problem"].get("s", -2))

            x_anchors = (y_anchors + 1.0) / 2.0
            R_mma_list = []

            for i in range(a_batch.shape[0]):
                rp = r_plus(a_batch[i], M)
                r_anchors = r_from_x(x_anchors, rp).detach().cpu().numpy()

                R_mma = get_mathematica_Rin(
                    float(a_batch[i]),
                    float(omega_batch[i]),
                    l, m, s,
                    r_anchors,
                )
                R_mma_list.append(
                    torch.tensor(R_mma, dtype=torch.complex128, device=self.device)
                )

            R_mma_batch = torch.stack(R_mma_list, dim=0)
            loss_anchor = compute_data_anchor_loss(
                self.model,
                self.physics_cfg,
                a_batch,
                omega_batch,
                y_anchors,
                R_mma_batch,
                relative=True,
            )
            info["loss_anchor"] = float(loss_anchor.item())

        anchor_weight = self.weight_anchor
        if loss_anchor is not None:
            anchor_weight = self._update_dynamic_anchor_weight(
                loss_pde, loss_anchor, global_step
            )

        total_loss = self._combine_losses(
            loss_pde,
            loss_seed=loss_seed,
            loss_anchor=loss_anchor,
            loss_var=loss_var,
            anchor_weight=anchor_weight,
            var_weight=var_weight

        )
        info["weight_anchor_dynamic"] = float(anchor_weight)
        
        self.optimizer.zero_grad()
        total_loss.backward()
        grad_sq = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                grad_sq += p.grad.detach().pow(2).sum().item()
        grad_norm = grad_sq ** 0.5

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        info["grad_norm"] = grad_norm

        info["total_loss"] = float(total_loss.item())
        if self.loss_balance_mode == "lbpinn":
            info["w_int"] = float(torch.exp(-self.log_sigma_interior).item())
            info["w_anchor"] = float(torch.exp(-self.log_sigma_anchor).item())

        return total_loss, info




    def _channels_to_complex(self, out: torch.Tensor) -> torch.Tensor:
        """
        将模型输出转成复数。
        约定:
        - 若最后一维是 2，则 out[...,0] 是实部，out[...,1] 是虚部
        - 若模型本来就返回 complex tensor，则直接返回
        """
        if torch.is_complex(out):
            return out
        if out.shape[-1] != 2:
            raise ValueError(
                f"Expected model output last dim = 2 for [Re, Im], got shape {tuple(out.shape)}"
            )
        return out[..., 0] + 1j * out[..., 1]


    def _show_prediction_vs_mma(
        self,
        a_val: torch.Tensor,
        omega_val: torch.Tensor,
        lambda_val: torch.Tensor,
        ramp_val: torch.Tensor,
        p_val: int,
        global_step: int,
    ):
        """
        在 r ∈ [viz_r_min, viz_r_max] 上画:
            R_pred(r) = U(r) * ( g(x(r)) * f(y(r)) + 1 )
        并与 Mathematica 的 R_in 对比。
        显示若干秒后自动关闭窗口。
        """
        if not self.viz_enabled:
            return

        # ---- 统一 dtype / device，跟随模型 ----
        model_param = next(self.model.parameters())
        dtype = model_param.dtype
        device = model_param.device

        # ---- 物理参数 ----
        a_scalar = float(a_val.detach().cpu().item())
        omega_scalar = float(omega_val.detach().cpu().item())

        problem_cfg = self.physics_cfg["problem"]
        M = float(problem_cfg.get("M", 1.0))
        l = int(problem_cfg.get("l", 2))
        m = int(problem_cfg.get("m", 2))
        s = int(problem_cfg.get("s", -2))

        a_t = torch.tensor(a_scalar, device=device, dtype=dtype)
        omega_t = torch.tensor(omega_scalar, device=device, dtype=dtype)

        if isinstance(ramp_val, torch.Tensor):
            ramp_t = ramp_val.detach().to(device=device, dtype=dtype)
        else:
            ramp_t = torch.tensor(float(ramp_val), device=device, dtype=dtype)

        rp = r_plus(a_t, M)

        # ---- r 网格 ----
        r_min = max(self.viz_r_min, float(rp.detach().cpu().item()) + 1.0e-4)
        r_max = self.viz_r_max

        if r_min >= r_max:
            print(f"[viz] skip: r_min={r_min:.6f} >= r_max={r_max:.6f}")
            return

        r_grid = torch.linspace(
            r_min, r_max, self.viz_num_points, device=device, dtype=dtype
        )

        # x = r_+ / r, y = 2x - 1
        x_grid = rp / r_grid
        y_grid = 2.0 * x_grid - 1.0

         # ---- Mathematica 参考值 ----
        r_np = r_grid.detach().cpu().numpy()
        try:
            R_mma = get_mathematica_Rin(
                a_scalar,
                omega_scalar,
                l,
                m,
                s,
                r_np,
            )
            R_mma_t = torch.as_tensor(R_mma, device=device, dtype=torch.complex128)
        except Exception as e:
            print(f"[viz] Mathematica evaluation failed at step {global_step}: {e}")
            self.model.train()
            return

        # ---- 前向 ----
        self.model.eval()
        with torch.no_grad():
            # 你的 PINN_MLP.forward(a, omega, y) 返回复数 f_complex
            f_pred = self.model(a_t.unsqueeze(0), omega_t.unsqueeze(0), y_grid).squeeze(0)

            # g(x) = exp(x-1)-1
            g_val, _, _ = g_factor(x_grid)

            # R' = g(x) * f(y) + 1
            h=h_factor(a_t, omega_t, m=m, M=M, s=s)
            Rprime_pred = g_val * f_pred + h

            # U = Leaver_prefactor * prefactor_Q
            P, P_r, P_rr = Leaver_prefactors(
                r_grid, a_t, omega_t, m=m, M=M, s=s
            )
            Q, Q_r, Q_rr = prefactor_Q(
                r_grid,
                a_t,
                omega_t,
                p=int(p_val),
                R_amp=ramp_t,
                M=M,
                s=s,
            )

            U, _, _ = U_prefactor(P, P_r, P_rr, Q, Q_r, Q_rr)

            R_pred = U * Rprime_pred
            f_from_mma=(R_mma_t/U-h)/g_val
            

       

        R_pred_np = R_pred.detach().cpu().numpy()
        R_mma_np = np.asarray(R_mma)
        f_pred_np = f_pred.detach().cpu().numpy()
        f_from_mma_np=f_from_mma.detach().cpu().numpy()
        # ---- 画图 ----
        plt.ion()
        fig, axes = plt.subplots(3, 2, figsize=(10, 10), sharex=True)
        axes = axes.ravel()

        axes[0].plot(r_np, np.real(R_pred_np), label="Pred Re(R)", lw=1.8)
        axes[0].plot(r_np, np.real(R_mma_np), "--", label="MMA Re(R)", lw=1.2)
        axes[0].set_ylabel("Re(R)")
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        axes[1].plot(r_np, np.imag(R_pred_np), label="Pred Im(R)", lw=1.8)
        axes[1].plot(r_np, np.imag(R_mma_np), "--", label="MMA Im(R)", lw=1.2)
        axes[1].set_ylabel("Im(R)")
        axes[1].legend()
        axes[1].grid(alpha=0.3)

        axes[2].plot(r_np, np.abs(R_pred_np), label="Pred |R|", lw=1.8)
        axes[2].plot(r_np, np.abs(R_mma_np), "--", label="MMA |R|", lw=1.2)
        axes[2].set_ylabel("|R|")
        axes[2].set_xlabel("r")
        axes[2].legend()
        axes[2].grid(alpha=0.3)

        axes[3].plot(r_np, np.real(f_pred_np), label="Pred Re(f)", lw=1.8)
        axes[3].plot(r_np, np.real(f_from_mma_np), "--", label="MMA Re(f)", lw=1.2)
        axes[3].set_ylabel("Re(f)")
        axes[3].legend()
        axes[3].grid(alpha=0.3)

        axes[4].plot(r_np, np.imag(f_pred_np), label="Pred Im(f)", lw=1.8)
        axes[4].plot(r_np, np.imag(f_from_mma_np), "--", label="MMA Im(f)", lw=1.2)
        axes[4].set_ylabel("Im(R)")
        axes[4].legend()
        axes[4].grid(alpha=0.3)

        axes[5].plot(r_np, np.abs(f_pred_np), label="Pred |f|", lw=1.8)
        axes[5].plot(r_np, np.abs(f_from_mma_np), "--", label="MMA |f|", lw=1.2)
        axes[5].set_ylabel("|f|")
        axes[5].set_xlabel("r")
        axes[5].legend()
        axes[5].grid(alpha=0.3)

        fig.suptitle(
            f"step={global_step}, a={a_scalar:.6f}, omega={omega_scalar:.6f}",
            fontsize=12
        )
        fig.tight_layout()

        # ---- 保存图片到 outputs/pinn/viz_compare/ ----
        if self.viz_save_enabled:
            filename = (
                f"step_{global_step:06d}"
                f"_a_{a_scalar:.6f}"
                f"_omega_{omega_scalar:.6f}.png"
            )
            save_path = self.viz_dir / filename
            fig.savefig(save_path, dpi=160, bbox_inches="tight")
            print(f"[viz] saved figure to {save_path}")

        # ---- 非阻塞显示 + 自动关闭 ----
        if self.viz_show_enabled:
            plt.show(block=False)
            plt.pause(self.viz_auto_close_sec)

        plt.close(fig)
        plt.close("all")
        self.model.train()


    def validate(self, a_val=None, omega_val=None):
        """在全r网格上验证"""
        self.model.eval()
        M = float(self.physics_cfg["problem"].get("M", 1.0))
        if a_val is None:
            a_val = torch.tensor(self.a_center, device=self.device, dtype=self.dtype)
        if omega_val is None:
            omega_val = torch.tensor(self.omega_center, device=self.device, dtype=self.dtype)
        
        

        rp = r_plus(a_val, M)
        r_min = float(rp) + 0.1
        r_max = 100.0

        x_grid = torch.linspace(float(rp)/r_max, float(rp)/r_min, self.val_n_points, device=self.device, dtype=self.dtype)
        y_grid = 2.0 * x_grid - 1.0
        y_grid = y_grid.clone().requires_grad_(True)

        a_batch = a_val.unsqueeze(0)
        omega_batch = omega_val.unsqueeze(0)

        lam = get_lambda_from_cfg(self.physics_cfg, self.cache, a_val, omega_val)
        p, ramp = get_ramp_and_p_from_cfg(self.physics_cfg, self.cache, a_val, omega_val)

        lambda_batch = lam.unsqueeze(0).to(device=self.device, dtype=torch.complex128)
        ramp_batch = ramp.unsqueeze(0).to(device=self.device, dtype=torch.complex128)

        loss, _ = pinn_residual_loss(
    self.model, self.physics_cfg, a_batch, omega_batch,
    lambda_batch, ramp_batch, int(p or 5),
    y_grid, torch.tensor([], device=self.device, dtype=self.dtype),
    weight_interior=1.0, weight_boundary=0.0
)

        return loss.item()

    def train_step(self):
        """单步训练"""
        self.model.train()
        self.optimizer.zero_grad()

        # 采样参数
        a_batch, omega_batch = sample_parameters(
            batch_size=self.batch_size,
            a_center=self.a_center,
            a_range=self.a_range,
            omega_center=self.omega_center,
            omega_range=self.omega_range,
            device=self.device,
            dtype=self.dtype,
        )

        # 采样点
        y_interior, y_boundary = sample_points_luna_style(
            n_interior=self.n_interior,
            n_boundary=self.n_boundary,
            boundary_layer_width=0.1,
            device=self.device,
            dtype=self.dtype,
        )

        # 需要梯度
        y_interior.requires_grad_(True)
        y_boundary.requires_grad_(True)

        # 计算 lambda 和 ramp
        lambda_list = []
        ramp_list = []
        p_val = None

        for i in range(self.batch_size):
            lam_i = get_lambda_from_cfg(self.physics_cfg, self.cache, a_batch[i], omega_batch[i])
            p_i, ramp_i = get_ramp_and_p_from_cfg(self.physics_cfg, self.cache, a_batch[i], omega_batch[i])

            lambda_list.append(lam_i)
            ramp_list.append(ramp_i)

            if p_val is None:
                p_val = p_i
            elif p_i != p_val:
                raise ValueError(f"Inconsistent p: {p_val} vs {p_i}")

        lambda_batch = torch.stack(lambda_list).to(device=self.device, dtype=torch.complex128)
        ramp_batch = torch.stack(ramp_list).to(device=self.device, dtype=torch.complex128)
        p = int(p_val or 5)

        # 计算PDE loss
        loss, info = pinn_residual_loss(
            model=self.model,
            cfg=self.physics_cfg,
            a_batch=a_batch,
            omega_batch=omega_batch,
            lambda_batch=lambda_batch,
            ramp_batch=ramp_batch,
            p=p,
            y_interior=y_interior,
            y_boundary=y_boundary,
            weight_interior=self.weight_interior,
            weight_boundary=self.weight_boundary,
        )
        self._maybe_restart_on_stall(info["total_loss"])
        total_loss = loss
        info['loss_anchor'] = 0.0

        # 反向传播
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        return info

    def train_step_with_anchor(self, step):
        """带锚点的训练步"""
        self.model.train()
        self.optimizer.zero_grad()

        # 采样参数
        a_batch, omega_batch = sample_parameters(
            batch_size=self.batch_size,
            a_center=self.a_center,
            a_range=self.a_range,
            omega_center=self.omega_center,
            omega_range=self.omega_range,
            device=self.device,
            dtype=self.dtype,
        )

        # 采样点
        y_interior, y_boundary = sample_points_luna_style(
            n_interior=self.n_interior,
            n_boundary=self.n_boundary,
            boundary_layer_width=0.1,
            device=self.device,
            dtype=self.dtype,
        )

        y_interior.requires_grad_(True)
        y_boundary.requires_grad_(True)

        # 计算 lambda 和 ramp
        lambda_list = []
        ramp_list = []
        p_val = None

        for i in range(self.batch_size):
            lam_i = get_lambda_from_cfg(self.physics_cfg, self.cache, a_batch[i], omega_batch[i])
            p_i, ramp_i = get_ramp_and_p_from_cfg(self.physics_cfg, self.cache, a_batch[i], omega_batch[i])
            lambda_list.append(lam_i)
            ramp_list.append(ramp_i)
            if p_val is None:
                p_val = p_i

        lambda_batch = torch.stack(lambda_list).to(device=self.device, dtype=torch.complex128)
        ramp_batch = torch.stack(ramp_list).to(device=self.device, dtype=torch.complex128)
        p = int(p_val or 5)

        # PDE loss
        loss_pde, info = pinn_residual_loss(
            model=self.model,
            cfg=self.physics_cfg,
            a_batch=a_batch,
            omega_batch=omega_batch,
            lambda_batch=lambda_batch,
            ramp_batch=ramp_batch,
            p=p,
            y_interior=y_interior,
            y_boundary=y_boundary,
            weight_interior=self.weight_interior,
            weight_boundary=self.weight_boundary,
        )

        # 锚点loss
        y_anchors = sample_anchor_points_gaussian_clusters(
            n_clusters=2, n_points_per_cluster=10, sigma=0.2,
            device=self.device, dtype=self.dtype
        )
        y_anchors.requires_grad_(True)

        # 获取Mathematica数据
        M = float(self.physics_cfg["problem"].get("M", 1.0))
        l = int(self.physics_cfg["problem"].get("l", 2))
        m = int(self.physics_cfg["problem"].get("m", 2))
        s = int(self.physics_cfg["problem"].get("s", -2))
        x_anchors = (y_anchors + 1.0) / 2.0

        R_mma_list = []
        for i in range(self.batch_size):
            rp = r_plus(a_batch[i], M)
            r_anchors = r_from_x(x_anchors, rp).detach().cpu().numpy()

            # 现在 get_mathematica_Rin 会严格在这组 r_anchors 上逐点求值
            R_mma = get_mathematica_Rin(
                float(a_batch[i]),
                float(omega_batch[i]),
                l, m, s,
                r_anchors
            )

            R_mma_torch = torch.tensor(
                R_mma,
                dtype=torch.complex128,
                device=self.device
            )
            R_mma_list.append(R_mma_torch)

        R_mma_batch = torch.stack(R_mma_list, dim=0)

        loss_anchor = compute_data_anchor_loss(
    self.model, self.physics_cfg, a_batch, omega_batch,
    y_anchors, R_mma_batch
)

        total_loss = loss_pde + self.weight_anchor * loss_anchor
        info['loss_anchor'] = loss_anchor.item()
        info['total_loss'] = total_loss.item()

        # 反向传播
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        return info

    def train(self, save_dir='outputs/pinn'):
        os.makedirs(save_dir, exist_ok=True)
        self.output_dir = Path(save_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.viz_dir = self.output_dir / self.viz_subdir
        self.viz_dir.mkdir(parents=True, exist_ok=True)
        best_val_loss = float('inf')
        no_improve_count = 0
        global_step = 0

        def save_checkpoint(path):
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'full_cfg': self.full_cfg,
                'physics_cfg': self.physics_cfg,
                'train_cfg': self.train_cfg,
                'loss_history': self.loss_history,
                'step_history': self.step_history,
                'val_loss_history': self.val_loss_history,
                'best_val_loss': best_val_loss,
                'global_step': global_step,
            }, path)

        # fixed_pool 模式只初始化一次
        fixed_pool = None
        if self.param_sampling_mode == "fixed_pool":
            fixed_pool = self._build_fixed_param_pool()
            print(f"[info] Built fixed parameter pool with {fixed_pool['a_all'].shape[0]} samples")

        # 计算总步数，用于 tqdm
        if self.param_sampling_mode == "fixed_pool":
            n_total = fixed_pool["a_all"].shape[0]
            steps_per_epoch = (n_total + self.batch_size - 1) // self.batch_size
        else:
            # random 模式下，每个 epoch 只生成一个参数 batch
            steps_per_epoch = 1

        total_steps = self.n_epochs * steps_per_epoch

        with tqdm(total=total_steps, desc="Training", dynamic_ncols=True) as pbar:
            for epoch in range(self.n_epochs):
                stage_changed = self._apply_curriculum(epoch)
                colloc_changed = self._apply_collocation_curriculum(epoch)
                if colloc_changed:
                    print(
                        f"[info] collocation curriculum switched: "
                        f"strategy={self.curr_interior_strategy}, "
                        f"n_interior={self.curr_n_interior}"
                    )
                if stage_changed and self.param_sampling_mode == "fixed_pool":
                    fixed_pool = self._build_fixed_param_pool()
                    print(
                        f"[info] curriculum switched: "
                        f"a_range={self.curr_a_range}, "
                        f"omega_range={self.curr_omega_range}, "
                        f"weight_anchor={self.weight_anchor}"
                    )
                param_batches = self._get_param_batches_for_epoch(fixed_pool=fixed_pool)
                epoch_losses = []

                for batch in param_batches:
                    global_step += 1

                    loss, info = self._run_one_training_batch(batch, global_step)
                    epoch_losses.append(float(loss.item()))

                    self.loss_history.append(info["total_loss"])
                    self.step_history.append(global_step)
                    
                    # ---- 周期性在线可视化 ----
                    if self.viz_enabled and global_step > 0 and global_step % self.viz_every_steps == 0:
                        try:
                            a_batch = batch["a_batch"]
                            omega_batch = batch["omega_batch"]
                            lambda_batch = batch["lambda_batch"]
                            ramp_batch = batch["ramp_batch"]
                            p = batch["p"]

                            self._show_prediction_vs_mma(
                                a_batch[0],
                                omega_batch[0],
                                lambda_batch[0],
                                ramp_batch[0],
                                p,
                                global_step,
                            )
                        except Exception as e:
                            print(f"[viz] failed at step {global_step}: {e}")
                    # 更新进度条
                    pbar.update(1)
                    pbar.set_postfix({
                        "total": f"{info['total_loss']:.3e}",
                        "int": f"{info.get('loss_interior', 0.0):.3e}",
                        "bd": f"{info.get('loss_boundary', 0.0):.3e}",
                        "anchor": f"{info.get('loss_anchor', 0.0):.3e}",
                        "grad": f"{info.get('grad_norm', 0.0):.3e}",
                        "var": f"{info.get('loss_var', 0.0):.3e}",
                        "sig": f"{info.get('sigma_var', 0.0):.3e}",
                    })

                    # 验证
                    if global_step % self.val_freq == 0:
                        val_loss = self.validate()
                        pbar.set_postfix({
                            "total": f"{info['total_loss']:.3e}",
                            "int": f"{info.get('loss_interior', 0.0):.3e}",
                            "bd": f"{info.get('loss_boundary', 0.0):.3e}",
                            "anchor": f"{info.get('loss_anchor', 0.0):.3e}",
                            "val": f"{val_loss:.3e}",
                        })

                        if val_loss < best_val_loss - self.early_stop_threshold:
                            best_val_loss = val_loss
                            no_improve_count = 0

                            # model_path = os.path.join(save_dir, "best_model.pt")
                            # torch.save({
                            #     'model_state_dict': self.model.state_dict(),
                            #     'optimizer_state_dict': self.optimizer.state_dict(),
                            #     'full_cfg': self.full_cfg,
                            #     'physics_cfg': self.physics_cfg,
                            #     'train_cfg': self.train_cfg,
                            #     'loss_history': self.loss_history,
                            #     'step_history': self.step_history,
                            # }, model_path)
                            model_path = os.path.join(save_dir, "best_model.pt")
                            save_checkpoint(model_path)
                        else:
                            no_improve_count += 1

                        if no_improve_count >= self.early_stop_patience:
                            print("\n[info] early stopping triggered")
                            final_model_path = os.path.join(save_dir, "final_model.pt")
                            save_checkpoint(final_model_path)
                            print(f"[info] final model saved to: {final_model_path}")
                            self.plot_loss_curve(save_dir)
                            return

                mean_epoch_loss = sum(epoch_losses) / len(epoch_losses)
                # tqdm.write(f"[epoch {epoch+1}] mean_train_loss = {mean_epoch_loss:.6e}")

        final_model_path = os.path.join(save_dir, "final_model.pt")
        save_checkpoint(final_model_path)
        print(f"[info] final model saved to: {final_model_path}")

        self.plot_loss_curve(save_dir)

    def _grad_norm(self, loss):
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward(retain_graph=True)

        grad_sq = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                grad_sq += p.grad.detach().pow(2).sum().item()

        return grad_sq ** 0.5


    def _update_dynamic_anchor_weight(self, loss_pde, loss_anchor, global_step):
        if not self.dynamic_balance_enabled or loss_anchor is None:
            return self.dynamic_anchor_weight

        if global_step % self.dynamic_balance_every != 0:
            return self.dynamic_anchor_weight

        g_pde = self._grad_norm(loss_pde)
        g_anchor = self._grad_norm(loss_anchor)

        ratio = g_pde / (g_anchor + 1e-12)
        ratio = max(self.w_anchor_min, min(self.w_anchor_max, ratio))
        self.dynamic_anchor_weight = float(ratio)
        return self.dynamic_anchor_weight

    # def train_random_sampling(self, save_dir='outputs/pinn'):
    #     """随机采样训练"""
    #     import time
    #     os.makedirs(save_dir, exist_ok=True)

    #     print(f"Training PINN (Random Sampling) on {self.device}")
    #     print(f"Parameter range: a={self.a_center}±{self.a_range}, ω={self.omega_center}±{self.omega_range}")
    #     print(f"Sampling: {self.n_interior} interior + {self.n_boundary}×2 boundary")
    #     print(f"Batch size: {self.batch_size}")
    #     print(f"Steps: {self.n_steps}\n")

    #     start_time = time.time()

    #     best_loss = float('inf')
    #     patience_counter = 0

    #     with tqdm(total=self.n_steps, desc="Training",
    #               bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:
    #         for step in range(1, self.n_steps + 1):
    #             # 每anchor_freq步使用锚点
    #             if step % self.anchor_freq == 0:
    #                 info = self.train_step_with_anchor(step)
    #             else:
    #                 info = self.train_step()

    #             # 每10步更新一次显示
    #             if step % 10 == 0:
    #                 elapsed = time.time() - start_time
    #                 postfix = {
    #                     'tot': f"{info['total_loss']:.2e}",
    #                     'int': f"{info['loss_interior']:.2e}",
    #                     'bd': f"{info['loss_boundary']:.2e}",
    #                 }
    #                 if info['loss_anchor'] > 0:
    #                     postfix['anch'] = f"{info['loss_anchor']:.2e}"
    #                 if hasattr(self, 'best_val_loss') and self.best_val_loss < float('inf'):
    #                     postfix['val'] = f"{self.best_val_loss:.2e}"
    #                 postfix['time'] = f"{elapsed:.1f}s"
    #                 pbar.set_postfix(postfix)

    #             # 每100步记录历史
    #             if step % 100 == 0:
    #                 self.loss_history.append(info['total_loss'])
    #                 self.step_history.append(step)

    #                 # 早停检查
    #                 if info['total_loss'] < best_loss - self.early_stop_threshold:
    #                     best_loss = info['total_loss']
    #                     patience_counter = 0
    #                 else:
    #                     patience_counter += 1

    #                 if patience_counter >= self.early_stop_patience:
    #                     print(f"\n早停触发：{patience_counter}步无改善")
    #                     break

    #             # 验证
    #             if step % self.val_freq == 0:
    #                 a_val = torch.tensor(self.a_center, device=self.device, dtype=self.dtype)
    #                 omega_val = torch.tensor(self.omega_center, device=self.device, dtype=self.dtype)
    #                 val_loss = self.validate(a_val, omega_val)
    #                 self.val_loss_history.append(val_loss)

    #                 if val_loss < self.best_val_loss:
    #                     self.best_val_loss = val_loss
    #                     if step % self.save_best_freq == 0:
    #                         best_path = os.path.join(save_dir, 'pinn_best.pt')
    #                         torch.save({'model_state_dict': self.model.state_dict(), 'step': step, 'val_loss': val_loss}, best_path)

    #             pbar.update(1)

    #     # 保存模型
    #     model_path = os.path.join(save_dir, 'pinn_model.pt')
    #     torch.save({
    #         'model_state_dict': self.model.state_dict(),
    #         'optimizer_state_dict': self.optimizer.state_dict(),
    #         'full_cfg': self.full_cfg,
    #         'physics_cfg': self.physics_cfg,
    #         'train_cfg': self.train_cfg,
    #         'loss_history': self.loss_history,
    #         'step_history': self.step_history,
    #     }, model_path)
    #     print(f"\nModel saved to: {model_path}")

    #     # 绘制loss曲线
    #     self.plot_loss_curve(save_dir)

    #     # 关闭Mathematica session
    #     from dataset.mathematica_anchor import close_mathematica_session
    #     close_mathematica_session()

    # def train_batch_gd(self, save_dir='outputs/pinn'):
    #     """批梯度下降训练"""
    #     import time
    #     os.makedirs(save_dir, exist_ok=True)

    #     print(f"Training PINN (Batch GD) on {self.device}")
    #     print(f"Parameter range: a={self.a_center}±{self.a_range}, ω={self.omega_center}±{self.omega_range}")
    #     print(f"Total param samples: {self.n_param_samples}, Batch size: {self.batch_size}")
    #     print(f"Epochs: {self.n_epochs}\n")

    #     # 预先采样固定的参数集
    #     a_all, omega_all = sample_parameters(
    #         batch_size=self.n_param_samples,
    #         a_center=self.a_center,
    #         a_range=self.a_range,
    #         omega_center=self.omega_center,
    #         omega_range=self.omega_range,
    #         device=self.device,
    #         dtype=self.dtype,
    #     )

    #     # 计算lambda和ramp
    #     cache = AuxCache()
    #     lambda_all = []
    #     ramp_all = []
    #     for i in range(self.n_param_samples):
    #         lam_i = get_lambda_from_cfg(self.physics_cfg, cache, a_all[i], omega_all[i])
    #         p_i, ramp_i = get_ramp_and_p_from_cfg(self.physics_cfg, cache, a_all[i], omega_all[i])
    #         lambda_all.append(lam_i)
    #         ramp_all.append(ramp_i)

    #     lambda_all = torch.stack(lambda_all).to(device=self.device, dtype=torch.complex128)
    #     ramp_all = torch.stack(ramp_all).to(device=self.device, dtype=torch.complex128)
    #     p = int(p_i or 5)

    #     n_batches = (self.n_param_samples + self.batch_size - 1) // self.batch_size
    #     start_time = time.time()
    #     best_loss = float('inf')
    #     patience_counter = 0
    #     step = 0

    #     with tqdm(total=self.n_epochs * n_batches, desc="Training") as pbar:
    #         for epoch in range(self.n_epochs):
    #             epoch_loss = 0.0

    #             for batch_idx in range(n_batches):
    #                 start_idx = batch_idx * self.batch_size
    #                 end_idx = min(start_idx + self.batch_size, self.n_param_samples)

    #                 a_batch = a_all[start_idx:end_idx]
    #                 omega_batch = omega_all[start_idx:end_idx]
    #                 lambda_batch = lambda_all[start_idx:end_idx]
    #                 ramp_batch = ramp_all[start_idx:end_idx]

    #                 # 每个batch重新采样空间点
    #                 y_interior, y_boundary = sample_points_luna_style(
    #                     n_interior=self.n_interior,
    #                     n_boundary=self.n_boundary,
    #                     boundary_layer_width=0.1,
    #                     device=self.device,
    #                     dtype=self.dtype,
    #                 )
    #                 y_interior.requires_grad_(True)
    #                 y_boundary.requires_grad_(True)

    #                 self.model.train()
    #                 self.optimizer.zero_grad()

    #                 step += 1

    #                 # PDE loss
    #                 loss_pde, info = pinn_residual_loss(
    #                     model=self.model,
    #                     cfg=self.physics_cfg,
    #                     a_batch=a_batch,
    #                     omega_batch=omega_batch,
    #                     lambda_batch=lambda_batch,
    #                     ramp_batch=ramp_batch,
    #                     p=p,
    #                     y_interior=y_interior,
    #                     y_boundary=y_boundary,
    #                     weight_interior=self.weight_interior,
    #                     weight_boundary=self.weight_boundary,
    #                 )

    #                 total_loss = loss_pde
    #                 info['loss_anchor'] = 0.0

    #                 # 每anchor_freq个batch使用锚点
    #                 if step % self.anchor_freq == 0:
    #                     y_anchors = sample_anchor_points_gaussian_clusters(
    #                         n_clusters=2, n_points_per_cluster=10, sigma=0.1,
    #                         device=self.device, dtype=self.dtype
    #                     )
    #                     y_anchors.requires_grad_(True)

    #                     M = float(self.physics_cfg["problem"].get("M", 1.0))
    #                     l = int(self.physics_cfg["problem"].get("l", 2))
    #                     m = int(self.physics_cfg["problem"].get("m", 2))
    #                     s = int(self.physics_cfg["problem"].get("s", -2))

    #                     R_mma_list = []
    #                     for i in range(a_batch.shape[0]):
    #                         rp = r_plus(a_batch[i], M)
    #                         x_anchors = (y_anchors + 1.0) / 2.0
    #                         r_anchors = r_from_x(x_anchors, rp).detach().cpu().numpy()

    #                         R_mma = get_mathematica_Rin(
    #                             float(a_batch[i]), float(omega_batch[i]),
    #                             l, m, s, r_anchors
    #                         )
    #                         R_mma_torch = torch.tensor(R_mma, dtype=torch.complex128, device=self.device)
    #                         R_mma_list.append(R_mma_torch)

    #                     R_mma_batch = torch.stack(R_mma_list, dim=0)

    #                     loss_anchor = compute_data_anchor_loss(
    #                         self.model, self.physics_cfg, a_batch, omega_batch,
    #                         y_anchors, R_mma_batch
    #                     )

    #                     total_loss = loss_pde + self.weight_anchor * loss_anchor
    #                     info['loss_anchor'] = loss_anchor.item()

    #                 info['total_loss'] = total_loss.item()

    #                 total_loss.backward()
    #                 torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
    #                 self.optimizer.step()

    #                 epoch_loss += info['total_loss']

    #                 if step % 10 == 0:
    #                     elapsed = time.time() - start_time
    #                     postfix = {
    #                         'ep': epoch+1,
    #                         'tot': f"{info['total_loss']:.2e}",
    #                         'int': f"{info['loss_interior']:.2e}",
    #                         'bd': f"{info['loss_boundary']:.2e}",
    #                         'anch': f"{info['loss_anchor']:.2e}",
    #                     }
    #                     if hasattr(self, 'best_val_loss') and self.best_val_loss < float('inf'):
    #                         postfix['val'] = f"{self.best_val_loss:.2e}"
    #                     postfix['time'] = f"{elapsed:.1f}s"
    #                     pbar.set_postfix(postfix)
    #                 pbar.update(1)

    #             # Epoch结束
    #             avg_loss = epoch_loss / n_batches
    #             self.loss_history.append(avg_loss)
    #             self.step_history.append(epoch)

    #             # 验证
    #             if epoch % self.val_freq == 0:
    #                 a_val = torch.tensor(self.a_center, device=self.device, dtype=self.dtype)
    #                 omega_val = torch.tensor(self.omega_center, device=self.device, dtype=self.dtype)
    #                 val_loss = self.validate(a_val, omega_val)
    #                 self.val_loss_history.append(val_loss)

    #                 if val_loss < self.best_val_loss:
    #                     self.best_val_loss = val_loss
    #                     if epoch % self.save_best_freq == 0:
    #                         best_path = os.path.join(save_dir, 'pinn_best.pt')
    #                         torch.save({'model_state_dict': self.model.state_dict(), 'epoch': epoch, 'val_loss': val_loss}, best_path)

    #             # 早停检查
    #             if avg_loss < best_loss - self.early_stop_threshold:
    #                 best_loss = avg_loss
    #                 patience_counter = 0
    #             else:
    #                 patience_counter += 1

    #             if patience_counter >= self.early_stop_patience:
    #                 print(f"\n早停触发：{patience_counter} epochs无改善")
    #                 break

    #     # 保存模型
    #     model_path = os.path.join(save_dir, 'pinn_model.pt')
    #     torch.save({
    #         'model_state_dict': self.model.state_dict(),
    #         'optimizer_state_dict': self.optimizer.state_dict(),
    #         'full_cfg': self.full_cfg,
    #         'physics_cfg': self.physics_cfg,
    #         'train_cfg': self.train_cfg,
    #         'loss_history': self.loss_history,
    #         'step_history': self.step_history,
    #     }, model_path)
    #     print(f"\nModel saved to: {model_path}")

    #     # 绘制loss曲线
    #     self.plot_loss_curve(save_dir)

    #     # 关闭Mathematica session
    #     from dataset.mathematica_anchor import close_mathematica_session
    #     close_mathematica_session()

    def plot_loss_curve(self, save_dir):
        """绘制loss曲线"""
        plt.figure(figsize=(10, 5))
        plt.semilogy(self.step_history, self.loss_history, label='Total Loss')
        plt.xlabel('Step')
        plt.ylabel('Loss')
        plt.title('PINN Training Loss')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        loss_path = os.path.join(save_dir, 'loss_curve.png')
        plt.savefig(loss_path, dpi=150)
        plt.close()
        print(f"Loss curve saved to: {loss_path}")


if __name__ == "__main__":
    cfg_path = "config/pinn_config.yaml"
    trainer = PINNTrainer(cfg_path, device='cpu')
    trainer.train()
