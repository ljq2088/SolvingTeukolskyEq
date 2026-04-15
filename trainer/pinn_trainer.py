"""
PINN Trainer
"""
import os
import math
import torch
import yaml
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datetime import datetime

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
    sample_points_chebyshev_grid,
)

from dataset.mathematica_anchor import get_mathematica_Rin




class PINNTrainer:
    # def __init__(self, cfg_path, device='cpu'):
    #     """
    #     Args:
    #         cfg_path: 配置文件路径
    #         device: 'cpu' 或 'cuda'
    #     """
    #     self.device = device
    #     # 加载总配置，并拆分 train / physics
    #     self.full_cfg = load_pinn_full_config(cfg_path)
    #     self.physics_cfg = self.full_cfg["physics"]
    #     self.train_cfg = self.full_cfg["train"]


    #     # 训练配置分组  
    #     param_cfg = self.train_cfg.get("param_sampling", {})
    #     model_cfg = self.train_cfg.get("model", {})
    #     sampling_cfg = self.train_cfg.get("sampling", {})
    #     self.param_sampling_mode = sampling_cfg.get("mode", "random")
    #     train_cfg = self.train_cfg.get("train", {})
    #     loss_cfg = self.train_cfg.get("loss", {})
    #     val_cfg = self.train_cfg.get("validation", {})
    #     early_cfg = self.train_cfg.get("early_stop", {})
    #     runtime_cfg = self.train_cfg.get("runtime", {})
    #     flat_var_cfg = self.train_cfg.get("flat_variance", {})

    #     sched_cfg = self.train_cfg.get("scheduler", {})
    #     self.scheduler_enabled = sched_cfg.get("enabled", False)
    #     self.scheduler_type = sched_cfg.get("type", "none")
    #     self.lr_warmup_steps = sched_cfg.get("warmup_steps", 0)
    #     self.lr_decay_start = sched_cfg.get("decay_start", 0)
    #     self.lr_decay_rate = sched_cfg.get("decay_rate", 1.0)
    #     self.min_lr = sched_cfg.get("min_lr", 1.0e-4)
    #     self.base_lr = train_cfg.get("lr", 1.0e-3)
    #     #平坦惩罚
    #     self.flat_var_enabled = flat_var_cfg.get("enabled", False)
    #     self.flat_var_target = flat_var_cfg.get("target", "Rprime")
    #     self.flat_var_kappa = flat_var_cfg.get("kappa", 20.0)
    #     self.flat_var_eps = flat_var_cfg.get("eps", 1.0e-12)
    #     self.flat_var_weight = flat_var_cfg.get("weight", 1.0e-3)
    #     self.flat_var_steps = flat_var_cfg.get("steps", 0)
    #     self.flat_var_decay_enabled = flat_var_cfg.get("decay_enabled", False)
    #     self.flat_var_warmup_steps = flat_var_cfg.get("warmup_steps", 0)
    #     self.flat_var_decay_start = flat_var_cfg.get("decay_start", 0)
    #     self.flat_var_decay_rate = flat_var_cfg.get("decay_rate", 1.0)
    #     self.flat_var_min_weight = flat_var_cfg.get("min_weight", 0.0)
    #     self.flat_var_base_weight = self.flat_var_weight


    #     self.flat_var_use_dedicated_points = flat_var_cfg.get("use_dedicated_points", True)
    #     self.flat_var_strategy = flat_var_cfg.get("strategy", "chebyshev")
    #     self.flat_var_n_points = flat_var_cfg.get("n_points", 64)
    #     # loss 权重
    #     self.weight_interior = loss_cfg.get("weight_interior", 1.0)
    #     self.weight_boundary = loss_cfg.get("weight_boundary", 10.0)
    #     self.weight_anchor = loss_cfg.get("weight_anchor", 1.0)
    #     # 采样配置
    #     self.n_interior = sampling_cfg.get("n_interior", 100)
    #     self.n_boundary = sampling_cfg.get("n_boundary", 20)
    #     self.batch_size = sampling_cfg.get("batch_size", 4)
    #     self.use_batch_gd = sampling_cfg.get("use_batch_gd", False)
    #     self.n_param_samples = sampling_cfg.get("n_param_samples", 20)
    #     self.n_epochs = sampling_cfg.get("n_epochs", 100)

    #     adaptive_cfg = self.train_cfg.get("adaptive_sampling", {})
    #     anchor_cfg = self.train_cfg.get("anchors", {})
    #     loss_balance_cfg = self.train_cfg.get("loss_balance", {})
    #     curriculum_cfg = self.train_cfg.get("curriculum", {})


    #     collocation_curr_cfg = self.train_cfg.get("collocation_curriculum", {})
    #     anti_trivial_cfg = self.train_cfg.get("anti_trivial", {})
    #     restart_cfg = self.train_cfg.get("restart", {})
    #     dynamic_balance_cfg = self.train_cfg.get("dynamic_balance", {})

    #     self.param_sampler = sampling_cfg.get("param_sampler", "sobol")

    #     self.interior_strategy = sampling_cfg.get("interior_strategy", "chebyshev")
    #     self.boundary_strategy = sampling_cfg.get("boundary_strategy", "none")
    #     self.article_uniform_cfg = sampling_cfg.get("article_uniform", {})

    #     self.collocation_curriculum_enabled = collocation_curr_cfg.get("enabled", False)
    #     self.collocation_curriculum_stages = collocation_curr_cfg.get("stages", [])
    #     self.curr_interior_strategy = self.interior_strategy
    #     self.curr_n_interior = self.n_interior

    #     self.anti_trivial_enabled = anti_trivial_cfg.get("enabled", False)
    #     self.seed_steps = anti_trivial_cfg.get("seed_steps", 0)
    #     self.seed_every = anti_trivial_cfg.get("seed_every", 10)
    #     self.n_seed = anti_trivial_cfg.get("n_seed", 8)
    #     self.seed_strategy = anti_trivial_cfg.get("seed_strategy", "article_uniform")
    #     self.weight_seed = anti_trivial_cfg.get("weight_seed", 0.1)
    #     self.seed_relative = anti_trivial_cfg.get("relative", False)

    #     self.restart_enabled = restart_cfg.get("enabled", False)
    #     self.stall_window = restart_cfg.get("stall_window", 300)
    #     self.stall_tol = restart_cfg.get("stall_tol", 1e-4)
    #     self.lr_decay_on_restart = restart_cfg.get("lr_decay_on_restart", 0.5)
    #     self._stall_best = float("inf")
    #     self._stall_count = 0

    #     self.dynamic_balance_enabled = dynamic_balance_cfg.get("enabled", False)
    #     self.dynamic_balance_every = dynamic_balance_cfg.get("every_steps", 20)
    #     self.w_anchor_min = dynamic_balance_cfg.get("w_anchor_min", 1e-3)
    #     self.w_anchor_max = dynamic_balance_cfg.get("w_anchor_max", 10.0)
    #     self.dynamic_anchor_weight = self.weight_anchor



    #     # dtype
    #     dtype_name = runtime_cfg.get("dtype", "float64")
    #     self.dtype = torch.float64 if dtype_name == "float64" else torch.float32

    #     # 参数范围
    #     self.a_center = param_cfg.get("a_center", 0.1)
    #     self.a_range = param_cfg.get("a_range", 0.01)
    #     self.omega_center = param_cfg.get("omega_center", 0.1)
    #     self.omega_range = param_cfg.get("omega_range", 0.01)



    #     # 训练配置
    #     self.n_steps = train_cfg.get("n_steps", 50000)
    #     self.lr = train_cfg.get("lr", 1e-3)
    #     self.anchor_freq = train_cfg.get("anchor_freq", 100)
    #     self.anchor_start_step = train_cfg.get("anchor_start_step", 0)
    #     self.n_anchors = train_cfg.get("n_anchors", 10)
    #     self.viz_enabled = train_cfg.get("viz_enabled", False)
    #     self.viz_every_steps = train_cfg.get("viz_every_steps", 500)
    #     self.viz_auto_close_sec = train_cfg.get("viz_auto_close_sec", 3.0)
    #     self.viz_num_points = train_cfg.get("viz_num_points", 400)
    #     self.viz_r_min = train_cfg.get("viz_r_min", 2.0)
    #     self.viz_r_max = train_cfg.get("viz_r_max", 100.0)
    #     self.viz_save_enabled = train_cfg.get("viz_save_enabled", True)
    #     self.viz_subdir = train_cfg.get("viz_subdir", "viz_compare")
    #     self.viz_show_enabled = train_cfg.get("viz_show_enabled", True)
    #     # 如果 trainer.train(save_dir=...) 之后会再覆盖主输出目录，这里先给默认值
    #     self.output_dir = Path(getattr(self, "output_dir", "outputs/pinn"))
    #     self.viz_dir = self.output_dir / self.viz_subdir
    #     self.viz_dir.mkdir(parents=True, exist_ok=True)



    #     # 早停
    #     self.early_stop_patience = early_cfg.get("patience", 500)
    #     self.early_stop_threshold = early_cfg.get("threshold", 1e-6)

    #     # 验证
    #     self.val_freq = val_cfg.get("val_freq", 50)
    #     self.val_n_points = val_cfg.get("val_n_points", 200)
    #     self.save_best_freq = val_cfg.get("save_best_freq", 50)



    #     self.adaptive_sampling_enabled = adaptive_cfg.get("enabled", True)
    #     self.candidate_method = adaptive_cfg.get("candidate_method", "sobol")
    #     self.n_candidates = adaptive_cfg.get("n_candidates", 512)
    #     self.refresh_freq = adaptive_cfg.get("refresh_freq", 50)
    #     self.adaptive_frac = adaptive_cfg.get("adaptive_frac", 0.7)
    #     self.normalize_residual = adaptive_cfg.get("normalize_residual", True)

    #     self.anchor_mode = anchor_cfg.get("mode", "sentinel_plus_adaptive")
    #     self.n_anchor_adaptive = anchor_cfg.get("n_adaptive", 6)
    #     self.n_anchor_random = anchor_cfg.get("n_random", 2)
    #     self.n_anchor_base = anchor_cfg.get("n_base", 8)

    #     self.loss_balance_mode = loss_balance_cfg.get("mode", "lbpinn")
    #     self.loss_balance_eps = loss_balance_cfg.get("eps", 1e-12)

    #     self.curriculum_enabled = curriculum_cfg.get("enabled", False)
    #     self.curriculum_stages = curriculum_cfg.get("stages", [])

    #     self.curr_a_range = self.a_range
    #     self.curr_omega_range = self.omega_range
    #     self._candidate_y = None    

    #             # ===== anchor 权重调度（仿 flat_variance）=====
    #     anchor_w_cfg = self.train_cfg.get("anchor_weight_schedule", {})
    #     self.anchor_w_enabled = anchor_w_cfg.get("enabled", False)
    #     self.anchor_w_steps = anchor_w_cfg.get("steps", 0)
    #     self.anchor_w_decay_enabled = anchor_w_cfg.get("decay_enabled", False)
    #     self.anchor_w_warmup_steps = anchor_w_cfg.get("warmup_steps", 0)
    #     self.anchor_w_decay_start = anchor_w_cfg.get("decay_start", 0)
    #     self.anchor_w_decay_rate = anchor_w_cfg.get("decay_rate", 1.0)
    #     self.anchor_w_min_weight = anchor_w_cfg.get("min_weight", 0.0)

    #     # ===== residual / interior 权重调度（与 flat 相反：逐步增大）=====
    #     interior_w_cfg = self.train_cfg.get("interior_weight_schedule", {})
    #     self.interior_w_enabled = interior_w_cfg.get("enabled", False)
    #     self.interior_w_steps = interior_w_cfg.get("steps", 0)
    #     self.interior_w_growth_enabled = interior_w_cfg.get("growth_enabled", False)
    #     self.interior_w_warmup_steps = interior_w_cfg.get("warmup_steps", 0)
    #     self.interior_w_growth_start = interior_w_cfg.get("growth_start", 0)
    #     self.interior_w_growth_rate = interior_w_cfg.get("growth_rate", 1.0)
    #     self.interior_w_max_weight = interior_w_cfg.get("max_weight", self.weight_interior)

    #     # 保存基础权重
    #     self.anchor_w_base_weight = self.weight_anchor
    #     self.interior_w_base_weight = self.weight_interior


    #     # 创建模型
    #     hidden_dims = model_cfg.get("hidden_dims", [128, 128, 128, 128])
    #     activation = model_cfg.get("activation", "silu")

    #     fourier_num_freqs = model_cfg.get("fourier_num_freqs", 8)
    #     fourier_scale = model_cfg.get("fourier_scale", 1.0)
    #     param_embed_dim = model_cfg.get("param_embed_dim", 64)
    #     use_film = model_cfg.get("use_film", True)
    #     use_residual = model_cfg.get("use_residual", True)

    #     self.model = PINN_MLP(
    #         hidden_dims=hidden_dims,
    #         activation=activation,
    #         fourier_num_freqs=fourier_num_freqs,
    #         fourier_scale=fourier_scale,
    #         param_embed_dim=param_embed_dim,
    #         use_film=use_film,
    #         use_residual=use_residual,
    #     ).to(device=self.device, dtype=self.dtype)

    #     # self.model = PINN_MLP(
    #     #     hidden_dims=hidden_dims,
    #     #     activation='tanh',
    #     # ).to(device=self.device, dtype=self.dtype)

    #     # 优化器
    #     extra_params = []
    #     if self.loss_balance_mode == "lbpinn":
    #         self.log_sigma_interior = torch.nn.Parameter(
    #             torch.zeros((), device=self.device, dtype=self.dtype)
    #         )
    #         self.log_sigma_anchor = torch.nn.Parameter(
    #             torch.zeros((), device=self.device, dtype=self.dtype)
    #         )
    #         extra_params = [self.log_sigma_interior, self.log_sigma_anchor]

    #     self.optimizer = torch.optim.Adam(
    #         list(self.model.parameters()) + extra_params,
    #         lr=self.lr,
    #     )

    #     # 缓存
    #     self.cache = AuxCache()

    #     # 历史记录
    #     self.loss_history = []
    #     self.step_history = []
    #     self.val_loss_history = []
    #     self.best_val_loss = float('inf')

    #     print(f"Parameter sampling mode: {self.param_sampling_mode}")

    def __init__(self, cfg_path, device='cpu'):
        self.device = device
        self.cfg_path = Path(cfg_path).resolve()
        self.project_root = self.cfg_path.parent.parent

        # ---- load config ----
        self.full_cfg = load_pinn_full_config(cfg_path)
        self.physics_cfg = self.full_cfg["physics"]
        self.cfg = self.full_cfg["train"]

        # 兼容旧代码
        self.train_cfg = self.cfg

        # ---- grouped init ----
        self._init_runtime(self.cfg.get("runtime", {}))
        self._init_parameter_space(self.cfg.get("parameter_space", {}))
        self._init_model_cfg(self.cfg.get("model", {}))
        self._init_training(self.cfg.get("training", {}))
        self._init_sampling(self.cfg.get("sampling", {}))
        self._init_curriculum(self.cfg.get("curriculum", {}))
        self._init_regularization(self.cfg.get("regularization", {}))
        self._init_restart(self.cfg.get("restart", {}))
        self._init_visualization(self.cfg.get("visualization", {}))
        self._init_initialization(self.cfg.get("initialization", {}))

        # ---- cache / histories ----
        self.cache = AuxCache()
        self.loss_history = []
        self.step_history = []
        self.val_loss_history = []
        self.val_worst_history = []
        self.best_val_loss = float('inf')
        self.latest_val_metrics = None
        self._validation_pool = None
        self._param_sample_skip = 0

        self.curr_a_range = self.a_range
        self.curr_omega_range = self.omega_range
        self.curr_interior_strategy = self.interior_strategy
        self.curr_n_interior = self.n_interior
        self._candidate_y = None

        # ---- model ----

        problem_cfg = self.physics_cfg.get("problem", {})
        M = float(problem_cfg.get("M", 1.0))
        m_mode = int(problem_cfg.get("m", 2))

        a_center_local = 0.5 * (self.model_a_min_local + self.model_a_max_local)
        a_half_range_local = 0.5 * (self.model_a_max_local - self.model_a_min_local)

        self._check_patch_consistency()

        self.model = PINN_MLP(
            hidden_dims=self.model_hidden_dims,
            activation=self.model_activation,
            fourier_num_freqs=self.model_fourier_num_freqs,
            fourier_scale=self.model_fourier_scale,
            param_embed_dim=self.model_param_embed_dim,
            use_film=self.model_use_film,
            use_residual=self.model_use_residual,
            a_center_local=a_center_local,
            a_half_range_local=a_half_range_local,
            omega_min_local=self.model_omega_min_local,
            omega_max_local=self.model_omega_max_local,
            M=M,
            m_mode=m_mode,
        ).to(device=self.device, dtype=self.dtype)

        print(f"[model] local patch: a in [{self.model_a_min_local}, {self.model_a_max_local}]")
        print(f"[model] local patch: omega in [{self.model_omega_min_local}, {self.model_omega_max_local}]")

        # ---- optimizer ----
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

        self._load_initial_checkpoint_if_needed()

        print(f"Parameter sampling mode: {self.param_sampling_mode}")
    def _init_runtime(self, runtime_cfg):
        dtype_name = runtime_cfg.get("dtype", "float64")
        self.dtype = torch.float64 if dtype_name == "float64" else torch.float32
    def _init_parameter_space(self, ps_cfg):
        a_cfg = ps_cfg.get("a", {})
        omega_cfg = ps_cfg.get("omega", {})
        domain_cfg = ps_cfg.get("domain", {})

        self.a_center = a_cfg.get("center", 0.1)
        self.a_range = a_cfg.get("range", 0.01)

        self.omega_center = omega_cfg.get("center", 0.1)
        self.omega_range = omega_cfg.get("range", 0.01)
        self.k_horizon_margin = float(domain_cfg.get("k_horizon_margin", 1.0e-2))
        self.param_resample_factor = int(domain_cfg.get("resample_factor", 4))
        self.param_resample_max_attempts = int(domain_cfg.get("max_attempts", 32))
    def _init_model_cfg(self, model_cfg):
        self.model_hidden_dims = model_cfg.get("hidden_dims", [128, 128, 128, 128])
        self.model_activation = model_cfg.get("activation", "silu")
        self.model_fourier_num_freqs = model_cfg.get("fourier_num_freqs", 2)
        self.model_fourier_scale = model_cfg.get("fourier_scale", 1.0)
        self.model_param_embed_dim = model_cfg.get("param_embed_dim", 64)
        self.model_use_film = model_cfg.get("use_film", True)
        self.model_use_residual = model_cfg.get("use_residual", True)

        self.model_a_min_local = model_cfg.get("a_min_local", 0.05)
        self.model_a_max_local = model_cfg.get("a_max_local", 0.2)
        self.model_omega_min_local = model_cfg.get("omega_min_local", 0.1)
        self.model_omega_max_local = model_cfg.get("omega_max_local", 1.0)
    def _check_patch_consistency(self):
        """
        检查当前训练参数域是否落在模型 patch 内。
        给一个很小的浮点容差，避免 0.099999999999 < 0.1 这种误判。
        """
        tol = 1.0e-12

        train_a_min = self.a_center - self.a_range
        train_a_max = self.a_center + self.a_range
        train_omega_min = self.omega_center - self.omega_range
        train_omega_max = self.omega_center + self.omega_range

        model_a_min = self.model_a_min_local
        model_a_max = self.model_a_max_local
        model_omega_min = self.model_omega_min_local
        model_omega_max = self.model_omega_max_local

        if train_a_min < model_a_min - tol or train_a_max > model_a_max + tol:
            raise ValueError(
                f"Training a-range [{train_a_min}, {train_a_max}] "
                f"falls outside model patch [{model_a_min}, {model_a_max}]"
            )

        if train_omega_min < model_omega_min - tol or train_omega_max > model_omega_max + tol:
            raise ValueError(
                f"Training omega-range [{train_omega_min}, {train_omega_max}] "
                f"falls outside model patch [{model_omega_min}, {model_omega_max}]"
            )
    def _init_training(self, training_cfg):
        optimizer_cfg = training_cfg.get("optimizer", {})
        anchor_cfg = training_cfg.get("anchor", {})
        scheduler_cfg = training_cfg.get("scheduler", {})
        loss_cfg = training_cfg.get("loss", {})
        loss_balance_cfg = training_cfg.get("loss_balance", {})
        val_cfg = training_cfg.get("validation", {})
        early_cfg = training_cfg.get("early_stop", {})
        ws_cfg = training_cfg.get("weight_schedule", {})

        # epochs
        self.n_epochs = training_cfg.get("epochs", 100)

        # optimizer
        self.lr = optimizer_cfg.get("lr", 1.0e-3)
        self.base_lr = self.lr

        # anchor trigger
        self.anchor_freq = anchor_cfg.get("freq", 100)
        self.anchor_start_step = anchor_cfg.get("start_step", 0)
        self.n_anchors = anchor_cfg.get("n_anchors", 10)

        # scheduler
        self.scheduler_enabled = scheduler_cfg.get("enabled", False)
        self.scheduler_type = scheduler_cfg.get("type", "none")
        self.lr_warmup_steps = scheduler_cfg.get("warmup_steps", 0)
        self.lr_decay_start = scheduler_cfg.get("decay_start", 0)
        self.lr_decay_rate = scheduler_cfg.get("decay_rate", 1.0)
        self.min_lr = scheduler_cfg.get("min_lr", 1.0e-4)

        # loss
        self.weight_interior = loss_cfg.get("weight_interior", 1.0)
        self.weight_boundary = loss_cfg.get("weight_boundary", 0.0)
        self.weight_anchor = loss_cfg.get("weight_anchor", 1.0)

        # loss balance
        self.loss_balance_mode = loss_balance_cfg.get("mode", "fixed")
        self.loss_balance_eps = loss_balance_cfg.get("eps", 1.0e-12)

        # validation
        self.val_freq = val_cfg.get("val_freq", 50)
        self.val_n_points = val_cfg.get("val_n_points", 200)
        self.save_best_freq = val_cfg.get("save_best_freq", 50)
        self.val_mode = val_cfg.get("mode", "fixed_pool")
        self.val_param_samples = val_cfg.get("n_param_samples", 16)
        self.val_param_sampler = val_cfg.get("param_sampler", "sobol")
        self.val_monitor = val_cfg.get("monitor", "mean")
        self.val_track_worst = val_cfg.get("track_worst", True)
        self.val_sobol_seed = val_cfg.get("sobol_seed", 4321)
        self.val_sobol_skip = val_cfg.get("sobol_skip", 0)

        # early stop
        self.early_stop_patience = early_cfg.get("patience", 500)
        self.early_stop_threshold = early_cfg.get("threshold", 1.0e-6)

        # weight schedule: anchor
        anchor_w_cfg = ws_cfg.get("anchor", {})
        self.anchor_w_enabled = anchor_w_cfg.get("enabled", False)
        self.anchor_w_steps = anchor_w_cfg.get("steps", 0)
        self.anchor_w_decay_enabled = anchor_w_cfg.get("decay_enabled", False)
        self.anchor_w_warmup_steps = anchor_w_cfg.get("warmup_steps", 0)
        self.anchor_w_decay_start = anchor_w_cfg.get("decay_start", 0)
        self.anchor_w_decay_rate = anchor_w_cfg.get("decay_rate", 1.0)
        self.anchor_w_min_weight = anchor_w_cfg.get("min_weight", 0.0)
        self.anchor_w_base_weight = self.weight_anchor

        # weight schedule: interior
        interior_w_cfg = ws_cfg.get("interior", {})
        self.interior_w_enabled = interior_w_cfg.get("enabled", False)
        self.interior_w_steps = interior_w_cfg.get("steps", 0)
        self.interior_w_growth_enabled = interior_w_cfg.get("growth_enabled", False)
        self.interior_w_warmup_steps = interior_w_cfg.get("warmup_steps", 0)
        self.interior_w_growth_start = interior_w_cfg.get("growth_start", 0)
        self.interior_w_growth_rate = interior_w_cfg.get("growth_rate", 1.0)
        self.interior_w_max_weight = interior_w_cfg.get("max_weight", self.weight_interior)
        self.interior_w_base_weight = self.weight_interior
    def _init_sampling(self, sampling_cfg):
        param_batch_cfg = sampling_cfg.get("parameter_batch", {})
        coll_cfg = sampling_cfg.get("collocation", {})
        interior_cfg = coll_cfg.get("interior", {})
        boundary_cfg = coll_cfg.get("boundary", {})
        adaptive_cfg = coll_cfg.get("adaptive", {})
        anchor_cfg = sampling_cfg.get("anchors", {})

        # parameter batch
        self.param_sampling_mode = param_batch_cfg.get("mode", "random")
        self.param_sampler = param_batch_cfg.get("sampler", "sobol")
        self.batch_size = param_batch_cfg.get("batch_size", 4)
        self.n_param_samples = param_batch_cfg.get("n_param_samples", 20)

        # collocation interior / boundary
        self.interior_strategy = interior_cfg.get("strategy", "rard")
        self.n_interior = interior_cfg.get("n_points", 100)

        self.boundary_strategy = boundary_cfg.get("strategy", "none")
        self.n_boundary = boundary_cfg.get("n_points", 0)

        # adaptive collocation
        self.adaptive_sampling_enabled = adaptive_cfg.get("enabled", True)
        self.candidate_method = adaptive_cfg.get("candidate_method", "sobol")
        self.n_candidates = adaptive_cfg.get("n_candidates", 512)
        self.refresh_freq = adaptive_cfg.get("refresh_freq", 50)
        self.adaptive_frac = adaptive_cfg.get("adaptive_frac", 0.7)
        self.normalize_residual = adaptive_cfg.get("normalize_residual", True)

        # anchor sampling
        self.anchor_mode = anchor_cfg.get("mode", "sentinel_plus_adaptive")
        self.n_anchor_base = anchor_cfg.get("n_base", 8)
        self.n_anchor_adaptive = anchor_cfg.get("n_adaptive", 6)
        self.n_anchor_random = anchor_cfg.get("n_random", 0)

        # 兼容旧调用
        self.article_uniform_cfg = {}
    def _init_curriculum(self, curriculum_cfg):
        param_curr_cfg = curriculum_cfg.get("parameter", {})
        colloc_curr_cfg = curriculum_cfg.get("collocation", {})

        self.curriculum_enabled = param_curr_cfg.get("enabled", False)
        self.curriculum_stages = param_curr_cfg.get("stages", [])

        self.collocation_curriculum_enabled = colloc_curr_cfg.get("enabled", False)
        self.collocation_curriculum_stages = colloc_curr_cfg.get("stages", [])
    def _init_regularization(self, reg_cfg):
        anti_cfg = reg_cfg.get("anti_trivial", {})
        dyn_cfg = reg_cfg.get("dynamic_balance", {})
        flat_cfg = reg_cfg.get("flat_variance", {})

        # anti-trivial
        self.anti_trivial_enabled = anti_cfg.get("enabled", False)
        self.seed_steps = anti_cfg.get("seed_steps", 0)
        self.seed_every = anti_cfg.get("seed_every", 10)
        self.n_seed = anti_cfg.get("n_seed", 8)
        self.seed_strategy = anti_cfg.get("seed_strategy", "article_uniform")
        self.weight_seed = anti_cfg.get("weight_seed", 0.1)
        self.seed_relative = anti_cfg.get("relative", False)

        # dynamic anchor balance
        self.dynamic_balance_enabled = dyn_cfg.get("enabled", False)
        self.dynamic_balance_every = dyn_cfg.get("every_steps", 20)
        self.w_anchor_min = dyn_cfg.get("w_anchor_min", 1.0e-3)
        self.w_anchor_max = dyn_cfg.get("w_anchor_max", 10.0)
        self.dynamic_anchor_weight = self.weight_anchor

        # flat variance
        self.flat_var_enabled = flat_cfg.get("enabled", False)
        self.flat_var_target = flat_cfg.get("target", "Rprime")
        self.flat_var_kappa = flat_cfg.get("kappa", 20.0)
        self.flat_var_eps = flat_cfg.get("eps", 1.0e-12)
        self.flat_var_weight = flat_cfg.get("weight", 1.0e-3)
        self.flat_var_steps = flat_cfg.get("steps", 0)

        self.flat_var_decay_enabled = flat_cfg.get("decay_enabled", False)
        self.flat_var_warmup_steps = flat_cfg.get("warmup_steps", 0)
        self.flat_var_decay_start = flat_cfg.get("decay_start", 0)
        self.flat_var_decay_rate = flat_cfg.get("decay_rate", 1.0)
        self.flat_var_min_weight = flat_cfg.get("min_weight", 0.0)
        self.flat_var_base_weight = self.flat_var_weight

        self.flat_var_use_dedicated_points = flat_cfg.get("use_dedicated_points", True)
        self.flat_var_strategy = flat_cfg.get("strategy", "article_uniform")
        self.flat_var_n_points = flat_cfg.get("n_points", 64)
                # head regularization
        head_reg_cfg = reg_cfg.get("head_reg", {})
        self.head_reg_enabled = head_reg_cfg.get("enabled", False)
        self.head_reg_weight_nl = head_reg_cfg.get("weight_nl", 1.0e-6)
        self.head_reg_weight_resp = head_reg_cfg.get("weight_resp", 1.0e-8)

        # parameter curvature regularization
        curv_cfg = reg_cfg.get("param_curvature", {})
        self.param_curv_enabled = curv_cfg.get("enabled", False)
        self.param_curv_weight = curv_cfg.get("weight", 1.0e-4)
        self.param_curv_delta_alpha = curv_cfg.get("delta_alpha", 0.05)
        self.param_curv_delta_xi = curv_cfg.get("delta_xi", 0.05)
        self.param_curv_n_probe_points = curv_cfg.get("n_probe_points", 32)
    def _init_restart(self, restart_cfg):
        self.restart_enabled = restart_cfg.get("enabled", False)
        self.stall_window = restart_cfg.get("stall_window", 300)
        self.stall_tol = restart_cfg.get("stall_tol", 1e-4)
        self.lr_decay_on_restart = restart_cfg.get("lr_decay_on_restart", 0.5)

        self._stall_best = float("inf")
        self._stall_count = 0
    def _init_visualization(self, viz_cfg):
        self.viz_show_enabled = viz_cfg.get("viz_show_enabled", True)
        self.viz_enabled = viz_cfg.get("viz_enabled", False)
        self.viz_every_steps = viz_cfg.get("viz_every_steps", 500)
        self.viz_auto_close_sec = viz_cfg.get("viz_auto_close_sec", 3.0)
        self.viz_num_points = viz_cfg.get("viz_num_points", 400)
        self.viz_r_min = viz_cfg.get("viz_r_min", 2.0)
        self.viz_r_max = viz_cfg.get("viz_r_max", 100.0)
        self.viz_save_enabled = viz_cfg.get("viz_save_enabled", True)
        self.viz_subdir = viz_cfg.get("viz_subdir", "viz_compare")

        self.output_dir = Path(getattr(self, "output_dir", "outputs/pinn"))
        self.viz_dir = self.output_dir / self.viz_subdir
        self.viz_dir.mkdir(parents=True, exist_ok=True)
        self.viz_start_time = None
    def _init_initialization(self, init_cfg):
        self.init_enabled = init_cfg.get("enabled", False)
        self.init_checkpoint = init_cfg.get("checkpoint", None)
        self.init_load_optimizer = init_cfg.get("load_optimizer", False)
        self.init_strict = init_cfg.get("strict", True)
    def _resolve_init_checkpoint_path(self) -> Path | None:
        if not self.init_checkpoint:
            return None

        ckpt_path = Path(self.init_checkpoint).expanduser()
        if ckpt_path.is_absolute():
            return ckpt_path

        return (self.project_root / ckpt_path).resolve()
    def _load_initial_checkpoint_if_needed(self):
        if not self.init_enabled:
            return

        ckpt_path = self._resolve_init_checkpoint_path()
        if ckpt_path is None:
            raise ValueError("initialization.enabled=True but initialization.checkpoint is missing")
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Initialization checkpoint not found: {ckpt_path}")

        checkpoint = torch.load(ckpt_path, map_location=self.device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        self.model.load_state_dict(state_dict, strict=self.init_strict)

        if self.init_load_optimizer and isinstance(checkpoint, dict) and "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        print(f"[init] loaded model weights from: {ckpt_path}")
        if self.init_load_optimizer:
            print("[init] optimizer state restored from checkpoint")
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

        pool = self._build_param_pool_with_metadata(
            target_size=self.n_param_samples,
            a_center=self.a_center,
            a_range=self.curr_a_range,
            omega_center=self.omega_center,
            omega_range=self.curr_omega_range,
            sampler=self.param_sampler,
            cache=AuxCache(),
            sobol_skip=self._param_sample_skip if self.param_sampler == "sobol" else 0,
            advance_main_skip=True,
        )
        a_all = pool["a_all"]
        omega_all = pool["omega_all"]
        lambda_all = pool["lambda_all"]
        ramp_all = pool["ramp_all"]
        p_val = pool["p"]

        print(
            f"[param_pool] a in [{a_all.min().item():.6f}, {a_all.max().item():.6f}], "
            f"omega in [{omega_all.min().item():.6f}, {omega_all.max().item():.6f}]"
        )

        return pool

    def _build_validation_pool(self):
        """
        构造固定验证参数池。
        验证始终覆盖完整 parameter_space，而不是 curriculum 的当前子区间。
        """
        if self.val_mode != "fixed_pool":
            raise ValueError(f"Unsupported validation mode: {self.val_mode}")

        return self._build_param_pool_with_metadata(
            target_size=self.val_param_samples,
            a_center=self.a_center,
            a_range=self.a_range,
            omega_center=self.omega_center,
            omega_range=self.omega_range,
            sampler=self.val_param_sampler,
            cache=AuxCache(),
            sobol_seed=self.val_sobol_seed,
            sobol_skip=self.val_sobol_skip,
            advance_main_skip=False,
        )


    def _format_float_tag(self, x: float, ndigits: int = 3) -> str:
        """
        用于文件夹命名，把浮点数转成安全字符串。
        例如 0.125 -> 0p125
        """
        return f"{x:.{ndigits}f}".replace("-", "m").replace(".", "p")


    def _build_viz_reference_sample(self, fixed_pool=None):
        """
        构造本次训练固定使用的可视化参考样本。
        逻辑：
        1) fixed_pool 模式：从整个参数池里选最接近训练中心(a_center, omega_center)的样本
        2) random 模式：直接用训练中心构造一个参考样本
        返回：
            {
                "a": ...,
                "omega": ...,
                "lambda": ...,
                "ramp": ...,
                "p": int,
            }
        """
        target_a = torch.tensor(self.a_center, device=self.device, dtype=self.dtype)
        target_omega = torch.tensor(self.omega_center, device=self.device, dtype=self.dtype)

        # ---- fixed_pool: 从参数池中挑一个固定参考点 ----
        if self.param_sampling_mode == "fixed_pool" and fixed_pool is not None:
            a_all = fixed_pool["a_all"]
            omega_all = fixed_pool["omega_all"]
            lambda_all = fixed_pool["lambda_all"]
            ramp_all = fixed_pool["ramp_all"]
            p_val = fixed_pool["p"]

            denom_a = max(float(self.a_range), 1.0e-12)
            denom_omega = max(float(self.omega_range), 1.0e-12)

            score = ((a_all - target_a) / denom_a) ** 2 + ((omega_all - target_omega) / denom_omega) ** 2
            idx = int(torch.argmin(score).item())

            sample = {
                "a": a_all[idx],
                "omega": omega_all[idx],
                "lambda": lambda_all[idx],
                "ramp": ramp_all[idx],
                "p": int(p_val),
            }
            return sample

        # ---- random: 直接用中心点构造 ----
        lam = get_lambda_from_cfg(self.physics_cfg, self.cache, target_a, target_omega)
        p_val, ramp = get_ramp_and_p_from_cfg(self.physics_cfg, self.cache, target_a, target_omega)

        sample = {
            "a": target_a,
            "omega": target_omega,
            "lambda": lam.to(device=self.device, dtype=torch.complex128),
            "ramp": ramp.to(device=self.device, dtype=torch.complex128),
            "p": int(p_val or 5),
        }
        return sample

    def _plot_parameter_domain(self, save_dir, fixed_pool=None):
        """
        绘制训练开始时的 (a, omega) 参数域示意图。
        显示：
        - 完整矩形 parameter_space
        - 共振线 omega = m * Omega_H(a)
        - 按 |k_horizon| < margin 剔除的危险带
        - 当前训练池 / 验证池散点
        """
        problem_cfg = self.physics_cfg.get("problem", {})
        M = float(problem_cfg.get("M", 1.0))
        m_mode = int(problem_cfg.get("m", 2))

        a_min = self.a_center - self.a_range
        a_max = self.a_center + self.a_range
        omega_min = self.omega_center - self.omega_range
        omega_max = self.omega_center + self.omega_range

        a_grid = np.linspace(a_min, a_max, 600)
        spin_gap = np.sqrt(np.clip(M * M - a_grid * a_grid, 1.0e-12, None))
        r_plus_grid = M + spin_gap
        omega_h = a_grid / (r_plus_grid * r_plus_grid + a_grid * a_grid)
        omega_res = m_mode * omega_h
        omega_lo = np.clip(omega_res - self.k_horizon_margin, omega_min, omega_max)
        omega_hi = np.clip(omega_res + self.k_horizon_margin, omega_min, omega_max)

        fig, ax = plt.subplots(figsize=(7.5, 6.0))
        ax.fill_between(
            a_grid,
            omega_lo,
            omega_hi,
            color="#d95f02",
            alpha=0.20,
            label=rf"excluded: $|k_H| < {self.k_horizon_margin:.1e}$",
        )
        ax.plot(a_grid, omega_res, color="#d95f02", lw=1.8, label=r"resonance: $\omega = m \Omega_H(a)$")

        if fixed_pool is not None:
            ax.scatter(
                fixed_pool["a_all"].detach().cpu().numpy(),
                fixed_pool["omega_all"].detach().cpu().numpy(),
                s=18,
                color="#1b9e77",
                alpha=0.85,
                label="training pool",
            )

        if self._validation_pool is not None:
            ax.scatter(
                self._validation_pool["a_all"].detach().cpu().numpy(),
                self._validation_pool["omega_all"].detach().cpu().numpy(),
                s=22,
                marker="x",
                color="#7570b3",
                alpha=0.9,
                label="validation pool",
            )

        ax.set_xlim(a_min, a_max)
        ax.set_ylim(omega_min, omega_max)
        ax.set_xlabel("a")
        ax.set_ylabel("omega")
        ax.set_title("Training Parameter Domain")
        ax.grid(alpha=0.25)
        ax.legend(loc="best", fontsize=9)
        fig.tight_layout()

        save_path = Path(save_dir) / "parameter_domain.png"
        fig.savefig(save_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        print(f"[train] parameter domain figure saved to: {save_path}")


    def _make_run_name(self) -> str:
        """
        用训练开始时间 + 当前训练范围生成 run 文件夹名。
        """
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        a_min = self.a_center - self.a_range
        a_max = self.a_center + self.a_range
        omega_min = self.omega_center - self.omega_range
        omega_max = self.omega_center + self.omega_range

        run_name = (
            f"{ts}"
            f"_a_{self._format_float_tag(a_min)}_{self._format_float_tag(a_max)}"
            f"_omega_{self._format_float_tag(omega_min)}_{self._format_float_tag(omega_max)}"
        )
        return run_name
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

            batch_pool = self._build_param_pool_with_metadata(
                target_size=self.batch_size,
                a_center=self.a_center,
                a_range=self.curr_a_range,
                omega_center=self.omega_center,
                omega_range=self.curr_omega_range,
                sampler=self.param_sampler,
                cache=self.cache,
                sobol_skip=self._param_sample_skip if self.param_sampler == "sobol" else 0,
                advance_main_skip=True,
            )

            batches.append({
                "a_batch": batch_pool["a_all"],
                "omega_batch": batch_pool["omega_all"],
                "lambda_batch": batch_pool["lambda_all"],
                "ramp_batch": batch_pool["ramp_all"],
                "p": batch_pool["p"],
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
        
    def _current_anchor_weight(self, global_step: int) -> float:
        """
        anchor 权重调度：完全仿照 flat_variance 的衰减逻辑
        """
        if not self.anchor_w_enabled:
            return self.anchor_w_base_weight

        if self.anchor_w_steps > 0 and global_step > self.anchor_w_steps:
            return self.anchor_w_min_weight

        if not self.anchor_w_decay_enabled:
            return self.anchor_w_base_weight

        if global_step < self.anchor_w_decay_start:
            return self.anchor_w_base_weight

        w = self.anchor_w_base_weight * (
            self.anchor_w_decay_rate ** (global_step - self.anchor_w_decay_start)
        )
        return max(self.anchor_w_min_weight, w)


    def _current_interior_weight(self, global_step: int) -> float:
        """
        residual / interior 权重调度：与 flat 相反，随 step 增长
        """
        if not self.interior_w_enabled:
            return self.interior_w_base_weight

        if self.interior_w_steps > 0 and global_step > self.interior_w_steps:
            return self.interior_w_max_weight

        if not self.interior_w_growth_enabled:
            return self.interior_w_base_weight

        if global_step < self.interior_w_growth_start:
            return self.interior_w_base_weight

        w = self.interior_w_base_weight * (
            self.interior_w_growth_rate ** (global_step - self.interior_w_growth_start)
        )
        return min(self.interior_w_max_weight, w)
    

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

    def _k_horizon_tensor(self, a_batch, omega_batch):
        problem_cfg = self.physics_cfg.get("problem", {})
        M = float(problem_cfg.get("M", 1.0))
        m_mode = int(problem_cfg.get("m", 2))
        spin_gap = torch.sqrt(torch.clamp(M * M - a_batch * a_batch, min=1.0e-12))
        r_plus_local = M + spin_gap
        omega_h = a_batch / (r_plus_local * r_plus_local + a_batch * a_batch)
        return omega_batch - m_mode * omega_h

    def _draw_param_candidates(
        self,
        batch_size,
        a_center,
        a_range,
        omega_center,
        omega_range,
        sampler,
        sobol_seed=None,
        sobol_skip=0,
    ):
        if sampler == "sobol":
            return sample_parameters_sobol(
                batch_size=batch_size,
                a_center=a_center,
                a_range=a_range,
                omega_center=omega_center,
                omega_range=omega_range,
                device=self.device,
                dtype=self.dtype,
                seed=1234 if sobol_seed is None else sobol_seed,
                skip=sobol_skip,
            )
        return sample_parameters(
            batch_size=batch_size,
            a_center=a_center,
            a_range=a_range,
            omega_center=omega_center,
            omega_range=omega_range,
            device=self.device,
            dtype=self.dtype,
        )

    def _sample_valid_param_batch(
        self,
        batch_size,
        a_center,
        a_range,
        omega_center,
        omega_range,
        sampler,
        sobol_seed=None,
        sobol_skip=0,
        advance_main_skip=True,
        return_next_skip=False,
    ):
        accepted_a = []
        accepted_omega = []
        next_skip = sobol_skip

        for _ in range(self.param_resample_max_attempts):
            if sum(x.numel() for x in accepted_a) >= batch_size:
                break

            need = batch_size - sum(x.numel() for x in accepted_a)
            draw_size = max(need * self.param_resample_factor, need)
            a_cand, omega_cand = self._draw_param_candidates(
                batch_size=draw_size,
                a_center=a_center,
                a_range=a_range,
                omega_center=omega_center,
                omega_range=omega_range,
                sampler=sampler,
                sobol_seed=sobol_seed,
                sobol_skip=next_skip,
            )

            if sampler == "sobol":
                next_skip += draw_size

            k_h = self._k_horizon_tensor(a_cand, omega_cand)
            valid_mask = torch.abs(k_h) >= self.k_horizon_margin
            if valid_mask.any():
                accepted_a.append(a_cand[valid_mask])
                accepted_omega.append(omega_cand[valid_mask])

        n_accepted = sum(x.numel() for x in accepted_a)
        if n_accepted < batch_size:
            raise RuntimeError(
                f"Unable to sample enough valid parameter points with |k_horizon| >= {self.k_horizon_margin}. "
                f"Requested {batch_size}, got {n_accepted}."
            )

        a_batch = torch.cat(accepted_a, dim=0)[:batch_size]
        omega_batch = torch.cat(accepted_omega, dim=0)[:batch_size]

        if sampler == "sobol" and advance_main_skip:
            self._param_sample_skip = next_skip

        if return_next_skip:
            return a_batch, omega_batch, next_skip
        return a_batch, omega_batch

    def _build_param_pool_with_metadata(
        self,
        target_size,
        a_center,
        a_range,
        omega_center,
        omega_range,
        sampler,
        cache,
        sobol_seed=None,
        sobol_skip=0,
        advance_main_skip=True,
    ):
        accepted_a = []
        accepted_omega = []
        lambda_list = []
        ramp_list = []
        p_val = None
        next_skip = sobol_skip
        n_fail = 0

        for _ in range(self.param_resample_max_attempts):
            if len(lambda_list) >= target_size:
                break

            need = target_size - len(lambda_list)
            a_batch, omega_batch, next_skip = self._sample_valid_param_batch(
                batch_size=need,
                a_center=a_center,
                a_range=a_range,
                omega_center=omega_center,
                omega_range=omega_range,
                sampler=sampler,
                sobol_seed=sobol_seed,
                sobol_skip=next_skip,
                advance_main_skip=False,
                return_next_skip=True,
            )

            for i in range(a_batch.shape[0]):
                try:
                    lam_i = get_lambda_from_cfg(
                        self.physics_cfg, cache, a_batch[i], omega_batch[i]
                    )
                    p_i, ramp_i = get_ramp_and_p_from_cfg(
                        self.physics_cfg, cache, a_batch[i], omega_batch[i]
                    )
                except Exception as exc:
                    n_fail += 1
                    print(
                        f"[param_pool] skip invalid sample a={float(a_batch[i].detach().cpu().item()):.6f}, "
                        f"omega={float(omega_batch[i].detach().cpu().item()):.6f}: {exc}"
                    )
                    continue

                if p_val is None:
                    p_val = int(p_i or 5)
                elif int(p_i or 5) != p_val:
                    raise ValueError("参数池中不同样本返回了不同的 p，当前实现不支持。")

                accepted_a.append(a_batch[i])
                accepted_omega.append(omega_batch[i])
                lambda_list.append(lam_i)
                ramp_list.append(ramp_i)

                if len(lambda_list) >= target_size:
                    break

        if len(lambda_list) < target_size:
            raise RuntimeError(
                f"Unable to build a valid parameter pool. Requested {target_size}, "
                f"built {len(lambda_list)}, skipped {n_fail} invalid samples."
            )

        if sampler == "sobol" and advance_main_skip:
            self._param_sample_skip = next_skip

        return {
            "a_all": torch.stack(accepted_a[:target_size], dim=0),
            "omega_all": torch.stack(accepted_omega[:target_size], dim=0),
            "lambda_all": torch.stack(lambda_list[:target_size], dim=0),
            "ramp_all": torch.stack(ramp_list[:target_size], dim=0),
            "p": int(p_val or 5),
        }

    def _sample_param_batch(self, batch_size, skip=0):
        base_skip = self._param_sample_skip if self.param_sampler == "sobol" else skip
        return self._sample_valid_param_batch(
            batch_size=batch_size,
            a_center=self.a_center,
            a_range=self.curr_a_range,
            omega_center=self.omega_center,
            omega_range=self.curr_omega_range,
            sampler=self.param_sampler,
            sobol_skip=base_skip,
            advance_main_skip=True,
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

       
        if not self.adaptive_sampling_enabled:
            y_interior = sample_interior_points(
                strategy="chebyshev",
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
        """
        anchor 点改为：
        1. 基础 Chebyshev-Lobatto 安全点（边界附近更密）
        2. 再叠加 residual top-k adaptive 点
        """
        y_parts = [
            sample_points_chebyshev_grid(
                n_points=self.n_anchor_base,
                y_min=-0.98,
                y_max=0.98,
                device=self.device,
                dtype=self.dtype,
            )
        ]

        if residual_score is not None and self.n_anchor_adaptive > 0 and self._candidate_y is not None:
            k = min(self.n_anchor_adaptive, self._candidate_y.numel())
            idx = torch.topk(residual_score, k=k).indices
            y_parts.append(self._candidate_y[idx].detach())

        y = torch.unique(torch.cat(y_parts, dim=0))
        y = torch.sort(y).values
        return y.clone().requires_grad_(True)


    def _combine_losses(
        self,
        loss_pde,
        loss_seed=None,
        loss_anchor=None,
        loss_var=None,
        interior_weight=None,
        anchor_weight=None,
        var_weight=None,
    ):
        wi = self.weight_interior if interior_weight is None else interior_weight

        if self.loss_balance_mode == "lbpinn":
            total = wi * (
                torch.exp(-self.log_sigma_interior) * loss_pde + self.log_sigma_interior
            )

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

        total = wi * loss_pde

        if loss_seed is not None:
            total = total + self.weight_seed * loss_seed

        if loss_anchor is not None:
            wa = self.weight_anchor if anchor_weight is None else anchor_weight
            total = total + wa * loss_anchor

        if loss_var is not None:
            wv = self.flat_var_weight if var_weight is None else var_weight
            total = total + wv * loss_var

        return total    
    
    def _compute_head_regularization(self):
        """
        头部正则：
        - 对 nl_head 用更强的 L2 正则
        - 对 a_head / omega_head 用较弱的 L2 正则
        目的：
        1. 初期先让一阶响应头学结构
        2. 防止 nl_head 过早吞掉所有参数变化
        """
        if not self.head_reg_enabled:
            zero = torch.zeros((), device=self.device, dtype=self.dtype)
            return zero, {
                "loss_head_reg": 0.0,
                "loss_head_reg_nl": 0.0,
                "loss_head_reg_resp": 0.0,
            }

        model = self.model

        nl_sq = (
            model.nl_head.weight.pow(2).mean()
            + model.nl_head.bias.pow(2).mean()
        )

        resp_sq = (
            model.a_head.weight.pow(2).mean()
            + model.a_head.bias.pow(2).mean()
            + model.omega_head.weight.pow(2).mean()
            + model.omega_head.bias.pow(2).mean()
        )

        loss_nl = self.head_reg_weight_nl * nl_sq
        loss_resp = self.head_reg_weight_resp * resp_sq
        loss_total = loss_nl + loss_resp

        info = {
            "loss_head_reg": float(loss_total.detach().cpu().item()),
            "loss_head_reg_nl": float(loss_nl.detach().cpu().item()),
            "loss_head_reg_resp": float(loss_resp.detach().cpu().item()),
        }
        return loss_total, info
    def _compute_param_curvature_regularization(self, a_batch, omega_batch, y_points):
        """
        用参数方向的二阶中心差分，约束局部曲率：
            f(p+δ) + f(p-δ) - 2 f(p)

        这里 δ 不是直接用物理单位，而是：
        - a 方向：通过 delta_alpha 转成物理 da
        - omega 方向：通过 delta_xi 转成对数频率扰动
        """
        if not self.param_curv_enabled:
            zero = torch.zeros((), device=self.device, dtype=self.dtype)
            return zero, {
                "loss_param_curv": 0.0,
                "loss_param_curv_a": 0.0,
                "loss_param_curv_omega": 0.0,
            }

        # 只取一部分 y 点，减少额外开销
        if y_points.numel() > self.param_curv_n_probe_points:
            idx = torch.linspace(
                0,
                y_points.numel() - 1,
                steps=self.param_curv_n_probe_points,
                device=y_points.device,
                dtype=torch.float64,
            ).round().long()
            y_probe = y_points[idx]
        else:
            y_probe = y_points

        model = self.model
        dtype = self.dtype
        device = self.device

        # 当前模型 patch 尺度
        da = self.param_curv_delta_alpha * model.a_half_range_local

        logw_min = math.log10(model.omega_min_local)
        logw_max = math.log10(model.omega_max_local)
        dlogw = 0.5 * (logw_max - logw_min) * self.param_curv_delta_xi

        # 当前点
        f0 = model(a_batch, omega_batch, y_probe)

        # ---------- a 方向 ----------
        a_plus = torch.clamp(a_batch + da, min=model.a_center_local - model.a_half_range_local,
                             max=model.a_center_local + model.a_half_range_local)
        a_minus = torch.clamp(a_batch - da, min=model.a_center_local - model.a_half_range_local,
                              max=model.a_center_local + model.a_half_range_local)

        f_a_plus = model(a_plus, omega_batch, y_probe)
        f_a_minus = model(a_minus, omega_batch, y_probe)

        curv_a = f_a_plus + f_a_minus - 2.0 * f0
        loss_curv_a = torch.mean(torch.abs(curv_a) ** 2)

        # ---------- omega 方向 ----------
        logw = torch.log10(torch.clamp(omega_batch, min=1.0e-12))
        logw_plus = torch.clamp(logw + dlogw, min=logw_min, max=logw_max)
        logw_minus = torch.clamp(logw - dlogw, min=logw_min, max=logw_max)

        omega_plus = torch.pow(
            torch.tensor(10.0, device=device, dtype=dtype),
            logw_plus
        )
        omega_minus = torch.pow(
            torch.tensor(10.0, device=device, dtype=dtype),
            logw_minus
        )

        f_w_plus = model(a_batch, omega_plus, y_probe)
        f_w_minus = model(a_batch, omega_minus, y_probe)

        curv_w = f_w_plus + f_w_minus - 2.0 * f0
        loss_curv_w = torch.mean(torch.abs(curv_w) ** 2)

        loss_total = self.param_curv_weight * (loss_curv_a + loss_curv_w)

        info = {
            "loss_param_curv": float(loss_total.detach().cpu().item()),
            "loss_param_curv_a": float(loss_curv_a.detach().cpu().item()),
            "loss_param_curv_omega": float(loss_curv_w.detach().cpu().item()),
        }
        return loss_total, info

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
        # ===== 当前 step 的动态权重 =====
        interior_weight = self._current_interior_weight(global_step)
        anchor_weight_sched = self._current_anchor_weight(global_step)

        loss_var = None
        info["loss_var"] = 0.0
        info["sigma_var"] = 0.0

        use_flat_var = (
            self.flat_var_enabled
           # and global_step <= self.flat_var_steps
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
                m = int(self.physics_cfg["problem"].get("m", 2))
            )

            info["loss_var"] = float(loss_var.detach().cpu().item())
            info["sigma_var"] = float(var_info["sigma_var"])

        var_weight = self._current_flat_var_weight(global_step)
        info["weight_var"] = float(var_weight)
        info["weight_interior"] = float(interior_weight)
        info["weight_anchor_sched"] = float(anchor_weight_sched)
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
            info["loss_seed"] = float(loss_seed.detach().cpu().item())

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
            info["loss_anchor"] = float(loss_anchor.detach().cpu().item())

        anchor_weight = anchor_weight_sched

        if loss_anchor is not None and self.dynamic_balance_enabled:
            dynamic_mult = self._update_dynamic_anchor_weight(
                loss_pde, loss_anchor, global_step
            )
            anchor_weight = anchor_weight_sched * dynamic_mult



        # =====================================================
        # 新增：结构相关正则
        # =====================================================
        loss_head_reg, head_reg_info = self._compute_head_regularization()
        info.update(head_reg_info)

        loss_param_curv, param_curv_info = self._compute_param_curvature_regularization(
            a_batch=a_batch,
            omega_batch=omega_batch,
            y_points=y_interior,
        )
        info.update(param_curv_info)


        total_loss = self._combine_losses(
            loss_pde,
            loss_seed=loss_seed,
            loss_anchor=loss_anchor,
            loss_var=loss_var,
            interior_weight=interior_weight,
            anchor_weight=anchor_weight,
            var_weight=var_weight,
        ) + loss_head_reg + loss_param_curv
        info["weight_anchor_dynamic"] = float(anchor_weight)
        
        self.optimizer.zero_grad()
        # ===== 记录乘权后的各项 loss，供进度条打印 =====
        if self.loss_balance_mode == "lbpinn":
            info["loss_interior_eff"] = float(
                (
                    interior_weight
                    * torch.exp(-self.log_sigma_interior)
                    * loss_pde
                ).detach().cpu().item()
            )

            if loss_anchor is not None:
                info["loss_anchor_eff"] = float(
                    (
                        anchor_weight
                        * torch.exp(-self.log_sigma_anchor)
                        * loss_anchor
                    ).detach().cpu().item()
                )
            else:
                info["loss_anchor_eff"] = 0.0

            if loss_var is not None:
                info["loss_var_eff"] = float((var_weight * loss_var).detach().cpu().item())
            else:
                info["loss_var_eff"] = 0.0
        else:
            info["loss_interior_eff"] = float((interior_weight * loss_pde).detach().cpu().item())

            if loss_anchor is not None:
                info["loss_anchor_eff"] = float((anchor_weight * loss_anchor).detach().cpu().item())
            else:
                info["loss_anchor_eff"] = 0.0

            if loss_var is not None:
                info["loss_var_eff"] = float((var_weight * loss_var).detach().cpu().item())
            else:
                info["loss_var_eff"] = 0.0
        total_loss.backward()
        grad_sq = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                grad_sq += p.grad.detach().pow(2).sum().item()
        grad_norm = grad_sq ** 0.5

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        info["grad_norm"] = grad_norm

        info["total_loss"] = float(total_loss.detach().cpu().item())
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
        左列：
            在 r ∈ [viz_r_min, viz_r_max] 上均匀采点，对比 R(r)

        右列：
            在对应区间的 y 上做 Chebyshev 安全采点，对比 R'(y)

        其中
            R' = g(x) * f(y) + h
            R  = U * R'
        """
        if not self.viz_enabled:
            return

        if self.viz_start_time is None:
            self.viz_start_time = datetime.now().strftime("%Y%m%d_%H%M%S")

        # ---- dtype / device ----
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
            ramp_t = ramp_val.detach().to(device=device, dtype=torch.complex128)
        else:
            ramp_t = torch.tensor(ramp_val, device=device, dtype=torch.complex128)

        rp = r_plus(a_t, M)

        # ---- 物理区间 ----
        r_min = max(self.viz_r_min, float(rp.detach().cpu().item()) + 1.0e-4)
        r_max = self.viz_r_max

        if r_min >= r_max:
            print(f"[viz] skip: r_min={r_min:.6f} >= r_max={r_max:.6f}")
            return

        # ============================================================
        # 第一列：R(r)，在 r 上均匀采点
        # ============================================================
        r_grid_uniform = torch.linspace(
            r_min, r_max, self.viz_num_points, device=device, dtype=dtype
        )
        x_grid_from_r = rp / r_grid_uniform
        y_grid_from_r = 2.0 * x_grid_from_r - 1.0

        # Mathematica 真值：直接在均匀 r 网格上求 R
        r_uniform_np = r_grid_uniform.detach().cpu().numpy()
        try:
            R_mma_r = get_mathematica_Rin(
                a_scalar,
                omega_scalar,
                l,
                m,
                s,
                r_uniform_np,
            )
            R_mma_r_t = torch.as_tensor(R_mma_r, device=device, dtype=torch.complex128)
        except Exception as e:
            print(f"[viz] Mathematica evaluation failed for R(r) at step {global_step}: {e}")
            self.model.train()
            return
       
        # ============================================================
        # 第二列：R'(y)，在 y 上 Chebyshev 采点
        # y 区间由 [r_min, r_max] 映射而来
        # ============================================================
        y_min = 2.0 * float(rp.detach().cpu().item()) / r_max - 1.0
        y_max = 2.0 * float(rp.detach().cpu().item()) / r_min - 1.0

        y_grid_cheb = sample_points_chebyshev_grid(
            n_points=self.viz_num_points,
            y_min=y_min,
            y_max=y_max,
            device=device,
            dtype=dtype,
        )

        x_grid_from_y = 0.5 * (y_grid_cheb + 1.0)
        r_grid_from_y = rp / x_grid_from_y

        # Mathematica 真值：先在 y-cheb 对应的 r 上求 R，再除以 U 得到 R'
        r_from_y_np = r_grid_from_y.detach().cpu().numpy()
        try:
            R_mma_y = get_mathematica_Rin(
                a_scalar,
                omega_scalar,
                l,
                m,
                s,
                r_from_y_np,
            )
            R_mma_y_t = torch.as_tensor(R_mma_y, device=device, dtype=torch.complex128)
        except Exception as e:
            print(f"[viz] Mathematica evaluation failed for R'(y) at step {global_step}: {e}")
            self.model.train()
            return

        # ============================================================
        # 网络前向
        # ============================================================
        self.model.eval()
        with torch.no_grad():
            h = h_factor(a_t, omega_t, m=m, M=M, s=s)

            # ---------- 左列：均匀 r -> 对应 y ----------
            f_pred_r = self.model(
                a_t.unsqueeze(0),
                omega_t.unsqueeze(0),
                y_grid_from_r,
            ).squeeze(0)

            g_val_r, _, _ = g_factor(x_grid_from_r)
            Rprime_pred_r = g_val_r * f_pred_r + h

            P_r, P_r_1, P_r_2 = Leaver_prefactors(
                r_grid_uniform, a_t, omega_t, m=m, M=M, s=s
            )
            Q_r, Q_r_1, Q_r_2 = prefactor_Q(
                r_grid_uniform,
                a_t,
                omega_t,
                p=int(p_val),
                R_amp=ramp_t,
                M=M,
                s=s,
            )
            U_r, _, _ = U_prefactor(P_r, P_r_1, P_r_2, Q_r, Q_r_1, Q_r_2)
            R_pred_r = U_r * Rprime_pred_r

            # ---------- 右列：Chebyshev y ----------
            f_pred_y = self.model(
                a_t.unsqueeze(0),
                omega_t.unsqueeze(0),
                y_grid_cheb,
            ).squeeze(0)

            g_val_y, _, _ = g_factor(x_grid_from_y)
            Rprime_pred_y = g_val_y * f_pred_y + h

            P_y, P_y_1, P_y_2 = Leaver_prefactors(
                r_grid_from_y, a_t, omega_t, m=m, M=M, s=s
            )
            Q_y, Q_y_1, Q_y_2 = prefactor_Q(
                r_grid_from_y,
                a_t,
                omega_t,
                p=int(p_val),
                R_amp=ramp_t,
                M=M,
                s=s,
            )
            U_y, _, _ = U_prefactor(P_y, P_y_1, P_y_2, Q_y, Q_y_1, Q_y_2)

            # Mathematica 的 R'(y)
            Rprime_from_mma_y = R_mma_y_t / U_y
        
       
        # ---- numpy for plotting ----
        r_uniform_np = r_grid_uniform.detach().cpu().numpy()
        y_cheb_np = y_grid_cheb.detach().cpu().numpy()

        R_pred_r_np = R_pred_r.detach().cpu().numpy()
        R_mma_r_np = np.asarray(R_mma_r)

        Rprime_pred_y_np = Rprime_pred_y.detach().cpu().numpy()
        Rprime_from_mma_y_np = Rprime_from_mma_y.detach().cpu().numpy()
        #查看Rprime_from_mmay_np最后几个值
        # print(f"[viz] Rprime_from_mma_y_np[-5:]: {Rprime_from_mma_y_np[-5:]}")
        # ============================================================
        # 画图
        # ============================================================
        plt.ion()
        fig, axes = plt.subplots(3, 2, figsize=(10, 10), sharex=False)

        # ---------------- 左列：R(r)，r均匀 ----------------
        axes[0, 0].plot(r_uniform_np, np.real(R_pred_r_np), label="Pred Re(R)", lw=1.8)
        axes[0, 0].plot(r_uniform_np, np.real(R_mma_r_np), "--", label="MMA Re(R)", lw=1.2)
        axes[0, 0].set_ylabel("Re(R)")
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)

        axes[1, 0].plot(r_uniform_np, np.imag(R_pred_r_np), label="Pred Im(R)", lw=1.8)
        axes[1, 0].plot(r_uniform_np, np.imag(R_mma_r_np), "--", label="MMA Im(R)", lw=1.2)
        axes[1, 0].set_ylabel("Im(R)")
        axes[1, 0].legend()
        axes[1, 0].grid(alpha=0.3)

        axes[2, 0].plot(r_uniform_np, np.abs(R_pred_r_np), label="Pred |R|", lw=1.8)
        axes[2, 0].plot(r_uniform_np, np.abs(R_mma_r_np), "--", label="MMA |R|", lw=1.2)
        axes[2, 0].set_ylabel("|R|")
        axes[2, 0].set_xlabel("r")
        axes[2, 0].legend()
        axes[2, 0].grid(alpha=0.3)

        # ---------------- 右列：R'(y)，y-Chebyshev ----------------
        axes[0, 1].plot(y_cheb_np, np.real(Rprime_pred_y_np), label="Pred Re(R')", lw=1.8)
        axes[0, 1].plot(y_cheb_np, np.real(Rprime_from_mma_y_np), "--", label="MMA Re(R')", lw=1.2)
        axes[0, 1].set_ylabel("Re(R')")
        axes[0, 1].set_xlim(-1, 1)
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)

        axes[1, 1].plot(y_cheb_np, np.imag(Rprime_pred_y_np), label="Pred Im(R')", lw=1.8)
        axes[1, 1].plot(y_cheb_np, np.imag(Rprime_from_mma_y_np), "--", label="MMA Im(R')", lw=1.2)
        axes[1, 1].set_ylabel("Im(R')")
        axes[1, 1].set_xlim(-1, 1)
        axes[1, 1].legend()
        axes[1, 1].grid(alpha=0.3)

        axes[2, 1].plot(y_cheb_np, np.abs(Rprime_pred_y_np), label="Pred |R'|", lw=1.8)
        axes[2, 1].plot(y_cheb_np, np.abs(Rprime_from_mma_y_np), "--", label="MMA |R'|", lw=1.2)
        axes[2, 1].set_ylabel("|R'|")
        axes[2, 1].set_xlabel("y")
        axes[2, 1].set_xlim(-1, 1)
        axes[2, 1].legend()
        axes[2, 1].grid(alpha=0.3)

        for i in range(3):
            axes[i, 0].set_xlabel("r")
            axes[i, 1].set_xlabel("y")

        fig.suptitle(
            f"step={global_step}, a={a_scalar:.6f}, omega={omega_scalar:.6f}",
            fontsize=12
        )
        fig.tight_layout()

        # ---- 保存图片 ----
        if self.viz_save_enabled:
            filename = (
                f"step_{global_step:06d}"
                f"_a_{a_scalar:.6f}"
                f"_omega_{omega_scalar:.6f}.png"
            )
            save_dir = self.viz_dir / self.viz_start_time
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / filename
            fig.savefig(save_path, dpi=160, bbox_inches="tight")
            print(f"[viz] saved figure to {save_path}")

        # ---- 非阻塞显示 + 自动关闭 ----
        if self.viz_show_enabled:
            plt.show(block=False)
            plt.pause(self.viz_auto_close_sec)

        plt.close(fig)
        plt.close("all")
        self.model.train()


    def _validate_single_case(
        self,
        a_val: torch.Tensor,
        omega_val: torch.Tensor,
        lambda_val: torch.Tensor | None = None,
        ramp_val: torch.Tensor | None = None,
        p_val: int | None = None,
    ):
        """在单个参数点上做 residual 验证。"""
        was_training = self.model.training
        self.model.eval()
        try:
            M = float(self.physics_cfg["problem"].get("M", 1.0))

            rp = r_plus(a_val, M)
            r_min = float(rp) + 0.1
            r_max = 100.0

            y_min = 2.0 * float(rp) / r_max - 1.0
            y_max = 2.0 * float(rp) / r_min - 1.0

            y_grid = sample_points_chebyshev_grid(
                n_points=self.val_n_points,
                y_min=y_min,
                y_max=y_max,
                device=self.device,
                dtype=self.dtype,
            )
            y_grid = y_grid.clone().requires_grad_(True)

            a_batch = a_val.unsqueeze(0)
            omega_batch = omega_val.unsqueeze(0)

            if lambda_val is None or ramp_val is None or p_val is None:
                lam = get_lambda_from_cfg(self.physics_cfg, self.cache, a_val, omega_val)
                p, ramp = get_ramp_and_p_from_cfg(self.physics_cfg, self.cache, a_val, omega_val)
            else:
                lam = lambda_val
                ramp = ramp_val
                p = p_val

            lambda_batch = lam.unsqueeze(0).to(device=self.device, dtype=torch.complex128)
            ramp_batch = ramp.unsqueeze(0).to(device=self.device, dtype=torch.complex128)

            loss, _ = pinn_residual_loss(
                self.model, self.physics_cfg, a_batch, omega_batch,
                lambda_batch, ramp_batch, int(p or 5),
                y_grid, torch.tensor([], device=self.device, dtype=self.dtype),
                weight_interior=1.0, weight_boundary=0.0
            )

            return float(loss.detach().cpu().item())
        finally:
            if was_training:
                self.model.train()

    def validate(self, a_val=None, omega_val=None):
        """
        若传入 a_val / omega_val，则验证单点；
        否则在固定验证池上做全参数空间验证并返回聚合指标。
        """
        if a_val is not None and omega_val is not None:
            return self._validate_single_case(a_val=a_val, omega_val=omega_val)

        if self._validation_pool is None:
            self._validation_pool = self._build_validation_pool()

        pool = self._validation_pool
        losses = []
        worst_idx = -1
        worst_loss = -float("inf")

        for i in range(pool["a_all"].shape[0]):
            loss_i = self._validate_single_case(
                a_val=pool["a_all"][i],
                omega_val=pool["omega_all"][i],
                lambda_val=pool["lambda_all"][i],
                ramp_val=pool["ramp_all"][i],
                p_val=pool["p"],
            )
            losses.append(loss_i)
            if loss_i > worst_loss:
                worst_loss = loss_i
                worst_idx = i

        val_mean = float(np.mean(losses)) if losses else float("inf")
        metrics = {
            "val_mean": val_mean,
            "val_worst": float(worst_loss if losses else float("inf")),
            "worst_a": float(pool["a_all"][worst_idx].detach().cpu().item()) if worst_idx >= 0 else None,
            "worst_omega": float(pool["omega_all"][worst_idx].detach().cpu().item()) if worst_idx >= 0 else None,
            "n_val_samples": int(len(losses)),
        }
        self.latest_val_metrics = metrics
        self.val_loss_history.append(metrics["val_mean"])
        self.val_worst_history.append(metrics["val_worst"])
        return metrics

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
        self.best_val_loss = float('inf')
        no_improve_count = 0
        global_step = 0
        self._validation_pool = self._build_validation_pool()

        def monitored_val(metrics):
            if self.val_monitor == "worst":
                return metrics["val_worst"]
            return metrics["val_mean"]

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
                'val_worst_history': self.val_worst_history,
                'best_val_loss': self.best_val_loss,
                'latest_val_metrics': self.latest_val_metrics,
                'global_step': global_step,
            }, path)

        def finalize_training(reason):
            final_model_path = os.path.join(save_dir, "final_model.pt")
            save_checkpoint(final_model_path)
            print(f"[info] training finished ({reason})")
            print(f"[info] final model saved to: {final_model_path}")
            self.plot_loss_curve(save_dir)

        # fixed_pool 模式只初始化一次
        fixed_pool = None
        if self.param_sampling_mode == "fixed_pool":
            fixed_pool = self._build_fixed_param_pool()
            print(f"[info] Built fixed parameter pool with {fixed_pool['a_all'].shape[0]} samples")
        print(f"[info] Built validation pool with {self._validation_pool['a_all'].shape[0]} samples")
        self._plot_parameter_domain(save_dir=save_dir, fixed_pool=fixed_pool)

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
                    epoch_losses.append(float(loss.detach().cpu().item()))

                    self.loss_history.append(info["total_loss"])
                    self.step_history.append(global_step)
                    
                    # ---- 周期性在线可视化 ----
                    if self.viz_enabled and global_step > 0 and (global_step % self.viz_every_steps == 0 or global_step ==1):
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
                        "int": f"{info.get('loss_interior_eff', 0.0):.3e}",
                        # "bd": f"{info.get('loss_boundary', 0.0):.3e}",
                        "anchor": f"{info.get('loss_anchor_eff', 0.0):.3e}",
                        "grad": f"{info.get('grad_norm', 0.0):.3e}",
                        # "var": f"{info.get('loss_var_eff', 0.0):.3e}",
                        # "w_int": f"{info.get('weight_interior', 0.0):.2e}",
                        # "w_anch": f"{info.get('weight_anchor_sched', 0.0):.2e}",
                        # "w_var": f"{info.get('weight_var', 0.0):.2e}",
                        "head": f"{info.get('loss_head_reg', 0.0):.3e}",
                        "curv": f"{info.get('loss_param_curv', 0.0):.3e}",
                        "grad": f"{info.get('grad_norm', 0.0):.3e}",
                    })

                    # 验证
                    if global_step % self.val_freq == 0:
                        val_metrics = self.validate()
                        val_metric = monitored_val(val_metrics)
                        pbar.set_postfix({
                            "total": f"{info['total_loss']:.3e}",
                            "int": f"{info.get('loss_interior_eff', 0.0):.3e}",
                            "bd": f"{info.get('loss_boundary', 0.0):.3e}",
                            "anchor": f"{info.get('loss_anchor_eff', 0.0):.3e}",
                            "var": f"{info.get('loss_var_eff', 0.0):.3e}",
                            "val": f"{val_metrics['val_mean']:.3e}",
                            "worst": f"{val_metrics['val_worst']:.3e}",
                        })

                        if val_metric < self.best_val_loss - self.early_stop_threshold:
                            self.best_val_loss = val_metric
                            no_improve_count = 0

                            model_path = os.path.join(save_dir, "best_model.pt")
                            save_checkpoint(model_path)
                        else:
                            no_improve_count += 1

                        if no_improve_count >= self.early_stop_patience:
                            print("\n[info] early stopping triggered")
                            finalize_training(reason="early_stop")
                            return

                mean_epoch_loss = sum(epoch_losses) / len(epoch_losses)
                # tqdm.write(f"[epoch {epoch+1}] mean_train_loss = {mean_epoch_loss:.6e}")

        finalize_training(reason="max_epochs")

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
