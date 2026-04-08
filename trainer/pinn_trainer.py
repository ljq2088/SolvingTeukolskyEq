"""
PINN Trainer
"""
import os
import torch
import yaml
from tqdm import tqdm
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config_loader import load_pinn_full_config
from model.pinn_mlp import PINN_MLP
from physical_ansatz.residual_pinn import pinn_residual_loss, compute_data_anchor_loss
from physical_ansatz.residual import AuxCache, get_lambda_from_cfg, get_ramp_and_p_from_cfg
from physical_ansatz.mapping import r_plus, r_from_x
from dataset.sampling import sample_points_luna_style, sample_parameters
from dataset.sampling import (
    sample_points_luna_style,
    sample_parameters,
    sample_parameters_sobol,
    build_candidate_pool_1d,
    sample_points_rard,
    get_sentinel_anchor_points,
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


        adaptive_cfg = self.train_cfg.get("adaptive_sampling", {})
        anchor_cfg = self.train_cfg.get("anchors", {})
        loss_balance_cfg = self.train_cfg.get("loss_balance", {})
        curriculum_cfg = self.train_cfg.get("curriculum", {})

        self.param_sampler = sampling_cfg.get("param_sampler", "sobol")

        

        # dtype
        dtype_name = runtime_cfg.get("dtype", "float64")
        self.dtype = torch.float64 if dtype_name == "float64" else torch.float32

        # 参数范围
        self.a_center = param_cfg.get("a_center", 0.1)
        self.a_range = param_cfg.get("a_range", 0.01)
        self.omega_center = param_cfg.get("omega_center", 0.1)
        self.omega_range = param_cfg.get("omega_range", 0.01)

        # 采样配置
        self.n_interior = sampling_cfg.get("n_interior", 100)
        self.n_boundary = sampling_cfg.get("n_boundary", 20)
        self.batch_size = sampling_cfg.get("batch_size", 4)
        self.use_batch_gd = sampling_cfg.get("use_batch_gd", False)
        self.n_param_samples = sampling_cfg.get("n_param_samples", 20)
        self.n_epochs = sampling_cfg.get("n_epochs", 100)

        # 训练配置
        self.n_steps = train_cfg.get("n_steps", 50000)
        self.lr = train_cfg.get("lr", 1e-3)
        self.anchor_freq = train_cfg.get("anchor_freq", 100)
        self.n_anchors = train_cfg.get("n_anchors", 10)

        # loss 权重
        self.weight_interior = loss_cfg.get("weight_interior", 1.0)
        self.weight_boundary = loss_cfg.get("weight_boundary", 10.0)
        self.weight_anchor = loss_cfg.get("weight_anchor", 1.0)

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


    def _sample_training_points(self, batch, global_step):
        if not self.adaptive_sampling_enabled:
            return sample_points_luna_style(
                n_interior=self.n_interior,
                n_boundary=self.n_boundary,
                device=self.device,
                dtype=self.dtype,
            ), None

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
            n_select=self.n_interior,
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


    def _combine_losses(self, loss_pde, loss_anchor=None):
        if self.loss_balance_mode == "lbpinn":
            total = torch.exp(-self.log_sigma_interior) * loss_pde + self.log_sigma_interior
            if loss_anchor is not None:
                total = total + self.weight_anchor * (
                    torch.exp(-self.log_sigma_anchor) * loss_anchor + self.log_sigma_anchor
                )
            return total

        total = self.weight_interior * loss_pde
        if loss_anchor is not None:
            total = total + self.weight_anchor * loss_anchor
        return total


    def _run_one_training_batch(self, batch, global_step):
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

        loss_anchor = None
        info["loss_anchor"] = 0.0

        if global_step % self.anchor_freq == 0:
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

        total_loss = self._combine_losses(loss_pde, loss_anchor)

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        info["total_loss"] = float(total_loss.item())
        if self.loss_balance_mode == "lbpinn":
            info["w_int"] = float(torch.exp(-self.log_sigma_interior).item())
            info["w_anchor"] = float(torch.exp(-self.log_sigma_anchor).item())

        return total_loss, info


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

                    # 更新进度条
                    pbar.update(1)
                    pbar.set_postfix({
                        "total": f"{info['total_loss']:.3e}",
                        "int": f"{info.get('loss_interior', 0.0):.3e}",
                        "bd": f"{info.get('loss_boundary', 0.0):.3e}",
                        "anchor": f"{info.get('loss_anchor', 0.0):.3e}",
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
