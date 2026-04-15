from __future__ import annotations

from pathlib import Path
import random
from typing import Any

import numpy as np
import torch

from config.config_loader import load_pinn_full_config
from model.pinn_mlp import PINN_MLP
from dataset.sampling import sample_points_chebyshev_grid
from physical_ansatz.residual import AuxCache, get_lambda_from_cfg, get_ramp_and_p_from_cfg
from physical_ansatz.residual_pinn import pinn_residual_loss
from domain.patch_cover import load_patch_cover, load_valid_chart_points
from physical_ansatz.residual_pinn import pinn_residual_loss, compute_data_anchor_loss
from physical_ansatz.mapping import r_plus, r_from_x
from mma.rin_sampler import MathematicaRinSampler

def _get_dtype(dtype_name: str):
    if dtype_name == "float32":
        return torch.float32
    return torch.float64


class AtlasPatchTrainer:
    """
    最小版 atlas patch trainer：

    - 只训练一个 patch
    - 参数点来自 patch 覆盖到的 safe 点
    - PDE 系数仍然用 raw (a, omega)
    - 模型 local coords 用 (u, v)
    - 先只跑 interior residual，不加 anchor / overlap
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
        anchor_weight: float = 1.0,
        n_anchor_y: int = 16,
        mma_interp_n_grid: int = 1200,
    ):
        self.device = torch.device(device)

        full_cfg = load_pinn_full_config(cfg_path)
        self.full_cfg = full_cfg
        self.physics_cfg = full_cfg["physics"]
        self.cfg = full_cfg["train"]
        self.train_cfg = self.cfg

        runtime_cfg = self.cfg.get("runtime", {})
        self.dtype = _get_dtype(runtime_cfg.get("dtype", "float64"))

        # ---------------------------------------------------------
        # 读取 model config（兼容当前 test2 结构）
        # ---------------------------------------------------------
        model_cfg = self.cfg.get("model", {})
        self.model_cfg = model_cfg

        # ---------------------------------------------------------
        # 读取训练相关 config（兼容旧风格/当前风格）
        # ---------------------------------------------------------
        sampling_cfg = self.cfg.get("sampling", {})
        if "collocation" in sampling_cfg:
            self.n_interior = sampling_cfg["collocation"]["interior"].get("n_points", 64)
            self.batch_size = sampling_cfg["parameter_batch"].get("batch_size", 4)
            adaptive_cfg = sampling_cfg["collocation"].get("adaptive", {})
            self.normalize_residual = adaptive_cfg.get("normalize_residual", False)
        else:
            self.n_interior = sampling_cfg.get("n_interior", 64)
            self.batch_size = sampling_cfg.get("batch_size", 4)

            adaptive_cfg = self.cfg.get("adaptive_sampling", {})
            self.normalize_residual = adaptive_cfg.get("normalize_residual", False)

        train_block = self.cfg.get("training", {})
        if "optimizer" in train_block:
            self.lr = train_block["optimizer"].get("lr", 1.0e-3)
        else:
            self.lr = self.cfg.get("train", {}).get("lr", 1.0e-3)

        self.grad_clip = 1.0

        # ---------------------------------------------------------
        # 读取 patch cover
        # ---------------------------------------------------------
        self.patch_cover = load_patch_cover(patch_json)
        patches = [p for p in self.patch_cover.patches if p.patch_id == patch_id]
        if len(patches) != 1:
            raise ValueError(f"patch_id={patch_id} not found in {patch_json}")
        self.patch = patches[0]

        # ---------------------------------------------------------
        # 读取所有 safe 点，并筛出属于当前 patch 的点
        # ---------------------------------------------------------
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
        # 建立模型：关键是 local_coord_mode='chart_uv'
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

            # 旧模式参数仍然保留，但在 chart_uv 下不会用到
            a_center_local=model_cfg.get("a_center_local", 0.125),
            a_half_range_local=model_cfg.get("a_half_range_local", 0.075),
            omega_min_local=model_cfg.get("omega_min_local", 0.1),
            omega_max_local=model_cfg.get("omega_max_local", 1.0),

            # 真正使用的局部 chart patch 参数
            u_center_local=self.patch.u_center,
            v_center_local=self.patch.v_center,
            u_half_range_local=self.patch.h_u,
            v_half_range_local=self.patch.h_v,

            M=M,
            m_mode=m_mode,
        ).to(device=self.device, dtype=self.dtype)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.cache = AuxCache()
        # ---------------------------------------------------------
        # Mathematica anchor
        # ---------------------------------------------------------
        self.anchor_enabled = bool(anchor_enabled)
        self.anchor_weight = float(anchor_weight)
        self.n_anchor_y = int(n_anchor_y)
        self.mma_interp_n_grid = int(mma_interp_n_grid)

        mma_cfg = self.cfg.get("mathematica", {})
        self.mma_kernel_path = mma_cfg.get("kernel_path", None)
        self.mma_wl_path_win = mma_cfg.get("wl_path_win", None)

        if self.anchor_enabled:
            if self.mma_kernel_path is None or self.mma_wl_path_win is None:
                raise ValueError(
                    "anchor_enabled=True requires cfg['mathematica']['kernel_path'] "
                    "and cfg['mathematica']['wl_path_win']"
                )
            self.mma_sampler = MathematicaRinSampler(
                kernel_path=self.mma_kernel_path,
                wl_path_win=self.mma_wl_path_win,
            )
        else:
            self.mma_sampler = None

    # ---------------------------------------------------------
    # patch 参数池采样
    # ---------------------------------------------------------
    def sample_param_batch(self):
        n_pool = len(self.patch_aw)
        replace = n_pool < self.batch_size
        idx = np.random.choice(n_pool, size=self.batch_size, replace=replace)

        aw = self.patch_aw[idx]
        uv = self.patch_uv[idx]

        a_batch = torch.tensor(aw[:, 0], device=self.device, dtype=self.dtype)
        omega_batch = torch.tensor(aw[:, 1], device=self.device, dtype=self.dtype)
        u_batch = torch.tensor(uv[:, 0], device=self.device, dtype=self.dtype)
        v_batch = torch.tensor(uv[:, 1], device=self.device, dtype=self.dtype)

        return a_batch, omega_batch, u_batch, v_batch

    # ---------------------------------------------------------
    # 当前最小版本：y 仍然先用统一 Chebyshev 点
    # 这里只是验证 atlas patch 通路，不在这一步解决全域 r_max 统一问题
    # ---------------------------------------------------------
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

    def query_mma_Rin_batch(self, a_batch, omega_batch, y_anchors):
        """
        对 batch 内每个 (a, omega)，在 y_anchors 对应的 r 点上查询 Mathematica 的 R_in。
        返回:
            R_mma_anchors: (B, N_anchor) complex128 torch tensor
        """
        problem_cfg = self.physics_cfg["problem"]
        M = float(problem_cfg.get("M", 1.0))
        s = int(problem_cfg.get("s", -2))
        l = int(problem_cfg.get("l", 2))
        m = int(problem_cfg.get("m", 2))

        x_anchors = 0.5 * (y_anchors + 1.0)

        rows = []
        for i in range(a_batch.shape[0]):
            a_i = float(a_batch[i].detach().cpu().item())
            omega_i = float(omega_batch[i].detach().cpu().item())

            rp_i = r_plus(a_batch[i], M)
            r_i = r_from_x(x_anchors, rp_i).detach().cpu().numpy()

            Rin_i = self.mma_sampler.interpolate_rin(
                s=s,
                l=l,
                m=m,
                a=a_i,
                omega=omega_i,
                r_query=r_i,
                n_grid=self.mma_interp_n_grid,
            )
            rows.append(Rin_i)

        arr = np.stack(rows, axis=0)   # (B, N_anchor)
        return torch.as_tensor(arr, device=self.device, dtype=torch.complex128)
    # ---------------------------------------------------------
    # 解析 lambda / ramp
    # ---------------------------------------------------------
    def resolve_aux_batch(self, a_batch, omega_batch):
        lambda_list = []
        ramp_list = []
        p_ref = None

        for i in range(a_batch.shape[0]):
            lam_i = get_lambda_from_cfg(
                self.physics_cfg,
                self.cache,
                a_batch[i],
                omega_batch[i],
            )
            p_i, ramp_i = get_ramp_and_p_from_cfg(
                self.physics_cfg,
                self.cache,
                a_batch[i],
                omega_batch[i],
            )

            if p_ref is None:
                p_ref = int(p_i)
            elif int(p_i) != p_ref:
                raise RuntimeError(
                    f"Inconsistent p across batch: got {p_ref} and {p_i}"
                )

            lambda_list.append(lam_i)
            ramp_list.append(ramp_i)

        lambda_batch = torch.stack(lambda_list, dim=0)
        ramp_batch = torch.stack(ramp_list, dim=0)
        return lambda_batch, ramp_batch, p_ref

    # ---------------------------------------------------------
    # 单步训练
    # ---------------------------------------------------------
    def train_one_step(self):
        self.model.train()
        self.optimizer.zero_grad()

        a_batch, omega_batch, u_batch, v_batch = self.sample_param_batch()
        lambda_batch, ramp_batch, p = self.resolve_aux_batch(a_batch, omega_batch)

        y_interior = self.sample_y_interior()
        y_boundary = torch.empty(0, device=self.device, dtype=self.dtype)

        loss_pde, info_pde = pinn_residual_loss(
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
            residual_scale_eps=1.0e-12,
            return_pointwise=False,
            u_batch=u_batch,
            v_batch=v_batch,
        )

        total_loss = loss_pde
        info = dict(info_pde)
        info["loss_anchor"] = 0.0

        if self.anchor_enabled:
            y_anchor = self.sample_y_anchor()
            R_mma_anchors = self.query_mma_Rin_batch(
                a_batch=a_batch,
                omega_batch=omega_batch,
                y_anchors=y_anchor,
            )

            loss_anchor = compute_data_anchor_loss(
                model=self.model,
                cfg=self.physics_cfg,
                a_batch=a_batch,
                omega_batch=omega_batch,
                y_anchors=y_anchor,
                R_mma_anchors=R_mma_anchors,
                relative=False,
                eps=1.0e-12,
                u_batch=u_batch,
                v_batch=v_batch,
            )

            total_loss = loss_pde + self.anchor_weight * loss_anchor
            info["loss_anchor"] = float(loss_anchor.detach().cpu().item())

        total_loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optimizer.step()

        info["total_loss"] = float(total_loss.detach().cpu().item())
        info["grad_norm"] = float(grad_norm.detach().cpu().item())
        info["patch_id"] = int(self.patch.patch_id)
        info["patch_component_id"] = int(self.patch.component_id)
        info["patch_pool_size"] = int(len(self.patch_aw))
        return info