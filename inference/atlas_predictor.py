from __future__ import annotations

import json
from pathlib import Path

import torch

from config.config_loader import load_pinn_full_config
from domain.atlas_builder import load_atlas, map_to_chart
from domain.patch_cover import load_patch_cover
from model.pinn_mlp import PINN_MLP


def _get_dtype(dtype_name: str):
    if dtype_name == "float32":
        return torch.float32
    return torch.float64


class AtlasPredictor:
    """
    使用 atlas registry 做完整多 patch 推理。
    当前策略：
      - 找出覆盖当前 (u,v) 的所有 patch
      - 每个 patch 单独预测
      - 用 tent weight 做平滑 blending
    """

    def __init__(
        self,
        registry_json: str,
        device: str = "cpu",
    ):
        self.registry_json = Path(registry_json).resolve()
        with open(self.registry_json, "r", encoding="utf-8") as f:
            self.registry = json.load(f)

        self.cfg_path = self.registry["cfg_path"]
        self.full_cfg = load_pinn_full_config(self.cfg_path)
        self.train_cfg = self.full_cfg["train"]
        self.physics_cfg = self.full_cfg["physics"]

        runtime_cfg = self.train_cfg.get("runtime", {})
        self.dtype = _get_dtype(runtime_cfg.get("dtype", "float64"))
        self.device = torch.device(device)

        self.patch_cover = load_patch_cover(self.registry["patch_json"])
        self.atlas = load_atlas(self.registry["atlas_json"])
        self.component = self.atlas.components[self.patch_cover.component_id]

        self.model_cfg = self.train_cfg.get("model", {})
        self.problem_cfg = self.physics_cfg.get("problem", {})
        self.M = float(self.problem_cfg.get("M", 1.0))
        self.m_mode = int(self.problem_cfg.get("m", 2))

        # lazy load
        self._models = {}

    def _find_patch_record(self, patch_id: int):
        for rec in self.registry["patch_records"]:
            if int(rec["patch_id"]) == int(patch_id):
                return rec
        raise KeyError(f"patch_id={patch_id} not found in registry")

    def _build_model_for_patch(self, patch):
        model = PINN_MLP(
            hidden_dims=self.model_cfg.get("hidden_dims", [128, 128, 128, 128]),
            activation=self.model_cfg.get("activation", "silu"),
            fourier_num_freqs=self.model_cfg.get("fourier_num_freqs", 2),
            fourier_scale=self.model_cfg.get("fourier_scale", 1.0),
            param_embed_dim=self.model_cfg.get("param_embed_dim", 64),
            use_film=self.model_cfg.get("use_film", True),
            use_residual=self.model_cfg.get("use_residual", True),

            local_coord_mode="chart_uv",

            a_center_local=self.model_cfg.get("a_center_local", 0.125),
            a_half_range_local=self.model_cfg.get("a_half_range_local", 0.075),
            omega_min_local=self.model_cfg.get("omega_min_local", 1.0e-4),
            omega_max_local=self.model_cfg.get("omega_max_local", 10.0),

            u_center_local=float(patch.u_center),
            v_center_local=float(patch.v_center),
            u_half_range_local=float(patch.h_u),
            v_half_range_local=float(patch.h_v),

            M=self.M,
            m_mode=self.m_mode,
        ).to(device=self.device, dtype=self.dtype)

        record = self._find_patch_record(int(patch.patch_id))
        ckpt_path = Path(record["best_model_path"])
        ckpt = torch.load(ckpt_path, map_location=self.device)
        state_dict = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        return model

    def _get_model(self, patch):
        pid = int(patch.patch_id)
        if pid not in self._models:
            self._models[pid] = self._build_model_for_patch(patch)
        return self._models[pid]

    def _candidate_patches(self, u: float, v: float):
        out = []
        for patch in self.patch_cover.patches:
            du = abs(u - float(patch.u_center))
            dv = abs(v - float(patch.v_center))
            if du <= float(patch.h_u) and dv <= float(patch.h_v):
                out.append(patch)
        return out

    def _tent_weight(self, patch, u: float, v: float) -> float:
        du = abs(u - float(patch.u_center)) / float(patch.h_u)
        dv = abs(v - float(patch.v_center)) / float(patch.h_v)
        wu = max(0.0, 1.0 - du)
        wv = max(0.0, 1.0 - dv)
        return wu * wv + 1.0e-12

    def predict_f(
        self,
        a: float,
        omega: float,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """
        返回 blended complex f(y; a, omega), shape=(N,)
        """
        u, v = map_to_chart(self.component, float(a), float(omega))
        patches = self._candidate_patches(u, v)
        if len(patches) == 0:
            raise RuntimeError(f"No patch covers point (a,omega)=({a},{omega}), mapped (u,v)=({u},{v})")

        a_t = torch.tensor([float(a)], device=self.device, dtype=self.dtype)
        omega_t = torch.tensor([float(omega)], device=self.device, dtype=self.dtype)
        u_t = torch.tensor([float(u)], device=self.device, dtype=self.dtype)
        v_t = torch.tensor([float(v)], device=self.device, dtype=self.dtype)
        y = y.to(device=self.device, dtype=self.dtype)

        preds = []
        weights = []
        with torch.no_grad():
            for patch in patches:
                model = self._get_model(patch)
                f_i = model(a_t, omega_t, y, u=u_t, v=v_t).squeeze(0)
                w_i = self._tent_weight(patch, float(u), float(v))
                preds.append(f_i)
                weights.append(w_i)

        wsum = sum(weights)
        out = None
        for f_i, w_i in zip(preds, weights):
            if out is None:
                out = (w_i / wsum) * f_i
            else:
                out = out + (w_i / wsum) * f_i
        return out
