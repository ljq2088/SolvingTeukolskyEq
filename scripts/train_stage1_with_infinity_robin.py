#!/usr/bin/env python3
"""Stage-1 refinement: Adam fine-tune + L-BFGS with exact infinity Robin loss.

Usage:
  # Preflight: diagnose current Robin residual
  python scripts/diagnose_infinity_robin_stage1.py --device cuda

  # Debug: verify all checks pass
  python scripts/debug_stage1_infinity_robin.py --config config/autoencoder_stage1_infinity_robin.yaml --device cuda

  # Adam smoke (20 epochs)
  python scripts/train_stage1_with_infinity_robin.py --config config/autoencoder_stage1_infinity_robin.yaml --device cuda --phase adam --epochs 20 --verbose

  # Adam full (50 epochs)
  python scripts/train_stage1_with_infinity_robin.py --config config/autoencoder_stage1_infinity_robin.yaml --device cuda --phase adam --epochs 50 --resume-checkpoint <adam_smoke_best> --verbose

  # L-BFGS smoke (5 epochs)
  python scripts/train_stage1_with_infinity_robin.py --config config/autoencoder_stage1_infinity_robin.yaml --device cuda --phase lbfgs --epochs 5 --resume-checkpoint <adam_best> --verbose

  # L-BFGS full (50 epochs)
  python scripts/train_stage1_with_infinity_robin.py --config config/autoencoder_stage1_infinity_robin.yaml --device cuda --phase lbfgs --epochs 50 --resume-checkpoint <lbfgs_smoke_best> --verbose
"""
import argparse
import copy
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.config_loader import load_pinn_full_config
from model.autoencoder_pinn import AutoencoderTeukolskyPINN
from physical_ansatz.infinity_robin import (
    analytic_c_inf,
    infinity_robin_loss,
    compute_S_and_Sy_at_infinity,
)
from physical_ansatz.stage1_endpoint import compute_stage1_S_and_Sy_at_infinity
from physical_ansatz.transform_y import (
    compose_reduced_shape_from_f,
    horizon_regularity_slope,
)
from physical_ansatz.residual_pinn import compute_f_derivatives_autograd
from physical_ansatz.teukolsky_coeffs import coeffs_x
from physical_ansatz.transform_y import transform_coeffs_x_to_y_S
from utils.compute_lambda_usage import compute_lambda


# ============================================================
# RefinementTrainer
# ============================================================
class RefinementTrainer:
    def __init__(self, config_path, device="cuda", resume_checkpoint=None, verbose=False):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.verbose = verbose
        self.dtype = torch.float64
        self.cdtype = torch.complex128

        full_cfg = load_pinn_full_config(config_path)
        self.cfg = full_cfg.get("train", full_cfg).get("stage1_infinity_robin", full_cfg.get("stage1_infinity_robin", {}))
        self.physics_cfg = full_cfg

        prob = full_cfg.get("physics", full_cfg).get("problem", full_cfg.get("problem", {}))
        self.M = float(prob.get("M", 1.0))
        self.ell = int(prob.get("l", 2))
        self.m_mode = int(prob.get("m", 2))
        self.s = int(prob.get("s", -2))

        self.base_checkpoint = self.cfg.get("base_checkpoint", "")
        if not self.base_checkpoint:
            import glob as _glob
            candidates = sorted(_glob.glob("outputs/autoencoder_stage1_rin_train/*/checkpoints/best_model.pt"))
            if candidates:
                self.base_checkpoint = candidates[-1]
                print(f"Auto-detected checkpoint: {self.base_checkpoint}")
            else:
                raise FileNotFoundError("No base_checkpoint in config and no checkpoint auto-detected")

        # Load model
        if resume_checkpoint:
            ckpt_path = resume_checkpoint
        else:
            ckpt_path = self.base_checkpoint
        print(f"Loading checkpoint: {ckpt_path}")
        self.model = self._load_model(ckpt_path)
        self.model.to(self.device)
        self.model.eval()

        # Frozen copy for drift loss
        self.frozen_model = copy.deepcopy(self.model)
        self.frozen_model.to(self.device)
        for p in self.frozen_model.parameters():
            p.requires_grad_(False)
        self.frozen_model.eval()

        # Apply freeze policy
        self._apply_freeze()

        # Output dir
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        phase_tag = "adam" if self.cfg.get("adam", {}).get("enabled", True) else "lbfgs"
        self.run_dir = Path(self.cfg.get("output_root", "outputs/stage1_infinity_robin")) / f"{timestamp}_{phase_tag}_refine"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "checkpoints").mkdir(exist_ok=True)
        (self.run_dir / "logs").mkdir(exist_ok=True)
        print(f"Run dir: {self.run_dir}")

        # Save config snapshot
        with open(self.run_dir / "config_snapshot.yaml", "w") as f:
            yaml.safe_dump(full_cfg, f)

        # Metrics log
        self.history_path = self.run_dir / "logs" / "history.jsonl"
        self.best_loss = float("inf")
        self.best_step = 0
        self.global_step = 0

        self._report_params()

    def _load_model(self, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model = AutoencoderTeukolskyPINN()
        sd = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
        model.load_state_dict(sd)
        return model

    def _apply_freeze(self):
        train_enc = self.cfg.get("train_encoder", True)
        train_rin = self.cfg.get("train_rin_decoder", True)
        train_amp = self.cfg.get("train_amplitude_net", False)
        train_up = self.cfg.get("train_up_decoder", False)
        train_down = self.cfg.get("train_down_decoder", False)

        for n, p in self.model.named_parameters():
            if n.startswith("encoder"):
                p.requires_grad_(train_enc)
            elif n.startswith("rin_decoder"):
                p.requires_grad_(train_rin)
            elif n.startswith("amplitude_net"):
                p.requires_grad_(train_amp)
            elif n.startswith("up_decoder"):
                p.requires_grad_(train_up)
            elif n.startswith("down_decoder"):
                p.requires_grad_(train_down)

    def _report_params(self):
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total params: {total}, Trainable: {trainable}")

    # ---- Collocation ----
    def build_y_grid(self, device=None, dtype=None):
        """Build fixed y-grid from config sampling regions."""
        if device is None:
            device = self.device
        if dtype is None:
            dtype = self.dtype
        sampling = self.cfg.get("sampling", {})
        pieces = []
        for region in ["near_infinity", "transition", "horizon_side"]:
            rcfg = sampling.get(region, {})
            y_lo = rcfg.get("y_min")
            y_hi = rcfg.get("y_max")
            n_y = rcfg.get("n_y", 32)
            if y_lo is not None and y_hi is not None and n_y > 0:
                y_piece = torch.linspace(y_lo, y_hi, n_y, device=device, dtype=dtype)
                pieces.append(y_piece)
        y_all = torch.cat(pieces)
        y_all = torch.unique(y_all, sorted=True)
        return y_all

    def build_random_y_batch(self, B, device=None, dtype=None):
        """Random collocation for Adam phase."""
        if device is None:
            device = self.device
        if dtype is None:
            dtype = self.dtype
        sampling = self.cfg.get("sampling", {})
        pieces = []
        n_total = 0
        for region in ["near_infinity", "transition", "horizon_side"]:
            rcfg = sampling.get(region, {})
            y_lo = rcfg.get("y_min")
            y_hi = rcfg.get("y_max")
            n_y = rcfg.get("n_y", 32)
            if y_lo is not None and y_hi is not None and n_y > 0:
                n_total += n_y
        y_all = torch.rand(B, n_total, device=device, dtype=dtype)
        # Scale to [-1, 0.999]
        y_all = y_all * 1.999 - 1.0
        return y_all

    # ---- Autograd shape workaround ----
    @staticmethod
    def _fix_autograd_shapes(fy, fyy):
        """Fix (B, B, N) -> (B, N) diagonal extraction from autograd bug."""
        if fy is not None and fy.ndim > 2 and fy.shape[0] == fy.shape[1]:
            fy = fy.diagonal(dim1=0, dim2=1).transpose(0, 1)
        if fyy is not None and fyy.ndim > 2 and fyy.shape[0] == fyy.shape[1]:
            fyy = fyy.diagonal(dim1=0, dim2=1).transpose(0, 1)
        return fy, fyy

    # ---- Loss functions ----
    def _compute_pde_residual(self, a_batch, omega_batch, lambda_batch, y_grid, u_batch, v_batch):
        """Compute pointwise PDE residual |D2*S_yy + D1*S_y + D0*S|^2."""
        B = a_batch.shape[0]
        N = y_grid.shape[1]

        f, fy, fyy = compute_f_derivatives_autograd(
            self.model, a_batch, omega_batch, y_grid, u_batch=u_batch, v_batch=v_batch,
        )

        # Build S and S-derivatives via the ansatz chain
        slope = horizon_regularity_slope(
            a=a_batch, omega=omega_batch, lambda_=lambda_batch,
            m=self.m_mode, M=self.M, s=self.s,
        )  # (B,)

        # Build S, Sy, Syy from f and autograd
        from physical_ansatz.transform_y import transform_coeffs_x_to_y_S
        from physical_ansatz.mapping import r_plus, r_from_x

        x_int = (y_grid + 1.0) / 2.0
        rp = r_plus(a_batch, self.M)
        r_int = rp.unsqueeze(-1) / x_int

        # Compute A coeffs then D coeffs
        D2, D1, D0 = transform_coeffs_x_to_y_S(
            A2=None, A1=None, A0=None,
            r=r_int, a=a_batch, omega=omega_batch,
            m=self.m_mode, M=self.M, s=self.s,
        )

        # Convert to complex
        D2_c = D2.to(dtype=self.cdtype)
        D1_c = D1.to(dtype=self.cdtype)
        D0_c = D0.to(dtype=self.cdtype)

        # Compute S and S-derivatives via the full ansatz
        S_vals, Sy_vals, Syy_vals = [], [], []
        for i in range(B):
            fi = f[i:i+1]
            fyi = fy[i:i+1]
            fyyi = fyy[i:i+1]
            yi = y_grid[i:i+1]
            sl_i = slope[i:i+1]

            Si = compose_reduced_shape_from_f(fi[0], yi[0], sl_i[0])

            # Use autograd to get Sy from S
            # Actually, we need S derivatives. Let's use the f-derivatives + ansatz
            # S(y) = g(x)*(h1(x)*f(y)+1)+1  where x=(y+1)/2
            # We have f, fy, fyy. Build S, Sy, Syy from the ansatz chain.
            from physical_ansatz.transform_y import g_factor, h1_factor

            x_i = x_int[i:i+1]
            h1_i = h1_factor(x_i[0])  # (h1, h1_x, h1_xx), each (N,)
            g_i = g_factor(x_i[0], sl_i[0])  # (g, g_x, g_xx), each (N,)

            h1_0, h1_x0, h1_xx0 = h1_i
            g_0, g_x0, g_xx0 = g_i

            # S = g * (h1*f + 1) + 1
            f_sc = fi[0].to(dtype=g_0.dtype)
            S_i = g_0 * (h1_0 * f_sc + 1.0) + 1.0

            # S' = g'*(h1*f+1) + g*(h1'*f + h1*f')
            fyi_sc = fyi[0].to(dtype=g_0.dtype)
            S_yi_1d = g_x0 * (h1_0 * f_sc + 1.0) + g_0 * (h1_x0 * f_sc + h1_0 * fyi_sc)
            # Note: fyi is df/dy, and x=(y+1)/2, so fx = 2*fy
            # But in the ansatz, S_x = g_x*(h1*f+1) + g*(h1_x*f + h1*f_x)
            # and f_x = 2*fy
            # So we need to adjust
            fxi_sc = 2.0 * fyi_sc
            S_xi_1d = g_x0 * (h1_0 * f_sc + 1.0) + g_0 * (h1_x0 * f_sc + h1_0 * fxi_sc)
            S_yi_1d_correct = S_xi_1d / 2.0  # since dx/dy = 1/2

            # S'' = g''*(h1*f+1) + 2*g'*(h1'*f + h1*f') + g*(h1''*f + 2*h1'*f' + h1*f'')
            # where all derivatives are with respect to x
            fxxi_sc = 4.0 * fyyi[0].to(dtype=g_0.dtype)  # f_xx = 4*f_yy
            S_xxi_1d = (g_xx0 * (h1_0 * f_sc + 1.0) +
                        2.0 * g_x0 * (h1_x0 * f_sc + h1_0 * fxi_sc) +
                        g_0 * (h1_xx0 * f_sc + 2.0 * h1_x0 * fxi_sc + h1_0 * fxxi_sc))
            S_yyi_1d = S_xxi_1d / 4.0  # S_yy = S_xx/4

            S_vals.append(S_i.unsqueeze(0))
            Sy_vals.append(S_yi_1d_correct.unsqueeze(0))
            Syy_vals.append(S_yyi_1d.unsqueeze(0))

        S_all = torch.cat(S_vals, dim=0)
        Sy_all = torch.cat(Sy_vals, dim=0)
        Syy_all = torch.cat(Syy_vals, dim=0)

        S_c = S_all.to(dtype=self.cdtype)
        Sy_c = Sy_all.to(dtype=self.cdtype)
        Syy_c = Syy_all.to(dtype=self.cdtype)

        pde_residual = D2_c * Syy_c + D1_c * Sy_c + D0_c * S_c
        pde_loss = (pde_residual.real ** 2 + pde_residual.imag ** 2).mean()

        return pde_loss, pde_residual

    def _compute_pde_loss_simple(self, a_batch, omega_batch, lambda_batch, y_grid, u_batch, v_batch):
        """Simpler PDE residual using the same logic as atlas_patch_trainer."""
        B = a_batch.shape[0]
        N = y_grid.shape[1]

        f, fy, fyy = compute_f_derivatives_autograd(
            self.model, a_batch, omega_batch, y_grid, u_batch=u_batch, v_batch=v_batch,
        )
        # Workaround for autograd bug: fy, fyy have shape (B, B, N) instead of (B, N)
        fy, fyy = self._fix_autograd_shapes(fy, fyy)

        slope = horizon_regularity_slope(
            a=a_batch, omega=omega_batch, lambda_=lambda_batch,
            m=self.m_mode, M=self.M, s=self.s,
        )

        x_int = (y_grid + 1.0) / 2.0
        r_plus_val = self.M + torch.sqrt(self.M ** 2 - a_batch ** 2 + 1e-12)
        r_int = r_plus_val.unsqueeze(-1) / x_int

        # Get A coeffs
        A2_list, A1_list, A0_list = [], [], []
        for i in range(B):
            A2_i, A1_i, A0_i = coeffs_x(
                x=x_int[i:i+1],
                a=a_batch[i],
                omega=omega_batch[i],
                m=self.m_mode,
                lambda_=lambda_batch[i],
                s=self.s,
                M=self.M,
            )
            A2_list.append(A2_i)
            A1_list.append(A1_i)
            A0_list.append(A0_i)

        # Transform A -> B via existing trainer logic
        # Use the same residual as the atlas trainer
        from physical_ansatz.transform_y import transform_coeffs_x_to_y

        # For simplicity: use the f-space residual (B2*f_yy + B1*f_y + B0*f = rhs)
        # This is equivalent and already implemented in the trainer
        f_c = f.to(dtype=self.cdtype)
        fy_c = fy.to(dtype=self.cdtype)
        fyy_c = fyy.to(dtype=self.cdtype)

        total_pde_loss = 0.0
        for i in range(B):
            B2, B1, B0, rhs = transform_coeffs_x_to_y(
                A2_list[i], A1_list[i], A0_list[i], y_grid[i:i+1], slope[i:i+1],
            )
            B2_c = B2.to(dtype=self.cdtype)
            B1_c = B1.to(dtype=self.cdtype)
            B0_c = B0.to(dtype=self.cdtype)
            rhs_c = rhs.to(dtype=self.cdtype) if rhs is not None else 0.0

            res = B2_c * fyy_c[i:i+1] + B1_c * fy_c[i:i+1] + B0_c * f_c[i:i+1] - rhs_c
            total_pde_loss += (res.real ** 2 + res.imag ** 2).mean()

        return total_pde_loss / B

    def compute_total_loss(self, a_batch, omega_batch, u_batch, v_batch, lambda_batch, y_grid):
        B = a_batch.shape[0]
        eps = self.cfg.get("loss", {}).get("eps", 1e-12)

        # PDE loss on all interior y (y > -1, handled by build_y_grid)
        # The random/build_y_grid never includes y=-1 exactly
        pde_loss = self._compute_pde_loss_simple(
            a_batch, omega_batch, lambda_batch, y_grid, u_batch, v_batch,
        )

        # Near-infinity PDE loss (extra weight on y < -0.95)
        near_pde_loss = torch.tensor(0.0, device=self.device)
        # TODO: implement per-row near-inf filtering if needed

        # Infinity Robin loss at y=-1
        ep = compute_stage1_S_and_Sy_at_infinity(
            self.model, a_batch, omega_batch, u=u_batch, v=v_batch,
            lambda_=lambda_batch, m=self.m_mode, M=self.M, s=self.s,
        )
        inf_loss = (ep["robin_rel"] ** 2).mean()

        # Drift loss: S_new vs S_old
        drift_loss = self._compute_drift_loss(
            a_batch, omega_batch, u_batch, v_batch, lambda_batch, y_grid,
        )

        loss_cfg = self.cfg.get("loss", {})
        w_pde = loss_cfg.get("weight_pde", 1.0)
        w_inf = loss_cfg.get("weight_inf_robin", 1.0)
        w_near = loss_cfg.get("weight_near_pde", 2.0)
        w_drift = loss_cfg.get("weight_drift", 0.1)

        total = w_pde * pde_loss + w_inf * inf_loss + w_near * near_pde_loss + w_drift * drift_loss

        return total, {
            "loss_pde": float(pde_loss.detach().cpu().item()),
            "loss_inf_robin": float(inf_loss.detach().cpu().item()),
            "loss_near_pde": float(near_pde_loss.detach().cpu().item()) if torch.is_tensor(near_pde_loss) else near_pde_loss,
            "loss_drift": float(drift_loss.detach().cpu().item()),
            "total_loss": float(total.detach().cpu().item()),
        }

    def _compute_drift_loss(self, a_batch, omega_batch, u_batch, v_batch, lambda_batch, y_grid):
        """Compute |S_new - S_old|^2 / (|S_old|^2 + eps)."""
        slope = horizon_regularity_slope(
            a=a_batch, omega=omega_batch, lambda_=lambda_batch,
            m=self.m_mode, M=self.M, s=self.s,
        )

        f_new, _, _ = compute_f_derivatives_autograd(
            self.model, a_batch, omega_batch, y_grid, u_batch=u_batch, v_batch=v_batch,
        )
        f_old, _, _ = compute_f_derivatives_autograd(
            self.frozen_model, a_batch, omega_batch, y_grid, u_batch=u_batch, v_batch=v_batch,
        )

        eps = self.cfg.get("loss", {}).get("eps", 1e-12)
        total_drift = 0.0
        for i in range(a_batch.shape[0]):
            S_new_i = compose_reduced_shape_from_f(f_new[i], y_grid[i], slope[i])
            S_old_i = compose_reduced_shape_from_f(f_old[i], y_grid[i], slope[i])
            S_old_abs2 = S_old_i.real ** 2 + S_old_i.imag ** 2
            S_diff_abs2 = (S_new_i - S_old_i).real ** 2 + (S_new_i - S_old_i).imag ** 2
            total_drift += (S_diff_abs2 / (S_old_abs2 + eps)).mean()

        return total_drift / a_batch.shape[0]

    # ---- Parameter sampling ----
    def _sample_parameters(self, batch_size):
        """Sample random (a, omega) for training. Uses uniform random within patch-0 range."""
        # Patch 0 range: a ∈ [0.01, 0.99], omega ∈ [0.0001, 1.0]
        a_vals = torch.rand(batch_size, device=self.device, dtype=self.dtype) * 0.98 + 0.01
        omega_vals = torch.rand(batch_size, device=self.device, dtype=self.dtype) * 0.9999 + 0.0001

        # Compute lambda for each
        lam_vals = []
        for i in range(batch_size):
            lam_i = compute_lambda(float(a_vals[i]), float(omega_vals[i]), self.ell, self.m_mode, s=self.s)
            lam_vals.append(torch.tensor(lam_i, device=self.device, dtype=self.cdtype))

        return a_vals, omega_vals, torch.stack(lam_vals)

    # ---- Optimizer ----
    def _build_adam_optimizer(self):
        adam_cfg = self.cfg.get("adam", {})
        lr_enc = adam_cfg.get("lr_encoder", 2e-7)
        lr_rin = adam_cfg.get("lr_rin_decoder", 1e-6)
        wd = adam_cfg.get("weight_decay", 0.0)

        enc_params = [p for n, p in self.model.named_parameters() if n.startswith("encoder") and p.requires_grad]
        rin_params = [p for n, p in self.model.named_parameters() if n.startswith("rin_decoder") and p.requires_grad]

        return torch.optim.Adam([
            {"params": enc_params, "lr": lr_enc},
            {"params": rin_params, "lr": lr_rin},
        ], weight_decay=wd)

    # ---- Training phases ----
    def train_adam(self, epochs):
        adam_cfg = self.cfg.get("adam", {})
        batch_size = adam_cfg.get("batch_size", 8)
        grad_clip = adam_cfg.get("grad_clip", 0.1)

        self.optimizer = self._build_adam_optimizer()
        print(f"\n{'='*60}")
        print(f"Adam phase: {epochs} epochs, batch_size={batch_size}")
        print(f"{'='*60}")

        for epoch in range(epochs):
            a_b, omega_b, lam_b = self._sample_parameters(batch_size)
            y_grid = self.build_random_y_batch(batch_size)

            u_b = None
            v_b = None

            self.optimizer.zero_grad()
            total_loss, info = self.compute_total_loss(
                a_b, omega_b, u_b, v_b, lam_b, y_grid,
            )
            total_loss.backward()

            gn = torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad], grad_clip,
            )
            self.optimizer.step()

            info["epoch"] = epoch
            info["grad_norm"] = float(gn.item()) if torch.is_tensor(gn) else float(gn)
            self.global_step = epoch

            if epoch == 0 or epoch % 5 == 0 or epoch == epochs - 1:
                self._log_step(epoch, info)

            # Save best
            if info["total_loss"] < self.best_loss:
                self.best_loss = info["total_loss"]
                self.best_step = epoch
                self._save_checkpoint("best_model.pt")

            if epoch % 100 == 0 or epoch == epochs - 1:
                self._save_checkpoint(f"step_{epoch:06d}.pt")

        self._save_checkpoint("latest_model.pt")
        print(f"\nAdam done. Best loss: {self.best_loss:.6f} at epoch {self.best_step}")

    def train_lbfgs(self, epochs):
        lbfgs_cfg = self.cfg.get("lbfgs", {})
        max_iter = lbfgs_cfg.get("max_iter_per_epoch", 20)
        history_size = lbfgs_cfg.get("history_size", 50)
        line_search_fn = lbfgs_cfg.get("line_search_fn", "strong_wolfe")
        restart_every = lbfgs_cfg.get("restart_every", 10)
        grad_clip = lbfgs_cfg.get("grad_clip", 0.1)

        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.LBFGS(
            trainable_params,
            lr=lbfgs_cfg.get("lr", 0.5),
            history_size=history_size,
            line_search_fn=line_search_fn,
            max_iter=max_iter,
            tolerance_grad=1e-12,
            tolerance_change=1e-14,
        )

        print(f"\n{'='*60}")
        print(f"L-BFGS phase: {epochs} epochs, max_iter={max_iter}")
        print(f"{'='*60}")

        # Fixed collocation for L-BFGS
        fixed_batch_size = lbfgs_cfg.get("fixed_batch_size", 16)
        a_fixed, omega_fixed, lam_fixed = self._sample_parameters(fixed_batch_size)
        y_fixed_raw = self.build_y_grid()
        y_fixed = y_fixed_raw.unsqueeze(0).expand(fixed_batch_size, -1)

        for epoch in range(epochs):
            # Restart: regenerate fixed grid
            if epoch > 0 and restart_every > 0 and epoch % restart_every == 0:
                a_fixed, omega_fixed, lam_fixed = self._sample_parameters(fixed_batch_size)
                y_fixed_raw = self.build_y_grid()
                y_fixed = y_fixed_raw.unsqueeze(0).expand(fixed_batch_size, -1)
                # Re-create L-BFGS optimizer on restart
                self.optimizer = torch.optim.LBFGS(
                    trainable_params,
                    lr=lbfgs_cfg.get("lr", 0.5) * (0.5 ** (epoch // restart_every)),
                    history_size=history_size,
                    line_search_fn=line_search_fn,
                    max_iter=max_iter,
                    tolerance_grad=1e-12,
                    tolerance_change=1e-14,
                )

            def closure():
                self.optimizer.zero_grad()
                total_loss, info = self.compute_total_loss(
                    a_fixed, omega_fixed, None, None, lam_fixed,
                    y_fixed,
                )
                total_loss.backward()

                # Clip gradients
                gn = torch.nn.utils.clip_grad_norm_(trainable_params, grad_clip)

                # Store info for logging
                closure._info = info
                closure._gn = float(gn.item()) if torch.is_tensor(gn) else float(gn)
                return total_loss

            loss_val = self.optimizer.step(closure)
            info = getattr(closure, "_info", {})
            gn_val = getattr(closure, "_gn", 0.0)

            info["epoch"] = epoch
            info["grad_norm"] = gn_val
            self.global_step = epoch

            total_loss_val = float(loss_val.detach().item()) if torch.is_tensor(loss_val) else float(loss_val)

            if total_loss_val < self.best_loss:
                self.best_loss = total_loss_val
                self.best_step = epoch
                self._save_checkpoint("best_model.pt")

            if epoch % 5 == 0 or epoch == epochs - 1:
                self._log_step(epoch, info)

            if epoch % 10 == 0 or epoch == epochs - 1:
                self._save_checkpoint(f"step_{epoch:06d}.pt")

        self._save_checkpoint("latest_model.pt")
        print(f"\nL-BFGS done. Best loss: {self.best_loss:.6f} at epoch {self.best_step}")

    def _log_step(self, step, info):
        msg = f"  step {step:4d}: "
        msg += f"tot={info.get('total_loss', 0):.4e}, "
        msg += f"pde={info.get('loss_pde', 0):.4e}, "
        msg += f"inf={info.get('loss_inf_robin', 0):.4e}, "
        msg += f"near={info.get('loss_near_pde', 0):.4e}, "
        msg += f"drift={info.get('loss_drift', 0):.4e}, "
        msg += f"g={info.get('grad_norm', 0):.2f}"
        print(msg)

        with open(self.history_path, "a") as f:
            f.write(json.dumps({**info, "step": step}) + "\n")

    def _save_checkpoint(self, name):
        path = self.run_dir / "checkpoints" / name
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "best_loss": self.best_loss,
            "best_step": self.best_step,
            "global_step": self.global_step,
        }, path)
        if self.verbose:
            print(f"  Saved checkpoint: {path}")


# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/autoencoder_stage1_infinity_robin.yaml")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--phase", type=str, default="adam", choices=["adam", "lbfgs"])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--resume-checkpoint", type=str, default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    trainer = RefinementTrainer(
        config_path=args.config,
        device=args.device,
        resume_checkpoint=args.resume_checkpoint,
        verbose=args.verbose,
    )

    if args.phase == "adam":
        adam_cfg = trainer.cfg.get("adam", {})
        epochs = args.epochs if args.epochs is not None else adam_cfg.get("epochs", 50)
        trainer.train_adam(epochs)
    elif args.phase == "lbfgs":
        lbfgs_cfg = trainer.cfg.get("lbfgs", {})
        epochs = args.epochs if args.epochs is not None else lbfgs_cfg.get("max_epochs", 50)
        trainer.train_lbfgs(epochs)


if __name__ == "__main__":
    main()
