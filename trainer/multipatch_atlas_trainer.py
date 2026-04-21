from __future__ import annotations

import json
import math
import shutil
from dataclasses import asdict
from pathlib import Path

from config.config_loader import load_pinn_full_config
from domain.patch_cover import load_patch_cover
from trainer.atlas_patch_trainer import AtlasPatchTrainer


class MultiPatchAtlasTrainer:
    """
    完整 v1:
      - 顺序训练所有 patch
      - 支持按中心/覆盖度排序
      - 支持从相邻已训练 patch warm start
      - 自动维护 atlas registry
      - 自动导出统一 best/latest 模型目录
    """

    def __init__(
        self,
        cfg_path: str,
        probe_json: str,
        atlas_json: str,
        patch_json: str,
        device: str = "cuda",
        anchor_enabled: bool = True,
        n_anchor_y: int = 4,
        verbose: bool = True,
    ):
        self.cfg_path = Path(cfg_path).resolve()
        self.probe_json = str(probe_json)
        self.atlas_json = str(atlas_json)
        self.patch_json = str(patch_json)
        self.device = device
        self.anchor_enabled = bool(anchor_enabled)
        self.n_anchor_y = int(n_anchor_y)
        self.verbose = bool(verbose)

        self.full_cfg = load_pinn_full_config(str(self.cfg_path))
        self.train_cfg = self.full_cfg["train"]
        self.mcfg = self.train_cfg.get("multipatch_training", {})

        self.patch_cover = load_patch_cover(self.patch_json)
        self.patches = list(self.patch_cover.patches)

        self.patch_order_mode = str(self.mcfg.get("patch_order", "center_out"))
        self.warmstart_from_neighbor = bool(self.mcfg.get("warmstart_from_neighbor", True))
        self.skip_existing = bool(self.mcfg.get("skip_existing", True))
        self.patch_steps = self.mcfg.get("patch_steps", None)
        self.output_root = Path(self.mcfg.get("output_root", "outputs/atlas_multipatch_train"))
        self.registry_name = str(self.mcfg.get("registry_name", "atlas_registry.json"))
        self.export_flat_models = bool(self.mcfg.get("export_flat_models", True))
        self.export_model_dirname = str(self.mcfg.get("export_model_dirname", "models"))
        self.max_patches = self.mcfg.get("max_patches", None)

        self.output_root.mkdir(parents=True, exist_ok=True)
        self.registry_path = self.output_root / self.registry_name

        self.patch_runs_root = self.output_root / "patch_runs"
        self.patch_runs_root.mkdir(parents=True, exist_ok=True)

        self.export_model_dir = self.output_root / self.export_model_dirname
        self.export_model_dir.mkdir(parents=True, exist_ok=True)

        self.registry = {
            "cfg_path": str(self.cfg_path),
            "probe_json": self.probe_json,
            "atlas_json": self.atlas_json,
            "patch_json": self.patch_json,
            "device": self.device,
            "anchor_enabled": self.anchor_enabled,
            "n_anchor_y": self.n_anchor_y,
            "patch_cover_meta": {
                "n_patches": int(self.patch_cover.n_patches),
                "component_id": int(self.patch_cover.component_id),
                "coverage_min": int(self.patch_cover.coverage_min),
                "coverage_mean": float(self.patch_cover.coverage_mean),
                "coverage_max": int(self.patch_cover.coverage_max),
                "n_points_with_overlap": int(self.patch_cover.n_points_with_overlap),
            },
            "patch_order_mode": self.patch_order_mode,
            "patch_records": [],
        }

        if self.registry_path.exists():
            with open(self.registry_path, "r", encoding="utf-8") as f:
                self.registry = json.load(f)

    def _vprint(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

    def _patch_sort_key(self, patch):
        if self.patch_order_mode == "pool_desc":
            return (-int(patch.n_safe_points_covered), int(patch.patch_id))

        if self.patch_order_mode == "patch_id":
            return int(patch.patch_id)

        # default: center_out
        du = float(patch.u_center) - 0.5
        dv = float(patch.v_center) - 0.5
        return (du * du + dv * dv, int(patch.patch_id))

    def _ordered_patches(self):
        patches = sorted(self.patches, key=self._patch_sort_key)
        if self.max_patches is not None:
            patches = patches[: int(self.max_patches)]
        return patches

    def _find_existing_record(self, patch_id: int):
        for rec in self.registry.get("patch_records", []):
            if int(rec["patch_id"]) == int(patch_id):
                return rec
        return None

    def _patches_overlap(self, p1, p2) -> bool:
        du = abs(float(p1.u_center) - float(p2.u_center))
        dv = abs(float(p1.v_center) - float(p2.v_center))
        return (du <= float(p1.h_u + p2.h_u)) and (dv <= float(p1.h_v + p2.h_v))

    def _find_warmstart_checkpoint(self, current_patch):
        """
        从已训练 patch 中找一个最合适的 best checkpoint：
          1) 优先与当前 patch 重叠
          2) 再按 uv 中心距离最小
          3) 再按 best_val_mean 最小
        """
        if not self.warmstart_from_neighbor:
            return None

        candidates = []
        for rec in self.registry.get("patch_records", []):
            if not rec.get("best_model_path"):
                continue
            best_path = Path(rec["best_model_path"])
            if not best_path.exists():
                continue

            prev_patch = None
            for p in self.patches:
                if int(p.patch_id) == int(rec["patch_id"]):
                    prev_patch = p
                    break
            if prev_patch is None:
                continue

            overlap = self._patches_overlap(current_patch, prev_patch)
            du = float(current_patch.u_center - prev_patch.u_center)
            dv = float(current_patch.v_center - prev_patch.v_center)
            d2 = du * du + dv * dv
            score = (
                0 if overlap else 1,
                d2,
                float(rec.get("best_val_mean", 1.0e99)),
            )
            candidates.append((score, str(best_path)))

        if len(candidates) == 0:
            return None

        candidates.sort(key=lambda x: x[0])
        return candidates[0][1]

    def _find_resume_run(self, patch_id: int):
        pattern = f"*_patch_{int(patch_id):03d}_comp_*"
        run_dirs = [p for p in self.patch_runs_root.glob(pattern) if p.is_dir()]
        if len(run_dirs) == 0:
            return None, None

        run_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        for run_dir in run_dirs:
            latest_ckpt = run_dir / "checkpoints" / "latest_model.pt"
            if not latest_ckpt.exists():
                continue
            if self._resume_run_is_healthy(run_dir):
                return str(latest_ckpt), str(run_dir)
            self._vprint(f"[multipatch] patch {patch_id}: skip unhealthy resume run {run_dir}")
        return None, None

    def _resume_run_is_healthy(self, run_dir: Path) -> bool:
        history_path = run_dir / "logs" / "history.jsonl"
        if not history_path.exists():
            return True

        try:
            tail = history_path.read_text(encoding="utf-8")[-8192:]
        except Exception:
            return True

        unhealthy_tokens = (
            '"total_loss": NaN',
            '"loss_pde": NaN',
            '"grad_norm": NaN',
            '"total_loss": Infinity',
            '"loss_pde": Infinity',
            '"grad_norm": Infinity',
        )
        return not any(tok in tail for tok in unhealthy_tokens)

    def _save_registry(self):
        with open(self.registry_path, "w", encoding="utf-8") as f:
            json.dump(self.registry, f, ensure_ascii=False, indent=2)

    def _export_models(self, patch_id: int, run_dir: Path):
        best_src = run_dir / "checkpoints" / "best_model.pt"
        latest_src = run_dir / "checkpoints" / "latest_model.pt"

        best_dst = self.export_model_dir / f"patch_{patch_id:03d}_best.pt"
        latest_dst = self.export_model_dir / f"patch_{patch_id:03d}_latest.pt"

        if best_src.exists():
            shutil.copy2(best_src, best_dst)
        if latest_src.exists():
            shutil.copy2(latest_src, latest_dst)

        return str(best_dst) if best_dst.exists() else None, str(latest_dst) if latest_dst.exists() else None

    def train_all(self):
        ordered = self._ordered_patches()
        self._vprint("=" * 100)
        self._vprint(f"[multipatch] total patches to train: {len(ordered)}")
        self._vprint("=" * 100)

        for patch in ordered:
            existing = self._find_existing_record(int(patch.patch_id))
            if self.skip_existing and existing is not None:
                best_path = existing.get("best_model_path", None)
                if best_path is not None and Path(best_path).exists():
                    self._vprint(f"[multipatch] skip patch {patch.patch_id}: existing best model found.")
                    continue

            resume_ckpt, resume_run_dir = self._find_resume_run(int(patch.patch_id))
            init_ckpt = None
            if resume_ckpt is not None and resume_run_dir is not None:
                self._vprint(f"[multipatch] patch {patch.patch_id}: resume from {resume_ckpt}")
            else:
                init_ckpt = self._find_warmstart_checkpoint(patch)
                if init_ckpt is not None:
                    self._vprint(f"[multipatch] patch {patch.patch_id}: warm start from {init_ckpt}")
                else:
                    self._vprint(f"[multipatch] patch {patch.patch_id}: cold start")

            trainer = AtlasPatchTrainer(
                cfg_path=str(self.cfg_path),
                probe_json=self.probe_json,
                atlas_json=self.atlas_json,
                patch_json=self.patch_json,
                patch_id=int(patch.patch_id),
                device=self.device,
                anchor_enabled=self.anchor_enabled,
                n_anchor_y=self.n_anchor_y,
                verbose=self.verbose,
                output_root=str(self.patch_runs_root),
                init_checkpoint=init_ckpt,
                init_load_optimizer=False,
                resume_checkpoint=resume_ckpt,
                resume_run_dir=resume_run_dir,
            )

            try:
                result = trainer.train(steps=self.patch_steps)
                run_dir = Path(result["run_dir"])

                best_export, latest_export = (None, None)
                if self.export_flat_models:
                    best_export, latest_export = self._export_models(
                        patch_id=int(patch.patch_id),
                        run_dir=run_dir,
                    )

                rec = {
                    "patch_id": int(patch.patch_id),
                    "component_id": int(patch.component_id),
                    "u_center": float(patch.u_center),
                    "v_center": float(patch.v_center),
                    "h_u": float(patch.h_u),
                    "h_v": float(patch.h_v),
                    "a_center": float(patch.a_center),
                    "omega_center": float(patch.omega_center),
                    "n_safe_points_covered": int(patch.n_safe_points_covered),
                    "run_dir": str(run_dir),
                    "best_val_mean": float(result["best_val_mean"]),
                    "best_model_path": best_export if best_export is not None else str(run_dir / "checkpoints" / "best_model.pt"),
                    "latest_model_path": latest_export if latest_export is not None else str(run_dir / "checkpoints" / "latest_model.pt"),
                    "warmstart_checkpoint": init_ckpt,
                    "final_val": result.get("final_val", None),
                }

                old = self._find_existing_record(int(patch.patch_id))
                if old is None:
                    self.registry["patch_records"].append(rec)
                else:
                    idx = None
                    for i, x in enumerate(self.registry["patch_records"]):
                        if int(x["patch_id"]) == int(patch.patch_id):
                            idx = i
                            break
                    self.registry["patch_records"][idx] = rec

                self._save_registry()
                self._vprint(f"[multipatch] finished patch {patch.patch_id}, best_val_mean={result['best_val_mean']:.6e}")

            finally:
                trainer.close()

        self._save_registry()
        self._vprint("=" * 100)
        self._vprint(f"[multipatch] registry saved to: {self.registry_path}")
        self._vprint("=" * 100)
        return str(self.registry_path)
