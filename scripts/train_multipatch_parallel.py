#!/usr/bin/env python3
"""
Parallel multipatch trainer for Phase 4 Atlas-PINN.

Strategy:
  - Patch 0 trained first (cold start, foundation for all others)
  - After patch 0 completes, remaining patches run in parallel
  - Each patch is a subprocess (isolated CUDA context)
  - GPU concurrency limited to avoid OOM

Usage:
  python scripts/train_multipatch_parallel.py \
    --cfg config/pinn_config_phase4_integral.yaml \
    --max_parallel 2
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.config_loader import load_pinn_full_config
from domain.patch_cover import load_patch_cover

# ============================================================
# helpers
# ============================================================
def _patches_overlap(p1, p2) -> bool:
    du = abs(float(p1.u_center) - float(p2.u_center))
    dv = abs(float(p1.v_center) - float(p2.v_center))
    return (du <= float(p1.h_u + p2.h_u)) and (dv <= float(p1.h_v + p2.h_v))


def _patch_sort_key(patch, mode: str = "center_out"):
    if mode == "pool_desc":
        return (-int(patch.n_safe_points_covered), int(patch.patch_id))
    if mode == "patch_id":
        return int(patch.patch_id)
    du = float(patch.u_center) - 0.5
    dv = float(patch.v_center) - 0.5
    return (du * du + dv * dv, int(patch.patch_id))


@dataclass
class PatchTask:
    patch_id: int
    u_center: float
    v_center: float
    h_u: float
    h_v: float
    n_safe_points: int
    warmstart_from: int | None  # patch_id to warmstart from, None = cold start
    priority: int = 0  # lower = higher priority


# ============================================================
# single-patch training worker (runs in subprocess)
# ============================================================
def _train_one_patch(
    cfg_path: str,
    probe_json: str,
    atlas_json: str,
    patch_json: str,
    patch_id: int,
    device: str,
    anchor_enabled: bool,
    n_anchor_y: int,
    output_root: str,
    init_checkpoint: str | None,
    init_load_optimizer: bool,
    steps: int | None,
    verbose: bool,
    model_type: str,
    cheb_N: int,
) -> dict:
    """Train a single patch. Returns result dict for registry."""
    # Re-import inside subprocess
    import sys as _sys
    _sys.path.insert(0, str(ROOT))
    from trainer.atlas_patch_trainer import AtlasPatchTrainer

    trainer = AtlasPatchTrainer(
        cfg_path=cfg_path,
        probe_json=probe_json,
        atlas_json=atlas_json,
        patch_json=patch_json,
        patch_id=patch_id,
        device=device,
        anchor_enabled=anchor_enabled,
        n_anchor_y=n_anchor_y,
        verbose=verbose,
        output_root=output_root,
        init_checkpoint=init_checkpoint,
        init_load_optimizer=init_load_optimizer,
        model_type=model_type,
        cheb_N=cheb_N,
    )

    try:
        result = trainer.train(steps=steps)
        run_dir = Path(result["run_dir"])

        best_src = run_dir / "checkpoints" / "best_model.pt"
        latest_src = run_dir / "checkpoints" / "latest_model.pt"

        export_dir = Path(output_root) / "models"
        export_dir.mkdir(parents=True, exist_ok=True)
        import shutil

        best_dst = export_dir / f"patch_{patch_id:03d}_best.pt"
        latest_dst = export_dir / f"patch_{patch_id:03d}_latest.pt"
        if best_src.exists():
            shutil.copy2(best_src, best_dst)
        if latest_src.exists():
            shutil.copy2(latest_src, latest_dst)

        patch = trainer.patch
        rec = {
            "patch_id": int(patch_id),
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
            "best_model_path": str(best_dst) if best_dst.exists() else str(best_src),
            "latest_model_path": str(latest_dst) if latest_dst.exists() else str(latest_src),
            "warmstart_checkpoint": init_checkpoint,
            "final_val": result.get("final_val", None),
        }
        return {"status": "ok", "record": rec}

    except Exception as e:
        return {"status": "error", "patch_id": int(patch_id), "error": str(e),
                "traceback": traceback.format_exc()}
    finally:
        try:
            trainer.close()
        except Exception:
            pass


# ============================================================
# main orchestrator
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Parallel multipatch trainer")
    parser.add_argument("--cfg", required=True, help="Path to PINN config YAML")
    parser.add_argument("--probe", default="outputs/domain/probe_l2_m2.json")
    parser.add_argument("--atlas", default="outputs/domain/atlas_l2_m2.json")
    parser.add_argument("--patch", default="outputs/domain/patch_cover_l2_m2.json")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_parallel", type=int, default=2, help="Max concurrent GPU trainings")
    parser.add_argument("--patch_ids", nargs="*", type=int, default=None,
                        help="Specific patch IDs to train (default: all)")
    parser.add_argument("--retrain_patch0", action="store_true",
                        help="Force retrain patch 0 even if existing")
    parser.add_argument("--output_root", default=None, help="Override output root")
    parser.add_argument("--steps", type=int, default=None, help="Override training steps")
    parser.add_argument("--anchor", action="store_true", default=False)
    parser.add_argument("--model_type", default="pinn_mlp")
    parser.add_argument("--cheb_N", type=int, default=64)
    parser.add_argument("--registry_name", default="atlas_registry_phase4_integral.json")
    args = parser.parse_args()

    cfg_path = str(Path(args.cfg).resolve())
    full_cfg = load_pinn_full_config(cfg_path)
    train_cfg = full_cfg["train"]
    mcfg = train_cfg.get("multipatch_training", {})
    atlas_cfg = train_cfg.get("atlas_training", {})

    output_root = Path(args.output_root or mcfg.get("output_root", "outputs/atlas_multipatch_phase4_integral"))
    output_root.mkdir(parents=True, exist_ok=True)
    registry_path = output_root / args.registry_name

    patch_cover = load_patch_cover(args.patch)
    patches = list(patch_cover.patches)

    patch_order_mode = mcfg.get("patch_order", "center_out")
    ordered = sorted(patches, key=lambda p: _patch_sort_key(p, patch_order_mode))

    print("=" * 100)
    print(f"[parallel] total patches available: {len(ordered)}")
    print(f"[parallel] output_root: {output_root}")
    print(f"[parallel] registry: {registry_path}")

    # Load existing registry
    registry = {"patch_records": []}
    if registry_path.exists():
        with open(registry_path, "r", encoding="utf-8") as f:
            registry = json.load(f)
    existing_ids = {int(r["patch_id"]) for r in registry.get("patch_records", [])}

    # Filter patches to train
    to_train = []
    if args.patch_ids:
        target_ids = set(args.patch_ids)
    else:
        target_ids = {int(p.patch_id) for p in ordered}

    for p in ordered:
        pid = int(p.patch_id)
        if pid not in target_ids:
            continue
        if pid in existing_ids and pid != 0:
            print(f"[parallel] skip patch {pid}: already in registry")
            continue
        if pid == 0 and pid in existing_ids and not args.retrain_patch0:
            print(f"[parallel] skip patch 0: already in registry (use --retrain_patch0 to force)")
            continue

        # Determine warmstart
        warmstart_from = None
        warmstart_ckpt = None
        if pid != 0:
            # Find best warmstart from completed patches
            candidates = []
            for rec in registry.get("patch_records", []):
                rpid = int(rec["patch_id"])
                if rpid not in existing_ids:
                    continue
                bp = rec.get("best_model_path")
                if not bp or not Path(bp).exists():
                    continue
                rp = next((pp for pp in patches if int(pp.patch_id) == rpid), None)
                if rp is None:
                    continue
                overlap = _patches_overlap(p, rp)
                du = float(p.u_center - rp.u_center)
                dv = float(p.v_center - rp.v_center)
                d2 = du * du + dv * dv
                candidates.append(((0 if overlap else 1, d2, float(rec.get("best_val_mean", 1e99))), str(bp)))
            if candidates:
                candidates.sort(key=lambda x: x[0])
                warmstart_ckpt = candidates[0][1]
                warmstart_from = candidates[0][1]  # simplified

        to_train.append(PatchTask(
            patch_id=pid,
            u_center=float(p.u_center),
            v_center=float(p.v_center),
            h_u=float(p.h_u),
            h_v=float(p.h_v),
            n_safe_points=int(p.n_safe_points_covered),
            warmstart_from=None,
            priority=0 if pid == 0 else 1,
        ))

    if not to_train:
        print("[parallel] all patches already trained. nothing to do.")
        return

    # Patch 0 must go first (cold start)
    patch0_tasks = [t for t in to_train if t.patch_id == 0]
    other_tasks = [t for t in to_train if t.patch_id != 0]

    # --- Phase 1: train patch 0 ---
    for t in patch0_tasks:
        print("=" * 100)
        print(f"[parallel] Phase 1: training patch 0 (cold start, {t.n_safe_points} safe points)")
        print("=" * 100)

        steps = args.steps or atlas_cfg.get("steps", 30000)
        result = _train_one_patch(
            cfg_path=cfg_path,
            probe_json=str(args.probe),
            atlas_json=str(args.atlas),
            patch_json=str(args.patch),
            patch_id=0,
            device=args.device,
            anchor_enabled=bool(args.anchor),
            n_anchor_y=4,
            output_root=str(output_root),
            init_checkpoint=None,
            init_load_optimizer=False,
            steps=steps,
            verbose=True,
            model_type=args.model_type,
            cheb_N=args.cheb_N,
        )

        if result["status"] == "ok":
            rec = result["record"]
            # Replace or append
            idx = None
            for i, r in enumerate(registry.get("patch_records", [])):
                if int(r["patch_id"]) == 0:
                    idx = i
                    break
            if idx is not None:
                registry["patch_records"][idx] = rec
            else:
                registry["patch_records"].append(rec)
            with open(registry_path, "w", encoding="utf-8") as f:
                json.dump(registry, f, ensure_ascii=False, indent=2)
            existing_ids.add(0)
            print(f"[parallel] patch 0 done. best_val_mean={rec['best_val_mean']:.6e}")
            # Update warmstart checkpoints for other tasks
            for ot in other_tasks:
                ot.warmstart_from = rec["best_model_path"]
        else:
            print(f"[parallel] patch 0 FAILED: {result.get('error')}")
            print(result.get("traceback", ""))
            return

    # --- Phase 2: parallel training of remaining patches ---
    if not other_tasks:
        print("[parallel] no remaining patches. done.")
        return

    print("=" * 100)
    print(f"[parallel] Phase 2: training {len(other_tasks)} patches in parallel "
          f"(max_concurrent={args.max_parallel})")
    print("=" * 100)

    def _save_registry():
        with open(registry_path, "w", encoding="utf-8") as f:
            json.dump(registry, f, ensure_ascii=False, indent=2)

    # Rebuild warmstart checkpoints for other tasks from updated registry
    patches_list = list(patches)
    for ot in other_tasks:
        candidates = []
        for rec in registry.get("patch_records", []):
            rpid = int(rec["patch_id"])
            bp = rec.get("best_model_path")
            if not bp or not Path(bp).exists():
                continue
            rp = next((pp for pp in patches_list if int(pp.patch_id) == rpid), None)
            if rp is None:
                continue
            cp = next((pp for pp in patches_list if int(pp.patch_id) == ot.patch_id), None)
            if cp is None:
                continue
            overlap = _patches_overlap(cp, rp)
            du = float(cp.u_center - rp.u_center)
            dv = float(cp.v_center - rp.v_center)
            d2 = du * du + dv * dv
            candidates.append(((0 if overlap else 1, d2, float(rec.get("best_val_mean", 1e99))), str(bp)))
        if candidates:
            candidates.sort(key=lambda x: x[0])
            ot.warmstart_from = candidates[0][1]

    steps = args.steps or atlas_cfg.get("steps", 15000)

    with ProcessPoolExecutor(max_workers=args.max_parallel) as executor:
        futures = {}
        for t in other_tasks:
            fut = executor.submit(
                _train_one_patch,
                cfg_path=cfg_path,
                probe_json=str(args.probe),
                atlas_json=str(args.atlas),
                patch_json=str(args.patch),
                patch_id=t.patch_id,
                device=args.device,
                anchor_enabled=bool(args.anchor),
                n_anchor_y=4,
                output_root=str(output_root),
                init_checkpoint=t.warmstart_from,
                init_load_optimizer=False,
                steps=steps,
                verbose=True,
                model_type=args.model_type,
                cheb_N=args.cheb_N,
            )
            futures[fut] = t.patch_id

        for fut in as_completed(futures):
            pid = futures[fut]
            try:
                result = fut.result()
            except Exception as e:
                print(f"[parallel] patch {pid} crashed: {e}")
                continue

            if result["status"] == "ok":
                rec = result["record"]
                idx = None
                for i, r in enumerate(registry.get("patch_records", [])):
                    if int(r["patch_id"]) == pid:
                        idx = i
                        break
                if idx is not None:
                    registry["patch_records"][idx] = rec
                else:
                    registry["patch_records"].append(rec)
                _save_registry()
                print(f"[parallel] patch {pid} done. best_val_mean={rec['best_val_mean']:.6e}")
            else:
                print(f"[parallel] patch {pid} FAILED: {result.get('error')}")

    _save_registry()
    print("=" * 100)
    print(f"[parallel] registry saved to: {registry_path}")
    registered = len(registry.get("patch_records", []))
    print(f"[parallel] done. {registered}/{len(patches)} patches in registry.")
    print("=" * 100)


if __name__ == "__main__":
    main()
