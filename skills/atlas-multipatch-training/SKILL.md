---
name: atlas-multipatch-training
description: Use when working on atlas patch training, multi-patch resume, MMA session handling during training, early-stop/save behavior, or domain scripts 04/05/06/07/08/09 under test/domain.
---

# Atlas Multipatch Training

Use this skill for the atlas patch training workflow in this repo.

## Scope

- `trainer/atlas_patch_trainer.py`
- `trainer/multipatch_atlas_trainer.py`
- `test/domain/04_smoke_train_one_patch.py`
- `test/domain/05_smoke_train_one_patch_with_anchor.py`
- `test/domain/06_train_one_patch_full.py`
- `test/domain/07_train_all_patches_full.py`
- `test/domain/08_smoke_eval_atlas_predictor.py`
- `test/domain/09_random_scan_teukolsky_solver.py`

## Current repo conventions

- `AtlasPatchTrainer` is the single-patch trainer.
- `MultiPatchAtlasTrainer` trains all patches sequentially.
- Patch identity is defined in chart coordinates `(u, v)`, not directly by `(a, omega)`.
- Multi-patch outputs live under `outputs/atlas_multipatch_train/`.
- Patch runs live under `outputs/atlas_multipatch_train/patch_runs/`.
- Completed patch metadata is tracked in `outputs/atlas_multipatch_train/atlas_registry.json`.

## Resume behavior

- Finished patches are skipped using `atlas_registry.json`.
- Interrupted patches resume from the latest run directory for the same `patch_id` if `checkpoints/latest_model.pt` exists.
- Resume logic depends on unchanged `patch_json` and unchanged `multipatch_training.output_root`.
- Resume restores:
  - model weights
  - optimizer state
  - `global_step`
  - `best_val_mean`

## MMA behavior during atlas training

- Training anchor should default to off unless explicitly requested.
- Visualization may use MMA, but should not block training.
- If visualization MMA fails, save a model-only figure instead of aborting the step.
- Do not recreate a new Mathematica kernel for ordinary numerical failures.
- Only reset the session on transport-level failures such as socket/WSTP issues.

## Output expectations

- Single-patch runs save:
  - `checkpoints/best_model.pt`
  - `checkpoints/latest_model.pt`
  - `logs/history.jsonl`
  - `logs/summary.json`
  - figures under `figures/`
- Multi-patch export copies:
  - `models/patch_XXX_best.pt`
  - `models/patch_XXX_latest.pt`

## Known workflow decisions

- `07_train_all_patches_full.py` should usually be run without `--anchor-enabled` unless MMA anchor stability is the actual test target.
- `atlas_training.viz_mma_enabled: false` is the safest setting when MMA kernels are unstable.
- `08_smoke_eval_atlas_predictor.py` should evaluate `R(r)` on an independent `r` grid and `R'(y)` on an independent `y` grid.
- `09_random_scan_teukolsky_solver.py` should sample random points from atlas chart support, not from the raw rectangular parameter box.

## Useful checks

- Read `outputs/atlas_multipatch_train/atlas_registry.json` before reasoning about which patches are complete.
- Inspect `patch_runs/*/checkpoints/latest_model.pt` before claiming resume is unavailable.
- If a patch “fails everywhere”, separate:
  - envelope/out-of-support failures
  - MMA transport failures
  - MMA numerical failures

