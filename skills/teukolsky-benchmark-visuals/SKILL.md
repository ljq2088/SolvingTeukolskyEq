---
name: teukolsky-benchmark-visuals
description: Use when working on TeukolskySolver vs MMA benchmark plots, random parameter scan visualizations, 3x3 comparison figures, relative-error columns, or repairing missing figures from saved npz outputs.
---

# Teukolsky Benchmark Visuals

Use this skill for benchmark visualization scripts in this repo.

## Scope

- `test/domain/08_smoke_eval_atlas_predictor.py`
- `test/domain/09_random_scan_teukolsky_solver.py`

## Naming convention

Use `TeukolskySolver` in plot titles and legends, not `Atlas`, when the script is acting as the solver-facing benchmark.

## Plot layout

Current preferred single-sample comparison figure is `3 x 3`:

- column 1: `R(r)` comparison
  - `Re(R)`
  - `Im(R)`
  - `|R|`
- column 2: `R'(y)` comparison
  - `Re(R')`
  - `Im(R')`
  - `|R'|`
- column 3: errors
  - pointwise relative error for `R`
  - pointwise relative error for `R'`
  - text summary box

## Grid policy

Do not derive the right-column `y` grid from the left-column `r` grid.

Use two independent grids:

- left column: uniform `r` grid
- right column: independent `y` grid, usually Chebyshev

Expected CLI knobs:

- `--n-r`
- `--n-y`
- `--y-grid-mode`
- `--viz-r-min`
- `--viz-r-max`

## Error metrics

Store at least:

- `rel_R_l2`
- `rel_Rprime_l2`
- `abs_R_max`
- `abs_Rprime_max`
- `rel_R_pointwise_mean`
- `rel_Rprime_pointwise_mean`

Pointwise relative error convention:

- `|pred - ref| / (|ref| + eps)`

## Random scan behavior

For `09_random_scan_teukolsky_solver.py`:

- set global seeds for `random`, `numpy`, and `torch`
- sample from atlas chart support, not the raw rectangular parameter box
- keep one `MathematicaRinSampler` alive for the whole scan when possible
- only reset session on transport-level failures

## Repair mode

The script should support a repair pass over existing outputs:

- if a sample folder contains `benchmark_data.npz` and `summary.json`
- but does not contain the latest comparison figure
- regenerate the figure without rerunning MMA

Useful CLI:

- normal scan: `--n-samples ...`
- repair only: `--repair-only`

## Output expectations

Per sample:

- `summary.json`
- `benchmark_data.npz`
- `teukolsky_solver_vs_mma.png`
- if failed: `error.json`

Run-level:

- `random_scatter.png`
- `scan_records.json`
- `summary.json`

