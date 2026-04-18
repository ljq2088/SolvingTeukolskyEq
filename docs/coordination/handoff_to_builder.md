# Handoff to Builder (agent2)

**Issued by**: agent1  
**Date**: 2026-04-19  
**Task ID**: TASK-001

---

## Objective
Verify that the existing data generation pipeline (`utils/amplitude_ratio.py` → `kerr_matcher`) produces correct Kerr scattering amplitudes, and write a clean single-point generation script.

## Why This Matters
Before generating any training dataset, we must confirm the numerical solver is correct. The reference value from pybhpt is known. If kerr_matcher disagrees, we need to fix it before wasting compute on a bad dataset.

## Scope
1. In WSL, run `utils/amplitude_ratio.py` for the reference point: `a=0.1, ω=0.1, l=2, m=2, s=-2`
2. Compare `A_ref/A_inc` against the pybhpt reference: `|A_ref/A_inc| ≈ 1.000025`
3. If kerr_matcher is broken or unavailable, fall back to `pybhpt` (conda env `gw_ml_env`) as the data source
4. Write `scripts/generate_one.py` that:
   - Takes (a, omega, l, m) as CLI args
   - Computes λ via `utils/compute_lambda.py`
   - Computes A_ref/A_inc
   - Prints result as JSON: `{"a": ..., "omega": ..., "l": ..., "m": ..., "lambda": ..., "A_ref_re": ..., "A_ref_im": ..., "A_inc_re": ..., "A_inc_im": ...}`
5. Commit `scripts/generate_one.py` to master

## Constraints
- Do NOT modify `utils/compute_lambda.py` or `utils/amplitude_ratio.py` without agent1 approval
- If kerr_matcher is unavailable, document this clearly and use pybhpt as fallback — do not silently substitute
- Script must run in WSL Ubuntu-22.04-D

## Required Checks
- [ ] `compute_lambda(a=0.1, omega=0.1, l=2, m=2, s=-2)` returns λ ≈ 3.9334
- [ ] Amplitude solver returns `|A_ref/A_inc|` within 1% of pybhpt reference
- [ ] `scripts/generate_one.py` runs without error and prints valid JSON

## Acceptance Criteria
- `generate_one.py` committed to master
- Output for reference point matches pybhpt to 1% or better
- Any discrepancy documented in `docs/coordination/handoff_from_builder.md`

## Reference Values (pybhpt, M=1)
```
a=0.1, omega=0.1, l=2, m=2, s=-2
lambda = 3.93335973
|A_ref| ≈ 17.04
|A_ref/A_inc| ≈ 1.000025
A_ref ≈ -11.51 - 12.57j
```
