# TeukSpecFiLM Patch Workflow Report

Date: 2026-06-05  
Branch: `research/high-accuracy-teukolsky-plan-20260528`  
Instruction source: `docs/instruction/teukspecfilm_implementation_plan_20260605.md`

## 1. Scope

This report records the work performed after the TeukSpecFiLM implementation instruction document. The goal was to move from monolithic experimental scripts toward a reusable patch-local spectral operator workflow:

```text
spectral branch teachers
  -> parameter-space Chebyshev decoders
  -> analytic multipoint amplitude solve
  -> optional frozen-decoder FiLM/Pirate residual corrector
```

The target mode remains:

```text
s = -2, l = m = 2,
a in a local patch,
omega in a local log-frequency patch,
R_in(r) with B_trans = 1.
```

## 2. Pure Spectral Status Recorded

A separate status note was added:

```text
docs/reports/pure_spectral_extreme_frequency_status_20260605.md
```

It records the current pure-spectral state before switching focus to TeukSpecFiLM:

- moderate frequency is reliable and suitable for the first patch-local decoder target;
- low frequency can be handled with far-field windows and AnMR, but amplitude scale separation is severe;
- high frequency still needs multi-domain/two-channel treatment rather than a single middle envelope;
- amplitudes should continue to be computed analytically by multipoint least squares, not by a neural head.

## 3. New Package Structure

A reusable `teukspec/` package was introduced.

Core modules:

```text
teukspec/core/chebyshev.py
teukspec/teachers/spectral_teacher.py
teukspec/decoders/param_cheb_decoder.py
teukspec/amplitude/multipoint_ls.py
teukspec/atlas/patch_expert.py
teukspec/correctors/film_pirate_corrector.py
```

Scripts added:

```text
scripts/fit_patch_param_cheb_decoders.py
scripts/validate_patch_solver.py
scripts/train_patch_corrector.py
scripts/evaluate_patch_vs_pybhpt.py
scripts/evaluate_patch_full_r_vs_pybhpt.py
```

The implementation keeps the old scripts intact and moves reusable logic into the new package.

## 4. First Moderate Patch Decoder

The first full patch decoder was trained with:

```text
a_center = 0.5
a_half_width = 0.12
logw_center = -1.5
logw_half_width = 0.25
n_param_side = 13
deg_a = 12
deg_logw = 12
N_z = 64
grid_kind = anmr
y_match = -0.25
match_width = 0.12
```

Output directory:

```text
outputs/param_cheb_patch_decoder/20260605_132655
```

### 4.1 Training-grid decoder fit

At the tensor-product parameter training nodes, all branches fit the teacher coefficient arrays near machine precision.

Representative summary:

```text
in   value_rel_median ~ 2.7e-15
down value_rel_median ~ 2.6e-15
up   value_rel_median ~ 1.0e-14
```

Teacher radial Chebyshev tails were also near double precision:

```text
in/down/up teacher tail medians ~ 1e-14 to 4e-14
```

### 4.2 Off-grid branch validation

Off-grid validation against direct spectral teachers gave:

```text
in:
  value_rel_median = 8.32e-13
  residual_median  = 5.32e-12

down:
  value_rel_median = 2.27e-13
  residual_median  = 2.80e-11

up:
  value_rel_median = 3.72e-5
  value_rel_max    = 1.26e-4
  residual_median  = 7.16e-6
  residual_max     = 9.22e-5
```

The first milestone passed for branch value/residual accuracy, but `up` was clearly the limiting branch.

## 5. FiLM/Pirate Corrector Trial

A gated residual corrector was implemented:

```text
teukspec/correctors/film_pirate_corrector.py
scripts/train_patch_corrector.py
```

Important design choices:

- this is not an autoencoder;
- the parameter-space Chebyshev decoder is frozen;
- the corrector starts from exact zero output;
- hard gates preserve boundary normalization:
  - `in`: `(1-z)^2`
  - `down/up`: `z^2`
- loss uses reduced-equation relative residual plus correction-size and Sobolev regularization;
- no teacher anchor is used in corrector training.

A short `up`-branch trial produced:

```text
outputs/param_cheb_patch_decoder/20260605_132655/correctors/up_20260605_203846
```

Final evaluation:

```text
initial before_median = 2.416e-6
final after_median    = 2.298e-6
initial before_max    = 1.023e-4
final after_max       = 1.063e-4
correction median     = 4.45e-8 relative to u_spec
correction max        = 3.01e-6 relative to u_spec
```

Conclusion: the current corrector did not produce a robust net improvement. Continuing to add steps is not justified until the decoder derivative evaluation and the training target are improved.

Implementation issue discovered:

- `eval_u_derivatives` is currently too slow because derivative evaluation is effectively pointwise and repeatedly reconstructs derivative information.
- Cache-based training was added to reduce repeated spectral-decoder calls, but the cache construction is still slow.
- Vectorizing decoder derivative evaluation is required before larger corrector experiments are efficient.

## 6. PyBHPT Evaluation

### 6.1 In-domain `R_in` validation

A pybhpt comparison script was added:

```text
scripts/evaluate_patch_vs_pybhpt.py
```

It evaluates only the `in` branch's valid radial domain:

```text
R_in = A_in u_in
```

For a 5x5 parameter grid over the patch, the result was:

```text
outputs/param_cheb_patch_decoder/20260605_132655/pybhpt_eval/20260605_210048
```

Summary:

```text
n_success = 25
n_failure = 0
rel_median_over_params = 1.44e-12
rel_p90_over_params    = 1.45e-12
rel_max_over_params    = 3.96e-12
```

This confirms that the `in` branch is not the current bottleneck.

Important caveat: some early pybhpt grids included parameter training nodes. Future acceptance evaluations must use strictly off-training parameter points and random/nontraining radial points.

### 6.2 Full-r piecewise validation

A full-r pybhpt comparison script was added:

```text
scripts/evaluate_patch_full_r_vs_pybhpt.py
```

It forms the piecewise physical solution:

```text
inner region: R = A_in u_in
outer region: R = B_inc A_down u_down + B_ref A_up u_up
```

with amplitudes from analytic multipoint LS.

For a 3x3 grid, the result was:

```text
outputs/param_cheb_patch_decoder/20260605_132655/full_r_pybhpt_eval/20260605_212755
```

Summary:

```text
n_success = 9
n_failure = 0
rel_median_over_params = 1.06e-9
rel_p90_over_params    = 1.29e-9
rel_max_over_params    = 3.74e-8
outer_median_over_params = 1.08e-9
inner_median_over_params = 1.48e-12
amp_ls_residual_median = 3.74e-10
amp_ls_residual_max    = 2.13e-8
```

This shows that the full-r physical solution can look good on regular parameter grids, but this does not by itself prove true off-grid amplitude accuracy.

The script was then updated to support:

```text
--sample-kind random
--n-param-random <N>
--seed <seed>
```

so future evaluations can avoid training nodes.

## 7. Amplitude Accuracy Diagnostics

Since pybhpt does not expose `B_inc/B_ref` through the current wrapper, amplitude accuracy was checked against direct spectral teachers on the same patch.

For the large patch:

```text
outputs/param_cheb_patch_decoder/20260605_132655/amplitude_teacher_eval/summary.json
```

The 7x7 parameter test produced:

```text
Binc_relerr_median = 9.62e-2
Binc_relerr_max    = 7.65e-1
Bref_relerr_median = 9.63e-2
Bref_relerr_max    = 7.65e-1
pred_ls_residual_median = 1.79e-2
pred_ls_residual_max    = 1.30e-1
cond_median = 2.85e5
cond_max    = 7.19e6
```

This is the most important failure found in this phase.

Interpretation:

- branch values can be accurate enough locally, especially for `in/down`;
- the `up` branch has larger parameter interpolation error;
- amplitude LS is ill-conditioned, often `1e5` to `1e6+`;
- therefore small parameter interpolation errors are amplified into large `B_inc/B_ref` errors;
- training-node performance is not representative for amplitude recovery.

## 8. Refined Smaller Patch Test

To test whether patch splitting is the correct direction, a smaller patch was trained:

```text
a_center = 0.5
a_half_width = 0.06
logw_center = -1.5
logw_half_width = 0.125
n_param_side = 9
deg_a = 8
deg_logw = 8
N_z = 64
```

Output directory:

```text
outputs/param_cheb_patch_decoder_refine/20260605_212912
```

Off-grid branch validation:

```text
in:
  value_rel_median = 4.29e-12
  residual_median  = 1.16e-11

down:
  value_rel_median = 1.02e-12
  residual_median  = 2.67e-11

up:
  value_rel_median = 3.83e-6
  value_rel_max    = 1.67e-5
  residual_median  = 3.25e-7
  residual_max     = 1.20e-6
```

Amplitude teacher comparison:

```text
outputs/param_cheb_patch_decoder_refine/20260605_212912/amplitude_teacher_eval/summary.json
```

Summary:

```text
Binc_relerr_median = 4.15e-3
Binc_relerr_max    = 1.74e-2
Bref_relerr_median = 4.15e-3
Bref_relerr_max    = 1.74e-2
pred_ls_residual_median = 7.80e-4
pred_ls_residual_max    = 3.02e-3
cond_median = 2.85e5
cond_max    = 1.42e6
```

Patch splitting improved the amplitude error by roughly one to two orders of magnitude, but still does not reach the desired `1e-4` level.

## 9. Current Bottleneck

The current bottleneck is:

```text
parameter off-grid amplitude recovery, dominated by up-branch interpolation and LS conditioning.
```

It is not:

```text
in-branch local accuracy.
```

The difficulty is structural:

1. `B_inc` is very large in this frequency range while `B_ref` is moderate.
2. The two-column amplitude LS can have condition number `1e5` to `1e6+`.
3. The `up` branch is the hardest regular factor to interpolate over `(a, log10 omega)`.
4. Small off-grid branch errors are amplified into large amplitude errors.
5. Regular parameter grids can hide this problem because they include training nodes or favorable structured points.

## 10. Required Next Steps

The immediate next steps should be:

1. Use strictly off-training random parameter points and random/nontraining radial points for pybhpt/full-r validation.
2. Continue splitting patches, especially in `log10 omega`, until amplitude teacher error approaches `1e-4`.
3. Track parameter-space Chebyshev coefficient tails for each branch; if tails do not decay, split rather than increasing degree blindly.
4. Improve `up` branch representation first:
   - smaller patch,
   - higher parameter resolution for `up` only,
   - or branch-specific parameter decoder degree.
5. Vectorize decoder derivative evaluation before running large FiLM/Pirate corrector studies.
6. Keep amplitudes analytic by multipoint zero-order LS; do not introduce a neural amplitude head.

## 11. Validation Commands Used

Main patch fit:

```bash
python scripts/fit_patch_param_cheb_decoders.py \
  --basis all \
  --a-center 0.5 --a-half-width 0.12 \
  --logw-center -1.5 --logw-half-width 0.25 \
  --n-param-side 13 --deg-a 12 --deg-logw 12 \
  --n 64 --grid-kind anmr \
  --y-match -0.25 --match-width 0.12 \
  --output-dir outputs/param_cheb_patch_decoder
```

Main patch validation:

```bash
python scripts/validate_patch_solver.py \
  --patch-dir outputs/param_cheb_patch_decoder/20260605_132655 \
  --n-offgrid 49
```

Full-r pybhpt validation:

```bash
python scripts/evaluate_patch_full_r_vs_pybhpt.py \
  --patch-dir outputs/param_cheb_patch_decoder/20260605_132655 \
  --n-param-side 3 --n-z 120 --z-min 1e-3 --timeout 45
```

Smaller patch fit:

```bash
python scripts/fit_patch_param_cheb_decoders.py \
  --basis all \
  --a-center 0.5 --a-half-width 0.06 \
  --logw-center -1.5 --logw-half-width 0.125 \
  --n-param-side 9 --deg-a 8 --deg-logw 8 \
  --n 64 --grid-kind anmr \
  --y-match -0.25 --match-width 0.12 \
  --output-dir outputs/param_cheb_patch_decoder_refine
```

## 12. Summary

The TeukSpecFiLM direction is still valid, but the acceptance criterion must be sharpened:

- branch value/residual accuracy alone is not enough;
- full-r pybhpt agreement on structured grids is not enough;
- amplitude accuracy on strictly off-training parameter points is the controlling metric for this patch.

The best current engineering direction is patch splitting plus stronger `up`-branch parameter representation, followed by random off-grid pybhpt/full-r validation and teacher-amplitude validation.
