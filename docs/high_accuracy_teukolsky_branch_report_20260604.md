# High-Accuracy Teukolsky Solver Branch Report

Date: 2026-06-04

Branch: `research/high-accuracy-teukolsky-plan-20260528`

Repository path: `/home/ljq/code/PINN/SolvingTeukolskyEq_autoencoder`

## 1. Goal of This Branch

This branch was created to reassess the current PINN/autoencoder approach and build a credible path toward a high-accuracy, high-speed solver for the Kerr Teukolsky radial equation at

- spin weight `s=-2`,
- fixed `l=m=2`,
- parameters `a∈[0,1)`, `ω∈(1e-4,10]`,
- radial domain `r∈[r_+,∞)`.

The desired final product is a fast surrogate that returns the ingoing homogeneous radial solution `R_in(r)` and the far-field amplitudes

- `B^inc` multiplying `r^{-1} exp(-iωr*)`,
- `B^ref` multiplying `r^3 exp(+iωr*)`,
- with `B^trans=1` at the horizon.

The final accuracy target is not agreement with a training loss or anchor loss, but the normalized Teukolsky equation residual

```text
|equation residual| / max(|second-derivative term|, |first-derivative term|, |zeroth-order term|)
```

using the solution itself, without removing its physical normalization.

## 2. Starting Point and Initial Problems

The branch started from the existing autoencoder/PINN Stage-1 codebase. The immediate context was the observation that Adam pretraining followed by L-BFGS refinement could collapse rapidly to a trivial or near-trivial solution.

A representative problematic run was

```text
outputs/stage1_lbfgs_refine/20260526_205009_patch_000_lbfgs
```

where the Adam checkpoint was loaded and then L-BFGS refinement caused the predicted solution to lose amplitude structure quickly. The key observations were:

1. The homogeneous Teukolsky equation admits scale degeneracies unless the normalization and regular branch constraints are enforced robustly.
2. A PINN can satisfy residual terms locally while drifting toward a physically wrong solution when the loss is ill-conditioned.
3. The `S` ansatz used near the horizon fixes formal boundary values, but numerical propagation of errors through a degenerate boundary equation can still destabilize the learned solution.
4. Adding simple anchor terms is undesirable for the final solver because the desired method should solve independently, not be permanently tied to pointwise truth data.

This led to a shift away from unconstrained direct PINNs toward spectral representations of regularized branch functions.

## 3. Benchmark and Convention Work

The branch established benchmark conventions and cross-checking scripts for pybhpt, Mathematica/MMA, and GSN.

Important files:

- `docs/reference_conventions.md`
- `scripts/cross_validate_benchmarks_omega01.py`
- `scripts/extract_highw_bref_from_gsn_profile.py`

Benchmark conclusions:

1. `pybhpt` is reliable for low and moderate frequencies and gives radial solution values, but does not expose all amplitude coefficients needed here.
2. Mathematica/MMA can provide amplitudes and is useful around lower/moderate frequency, but calling it from WSL requires care with Windows kernels and kernel shutdown.
3. GSN is useful for high-frequency amplitude checks, especially around `ω=10`.
4. Around `ω≈0.1`, cross-validation between available tools is the most useful anchor for convention checks.

A high-frequency reference used repeatedly was

```text
a=0.5, ω=10, s=-2, l=m=2
B_inc ≈ 24.675049712429978 - 5.00013326071889 i
B_ref ≈ -6.457956617151793e-09 - 8.144599106575082e-09 i
λ ≈ -26.85215496939334
```

## 4. Spectral Method Investigation

A major part of the branch was devoted to understanding whether pure spectral methods could solve the branch functions and amplitudes accurately.

Important files:

- `utils/amplitude.py`
- `utils/matlcheb.py`
- `utils/GF_adaptive_match.m`
- `utils/MatlCheb-main.zip`
- `scripts/schwarzschild_gf_adaptive_match.py`
- `scripts/kerr_s0_spectral_match.py`
- `scripts/diagnose_spectral_conditioning.py`
- `scripts/spectral_marching_farfield_s2.py`
- `scripts/three_patch_raw_mp_value_fit.py`

### 4.1 Schwarzschild / MatlCheb Reference

The MATLAB script `utils/GF_adaptive_match.m` and the `MatlCheb-main.zip` utilities were used as a guide. The key transferable ideas were:

1. Use Chebyshev-Gauss-Lobatto collocation.
2. Split the physical domain into inner and outer pieces.
3. At low frequency, use analytic mesh refinement (AnMR), implemented as a sinh-type map.
4. Perform matching at a frequency-dependent or manually selected intermediate compactified coordinate.
5. Inspect Chebyshev coefficient decay as a first-class convergence diagnostic.

This motivated the Python port `utils/matlcheb.py` and the additions to `utils/amplitude.py`.

### 4.2 AnMR and Conditioning

The branch added AnMR support to `utils/amplitude.py`:

- `cheb_D_sinh`
- `adaptive_match_z`
- `_domain_D`
- row/column equilibrated linear solve
- optional `grid_kind={linear,anmr,auto}`

This fixed an important problem: at low frequency, the outer `down` branch on a linear `z` grid had poor Chebyshev tail decay. For the current local patch, representative numbers were:

```text
linear N=40 down_tail_rel median ≈ 1e-7
AnMR  N=64 down_tail_rel median ≈ 1e-14
```

This is close to double-precision roundoff. The result confirmed that the spectral method itself can produce smooth branch functions when the coordinate map is appropriate.

## 5. Regular Branch Variables and Ansatz Corrections

A substantial conceptual correction was made around the horizon branch.

The old PINN variable `S` was related to the physical ingoing solution by a Leaver-style prefactor. For high frequency, `S` can show strong oscillation even when the physically more natural regular factor is smooth.

The better variable for the ingoing branch is

```text
u_in = R_in / A_in
A_in = Δ^2 exp(-i k r*)
```

For the far-field branches,

```text
u_down = R_down / A_down,   A_down = r^{-1} exp(-iωr*)
u_up   = R_up   / A_up,     A_up   = r^3 exp(+iωr*)
```

Important diagnostic scripts:

- `scripts/analyze_branch_u_convergence.py`
- `scripts/match_inA_spectral_overlap_amplitudes.py`
- `scripts/match_inA_mp_overlap_amplitudes.py`

The high-frequency `ω=10` diagnostic showed:

1. `S` oscillations are not necessarily plotting bugs; they can be induced by the prefactor choice.
2. `u_in=R/A_in` is much smoother and has better Chebyshev convergence.
3. Direct two-domain overlap can recover `B_inc` well in some settings but tends to collapse `B_ref` toward zero when `B_ref` is many orders of magnitude smaller than `B_inc`.

## 6. High-Frequency Amplitude Extraction Attempts

The branch explored several strategies for extracting `B^ref` at high frequency.

Important outputs:

- `outputs/inA_spectral_overlap_amplitude_match/*`
- `outputs/inA_mp_overlap_amplitude_match/*`
- `outputs/three_patch_raw_mp_value_fit/*`

Important scripts:

- `scripts/match_inA_spectral_overlap_amplitudes.py`
- `scripts/match_inA_mp_overlap_amplitudes.py`
- `scripts/three_patch_raw_mp_value_fit.py`

### 6.1 Direct Overlap

Using correct `u_in=R/A_in`, direct overlap matching near intermediate/far regions could give excellent `B_inc`, but `B_ref` often collapsed to values close to zero. Moving farther toward infinity increased sensitivity to the reflected channel but made the propagated ingoing solution numerically unstable.

### 6.2 Three-Patch Matching

A three-patch approach was tested:

1. Solve `u_in` near the horizon.
2. Propagate through a middle raw or phase-extracted region.
3. Fit against `up/down` near infinity with multipoint value-only least squares.

Variants included:

- raw middle propagation,
- right-boundary middle BVP,
- middle extraction of `exp(+iωr*)`,
- middle extraction of `exp(-iωr*)`.

The phase-extracted tests did not solve the high-frequency amplitude problem. Representative runs:

```text
outputs/three_patch_raw_mp_value_fit/20260601_101554  # + exp(iωr*)
outputs/three_patch_raw_mp_value_fit/20260601_101929  # - exp(iωr*)
```

Both showed poor fit residuals and large amplitude errors. The conclusion was that a single extracted phase is insufficient because the middle region contains a mixture of two channels. A future pure spectral method may need a two-channel WKB/phase basis or a better conditioned amplitude invariant.

## 7. Spectral-PINN and Coefficient-Surrogate Work

The branch then shifted toward a spectral surrogate strategy:

1. Use spectral collocation to generate teacher Chebyshev coefficients for regular branch functions.
2. Train a network or decoder to map `(a, logω)` to branch coefficients.
3. Evaluate branch functions quickly via Chebyshev series.
4. Compute amplitudes by analytic multipoint least-squares matching, not by a neural amplitude head.

Important scripts:

- `scripts/train_horizon_spectral_coeff.py`
- `scripts/train_infinity_spectral_coeff.py`
- `scripts/train_patch_spectral_decoder.py`
- `scripts/train_up_spectral_envelope.py`
- `scripts/fit_up_param_cheb_decoder.py`

### 7.1 Horizon and Infinity Local Tests

Local spectral coefficient training showed that branch functions are indeed smooth in suitable variables. However, generic MLPs had difficulty learning the coefficient field uniformly over even a moderate parameter patch, especially for `u_up`.

### 7.2 Patch Spectral Decoder Prototype

The combined patch decoder was implemented in

```text
scripts/train_patch_spectral_decoder.py
```

It generated AnMR teacher coefficients for `in/down/up` over a patch centered at

```text
a=0.5, log10(ω)=-1.5
```

with patch widths roughly

```text
a∈[0.38,0.62]
log10(ω)∈[-1.75,-1.25]
```

Representative output:

```text
outputs/patch_spectral_decoder/20260602_181328
```

Key numbers:

```text
teacher Chebyshev tails:
  in   median ≈ 2.09e-14
  down median ≈ 1.38e-14
  up   median ≈ 4.86e-14

network value_rel_rms:
  in   median ≈ 2.51e-3, max ≈ 4.75e-3
  down median ≈ 2.99e-4, max ≈ 4.75e-4
  up   median ≈ 1.60e-1, max ≈ 3.24e-1
```

The teacher was spectrally converged, but the ordinary neural coefficient map was not. In particular, the `up` branch was the dominant bottleneck.

### 7.3 Decision: No Neural Amplitude Head

An amplitude head was initially tested, but this is no longer the recommended path. The user explicitly clarified that amplitudes should be computed by a fast analytic solve, not by a neural network. The branch now treats the neural/spectral surrogate as a branch-function generator only.

The correct downstream strategy is:

```text
network/spectral decoder -> u_in, u_down, u_up -> analytic multipoint LS -> B_inc, B_ref
```

## 8. Why Ordinary MLPs Failed for Spectral Coefficients

The core issue is that Chebyshev coefficients should decay geometrically for analytic functions:

```text
|c_k| ~ C ρ^{-k}
```

A plain MLP directly outputting a coefficient vector does not enforce this structure. It can fit low-order modes while producing no meaningful natural spectral tail. Artificially zero-padding high modes is not true spectral convergence.

The branch tested a stronger MLP with

- per-mode coefficient normalization,
- full coefficient output,
- Fourier features in parameter space,
- full-patch training,
- worst-case penalty,
- value-space loss,
- tail-weighted loss.

Script:

```text
scripts/train_up_spectral_envelope.py
```

Representative output:

```text
outputs/up_spectral_envelope/20260603_205513
```

Result:

```text
up value_rel_l2 median ≈ 3.32e-2
up value_rel_l2 max    ≈ 5.96e-2
```

This improved over the first MLP but still missed the desired accuracy by orders of magnitude.

## 9. Successful Parameter-Space Chebyshev Decoder

The breakthrough was to exploit analyticity in parameter space directly. Instead of using a generic MLP, the branch added a tensor-product Chebyshev decoder over `(a, log10ω)`.

Script:

```text
scripts/fit_up_param_cheb_decoder.py
```

This fits the map

```text
(a, log10ω) -> {Chebyshev coefficients in y}
```

using a tensor-product Chebyshev expansion in the parameter patch.

Representative output:

```text
outputs/up_param_cheb_decoder/20260603_210633
```

Training-node results:

```text
coeff_rel_median ≈ 7.23e-15
coeff_rel_max    ≈ 1.39e-12
value_rel_median ≈ 7.53e-15
value_rel_max    ≈ 1.38e-12
teacher_tail_median ≈ 3.99e-14
teacher_tail_max    ≈ 3.99e-13
```

Off-grid 7×7 validation inside the patch:

```text
offgrid_coeff_rel_median ≈ 5.04e-5
offgrid_coeff_rel_max    ≈ 2.48e-4
offgrid_value_rel_median ≈ 5.00e-5
offgrid_value_rel_max    ≈ 2.47e-4
```

Figures:

```text
outputs/up_param_cheb_decoder/20260603_210633/figures/up_param_cheb_vs_teacher_coeff_decay_center.png
outputs/up_param_cheb_decoder/20260603_210633/figures/up_parameter_cheb_coeff_decay.png
outputs/up_param_cheb_decoder/20260603_210633/figures/up_offgrid_value_rel_error_heatmap.png
```

This is the first approach on this branch that achieves strong uniform accuracy for the difficult `up` branch on the test patch.

## 10. Current Best Understanding

### 10.1 What Works

1. Regular branch variables `u_in`, `u_down`, `u_up` are the right objects for spectral representation.
2. AnMR is essential at low frequency for outer-domain spectral convergence.
3. Direct spectral collocation gives teacher branch coefficients with tails near double precision.
4. Parameter-space Chebyshev decoders are far better than generic MLPs for this patch.
5. Amplitudes should be computed by analytic multipoint matching, not learned directly.

### 10.2 What Does Not Yet Work

1. Direct PINNs remain vulnerable to trivial-solution collapse and poor conditioning.
2. Generic MLP coefficient heads do not naturally learn exponentially decaying Chebyshev coefficient sequences.
3. High-frequency `B^ref` extraction remains difficult when `|B^ref/B^inc|` is extremely small.
4. Single-phase middle propagation is not enough for high-frequency amplitude extraction.
5. Current tensor-product Chebyshev decoder has only been demonstrated on one moderate patch, not the full target domain.

## 11. Important Files Added or Modified

### Documentation

- `docs/high_accuracy_teukolsky_solver_strategy_20260528.md`
- `docs/reference_conventions.md`
- `docs/high_accuracy_teukolsky_branch_report_20260604.md`
- `notes/spectral_surrogate_lessons.md`

### Spectral utilities

- `utils/amplitude.py`
- `utils/matlcheb.py`
- `utils/GF_adaptive_match.m`
- `utils/MatlCheb-main.zip`

### Benchmark and spectral diagnostics

- `scripts/cross_validate_benchmarks_omega01.py`
- `scripts/extract_highw_bref_from_gsn_profile.py`
- `scripts/diagnose_spectral_conditioning.py`
- `scripts/schwarzschild_gf_adaptive_match.py`
- `scripts/kerr_s0_spectral_match.py`
- `scripts/analyze_branch_u_convergence.py`

### Amplitude extraction experiments

- `scripts/match_inA_spectral_overlap_amplitudes.py`
- `scripts/match_inA_mp_overlap_amplitudes.py`
- `scripts/three_patch_raw_mp_value_fit.py`
- `scripts/spectral_marching_farfield_s2.py`

### Spectral surrogate experiments

- `scripts/test_horizon_local_spectral.py`
- `scripts/test_horizon_local_spectral_pinn.py`
- `scripts/train_horizon_spectral_coeff.py`
- `scripts/train_horizon_spectral_teacher.py`
- `scripts/train_infinity_spectral_coeff.py`
- `scripts/train_patch_spectral_decoder.py`
- `scripts/train_up_spectral_envelope.py`
- `scripts/fit_up_param_cheb_decoder.py`

## 12. Recommended Next Steps

1. Replace `in/down/up` MLP coefficient heads with parameter-space Chebyshev decoders.
2. Extend the current successful `up` decoder approach to `in` and `down`.
3. Use off-grid validation as the primary surrogate check for each parameter patch.
4. Use analytic multipoint least-squares to compute `B_inc` and `B_ref` from decoded branches.
5. Test amplitude recovery again on the current moderate patch after all three branch decoders are accurate.
6. Only then expand to a patch cover in `(a,logω)`.
7. For extreme `ω≈1e-4` and `ω≈10`, revisit conditioning and possibly introduce different asymptotic variables or multi-channel phase bases.

## 13. Summary

The branch began as an attempt to repair a direct PINN/L-BFGS training pipeline. The work showed that direct PINNs are poorly conditioned for this problem and can collapse or learn physically wrong homogeneous solutions. The project then moved toward a spectral representation of analytically regular branch functions.

The most important technical result so far is that the regular branch functions have excellent Chebyshev convergence when using appropriate variables and AnMR grids, and that a tensor-product Chebyshev decoder in parameter space can reproduce the difficult `up` branch on a moderate patch with off-grid relative errors below `3e-4`.

The current direction is therefore:

```text
spectral branch teachers -> parameter-space Chebyshev decoders -> analytic multipoint amplitude solve
```

not

```text
direct PINN residual training -> neural amplitude head
```
