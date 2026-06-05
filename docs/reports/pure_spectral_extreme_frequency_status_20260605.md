# Pure Spectral Extreme-Frequency Status

Date: 2026-06-05  
Branch: `research/high-accuracy-teukolsky-plan-20260528`

## Scope

This note records the current state of the pure spectral Teukolsky experiments before switching implementation focus to `spec-pinn + FiLM-MLP` / TeukSpecFiLM.

Fixed mode unless otherwise stated:

- `s = -2`
- `l = m = 2`
- Kerr `a = 0.5`
- horizon normalization `B_trans = 1`
- branch convention:
  - `R_in = A_in u_in`, `A_in = Δ^2 exp(-i k r*)`
  - `R_down = A_down u_down`, `A_down = r^-1 exp(-i ω r*)`
  - `R_up = A_up u_up`, `A_up = r^3 exp(+i ω r*)`

The final validation metric remains the reduced equation relative residual

```text
|B2 u_zz + B1 u_z + B0 u| / max(|B2 u_zz|, |B1 u_z|, |B0 u|).
```

## Moderate Frequency Baseline

For `ω ≈ 0.1`, the current pure spectral branch solve is reliable.

Observed status:

- `A_in u_in` agrees with external references in its valid inner domain to approximately double-precision/near-double-precision levels.
- Outer `A_down/A_up` branch combination agrees with reference `R_in` when the reference amplitudes are used.
- Raw `R` middle-domain four-boundary spectral solve works for `ω=0.1` with median relative error around `4.6e-11` in the previous test.

Conclusion: the branch equations, analytic factors, and amplitude convention are consistent in the moderate-frequency regime.

## High Frequency, `ω = 10`

### What works

Near-boundary spectral branch solves are locally accurate:

- inner `A_in u_in` matches GSN well in the inner reliable region;
- outer `A_down/A_up` branches are reliable only sufficiently far out;
- coefficient tails can reach near machine precision in local regions, especially for the `in` branch.

GSN amplitude values used in diagnostics:

```text
B_inc = 24.675049712429978 - 5.00013326071889 i
B_ref = -6.457956617151793e-09 - 8.144599106575082e-09 i
B_trans = 1
λ = -26.85215496939334
```

### What fails

A single extracted-phase middle variable fails at high frequency. The middle region contains a mixture of incoming/outgoing channels; forcing it into one envelope creates an unstable or inaccurate representation.

Key observations:

- `R/P` with the Leaver factor does not solve the high-frequency bridge problem.
- `P_up` is worse than the standard `A_up` envelope for the tested high-frequency outer branch.
- Single-sided Cauchy spectral propagation in the middle is unstable.
- Even two-end value BVP with `P` improves the issue but still does not reach the required accuracy for `ω=10`.
- Raw `R` middle-domain four-boundary solve improves with increasing `N`, but still requires more resolution or subdivision at `ω=10`.

Representative previous results:

```text
raw R middle solve, ω=10:
N=500 median relerr ~ 3.9e-1
N=650 median relerr ~ 2.1e-1
N=800 median relerr ~ 1.3e-1
```

### Current high-frequency diagnosis

The failure is not primarily a branch-factor algebra error. It is a representation/conditioning issue:

- oscillatory middle-region physics is genuinely two-channel;
- `B_ref` can be many orders smaller than `B_inc`, making amplitude extraction sensitive;
- near-infinity and middle coefficient matrices become strongly convection/reaction dominated;
- reliable inner and outer spectral domains may not overlap at high frequency.

Recommended pure-spectral direction for high frequency:

1. use three or more radial subdomains;
2. represent the middle by a two-channel phase basis, at minimum `A_up` and `A_down`;
3. avoid neural amplitude heads;
4. select subdomain interfaces by residual, Abel/Wronskian consistency, and LS amplitude residual;
5. use high-precision or strongly equilibrated LS when `|B_ref/B_inc|` is tiny.

## Low Frequency, `ω = 1e-4`

### Corrected latest result

The earlier conclusion that `up/down + MMA amplitudes` failed was due to an MMA `InputForm` parsing bug: each sample row contained duplicate `r` and an extra `{0,0,0}` vector, and the old parser grouped numbers incorrectly.

After fixing the parser, the result is consistent.

MMA amplitudes:

```text
B_inc = -1.543534727466664e20 + 3.4164322422445716e20 i
B_ref = 10210.782613860754 - 22813.76866596801 i
B_trans = 1
```

Best tested outer solve / zero-order multipoint fit:

```text
z1 = 0.002
Nout = 120
points = 96
B_inc relative error = 2.82e-5
B_ref relative error = 8.74e-9
fit relative L2 = 3.90e-8
condition number = 1.04
```

Artifacts:

```text
outputs/omega1e4_zero_order_outer_match/20260605_a05/summary.json
outputs/omega1e4_zero_order_outer_match/20260605_a05/zero_order_fit_results.csv
outputs/omega1e4_zero_order_outer_match/20260605_a05/figures/mma_combo_relerr_vs_Nout.png
outputs/omega1e4_zero_order_outer_match/20260605_a05/figures/zero_order_fit_ranked_errors.png
```

### Low-frequency diagnosis

The `up/down` factors and MMA amplitude convention are consistent. The usable outer fitting window must remain very far out. Pushing the outer domain too deep degrades amplitude extraction.

Low-frequency difficulty comes from scale separation:

```text
|B_inc| ~ 3.75e20
|B_ref| ~ 2.5e4
```

This makes physical `R_in` dominated by the incoming component, while the reflected component is many orders smaller. Stable extraction therefore requires:

- far-field windows;
- AnMR/adaptive outer grids;
- row/column-scaled multipoint zero-order LS;
- avoiding derivative matching unless derivative benchmarks are independently validated.

## Current Pure-Spectral Status

- Moderate frequency: reliable enough to serve as the first TeukSpecFiLM development target.
- Low frequency: outer `up/down` branch plus zero-order multipoint LS is viable if the window is far enough and parsing/reference conventions are handled carefully.
- High frequency: local branch solves are useful, but a global pure-spectral bridge still needs multi-domain/two-channel treatment.

## Handoff to TeukSpecFiLM

Implementation should now focus on the moderate patch first:

```text
a_center = 0.5
logw_center = -1.5
a_half_width = 0.12
logw_half_width = 0.25
N_z = 64
n_param_side = 9
deg_a = deg_logw = 8
```

The first milestone is not a global solver. It is:

1. build all-branch parameter-space Chebyshev decoders;
2. validate off-grid branch value error and reduced residual;
3. compute amplitudes by analytic multipoint zero-order LS;
4. only then add a frozen-decoder FiLM/Pirate residual corrector.
