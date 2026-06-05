# Spectral experiments on Leaver `P` factor, middle propagation, and high-frequency matching

Date: 2026-06-05

Branch: `research/high-accuracy-teukolsky-plan-20260528`

Repository path: `/home/ljq/code/PINN/SolvingTeukolskyEq_autoencoder`

## Scope

This report records the recent pure-spectral numerical experiments around the Leaver factor `P`, the high-frequency `ω=10` failure mode, and the attempted three-segment matching strategy for Kerr Teukolsky `s=-2, l=m=2`.

The experiments were designed to answer four questions:

1. Whether replacing the `up` or `in` asymptotic factors by the Leaver factor `P` improves high-frequency spectral convergence.
2. Whether `R_in/P` can be propagated from the horizon toward the far-field region and then used to extract `B^inc` and `B^ref`.
3. Whether the failed middle propagation is caused by an incorrect interface conversion or by a wrong `P`-reduced equation.
4. Whether the coefficient structure of the `P`-reduced equation explains the observed numerical instability.

The main conclusion is negative: `P` is useful as an endpoint asymptotic factor, but it is not a good middle propagation variable for extracting high-frequency amplitudes. The middle equation becomes strongly reaction/convection dominated, and single-sided Cauchy-type Chebyshev collocation is numerically unstable even at intermediate frequency.

## Conventions

The radial equation used in these tests is

```text
Δ R_rr - Δ_r R_r + V R = 0,
V = (K^2 + 4 i (r-M) K) / Δ - 8 i ω r - λ
```

with `s=-2`.

The physical branch factors are

```text
A_in   = Δ^2 exp(-i k r*)
A_down = r^-1 exp(-i ω r*)
A_up   = r^3 exp(+i ω r*)
```

and the Leaver factor is

```text
P(r) = exp(i ω r) (r-r_+)^pp (r-r_-)^pm
pp = -s - i σ
pm = -1 - s + 2 i ω + i σ
σ = (2 ω r_+ - a m) / (r_+ - r_-)
```

For `R=Pψ`, with `q=P_r/P`, the `z=r_+/r` equation is

```text
B2 ψ_zz + B1 ψ_z + B0 ψ = 0
B2 = Δ z_r^2
B1 = Δ (z_rr + 2 q z_r) - Δ_r z_r
B0 = Δ (q_r + q^2) - Δ_r q + V
```

The interface conversion from the inner `A_in` variable to `P` is

```text
ψ = (A_in/P) u_in
ψ_z = (A_in/P) [u_in,z + z_r (q_A - q_P) u_in]
q_A = A_in,r / A_in
q_P = P_r / P
```

This conversion was verified numerically at `z=0.4`: for both `ω=10` and `ω=0.1`, the relative mismatch in `R` and `R_z` was at roundoff level.

## Experiment 1: `P_up` and `P_in` basis vs original `A` basis

Output directory:

```text
outputs/leaver_P_basis_w10/20260605_a05_w10_analytic_bc
```

Key files:

```text
outputs/leaver_P_basis_w10/20260605_a05_w10_analytic_bc/summary.json
outputs/leaver_P_basis_w10/20260605_a05_w10_analytic_bc/basis_compare.csv
outputs/leaver_P_basis_w10/20260605_a05_w10_analytic_bc/N_sweep_tail.csv
outputs/leaver_P_basis_w10/20260605_a05_w10_analytic_bc/figures/P_nonunit_vs_A_coeff_decay.png
outputs/leaver_P_basis_w10/20260605_a05_w10_analytic_bc/figures/P_vs_A_tail_vs_N.png
```

The corrected boundary convention is important:

- For the `in` branch, `R_in/P` does not have boundary value `1` at the horizon. It has boundary value `A_in/P -> h2`, matching the previous hard normalization convention.
- For the `up` branch, `R_up/P` does not have boundary value `1` at infinity. It has boundary value `A_up/P -> C_up`.
- The one-sided derivative is obtained from the degenerate endpoint equation, equivalently from differentiating `(A_branch/P) u_branch`.

For `a=0.5, ω=10, N=220`, the coefficient tails were:

| basis | tail ratio |
|---|---:|
| `P_up` | `1.33e-12` |
| `A_up` | `2.23e-14` |
| `P_in` | `3.34e-14` |
| `A_in` | `8.35e-15` |

Interpretation:

- `P_in` is comparable to `A_in` locally near the horizon.
- `P_up` is worse than `A_up` because the corrected `P_up` boundary derivative is large. For `a=0.5,ω=10`, `u_z(0)` for `R_up/P` is approximately `-37.6-12.7i`.
- Changing the outgoing branch from `A_up` to `P` does not improve high-frequency spectral convergence.

## Experiment 2: GSN amplitude reconstruction with `P_up`

Output directory:

```text
outputs/gsn_amp_pfactor_reconstruct_w10/20260605_a05_w10
```

Key files:

```text
outputs/gsn_amp_pfactor_reconstruct_w10/20260605_a05_w10/summary.json
outputs/gsn_amp_pfactor_reconstruct_w10/20260605_a05_w10/reconstruction_compare.csv
outputs/gsn_amp_pfactor_reconstruct_w10/20260605_a05_w10/amplitude_fit_compare.csv
outputs/gsn_amp_pfactor_reconstruct_w10/20260605_a05_w10/figures/A_vs_Pmix_reconstruction_relerr.png
outputs/gsn_amp_pfactor_reconstruct_w10/20260605_a05_w10/figures/Pup_vs_Aup_channel_relerr.png
```

The comparison used GSN amplitudes:

```text
B_inc = 24.675049712429978 - 5.00013326071889 i
B_ref = -6.457956617151793e-09 - 8.144599106575082e-09 i
B_trans = 1
λ = -26.85215496939334
```

Two reconstructions were compared:

```text
R_A    = B_inc A_down u_down + B_ref A_up u_up
R_Pmix = B_inc A_down u_down + B_ref P u_Pup
```

The median relative errors against GSN were effectively identical:

| range | `A` basis | `Pmix` basis |
|---|---:|---:|
| global | `3.194285e-2` | `3.194285e-2` |
| `r∈[1000,3000]` | `1.09995e-4` | `1.10004e-4` |
| `r∈[3000,5000]` | `6.84645e-5` | `6.84623e-5` |

Refitting amplitudes using the two bases also gave essentially identical results. For `r∈[1000,3000]`:

| basis | `B_ref` relative error |
|---|---:|
| `A` | `2.87443e-5` |
| `Pmix` | `2.87451e-5` |

Interpretation:

- `P_up` is only a different representation of the same outgoing channel.
- It does not improve the conditioning of extracting the tiny `B_ref` at high frequency.

## Experiment 3: Extending `P_in` toward infinity

Output directory:

```text
outputs/pin_extend_to_infinity_w10/20260605_a05_w10
```

Key files:

```text
outputs/pin_extend_to_infinity_w10/20260605_a05_w10/summary.json
outputs/pin_extend_to_infinity_w10/20260605_a05_w10/pin_extend_summary.csv
outputs/pin_extend_to_infinity_w10/20260605_a05_w10/figures/pin_extend_relerr_vs_gsn.png
outputs/pin_extend_to_infinity_w10/20260605_a05_w10/figures/pin_extend_coeff_decay.png
```

This experiment solved a single-domain `R=Pψ` problem from the horizon to progressively larger `rmax`.

Results:

| `rmax` | tail ratio | median relative error vs GSN |
|---:|---:|---:|
| 50 | `1.87e-2` | `2.81e1` |
| 100 | `1.25e-2` | `4.69e1` |
| 300 | `7.31e-3` | `1.50e2` |
| 1000 | `4.59e-3` | `4.16e2` |
| 5000 | `2.70e-3` | `1.72e2` |

Interpretation:

- `P_in` is reliable only in the horizon-local region.
- At infinity, `R_in/P` contains the `A_down/P` component, which behaves like a fast oscillatory term after compactification.
- A global Chebyshev representation of `R_in/P` is therefore unsuitable.

## Experiment 4: Three-segment matching with `P` in the middle

Two versions were tested.

The first version used `P` at the horizon and the middle:

```text
outputs/three_segment_p_middle_w10/20260605_a05_w10
```

The corrected version used `A_in u_in` at the horizon, converted to `Pψ` at `z=0.4`, then propagated in the middle:

```text
outputs/three_segment_ain_p_middle_w10/20260605_a05_w10
```

Key corrected-version files:

```text
outputs/three_segment_ain_p_middle_w10/20260605_a05_w10/summary.json
outputs/three_segment_ain_p_middle_w10/20260605_a05_w10/three_segment_fit.csv
outputs/three_segment_ain_p_middle_w10/20260605_a05_w10/figures/middle_P_coeff_decay.png
outputs/three_segment_ain_p_middle_w10/20260605_a05_w10/figures/amplitude_error_vs_match_radius.png
```

The corrected setup was:

1. Inner segment: solve `R=A_in u_in` on `z∈[0.4,1]`.
2. Interface conversion at `z=0.4`: compute `ψ` and `ψ_z` using the formula above.
3. Middle segment: solve `R=Pψ` on `z∈[z1,0.4]`.
4. Outer segment: solve `A_down u_down` and `A_up u_up` near infinity.
5. Amplitudes: fit only function values over a window near `z1`:

```text
Pψ = B_inc A_down u_down + B_ref A_up u_up
```

Despite very small fitting residuals, the fitted amplitudes were wrong:

| `z1` | `B_inc` relative error | `B_ref` relative error | middle tail |
|---:|---:|---:|---:|
| 0.08 | `~1.000` | `1.678e6` | `6.35e-6` |
| 0.05 | `~1.000` | `1.678e6` | `4.31e-6` |
| 0.0373 | `~1.000` | `1.678e6` | `2.83e-6` |
| 0.025 | `~1.000` | `1.678e6` | `2.00e-6` |
| 0.0187 | `~1.000` | `1.678e6` | `1.45e-6` |

Direct comparison of the middle solution against GSN in the far matching window showed that the middle `Pψ` solution itself was already wrong. For the `z1=0.0187` case, in a window around `r≈56–97`, the median relative error of `R=Pψ` against GSN was about `2.18e4`.

Interpretation:

- The failure is not caused by derivative matching versus value matching.
- The corrected value-only multipoint fit still fails because the middle propagated function is already inaccurate.

## Experiment 5: Validation against GSN and `ω=0.1` middle-frequency check

Output directory:

```text
outputs/validate_in_middle_gsn/20260605_a05_w10_w0p1
```

Key files:

```text
outputs/validate_in_middle_gsn/20260605_a05_w10_w0p1/summary.json
outputs/validate_in_middle_gsn/20260605_a05_w10_w0p1/figures/in_middle_relerr_omega_10p0ppng
outputs/validate_in_middle_gsn/20260605_a05_w10_w0p1/figures/in_middle_relerr_omega_0p1ppng
```

The `A_in u_in` inner branch was validated against GSN.

| frequency | segment | median relative error vs GSN | max relative error vs GSN |
|---:|---|---:|---:|
| `ω=10` | inner `z∈[0.4,0.98]` | `1.19e-7` | `6.08e-6` |
| `ω=0.1` | inner `z∈[0.4,0.98]` | `7.35e-12` | `7.68e-12` |

The interface conversion was also checked:

| frequency | `R` interface relative mismatch | `R_z` interface relative mismatch |
|---:|---:|---:|
| `ω=10` | `1.24e-16` | `0` |
| `ω=0.1` | `1.99e-16` | `1.88e-16` |

However, middle propagation with right-end `ψ,ψ_z` conditions failed:

| frequency | middle `Pψ` median relative error vs GSN |
|---:|---:|
| `ω=10` | `1.69e2` |
| `ω=0.1` | `1.67e-1` |

The full `A_in` single-domain extension was used as a control:

| frequency | full `A_in` median relative error vs GSN |
|---:|---:|
| `ω=10` | `1.99e-6` |
| `ω=0.1` | `1.68e-12` |

Interpretation:

- The `in` branch is correct.
- The interface conversion is correct.
- The `P` middle propagation strategy is the failing component.

## Experiment 6: `P` middle equation manufactured-solution diagnostics

Output directory:

```text
outputs/diagnose_p_middle_equation/20260605_a05
```

Key files:

```text
outputs/diagnose_p_middle_equation/20260605_a05/summary.json
outputs/diagnose_p_middle_equation/20260605_a05/p_middle_bvp_diagnostics.csv
outputs/diagnose_p_middle_equation/20260605_a05/figures/p_middle_bvp_diagnostics_w0p1ppng
outputs/diagnose_p_middle_equation/20260605_a05/figures/p_middle_bvp_diagnostics_w10p0ppng
```

This test used a known good `A_in` solution, converted it to `ψ=R/P`, and then solved the same middle `P` equation with different boundary-condition types.

For `ω=0.1` on `[z1,z2]=[0.02,0.4]`:

| boundary type | median error vs transformed full `A_in` |
|---|---:|
| right-end Cauchy `ψ(z2),ψ_z(z2)` | `1.65e-1` |
| two-end value BVP `ψ(z1),ψ(z2)` | `2.67e-13` |
| mixed `ψ(z1),ψ_z(z2)` | `3.24e-3` |

For `ω=10` on `[z1,z2]=[0.0187,0.4]`:

| boundary type | median error vs transformed full `A_in` |
|---|---:|
| right-end Cauchy `ψ(z2),ψ_z(z2)` | `8.54` |
| two-end value BVP `ψ(z1),ψ(z2)` | `5.56e-2` |
| mixed `ψ(z1),ψ_z(z2)` | `9.77e-1` |

Interpretation:

- The `P` equation coefficients are not the primary error source. At `ω=0.1`, the two-end value BVP recovers the manufactured solution to `1e-13`.
- The right-end Cauchy formulation is the unstable part.
- At `ω=10`, even the two-end BVP becomes difficult because `ψ=R/P` is a poor high-frequency middle variable.

An MMA comparison was attempted, but `WolframKernel` was not discoverable in the current WSL environment. The attempted output is:

```text
outputs/diagnose_p_middle_equation/20260605_a05/mma_gsn_omega0p1_compare.json
```

The file records the GSN `ω=0.1` amplitudes and the `kernel-not-found` status.

## Experiment 7: Coefficient plots for the `P`-reduced equation

Output directory:

```text
outputs/p_equation_coefficients_w10/20260605_a05_w10
```

Key files:

```text
outputs/p_equation_coefficients_w10/20260605_a05_w10/summary.json
outputs/p_equation_coefficients_w10/20260605_a05_w10/figures/p_coefficients_real_imag_linear_z.png
outputs/p_equation_coefficients_w10/20260605_a05_w10/figures/p_coefficients_magnitude.png
outputs/p_equation_coefficients_w10/20260605_a05_w10/figures/p_coefficients_endpoint_scaling.png
outputs/p_equation_coefficients_w10/20260605_a05_w10/figures/p_coefficient_ratios.png
outputs/p_equation_coefficients_w10/20260605_a05_w10/figures/p_effective_slope_minus_B0_over_B1.png
```

For `a=0.5,ω=10`, in the tested middle interval `z∈[0.0187,0.4]`:

| coefficient or ratio | median |
|---|---:|
| `|B2|` | `3.42e-2` |
| `|B1|` | `3.40e1` |
| `|B0|` | `1.81e3` |
| `|B1/B2|` | `9.95e2` |
| `|B0/B2|` | `5.31e4` |
| `|B0/B1|` | `5.33e1` |

Near infinity, for `z<0.0187`:

| coefficient or ratio | median |
|---|---:|
| `|B2|` | `1.20e-8` |
| `|B1|` | `3.73e1` |
| `|B0|` | `1.48e3` |
| `|B1/B2|` | `3.10e9` |
| `|B0/B2|` | `1.23e11` |
| `|B0/B1|` | `3.97e1` |

Interpretation:

- The `P` equation is not a balanced second-order equation in the middle/far region.
- `B2` is small while `B1` and `B0` remain large.
- Locally the equation behaves more like `B1 ψ_z + B0 ψ ≈ 0`.
- The effective slope scale `-B0/B1` is typically `O(50)` in the tested middle interval.
- This creates a stiff, direction-sensitive propagation problem. Small errors in the single-sided Cauchy conditions or collocation solution project into the wrong mode and grow rapidly.

## Final interpretation

The experiments separate the failure into three layers:

1. The `in` branch itself is correct. It matches GSN well in the horizon and middle region.
2. The conversion from `A_in u_in` to `Pψ` is correct at the interface.
3. The `P`-reduced middle equation is algebraically consistent, but single-sided propagation is numerically unstable.

The central problem is therefore not a simple algebraic sign error in the middle coefficients. The problem is that `R/P` is a poor middle propagation variable, especially at high frequency. It makes the equation strongly first-order/reaction dominated after compactification and does not provide a stable overlap procedure for recovering `B_inc` and the tiny high-frequency `B_ref`.

## Recommended next steps

1. Do not use `R/P` as the middle segment propagation variable for high-frequency amplitude extraction.
2. Use `A_in`-based propagation for the ingoing solution where it remains accurate, or switch to a variable with balanced coefficient magnitudes in the middle.
3. Replace single-sided Cauchy-type Chebyshev collocation by either:
   - two-sided BVP constraints,
   - local multiple shooting,
   - Riccati/log-derivative propagation,
   - or high-precision transfer matrices with explicit mode filtering.
4. Keep `A_down` and `A_up` as the far-field branch variables for amplitude extraction.
5. Use `P` only as a local endpoint asymptotic tool unless a better-conditioned middle formulation is derived.

