# TeukSpecFiLM Implementation Plan

**Project:** `SolvingTeukolskyEq`  
**Target branch:** `research/high-accuracy-teukolsky-plan-20260528`  
**Prepared for:** Codex implementation  
**Date:** 2026-06-05  

---

## 0. Executive Summary

The current branch has already shown that direct PINN training is not the right core strategy for the high-accuracy Teukolsky radial solver. The reliable direction is:

```text
spectral branch teachers
    -> patch-local parameter-space Chebyshev decoders
    -> analytic multipoint amplitude solve
    -> optional FiLM/Pirate residual corrector
```

The proposed implementation is named **TeukSpecFiLM**:

```text
TeukSpecFiLM
= Patch-local parameter-space Chebyshev spectral operator
+ frozen spectral branch solver
+ FiLM/Pirate residual corrector
+ analytic multipoint amplitude solver
+ residual / Abel-Wronskian validator
```

The mathematical target is the ingoing homogeneous solution of the Kerr radial Teukolsky equation for fixed

```text
s = -2, l = m = 2,
a in [0, 1), omega in (1e-4, 10],
r in [r_+(a), infinity).
```

The solver should ultimately return:

```text
R_in(r; a, omega)
B_inc(a, omega)
B_ref(a, omega)
```

where the far-field convention is

```text
R_in ~ B_inc r^{-1} exp(-i omega r*) + B_ref r^3 exp(+i omega r*)
```

with horizon normalization `B_trans = 1`.

The current branch has already demonstrated that:

1. regular branch variables are smooth and spectrally convergent;
2. AnMR is essential at low frequency;
3. ordinary MLP coefficient heads fail to learn the coefficient field uniformly;
4. tensor-product Chebyshev decoders in parameter space are much better;
5. amplitudes should be computed by analytic multipoint least squares, not by a neural amplitude head;
6. FiLM/MLP should be retained only as a frozen-spectral-solver residual corrector, not as the main solver.

---

## 1. Existing Evidence from the Current Branch

### 1.1 Current goal

The branch report defines the long-term target as a high-accuracy, high-speed surrogate for `R_in` and far-field amplitudes over

```text
a in [0, 1), omega in (1e-4, 10], r in [r_+, infinity).
```

The final accuracy metric is the normalized Teukolsky equation residual, not merely agreement with training data:

```text
|equation residual| / max(|second-derivative term|,
                          |first-derivative term|,
                          |zeroth-order term|)
```

This should be evaluated on the physical solution, without changing its physical normalization.

### 1.2 Direct PINN failure mode

The branch report records that Adam pretraining followed by L-BFGS refinement can collapse rapidly to a trivial or near-trivial homogeneous solution. The reasons are structural:

- homogeneous equations have scale degeneracies unless normalization is strongly enforced;
- PINNs can find low residual but physically wrong functions;
- degenerate boundary equations propagate numerical errors;
- simple anchor terms do not solve the core conditioning problem.

Therefore, direct residual PINN should not be the primary solver.

### 1.3 Correct regular branch variables

The current branch has corrected the main variable choice. Use the following regular factors:

```text
u_in   = R_in   / A_in,     A_in   = Delta^2 exp(-i k r*)
u_down = R_down / A_down,   A_down = r^{-1} exp(-i omega r*)
u_up   = R_up   / A_up,     A_up   = r^3 exp(+i omega r*)
```

with

```text
k = omega - m Omega_H.
```

In the code, these are implemented in `utils/amplitude.py` as:

```python
A_down(r, mode) = r**(-1) * exp(-1j * omega * r_star)
A_up(r, mode)   = r**3    * exp(+1j * omega * r_star)
A_in(r, mode)   = Delta(r)**2 * exp(-1j * k_hor * r_star)
```

The reduced branch equation is then written as

```text
B2(z; p) u_zz + B1(z; p) u_z + B0(z; p) u = 0,
```

where `p = (a, omega, ell, m, s, lambda)`.

### 1.4 Spectral teachers are reliable in the right variables

The current branch shows that direct spectral collocation gives teacher coefficients with Chebyshev tails near double precision when using the correct variables and grids. In the moderate test patch, teacher tails were approximately:

```text
in   median ~ 2e-14
down median ~ 1e-14
up   median ~ 5e-14
```

The dominant failure was not the teacher but the neural coefficient map.

### 1.5 Ordinary MLP coefficient heads failed

The existing `ParamToCoeff` class in `scripts/train_patch_spectral_decoder.py` is a basic MLP:

```text
(a, log10 omega)
    -> raw or poly/Fourier features
    -> SiLU MLP
    -> complex Chebyshev coefficient vector
```

This produced acceptable results for some easier branches but failed badly for `u_up`; the reported `up` value error was still at the percent-to-tens-of-percent level. The reason is that a generic MLP has no built-in mechanism enforcing the geometric decay of Chebyshev coefficients.

### 1.6 Parameter-space Chebyshev decoder succeeded

The breakthrough was `scripts/fit_up_param_cheb_decoder.py`, which fits

```text
(a, log10 omega) -> {Chebyshev coefficients in z/y}
```

using a tensor-product Chebyshev expansion in parameter space. On the current moderate patch, it achieved off-grid errors around

```text
median ~ 5e-5
max    ~ 2.5e-4
```

for the difficult `up` branch. This is the first model class in the branch that demonstrated strong uniform accuracy for `up`.

---

## 2. Design Principle

The primary design principle is:

> Use spectral structure for the solution manifold, and use neural networks only where they add value.

That means:

1. use Chebyshev collocation for local branch teachers;
2. use tensor-product Chebyshev expansions in parameter space for the main surrogate;
3. compute amplitudes by analytic multipoint least squares;
4. use FiLM/Pirate MLP only as a small residual correction after freezing the spectral decoder.

Do **not** use:

```text
direct PINN residual training -> neural amplitude head
```

as the primary architecture.

---

## 3. Mathematical Representation

### 3.1 Radial coordinate

Use the current compactified coordinate

```text
z = r_+ / r,
y = 2 z - 1.
```

Thus:

```text
z = 1     horizon
z = 0     infinity
y = 1     horizon
y = -1    infinity
```

Most current code uses `z`; keep `z` in implementation and convert to `y` only when needed for compatibility.

### 3.2 Branch decomposition

For each basis branch `basis in {in, down, up}`:

```text
R_basis(r; p) = A_basis(r; p) u_basis(z; p),
```

where the analytic factors are:

```text
A_in   = Delta^2 exp(-i k r*)
A_down = r^{-1} exp(-i omega r*)
A_up   = r^3 exp(+i omega r*)
```

and the regular branch functions satisfy:

```text
B2 u_zz + B1 u_z + B0 u = 0.
```

### 3.3 Boundary normalization

The local spectral teachers should impose endpoint normalization and derivative conditions from the reduced degenerate equation:

For infinity-side branches:

```text
u_down(0) = 1,   u_down,z(0) = boundary_du_exact(mode, "down", "left")
u_up(0)   = 1,   u_up,z(0)   = boundary_du_exact(mode, "up",   "left")
```

For horizon-side branch:

```text
u_in(1) = 1,     u_in,z(1) = boundary_du_exact(mode, "in", "right")
```

These conditions are already implemented in `utils/amplitude.py` via `boundary_du_exact` and `solve_basis_domain`.

### 3.4 Far-field amplitude extraction

After evaluating `u_in`, `u_down`, and `u_up` on an overlap/matching window, reconstruct physical branch values:

```text
R_in   = A_in   u_in
R_down = A_down u_down
R_up   = A_up   u_up
```

Then solve the multipoint linear least-squares system:

```text
R_in(z_j) = B_inc R_down(z_j) + B_ref R_up(z_j).
```

This is the amplitude solve. It must remain analytic / numerical, not a neural amplitude head.

---

## 4. Proposed Architecture: TeukSpecFiLM

### 4.1 High-level structure

Implement the final system as:

```text
TeukSpecFiLM
  ├── PatchRouter
  ├── PatchExpert[patch_id]
  │     ├── ParamChebDecoder[in]
  │     ├── ParamChebDecoder[down]
  │     ├── ParamChebDecoder[up]
  │     ├── optional FiLMPirateCorrector[in/down/up]
  │     ├── analytic_multipoint_ls_amplitude_solver
  │     └── residual / Abel-Wronskian validator
  └── fallback hooks for pybhpt / MMA / GSN if available
```

### 4.2 Main solver: parameter-space Chebyshev decoder

For each parameter patch, radial segment, and branch, represent

```text
u_b(z; p) = sum_{n=0}^{N_z} c_{b,n}(p) T_n(xi_z),
```

where

```text
p = (a, zeta), zeta = log10(omega).
```

Instead of learning `c_n(p)` with a generic MLP, represent it spectrally in parameter space:

```text
c_{b,n}(p)
  = sum_{alpha=0}^{N_a} sum_{beta=0}^{N_zeta}
      C_{b,n,alpha,beta} T_alpha(xi_a) T_beta(xi_zeta).
```

This is a patch-local tensor-product Chebyshev operator.

#### Required class

Create:

```text
teukspec/decoders/param_cheb_decoder.py
```

with class:

```python
class ParamChebDecoder:
    def __init__(self, decoder_coeffs, a_center, a_half_width,
                 logw_center, logw_half_width, z_domain,
                 n_z, deg_a, deg_logw, basis, grid_kind):
        ...

    def eval_coeff(self, a, logw):
        """Return complex Chebyshev-in-z coefficient vector."""

    def eval_u(self, z, a, logw):
        """Evaluate u(z; a, logw) by Clenshaw or Vandermonde."""

    def eval_u_derivatives(self, z_nodes, a, logw):
        """Return u, u_z, u_zz using spectral differentiation."""

    def save_npz(self, path):
        ...

    @classmethod
    def load_npz(cls, path):
        ...
```

Implementation can reuse the logic in `scripts/fit_up_param_cheb_decoder.py`:

```python
tensor_vandermonde
fit_tensor_decoder
eval_tensor_decoder
```

but move these into reusable library modules.

### 4.3 Optional second-stage corrector: FiLM/Pirate residual network

After training/fitting and freezing the spectral decoder, train a small residual correction:

```text
u_total(z; p) = u_spec(z; p) + delta_u(z; p).
```

The correction is represented as:

```text
delta_u(z; p) = gate_b(z) * N_theta(z, p),
```

where `gate_b(z)` enforces zero correction at the physical boundary so that normalization and derivative constraints are not broken.

For `in` branch on `[z_L, 1]`:

```text
gate_in(z) = (1 - z)^2
```

For `down/up` branches on `[0, z_R]`:

```text
gate_out(z) = z^2
```

If both endpoints should be protected on a finite segment:

```text
gate_two_sided(z) = (z - z_L)^2 (z_R - z)^2.
```

#### Required class

Create:

```text
teukspec/correctors/film_pirate_corrector.py
```

with:

```python
class FiLMPirateCorrector(nn.Module):
    def __init__(self, feature_dim, hidden_dim=64, depth=4,
                 fourier_bands=4, basis="in", z_left=0.0, z_right=1.0,
                 alpha_init=0.0):
        ...

    def forward_raw(self, z, features):
        """Return raw complex correction before hard gate."""

    def gate(self, z):
        """Hard boundary gate."""

    def forward(self, z, features):
        """Return complex delta_u."""
```

Each residual block should use adaptive residual gating:

```text
h_{l+1} = (1 - alpha_l) h_l + alpha_l F_l(h_l, gamma_l(p), beta_l(p))
```

with:

```text
alpha_l initialized to 0 or 1e-3.
final output layer initialized to zero.
```

This ensures the corrector starts from `delta_u = 0`, so the model initially equals the frozen spectral solution.

---

## 5. Corrector Training Formulation

### 5.1 Reduced-equation residual

Train the corrector in the reduced `u` equation, not in the full `R` equation.

For each branch:

```text
res = B2 u_zz + B1 u_z + B0 u.
```

where:

```text
u    = u_spec + delta_u
u_z  = u_spec_z + delta_u_z
u_zz = u_spec_zz + delta_u_zz
```

Use spectral derivatives for `u_spec` and autograd derivatives for `delta_u`.

### 5.2 Relative residual loss

Use the project metric:

```text
rel = |res| / max(|B2 u_zz|, |B1 u_z|, |B0 u|, eps)
```

Training loss:

```text
L_res = mean(rel^2).
```

This should match the convention in current `residual_loss` and `solve_teacher`.

### 5.3 Correction-size regularization

Since the spectral solution should already be close, the corrector should remain small:

```text
L_amp = mean(|delta_u|^2 / (|u_spec|^2 + eps)).
```

### 5.4 Sobolev regularization

Prevent the corrector from introducing high-frequency noise:

```text
L_sob = mean(|delta_u_z|^2 + eta |delta_u_zz|^2).
```

### 5.5 Total corrector loss

Use:

```text
L = L_res + lambda_amp L_amp + lambda_sob L_sob.
```

Recommended initial values:

```text
lambda_amp = 1e-2 to 1e-1 initially, then decay to 1e-4 to 1e-3
lambda_sob = 1e-6 to 1e-4 depending on scale
```

### 5.6 Training schedule

```text
1. Load frozen spectral decoder.
2. Freeze all decoder parameters.
3. Initialize corrector with zero output.
4. Train with AdamW on residual collocation points.
5. Use residual-adaptive refinement:
   - scan parameter patch and z nodes;
   - add collocation points where rel residual is largest;
   - optionally split patch if residual remains localized.
6. Optional final optimizer: LBFGS only for corrector, with strong small-correction regularization.
```

Do **not** run L-BFGS on the full model from random or weakly constrained initialization.

---

## 6. Patch and Domain Design

### 6.1 Moderate frequency: two-domain structure

For moderate patches, use the current two-overlap structure:

```text
in:   [z_left, 1]
down: [0, z_right]
up:   [0, z_right]
```

where

```text
z_match = 0.5 * (y_match + 1)
z_left  = z_match - 0.5 * match_width
z_right = z_match + 0.5 * match_width
```

Default for current moderate patch:

```text
y_match = -0.25
z_match = 0.375
match_width = 0.12
N_z = 64
grid_kind = anmr or auto
```

For mid-frequency patches, `N_z = 64` is the first choice. Increase only if off-grid residual or Chebyshev tail requires it.

### 6.2 Low frequency: two-domain with AnMR

Low frequency requires AnMR. Use:

```text
grid_kind = anmr
```

and the existing adaptive matching rule:

```text
r_match = 3 M + omega^{-1/2}
z_match = r_+ / r_match.
```

Recommended low-frequency settings:

```text
omega in [1e-4, 1e-2]
outer N_z = 96 to 128
inner N_z = 64 to 96
```

Use AnMR for outer branches by default. Linear grid should only be used for diagnostics.

### 6.3 High frequency: three-domain with two-channel middle basis

High frequency cannot be solved robustly using a single extracted phase in the middle region. The current branch report found that single-phase middle extraction fails because the middle region contains a mixture of two channels.

Use three domains:

```text
outer:  [0, z1]
middle: [z1, z2]
inner:  [z2, 1]
```

In the middle domain, do not use a single envelope. Use a two-channel phase basis:

```text
R_mid(r)
  = C_plus  A_plus(r)  v_plus(z)
  + C_minus A_minus(r) v_minus(z),
```

with, at minimum,

```text
A_plus  = r^3 exp(+i omega r*)
A_minus = r^{-1} exp(-i omega r*)
```

or a higher-order WKB-improved pair later.

Recommended high-frequency settings:

```text
omega in [1, 10]
inner  N_z = 80 to 128
middle N_z = 96 to 160
outer  N_z = 96 to 160
```

Choose `z1, z2` by scanning:

```text
relative residual
condition number
Abel/Wronskian consistency
amplitude LS residual
```

Do not optimize high-frequency `B_ref` using a neural amplitude head.

---

## 7. Parameter Patch Atlas

### 7.1 Main coordinates

Use:

```text
zeta = log10(omega)
```

and patch in `(a, zeta)`.

Add physical features for the corrector and diagnostics:

```text
r_plus, r_minus, delta_h, Omega_H, k = omega - m Omega_H, lambda_value,
possibly log|k| or signed k feature near the superradiant seam.
```

### 7.2 Initial patch cover

Frequency direction:

```text
zeta in [-4, -3]
zeta in [-3, -2]
zeta in [-2, -1]
zeta in [-1,  0]
zeta in [ 0,  1]
```

Spin direction:

```text
a in [0.00, 0.40]
a in [0.30, 0.70]
a in [0.60, 0.90]
a in [0.85, 0.97]
a in [0.94, 0.99]
a in [0.98, 0.999]
```

Use overlaps for blending and validation.

### 7.3 Superradiant seam patches

The surface

```text
k = omega - m Omega_H = 0
```

is a special seam. Add narrow-band patches around it. These patches may require smaller widths and stronger validation.

### 7.4 Patch acceptance and splitting

For each patch, compute:

```text
teacher Chebyshev tail
parameter-space Chebyshev tail
off-grid value error
off-grid reduced residual
amplitude LS residual
Abel/Wronskian residual
```

If parameter-space Chebyshev tail does not decay, split the patch rather than increasing degree indefinitely.

---

## 8. Hyperparameter Recommendations

### 8.1 Parameter-space decoder

Start with:

```text
n_param_side = 9
deg_a = 8
deg_logw = 8
N_z = 64
```

For harder patches:

```text
n_param_side = 11 to 13
deg_a = 10 to 12
deg_logw = 10 to 12
```

For near-extremal or high-frequency patches:

```text
n_param_side = 13 to 17
deg_a, deg_logw = 12 to 16
```

Use coefficient decay, not guesswork, to decide.

### 8.2 Radial spectral nodes

```text
mid-frequency: N_z = 64, then 80/96 if needed
low-frequency: outer N_z = 96 to 128, inner N_z = 64 to 96
high-frequency: middle/outer N_z = 96 to 160 or higher after diagnostics
```

### 8.3 Matching window

Mid-frequency default:

```text
z_match = 0.35 to 0.45
match_width = 0.10 to 0.18
```

Current default:

```text
y_match = -0.25, z_match = 0.375, match_width = 0.12
```

Low-frequency:

```text
use adaptive_match_z(mode)
```

High-frequency:

```text
use z1, z2 two-match structure;
select by diagnostic scan.
```

---

## 9. Proposed Repository Structure

Add a reusable package instead of continuing with monolithic scripts:

```text
teukspec/
  __init__.py

  core/
    coordinates.py          # z, y, r, r*, horizon utilities
    mode.py                 # wrapper or import bridge to utils.mode.KerrMode
    factors.py              # A_in, A_down, A_up wrappers
    coefficients.py         # reduced B2,B1,B0 wrappers
    residual.py             # reduced residual and relative residual
    chebyshev.py            # Clenshaw, Cheb matrices, tensor Vandermonde

  teachers/
    spectral_teacher.py     # build branch teachers using solve_basis_domain
    dataset.py              # save/load teacher coefficient grids

  decoders/
    param_cheb_decoder.py   # core parameter-space Cheb decoder
    fit_param_decoder.py    # fit all branches for one patch

  correctors/
    film_pirate_corrector.py
    train_corrector.py

  amplitude/
    multipoint_ls.py        # analytic B_inc/B_ref solve
    diagnostics.py          # fit residuals, condition number

  atlas/
    patch_config.py
    patch_expert.py
    patch_router.py
    blending.py

  validation/
    residual_scan.py
    offgrid_validation.py
    abel_wronskian.py
    report.py
```

Scripts should become thin wrappers:

```text
scripts/
  fit_patch_decoders.py
  train_patch_corrector.py
  validate_patch.py
  build_patch_atlas.py
  evaluate_solver.py
```

Keep existing scripts for reproducibility, but move reusable logic out of them.

---

## 10. Concrete Codex Implementation Tasks

### Task 1: Extract reusable Chebyshev utilities

Move or copy reusable logic from `scripts/fit_up_param_cheb_decoder.py` into:

```text
teukspec/core/chebyshev.py
```

Required functions:

```python
cheb_lobatto_physical(center, half_width, n_side)
tensor_vandermonde(xi_a, xi_w, deg_a, deg_w)
fit_tensor_decoder(coeffs, xi_a, xi_w, deg_a, deg_w)
eval_tensor_decoder(decoder, xi_a, xi_w)
cheb_eval_matrix(n, xi)
clenshaw_eval(coeff, xi)
cheb_derivative_matrices(n, z_left, z_right, grid_kind, mode, domain)
```

### Task 2: Generalize `fit_up_param_cheb_decoder.py` to all branches

Create:

```text
scripts/fit_patch_param_cheb_decoders.py
```

Capabilities:

```text
--basis all / in / down / up
--a-center, --a-half-width
--logw-center, --logw-half-width
--n-param-side
--deg-a, --deg-logw
--n
--grid-kind
--y-match
--match-width
--output-dir
```

Output:

```text
patch_config.json
decoder_in.npz
decoder_down.npz
decoder_up.npz
teacher_metrics.csv
param_decoder_eval.csv
offgrid_validation.csv
figures/*
```

### Task 3: Implement `ParamChebDecoder`

Create:

```text
teukspec/decoders/param_cheb_decoder.py
```

It should:

1. load `.npz` decoder files;
2. evaluate coefficient vectors at arbitrary `(a, logw)` inside patch;
3. evaluate `u(z)`, `u_z(z)`, `u_zz(z)`;
4. expose metadata: basis, z-domain, degrees, grid kind, patch center/width.

### Task 4: Implement analytic amplitude solver

Create:

```text
teukspec/amplitude/multipoint_ls.py
```

Function:

```python
def solve_binc_bref_from_branches(mode, z_values, u_in, u_down, u_up):
    """
    Reconstruct physical R branches and solve:
        R_in = B_inc R_down + B_ref R_up
    by scaled complex least squares.
    Return B_inc, B_ref, residual, cond.
    """
```

Use row/column scaling where needed. Reuse `_solve_equilibrated` or related stable utilities.

### Task 5: Implement patch expert

Create:

```text
teukspec/atlas/patch_expert.py
```

Class:

```python
class PatchExpert:
    def __init__(self, patch_dir, enable_corrector=False):
        ...

    def eval_branches(self, a, omega, z_values):
        """Return u_in/down/up and derivatives."""

    def eval_R_in(self, a, omega, r_values):
        """Return R_in using in-branch representation where valid."""

    def solve_amplitudes(self, a, omega, z_values=None):
        """Return B_inc, B_ref using analytic LS."""

    def residual_scan(self, a, omega, branch, z_values):
        """Return reduced relative residual."""
```

### Task 6: Implement FiLM/Pirate corrector

Create:

```text
teukspec/correctors/film_pirate_corrector.py
teukspec/correctors/train_corrector.py
scripts/train_patch_corrector.py
```

The corrector must:

1. load frozen `PatchExpert` / decoders;
2. compute `u_spec`, `u_spec_z`, `u_spec_zz`;
3. produce hard-gated `delta_u`;
4. compute autograd derivatives of `delta_u`;
5. train on relative reduced residual;
6. save `corrector_{basis}.pt` and training diagnostics.

### Task 7: Implement validation report

Create:

```text
scripts/validate_patch_solver.py
```

Report:

```text
teacher tail statistics
parameter decoder coefficient tail
off-grid value relative error
off-grid relative reduced residual
amplitude LS residual
B_inc/B_ref comparison if external benchmark available
Abel/Wronskian residual where applicable
```

Save:

```text
validation_summary.json
validation_points.csv
figures/residual_heatmap.png
figures/value_error_heatmap.png
figures/amplitude_error.png
```

---

## 11. Acceptance Criteria

### 11.1 Moderate patch acceptance

For the current moderate patch around

```text
a_center = 0.5
logw_center = -1.5
a_half_width = 0.12
logw_half_width = 0.25
```

achieve, before corrector:

```text
teacher tail median < 1e-12
in/down/up off-grid value_rel_median < 1e-4
in/down/up off-grid value_rel_max < 5e-4
reduced residual median < 1e-5
```

After corrector:

```text
reduced residual improves by at least 3x, preferably 10x
correction norm median < 1e-2 relative to u_spec
boundary normalization unchanged to machine precision or near it
amplitude LS residual does not degrade
```

### 11.2 Amplitude acceptance

For patches where reliable external amplitude references exist:

```text
B_inc relative error < 1e-4 initially
B_ref relative error < 1e-4 where |B_ref| is not extremely tiny
```

For high-frequency cases where `|B_ref/B_inc|` is extremely small, track absolute error and scaled LS residual in addition to relative error.

### 11.3 Global atlas acceptance

For patch atlas:

```text
continuous output across overlaps
no patch seam jumps larger than local validation tolerance
fallback flag raised for unresolved difficult regions
residual validator callable in certified mode
```

---

## 12. Implementation Notes and Pitfalls

### 12.1 Do not train amplitude heads

The branch report explicitly moves away from neural amplitude heads. Amplitudes must be derived from the decoded branches through analytic multipoint LS.

### 12.2 Do not train FiLM corrector from scratch

The corrector must start from a frozen spectral decoder. Its output must be initialized to zero.

### 12.3 Do not let the corrector break boundary conditions

Always multiply the corrector by a hard gate. For example:

```text
in branch:   delta_u = (1 - z)^2 N_theta(z,p)
out branches: delta_u = z^2 N_theta(z,p)
```

### 12.4 Relative residual is the main metric

Small value error can still produce large derivative residual. Always validate both value error and reduced equation residual.

### 12.5 Split patches when parameter Chebyshev tail does not decay

Do not keep increasing `deg_a` and `deg_logw` indefinitely. If parameter-space coefficients do not decay, the patch is too large or crosses a singular/difficult regime.

### 12.6 Use high precision diagnostics for high-frequency amplitude extraction

When `B_ref` is many orders of magnitude smaller than `B_inc`, ordinary double precision LS can be misleading. Reuse high-precision/scaled utilities where possible.

---

## 13. Recommended Development Order

1. **Library extraction**: move Chebyshev/tensor decoder logic from scripts into `teukspec/`.
2. **All-branch parameter decoder**: generalize the successful `up` decoder to `in/down/up`.
3. **PatchExpert**: create a reusable object that loads decoders and evaluates branches.
4. **Analytic amplitude LS**: implement and test `B_inc/B_ref` extraction from decoded branches.
5. **Validation script**: produce consistent patch-level reports.
6. **FiLM/Pirate corrector**: add optional residual correction after freezing decoders.
7. **Patch atlas**: cover mid-frequency domain first.
8. **Low-frequency AnMR atlas**: extend with adaptive matching.
9. **High-frequency three-domain/two-channel basis**: implement only after mid/low are stable.

---

## 14. Minimal First Milestone

The first Codex milestone should be narrow:

```text
Implement all-branch parameter-space Chebyshev decoders and patch-level validation
for the existing moderate patch.
```

Do not implement global atlas or high-frequency three-domain support first.

Concrete command target:

```bash
python scripts/fit_patch_param_cheb_decoders.py \
  --basis all \
  --a-center 0.5 \
  --a-half-width 0.12 \
  --logw-center -1.5 \
  --logw-half-width 0.25 \
  --n-param-side 9 \
  --deg-a 8 \
  --deg-logw 8 \
  --n 64 \
  --grid-kind anmr \
  --y-match -0.25 \
  --match-width 0.12 \
  --output-dir outputs/param_cheb_patch_decoder
```

Then validate:

```bash
python scripts/validate_patch_solver.py \
  --patch-dir outputs/param_cheb_patch_decoder/<run_id> \
  --n-offgrid 49
```

Only after this passes should the FiLM/Pirate residual corrector be added:

```bash
python scripts/train_patch_corrector.py \
  --patch-dir outputs/param_cheb_patch_decoder/<run_id> \
  --basis all \
  --steps 5000 \
  --lr 1e-4 \
  --lambda-amp 1e-2 \
  --lambda-sob 1e-5
```

---

## 15. Long-Term Solver Interface

The final public interface should look like:

```python
from teukspec import TeukSpecFiLMSolver

solver = TeukSpecFiLMSolver.load("outputs/atlas/midfreq_v1")

R = solver.R_in(a=0.5, omega=0.0316227766, r=r_values, mode="fast")
B_inc, B_ref = solver.amplitudes(a=0.5, omega=0.0316227766)

cert = solver.certify(a=0.5, omega=0.0316227766)
print(cert["residual_median"], cert["residual_max"], cert["fallback_recommended"])
```

Modes:

```text
fast:
  use frozen spectral decoders only

corrected:
  use spectral decoders + FiLM corrector

certified:
  compute residual/Abel diagnostics and return warning/fallback flag
```

---

## 16. Final Direction

The correct implementation target is not a larger MLP and not a pure PINN. The target is:

```text
patch-local spectral operator learning
    + physics-informed residual correction
    + analytic amplitude recovery
    + rigorous residual diagnostics
```

This approach matches the evidence from the branch and directly addresses the observed bottlenecks:

- ordinary MLP coefficient heads fail for `up`;
- parameter-space Chebyshev decoders work on the test patch;
- direct PINNs are ill-conditioned;
- FiLM/MLP can still be useful as a small residual corrector;
- amplitude heads should be removed in favor of analytic LS.

