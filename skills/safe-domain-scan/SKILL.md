---
name: safe-domain-scan
description: Use when working on safe-domain scans over (a, omega), including lambda, ramp, Mathematica NumericalIntegration, Mathematica MST, pybhpt, k_horizon margins, timeout behavior, and resumable scan outputs.
---

# Safe Domain Scan

Use this skill for safe-domain scanning and comparison scripts.

## Scope

- `pybhpt/plot_safe_domain_scan_pybhpt.py`
- `mma/Radial_Function.wl`
- `mma/rin_sampler.py`
- external reference script:
  - `.../kerr_matcher_project/tests/plot_safe_domain_scan.py`

## Domain meaning

The current scan compares existence/finite-value criteria, not normalization equality.

Modes:

- `lambda`: whether angular separation constant can be computed and is finite
- `ramp`: whether `R_amp` from matcher is finite
- `mma_num`: Mathematica `R_in` using `Method -> "NumericalIntegration"`
- `mma_mst`: Mathematica `R_in` using `Method -> "MST"`
- `pybhpt`: `pybhpt.radial.RadialTeukolsky(...).solve()` then `radialsolutions("In")`

For `mma_*` and `pybhpt`, the comparison is:

- same `(a, omega)`
- same reference radii `r_ref`
- check computability and finite values only

## Important physics heuristic

- Dangerous region is near `k_horizon = omega - m * Omega_H(a) = 0`
- Scripts should keep a configurable margin such as `k_margin = 1e-2`
- Plot the resonance line and the excluded band

## Mathematica side

`mma/Radial_Function.wl` contains both NumericalIntegration and MST helpers.

Expected helper names:

- `SampleRinAtPoints`
- `SampleRinAtPointsMST`
- `SampleRinOnGrid`
- `SampleRinOnGridMST`

If MST mode fails everywhere, first verify the external Mathematica file actually contains the newly added `...MST` functions and that the Windows-side path points to the updated file.

## Timeouts and skipping

- MMA points should be skippable on failure.
- `pybhpt` requires subprocess-based timeout; Python `signal` alone is not reliable for blocking compiled code.
- The scan script should checkpoint partial state per mode:
  - `partial_<mode>.npz`
  - `partial_<mode>_failures.json`
- On rerun, default behavior should resume from existing partials unless explicitly disabled.

## Output expectations

- figure:
  - `safe_domain_five_modes.png`
- data:
  - `safe_domain_scan_data.npz`
- summary:
  - `safe_domain_scan_summary.json`

## Common failure interpretation

- `mma_*:TimeoutError` or `mma_*:RuntimeError` can be acceptable and should count as “unsafe”, not fatal script errors.
- `pybhpt:HardTimeoutError` means the subprocess timeout worked correctly.
- If many MMA points start failing after one socket error, inspect session reset logic rather than the parameter domain itself.

