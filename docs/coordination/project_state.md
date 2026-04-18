# Project State

**Last updated**: 2026-04-19  
**Updated by**: agent1

## Current Phase
Phase 0 — Bootstrap & Data Pipeline

## Repository
https://github.com/ljq2088/SolvingTeukolskyEq

## Project Goal
Build a neural network interpolation model for Kerr gravitational-wave scattering amplitudes (s=-2 Teukolsky equation). The model takes (a, ω, l, m) as input and outputs the complex amplitude ratio A_ref/A_inc.

## Existing Infrastructure (confirmed from repo)
- `utils/compute_lambda.py` — computes SWSH eigenvalue λ via GremLinEqRe
- `utils/amplitude_ratio.py` — computes amplitude ratio via kerr_matcher (spectral method)
- `docs/boundary_coefficients.md` — Teukolsky equation coefficients in x=r_+/r coordinates
- `model/`, `trainer/`, `dataset/` — skeleton directories, mostly empty
- `physical_ansatz/` — placeholder for physics-informed constraints

## Key Dependencies (WSL only)
- `GremLinEqRe` at `/home/ljq/code/Teukolsky_based/GremLinEqRe`
- `kerr_matcher` at `/home/ljq/code/radial_flow/spec_flow_method_Kerr/kerr_matcher_project/src`
- `pybhpt` in conda env `gw_ml_env`

## Normalization Convention (from pybhpt reference run)
- R_in ~ Δ² · exp(-ik·r_*) at horizon (B_trans = 1)
- R_in ~ A_inc · r^{-3} · exp(-iω·r_*) + A_ref · r³ · exp(+iω·r_*) at infinity
- Reference values (M=1, a=0.1, ω=0.1, l=2, m=2, s=-2): |A_ref| ≈ 17.04, |A_ref/A_inc| ≈ 1.000025

## Open Issues
1. kerr_matcher solver not yet verified against pybhpt reference
2. No training dataset exists yet
3. NN architecture not decided
4. No CLAUDE.md in repo

## Completed
- [x] Repo structure surveyed
- [x] Existing utilities identified
- [x] Normalization convention established from pybhpt
