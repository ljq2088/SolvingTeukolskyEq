# Project Roadmap

**Project**: Kerr Scattering Amplitude Neural Network Interpolator  
**Owner**: agent1  
**Last updated**: 2026-04-19

---

## Phase 0 — Bootstrap & Verification (current)
**Goal**: Confirm the data generation pipeline works end-to-end for a single parameter point.

Deliverables:
- [ ] Verify `kerr_matcher` produces correct A_ref/A_inc against pybhpt reference
- [ ] Confirm `compute_lambda` works for the full parameter range
- [ ] Define parameter space (a, ω, l, m) bounds
- [ ] Write `scripts/generate_one.py` — single-point data generation script

Acceptance: `generate_one.py` runs for (a=0.1, ω=0.1, l=2, m=2) and matches pybhpt to 1e-6 relative error.

---

## Phase 1 — Dataset Generation
**Goal**: Generate a training/validation/test dataset covering the target parameter space.

Deliverables:
- [ ] Define parameter grid or sampling strategy (uniform / quasi-random)
- [ ] Write `scripts/generate_dataset.py` — batch generation with parallelism
- [ ] Store dataset in `data/` as HDF5 or numpy arrays
- [ ] Dataset size: TBD (start with ~10k points)

Acceptance: Dataset covers (a ∈ [0, 0.9], ω ∈ [0.05, 0.5], l ∈ {2,3,4}, m ∈ {-l..l}) with no NaN/Inf.

---

## Phase 2 — NN Architecture & Training
**Goal**: Train a neural network that interpolates A_ref/A_inc(a, ω, l, m).

Deliverables:
- [ ] Choose architecture (MLP baseline, possibly physics-informed)
- [ ] Implement in `model/`
- [ ] Training loop in `trainer/`
- [ ] Validation loss < 1e-4 relative error on held-out set

---

## Phase 3 — Validation & Integration
**Goal**: Validate NN against independent pybhpt calculations; integrate into EMRI pipeline.

Deliverables:
- [ ] Benchmark NN vs pybhpt on 100 random test points
- [ ] Document speedup factor
- [ ] Integration example in `notebooks/`

---

## Out of Scope (for now)
- s ≠ -2 spin weights
- Superradiant regime (ω < m·Ω_H) — handle separately if needed
- Eccentric/inclined orbits (different l,m mixing)
