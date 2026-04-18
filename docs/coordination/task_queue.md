# Task Queue

**Last updated**: 2026-04-19  
**Updated by**: agent1

---

## Active Tasks

### TASK-001 — Verify data generation pipeline
**Status**: assigned to agent2  
**Priority**: P0 (blocker for everything else)  
**See**: `docs/coordination/handoff_to_builder.md`

---

## Pending Tasks (not yet assigned)

### TASK-002 — Define parameter space and sampling strategy
**Status**: pending  
**Depends on**: TASK-001  
**Description**: Decide (a, ω, l, m) bounds, grid density, and sampling method for dataset generation.

### TASK-003 — Batch dataset generation script
**Status**: pending  
**Depends on**: TASK-001, TASK-002  
**Description**: Write `scripts/generate_dataset.py` with parallelism.

### TASK-004 — NN architecture design
**Status**: pending  
**Depends on**: TASK-002  
**Description**: Decide MLP depth/width, input encoding, output representation (real+imag vs abs+phase).

---

## Completed Tasks

_(none yet)_
