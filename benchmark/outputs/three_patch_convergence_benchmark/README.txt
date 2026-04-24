Three-patch spectral convergence benchmark
========================================

a values: [0.1, 0.3, 0.5, 0.7, 0.9, 0.99]
omega values (baseline): [0.001, 0.01, 0.1, 1.0, 10.0]
omega values (match sweep): [0.001, 0.01]
N_edge values: [32, 48, 64, 80, 96, 128]
middle patch order is fixed to 2*N_edge
baseline match pair: (z1, z2) = (0.10, 0.90)
match sweep pairs: [(0.1, 0.9), (0.1, 0.8), (0.05, 0.9), (0.1, 0.95), (0.2, 0.9)]

CSV files:
  - baseline_detail.csv
  - baseline_convergence.csv
  - match_sweep_detail.csv
  - match_sweep_convergence.csv

Plots:
  * baseline: for each a, curves over omega at fixed (z1, z2) = (0.10, 0.90)
  * match_sweep: for each (a, omega), curves over the five requested match-point pairs
