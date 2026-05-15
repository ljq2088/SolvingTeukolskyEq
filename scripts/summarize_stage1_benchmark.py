#!/usr/bin/env python3
"""Summarize Stage-1 benchmark results from a training run directory.

Usage:
  python scripts/summarize_stage1_benchmark.py \
    --run-dir outputs/autoencoder_stage1_logw_v2_train/20260514_193357_patch_000_comp_0_u_0.500_v_0.603
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np


def _find_best_benchmark(history_path: Path):
    """Scan history.jsonl for the step with best benchmark data."""
    best_step = None
    best_val_mean = float("inf")
    best_entry = None
    last_bench_entry = None

    with open(history_path) as f:
        for line in f:
            d = json.loads(line)
            if d.get("benchmark_median_rel_err_R") is None:
                continue
            last_bench_entry = d
            vm = d.get("val_mean", float("inf"))
            if vm is not None and vm < best_val_mean:
                best_val_mean = vm
                best_entry = d
                best_step = d.get("step")

    return best_entry, last_bench_entry


def summarize(run_dir: str):
    run_dir = Path(run_dir)
    summary_path = run_dir / "logs" / "summary.json"
    history_path = run_dir / "logs" / "history.jsonl"

    if not summary_path.exists():
        print(f"ERROR: {summary_path} not found")
        return

    with open(summary_path) as f:
        summary = json.load(f)

    best_val_mean = summary.get("best_val_mean")
    final_val = summary.get("latest_val_metrics", {})
    final_val_mean = final_val.get("val_mean")
    best_cases = summary.get("best_val_cases", [])
    final_cases = final_val.get("case_metrics", [])

    # Try to get benchmark data from history if summary doesn't have it
    bench_from_history = None
    if history_path.exists():
        _, bench_from_history = _find_best_benchmark(history_path)

    final_bench_median = final_val.get("benchmark_median_rel_err_R")
    final_bench_max = final_val.get("benchmark_max_rel_err_R")

    # Build case table
    case_table = []
    for bc in best_cases:
        ci = bc["case_index"]
        row = {
            "case": ci,
            "a": bc["a"],
            "omega": bc["omega"],
            "best_y_mean": bc["best_y_mean"],
            "best_step": bc["best_step"],
            "median_rel_err_R": None,
            "max_rel_err_R": None,
        }
        # Try from final val metrics first
        if ci < len(final_cases):
            row["median_rel_err_R"] = final_cases[ci].get("median_rel_err_R")
            row["max_rel_err_R"] = final_cases[ci].get("max_rel_err_R")
        case_table.append(row)

    # Sort by omega
    case_table.sort(key=lambda r: r["omega"])

    # Low-omega cases
    omega_vals = [r["omega"] for r in case_table]
    omega_median = float(np.median(omega_vals))
    low_omega_cases = [r for r in case_table if r["omega"] < omega_median * 0.7]
    high_omega_cases = [r for r in case_table if r not in low_omega_cases]

    # Build markdown
    lines = []
    lines.append("# Stage-1 Benchmark Summary")
    lines.append("")
    lines.append(f"**Run dir:** `{run_dir}`")
    lines.append(f"**best_val_mean:** {best_val_mean:.6f}" if best_val_mean else "**best_val_mean:** N/A")
    lines.append(f"**final_val_mean:** {final_val_mean:.6f}" if final_val_mean else "**final_val_mean:** N/A")
    lines.append(f"**benchmark_median_rel_err_R:** {final_bench_median:.4e}" if final_bench_median is not None else "**benchmark_median_rel_err_R:** N/A")
    lines.append(f"**benchmark_max_rel_err_R:** {final_bench_max:.4e}" if final_bench_max is not None else "**benchmark_max_rel_err_R:** N/A")

    if bench_from_history:
        lines.append("")
        lines.append("## Last Benchmark Entry")
        lines.append(f"- step: {bench_from_history['step']}")
        lines.append(f"- val_mean: {bench_from_history.get('val_mean', 'N/A')}")
        lines.append(f"- benchmark_median_rel_err_R: {bench_from_history.get('benchmark_median_rel_err_R', 'N/A')}")
        lines.append(f"- benchmark_max_rel_err_R: {bench_from_history.get('benchmark_max_rel_err_R', 'N/A')}")

    lines.append("")
    lines.append("## All Cases (sorted by omega)")
    lines.append("")
    lines.append("| case | a | omega | best_y_mean | best_step | median_rel_err_R | max_rel_err_R |")
    lines.append("|------|---|-------|-------------|-----------|------------------|---------------|")
    for r in case_table:
        mre = f"{r['median_rel_err_R']:.4e}" if r['median_rel_err_R'] is not None else "N/A"
        mxe = f"{r['max_rel_err_R']:.4e}" if r['max_rel_err_R'] is not None else "N/A"
        lines.append(
            f"| {r['case']} | {r['a']:.4f} | {r['omega']:.6f} "
            f"| {r['best_y_mean']:.6f} | {r['best_step']} "
            f"| {mre} | {mxe} |"
        )

    lines.append("")
    lines.append("## Low-Omega Cases")
    if low_omega_cases:
        lines.append("")
        lines.append("| case | a | omega | best_y_mean | median_rel_err_R | max_rel_err_R |")
        lines.append("|------|---|-------|-------------|------------------|---------------|")
        for r in low_omega_cases:
            mre = f"{r['median_rel_err_R']:.4e}" if r['median_rel_err_R'] is not None else "N/A"
            mxe = f"{r['max_rel_err_R']:.4e}" if r['max_rel_err_R'] is not None else "N/A"
            lines.append(
                f"| {r['case']} | {r['a']:.4f} | {r['omega']:.6f} "
                f"| {r['best_y_mean']:.6f} | {mre} | {mxe} |"
            )
    else:
        lines.append("No low-omega cases detected.")

    # Metrics
    valid_errs = [r["median_rel_err_R"] for r in case_table if r["median_rel_err_R"] is not None]
    if valid_errs:
        lines.append("")
        lines.append("## Summary Metrics")
        lines.append(f"- median median_rel_err_R: {float(np.median(valid_errs)):.4e}")
        lines.append(f"- max median_rel_err_R: {float(np.max(valid_errs)):.4e}")
        lines.append(f"- mean median_rel_err_R: {float(np.mean(valid_errs)):.4e}")

    out_path = run_dir / "logs" / "benchmark_summary.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(lines) + "\n"
    out_path.write_text(content, encoding="utf-8")

    print(f"Written: {out_path}")
    print()
    print(content)


def main():
    parser = argparse.ArgumentParser(description="Summarize Stage-1 benchmark")
    parser.add_argument("--run-dir", type=str, required=True)
    args = parser.parse_args()
    summarize(args.run_dir)


if __name__ == "__main__":
    main()
