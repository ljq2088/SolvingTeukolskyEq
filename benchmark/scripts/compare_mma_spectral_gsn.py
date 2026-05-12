#!/usr/bin/env python3
"""
Compare Mathematica, Spectral method, and GSN for R_in computation.

Purpose:
- Verify consistency in ω ∈ [1e-2, 1.0]
- Determine which method to use as benchmark:
  - Low frequency (ω < 1e-2): Use Mathematica
  - High frequency (ω >= 1e-2): Use GSN

Usage:
    python compare_mma_spectral_gsn.py --omega-min 0.01 --omega-max 1.0 --n-omega 20
"""
import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

# Add project root to path
SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from benchmark.utils.mathematica_rin_sampler import MathematicaRinSampler
from benchmark.utils.spectral_rin_sampler import SpectralRinSampler
from benchmark.scripts.benchmark_9patch_vs_gsn_spectral import gsn_eval_rin


def compare_three_methods(
    s: int,
    l: int,
    m: int,
    a: float,
    omega: float,
    r_grid: np.ndarray,
    M: float = 1.0,
    mma_sampler: Optional[MathematicaRinSampler] = None,
    spectral_sampler: Optional[SpectralRinSampler] = None,
    gsn_timeout: float = 30.0,
    mma_timeout: float = 30.0,
):
    """
    Compare R_in from three methods at a single (a, omega) point.

    Returns:
        dict with keys: 'mma', 'spectral', 'gsn', 'r_grid'
        Each method returns complex array or None if failed
    """
    results = {
        'a': a,
        'omega': omega,
        'r_grid': r_grid,
        'mma': None,
        'spectral': None,
        'gsn': None,
        'mma_time': None,
        'spectral_time': None,
        'gsn_time': None,
    }

    # 1. Mathematica
    if mma_sampler is not None:
        print(f"  [MMA] Computing R_in at (a={a:.4f}, ω={omega:.6f})...", end=' ', flush=True)
        t0 = time.time()
        try:
            R_mma = mma_sampler.sample_Rin(
                a=a, omega=omega, r_list=r_grid.tolist(),
                s=s, l=l, m=m, M=M, timeout_sec=mma_timeout
            )
            if R_mma is not None:
                results['mma'] = np.array(R_mma)
                results['mma_time'] = time.time() - t0
                print(f"✓ ({results['mma_time']:.2f}s)")
            else:
                print("✗ (failed)")
        except Exception as e:
            print(f"✗ (error: {e})")

    # 2. Spectral method
    if spectral_sampler is not None:
        print(f"  [Spectral] Computing R_in...", end=' ', flush=True)
        t0 = time.time()
        try:
            R_spec = spectral_sampler.sample_Rin(
                a=a, omega=omega, r_list=r_grid.tolist(),
                s=s, l=l, m=m, M=M
            )
            if R_spec is not None:
                results['spectral'] = np.array(R_spec)
                results['spectral_time'] = time.time() - t0
                print(f"✓ ({results['spectral_time']:.2f}s)")
            else:
                print("✗ (failed)")
        except Exception as e:
            print(f"✗ (error: {e})")

    # 3. GSN
    print(f"  [GSN] Computing R_in...", end=' ', flush=True)
    t0 = time.time()
    try:
        R_gsn = gsn_eval_rin(s, l, m, a, omega, r_grid, M=M, timeout=gsn_timeout)
        if R_gsn is not None:
            results['gsn'] = R_gsn
            results['gsn_time'] = time.time() - t0
            print(f"✓ ({results['gsn_time']:.2f}s)")
        else:
            print("✗ (failed)")
    except Exception as e:
        print(f"✗ (error: {e})")

    return results


def compute_relative_errors(R_ref: np.ndarray, R_test: np.ndarray, eps: float = 1e-14):
    """Compute relative errors between reference and test."""
    abs_err = np.abs(R_test - R_ref)
    rel_err = abs_err / np.maximum(np.abs(R_ref), eps)
    return {
        'max_abs': np.max(abs_err),
        'mean_abs': np.mean(abs_err),
        'max_rel': np.max(rel_err),
        'mean_rel': np.mean(rel_err),
        'median_rel': np.median(rel_err),
    }


def plot_comparison(results_list: list, out_path: Path):
    """Plot comparison of three methods across omega range."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    omega_vals = [r['omega'] for r in results_list]

    # Collect error statistics
    mma_vs_gsn_max = []
    mma_vs_gsn_mean = []
    spec_vs_gsn_max = []
    spec_vs_gsn_mean = []
    mma_vs_spec_max = []
    mma_vs_spec_mean = []

    for res in results_list:
        R_mma = res.get('mma')
        R_spec = res.get('spectral')
        R_gsn = res.get('gsn')

        if R_mma is not None and R_gsn is not None:
            err = compute_relative_errors(R_gsn, R_mma)
            mma_vs_gsn_max.append(err['max_rel'])
            mma_vs_gsn_mean.append(err['mean_rel'])
        else:
            mma_vs_gsn_max.append(np.nan)
            mma_vs_gsn_mean.append(np.nan)

        if R_spec is not None and R_gsn is not None:
            err = compute_relative_errors(R_gsn, R_spec)
            spec_vs_gsn_max.append(err['max_rel'])
            spec_vs_gsn_mean.append(err['mean_rel'])
        else:
            spec_vs_gsn_max.append(np.nan)
            spec_vs_gsn_mean.append(np.nan)

        if R_mma is not None and R_spec is not None:
            err = compute_relative_errors(R_spec, R_mma)
            mma_vs_spec_max.append(err['max_rel'])
            mma_vs_spec_mean.append(err['mean_rel'])
        else:
            mma_vs_spec_max.append(np.nan)
            mma_vs_spec_mean.append(np.nan)

    # Plot 1: MMA vs GSN
    axes[0, 0].semilogy(omega_vals, mma_vs_gsn_max, 'o-', label='Max rel err', color='C0')
    axes[0, 0].semilogy(omega_vals, mma_vs_gsn_mean, 's-', label='Mean rel err', color='C1')
    axes[0, 0].axhline(0.01, color='red', linestyle='--', alpha=0.5, label='1% threshold')
    axes[0, 0].set_xlabel('ω')
    axes[0, 0].set_ylabel('Relative Error')
    axes[0, 0].set_title('Mathematica vs GSN')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    # Plot 2: Spectral vs GSN
    axes[0, 1].semilogy(omega_vals, spec_vs_gsn_max, 'o-', label='Max rel err', color='C0')
    axes[0, 1].semilogy(omega_vals, spec_vs_gsn_mean, 's-', label='Mean rel err', color='C1')
    axes[0, 1].axhline(0.01, color='red', linestyle='--', alpha=0.5, label='1% threshold')
    axes[0, 1].set_xlabel('ω')
    axes[0, 1].set_ylabel('Relative Error')
    axes[0, 1].set_title('Spectral vs GSN')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    # Plot 3: MMA vs Spectral
    axes[1, 0].semilogy(omega_vals, mma_vs_spec_max, 'o-', label='Max rel err', color='C0')
    axes[1, 0].semilogy(omega_vals, mma_vs_spec_mean, 's-', label='Mean rel err', color='C1')
    axes[1, 0].axhline(0.01, color='red', linestyle='--', alpha=0.5, label='1% threshold')
    axes[1, 0].set_xlabel('ω')
    axes[1, 0].set_ylabel('Relative Error')
    axes[1, 0].set_title('Mathematica vs Spectral')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)

    # Plot 4: Computation time
    mma_times = [r.get('mma_time', np.nan) for r in results_list]
    spec_times = [r.get('spectral_time', np.nan) for r in results_list]
    gsn_times = [r.get('gsn_time', np.nan) for r in results_list]

    axes[1, 1].plot(omega_vals, mma_times, 'o-', label='Mathematica', color='C0')
    axes[1, 1].plot(omega_vals, spec_times, 's-', label='Spectral', color='C1')
    axes[1, 1].plot(omega_vals, gsn_times, '^-', label='GSN', color='C2')
    axes[1, 1].set_xlabel('ω')
    axes[1, 1].set_ylabel('Time (s)')
    axes[1, 1].set_title('Computation Time')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[saved] Comparison plot -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare MMA/Spectral/GSN for R_in")
    parser.add_argument("--s", type=int, default=-2)
    parser.add_argument("--l", type=int, default=2)
    parser.add_argument("--m", type=int, default=2)
    parser.add_argument("--a", type=float, default=0.5, help="Spin parameter")
    parser.add_argument("--M", type=float, default=1.0)

    parser.add_argument("--omega-min", type=float, default=0.01, help="Min frequency")
    parser.add_argument("--omega-max", type=float, default=1.0, help="Max frequency")
    parser.add_argument("--n-omega", type=int, default=20, help="Number of omega points")
    parser.add_argument("--omega-log", action="store_true", help="Use log spacing for omega")

    parser.add_argument("--n-r", type=int, default=200, help="Number of r points")
    parser.add_argument("--r-max", type=float, default=500.0)

    parser.add_argument("--mma-kernel", type=str, default="/mnt/f/mma/WolframKernel.exe")
    parser.add_argument("--mma-wl", type=str, default="F:/EMRI/Radial_flow/Radial_Function.wl")
    parser.add_argument("--mma-timeout", type=float, default=30.0)
    parser.add_argument("--gsn-timeout", type=float, default=30.0)

    parser.add_argument("--spectral-N", type=int, default=64)
    parser.add_argument("--spectral-z-m", type=float, default=0.3)

    parser.add_argument("--skip-mma", action="store_true")
    parser.add_argument("--skip-spectral", action="store_true")
    parser.add_argument("--skip-gsn", action="store_true")

    parser.add_argument("--out-dir", type=str, default="benchmark/outputs/method_comparison")

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Generate omega grid
    if args.omega_log:
        omega_grid = np.logspace(np.log10(args.omega_min), np.log10(args.omega_max), args.n_omega)
    else:
        omega_grid = np.linspace(args.omega_min, args.omega_max, args.n_omega)

    # Generate r grid
    rp = 1.0 + np.sqrt(1.0 - args.a**2)
    r_grid = np.linspace(rp + 1e-4, args.r_max, args.n_r)

    print("=" * 80)
    print("Comparing Mathematica / Spectral / GSN for R_in computation")
    print("=" * 80)
    print(f"Parameters: s={args.s}, l={args.l}, m={args.m}, a={args.a:.4f}, M={args.M}")
    print(f"Omega range: [{args.omega_min:.6f}, {args.omega_max:.6f}] ({args.n_omega} points)")
    print(f"r grid: [{r_grid[0]:.4f}, {r_grid[-1]:.4f}] ({args.n_r} points)")
    print("=" * 80)

    # Initialize samplers (持续使用一个 kernel)
    mma_sampler = None
    spectral_sampler = None

    if not args.skip_mma:
        print("[init] Mathematica sampler...")
        mma_sampler = MathematicaRinSampler(
            kernel_path=args.mma_kernel,
            wl_path_win=args.mma_wl,
        )
        print("  ✓ Mathematica kernel started")

    if not args.skip_spectral:
        print("[init] Spectral sampler...")
        spectral_sampler = SpectralRinSampler(
            N=args.spectral_N,
            z_m=args.spectral_z_m,
        )
        print("  ✓ Spectral sampler ready")

    # Run comparison
    results_list = []

    for i, omega in enumerate(omega_grid):
        print(f"\n[{i+1}/{len(omega_grid)}] ω = {omega:.6f}")

        res = compare_three_methods(
            s=args.s, l=args.l, m=args.m,
            a=args.a, omega=omega, r_grid=r_grid, M=args.M,
            mma_sampler=mma_sampler,
            spectral_sampler=spectral_sampler,
            gsn_timeout=args.gsn_timeout,
            mma_timeout=args.mma_timeout,
        )

        results_list.append(res)

        # Print comparison if all three succeeded
        R_mma = res.get('mma')
        R_spec = res.get('spectral')
        R_gsn = res.get('gsn')

        if R_mma is not None and R_gsn is not None:
            err = compute_relative_errors(R_gsn, R_mma)
            print(f"    MMA vs GSN: max_rel={err['max_rel']:.4e}, mean_rel={err['mean_rel']:.4e}")

        if R_spec is not None and R_gsn is not None:
            err = compute_relative_errors(R_gsn, R_spec)
            print(f"    Spectral vs GSN: max_rel={err['max_rel']:.4e}, mean_rel={err['mean_rel']:.4e}")

        if R_mma is not None and R_spec is not None:
            err = compute_relative_errors(R_spec, R_mma)
            print(f"    MMA vs Spectral: max_rel={err['max_rel']:.4e}, mean_rel={err['mean_rel']:.4e}")

    # Cleanup
    if mma_sampler is not None:
        mma_sampler.close()
        print("\n[cleanup] Mathematica kernel closed")

    # Save results
    results_json = out_dir / f"comparison_a{args.a:.3f}_omega{args.omega_min:.4f}to{args.omega_max:.4f}.json"

    # Convert numpy arrays to lists for JSON serialization
    results_serializable = []
    for res in results_list:
        res_copy = res.copy()
        for key in ['r_grid', 'mma', 'spectral', 'gsn']:
            if res_copy.get(key) is not None:
                arr = res_copy[key]
                if np.iscomplexobj(arr):
                    res_copy[key] = {'real': arr.real.tolist(), 'imag': arr.imag.tolist()}
                else:
                    res_copy[key] = arr.tolist()
        results_serializable.append(res_copy)

    with open(results_json, 'w') as f:
        json.dump({
            'parameters': {
                's': args.s, 'l': args.l, 'm': args.m,
                'a': args.a, 'M': args.M,
                'omega_min': args.omega_min,
                'omega_max': args.omega_max,
                'n_omega': args.n_omega,
            },
            'results': results_serializable,
        }, f, indent=2)

    print(f"\n[saved] Results -> {results_json}")

    # Plot
    plot_path = out_dir / f"comparison_a{args.a:.3f}_omega{args.omega_min:.4f}to{args.omega_max:.4f}.png"
    plot_comparison(results_list, plot_path)

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    # Determine recommended method for each omega range
    print("\nRecommended benchmark method:")
    for res in results_list:
        omega = res['omega']
        R_mma = res.get('mma')
        R_gsn = res.get('gsn')

        if R_mma is not None and R_gsn is not None:
            err = compute_relative_errors(R_gsn, R_mma)
            if err['max_rel'] < 0.01:
                method = "GSN (consistent with MMA)"
            else:
                method = f"MMA (GSN differs by {err['max_rel']:.2%})"
        elif R_gsn is not None:
            method = "GSN (MMA unavailable)"
        elif R_mma is not None:
            method = "MMA (GSN unavailable)"
        else:
            method = "NONE (both failed)"

        print(f"  ω = {omega:.6f}: {method}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
