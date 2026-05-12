#!/usr/bin/env python3
"""
Simplified comparison of Mathematica, Spectral, and GSN for R_in.
Uses existing mma/rin_sampler.py infrastructure.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from mma.rin_sampler import MathematicaRinSampler
from benchmark.scripts.benchmark_9patch_vs_gsn_spectral import gsn_eval_rin


def spectral_eval_rin(s, l, m, a, omega, r_grid, M=1.0, N=64, z_m=0.3):
    """Evaluate R_in using spectral method."""
    try:
        from physical_ansatz.spectral_solver import SpectralSolver
        solver = SpectralSolver(s=s, l=l, m=m, a=a, omega=omega, M=M, N=N, z_m=z_m)
        R_in = solver.evaluate_Rin(r_grid)
        return R_in
    except Exception as e:
        print(f"    Spectral failed: {e}")
        return None


def compare_at_omega(
    s, l, m, a, omega, r_grid, M,
    mma_sampler, use_spectral, gsn_timeout, mma_timeout
):
    """Compare three methods at one omega."""
    results = {'a': a, 'omega': omega, 'r_grid': r_grid}

    # MMA
    if mma_sampler:
        print(f"  [MMA]", end=' ', flush=True)
        t0 = time.time()
        try:
            r_mma, R_mma = mma_sampler.sample_rin_on_grid(
                s=s, l=l, m=m, a=a, omega=omega,
                rmin=r_grid[0], rmax=r_grid[-1], npts=len(r_grid)
            )
            # Interpolate to exact r_grid
            R_mma_interp = np.interp(r_grid, r_mma, R_mma.real) + 1j * np.interp(r_grid, r_mma, R_mma.imag)
            results['mma'] = R_mma_interp
            results['mma_time'] = time.time() - t0
            print(f"✓ ({results['mma_time']:.2f}s)")
        except Exception as e:
            print(f"✗ ({e})")
            results['mma'] = None

    # Spectral
    if use_spectral:
        print(f"  [Spectral]", end=' ', flush=True)
        t0 = time.time()
        R_spec = spectral_eval_rin(s, l, m, a, omega, r_grid, M)
        if R_spec is not None:
            results['spectral'] = R_spec
            results['spectral_time'] = time.time() - t0
            print(f"✓ ({results['spectral_time']:.2f}s)")
        else:
            results['spectral'] = None

    # GSN
    print(f"  [GSN]", end=' ', flush=True)
    t0 = time.time()
    R_gsn = gsn_eval_rin(s, l, m, a, omega, r_grid, M=M, timeout=gsn_timeout)
    if R_gsn is not None:
        results['gsn'] = R_gsn
        results['gsn_time'] = time.time() - t0
        print(f"✓ ({results['gsn_time']:.2f}s)")
    else:
        results['gsn'] = None

    # Compare
    if results.get('mma') is not None and results.get('gsn') is not None:
        err = np.abs(results['mma'] - results['gsn']) / np.maximum(np.abs(results['gsn']), 1e-14)
        print(f"    MMA vs GSN: max={np.max(err):.4e}, mean={np.mean(err):.4e}")

    if results.get('spectral') is not None and results.get('gsn') is not None:
        err = np.abs(results['spectral'] - results['gsn']) / np.maximum(np.abs(results['gsn']), 1e-14)
        print(f"    Spectral vs GSN: max={np.max(err):.4e}, mean={np.mean(err):.4e}")

    return results


def plot_results(results_list, out_path):
    """Plot comparison."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    omega_vals = [r['omega'] for r in results_list]

    # Collect errors
    mma_gsn_max, mma_gsn_mean = [], []
    spec_gsn_max, spec_gsn_mean = [], []

    for res in results_list:
        R_mma, R_spec, R_gsn = res.get('mma'), res.get('spectral'), res.get('gsn')

        if R_mma is not None and R_gsn is not None:
            err = np.abs(R_mma - R_gsn) / np.maximum(np.abs(R_gsn), 1e-14)
            mma_gsn_max.append(np.max(err))
            mma_gsn_mean.append(np.mean(err))
        else:
            mma_gsn_max.append(np.nan)
            mma_gsn_mean.append(np.nan)

        if R_spec is not None and R_gsn is not None:
            err = np.abs(R_spec - R_gsn) / np.maximum(np.abs(R_gsn), 1e-14)
            spec_gsn_max.append(np.max(err))
            spec_gsn_mean.append(np.mean(err))
        else:
            spec_gsn_max.append(np.nan)
            spec_gsn_mean.append(np.nan)

    # Plot
    axes[0, 0].semilogy(omega_vals, mma_gsn_max, 'o-', label='Max', color='C0')
    axes[0, 0].semilogy(omega_vals, mma_gsn_mean, 's-', label='Mean', color='C1')
    axes[0, 0].axhline(0.01, color='red', linestyle='--', alpha=0.5)
    axes[0, 0].set_xlabel('ω')
    axes[0, 0].set_ylabel('Relative Error')
    axes[0, 0].set_title('MMA vs GSN')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    axes[0, 1].semilogy(omega_vals, spec_gsn_max, 'o-', label='Max', color='C0')
    axes[0, 1].semilogy(omega_vals, spec_gsn_mean, 's-', label='Mean', color='C1')
    axes[0, 1].axhline(0.01, color='red', linestyle='--', alpha=0.5)
    axes[0, 1].set_xlabel('ω')
    axes[0, 1].set_ylabel('Relative Error')
    axes[0, 1].set_title('Spectral vs GSN')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    # Times
    mma_times = [r.get('mma_time', np.nan) for r in results_list]
    spec_times = [r.get('spectral_time', np.nan) for r in results_list]
    gsn_times = [r.get('gsn_time', np.nan) for r in results_list]

    axes[1, 0].plot(omega_vals, mma_times, 'o-', label='MMA', color='C0')
    axes[1, 0].plot(omega_vals, spec_times, 's-', label='Spectral', color='C1')
    axes[1, 0].plot(omega_vals, gsn_times, '^-', label='GSN', color='C2')
    axes[1, 0].set_xlabel('ω')
    axes[1, 0].set_ylabel('Time (s)')
    axes[1, 0].set_title('Computation Time')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)

    # Recommendation
    axes[1, 1].axis('off')
    text = "Recommended benchmark:\n\n"
    for res in results_list:
        omega = res['omega']
        R_mma, R_gsn = res.get('mma'), res.get('gsn')
        if R_mma is not None and R_gsn is not None:
            err = np.max(np.abs(R_mma - R_gsn) / np.maximum(np.abs(R_gsn), 1e-14))
            method = "GSN" if err < 0.01 else f"MMA (GSN err {err:.2%})"
        elif R_gsn is not None:
            method = "GSN"
        elif R_mma is not None:
            method = "MMA"
        else:
            method = "NONE"
        text += f"ω={omega:.4f}: {method}\n"

    axes[1, 1].text(0.1, 0.9, text, transform=axes[1, 1].transAxes,
                    fontsize=9, verticalalignment='top', family='monospace')

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[saved] {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--s", type=int, default=-2)
    parser.add_argument("--l", type=int, default=2)
    parser.add_argument("--m", type=int, default=2)
    parser.add_argument("--a", type=float, default=0.5)
    parser.add_argument("--M", type=float, default=1.0)

    parser.add_argument("--omega-min", type=float, default=0.01)
    parser.add_argument("--omega-max", type=float, default=1.0)
    parser.add_argument("--n-omega", type=int, default=15)

    parser.add_argument("--n-r", type=int, default=200)
    parser.add_argument("--r-max", type=float, default=500.0)

    parser.add_argument("--mma-kernel", type=str, default="/mnt/f/mma/WolframKernel.exe")
    parser.add_argument("--mma-wl", type=str, default="F:/EMRI/Radial_flow/Radial_Function.wl")
    parser.add_argument("--mma-timeout", type=float, default=60.0)
    parser.add_argument("--gsn-timeout", type=float, default=60.0)

    parser.add_argument("--skip-mma", action="store_true")
    parser.add_argument("--skip-spectral", action="store_true")

    parser.add_argument("--out-dir", type=str, default="benchmark/outputs/method_comparison")

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    omega_grid = np.linspace(args.omega_min, args.omega_max, args.n_omega)
    rp = 1.0 + np.sqrt(1.0 - args.a**2)
    r_grid = np.linspace(rp + 1e-4, args.r_max, args.n_r)

    print("=" * 80)
    print("MMA / Spectral / GSN Comparison")
    print("=" * 80)
    print(f"a={args.a}, ω ∈ [{args.omega_min}, {args.omega_max}] ({args.n_omega} points)")
    print("=" * 80)

    # Init MMA sampler (持续一个 kernel)
    mma_sampler = None
    if not args.skip_mma:
        print("[init] MMA sampler...")
        mma_sampler = MathematicaRinSampler(
            kernel_path=args.mma_kernel,
            wl_path_win=args.mma_wl,
            timeout_sec=args.mma_timeout
        )
        print("  ✓ MMA kernel started")

    # Run comparison
    results_list = []
    for i, omega in enumerate(omega_grid):
        print(f"\n[{i+1}/{len(omega_grid)}] ω = {omega:.6f}")
        res = compare_at_omega(
            args.s, args.l, args.m, args.a, omega, r_grid, args.M,
            mma_sampler, not args.skip_spectral, args.gsn_timeout, args.mma_timeout
        )
        results_list.append(res)

    # Cleanup
    if mma_sampler:
        mma_sampler.close()
        print("\n[cleanup] MMA kernel closed")

    # Save
    out_json = out_dir / f"comparison_a{args.a:.3f}.json"
    results_save = []
    for res in results_list:
        res_copy = res.copy()
        for k in ['r_grid', 'mma', 'spectral', 'gsn']:
            if res_copy.get(k) is not None:
                arr = res_copy[k]
                if np.iscomplexobj(arr):
                    res_copy[k] = {'real': arr.real.tolist(), 'imag': arr.imag.tolist()}
                else:
                    res_copy[k] = arr.tolist()
        results_save.append(res_copy)

    with open(out_json, 'w') as f:
        json.dump({'parameters': vars(args), 'results': results_save}, f, indent=2)
    print(f"\n[saved] {out_json}")

    # Plot
    out_png = out_dir / f"comparison_a{args.a:.3f}.png"
    plot_results(results_list, out_png)

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()
