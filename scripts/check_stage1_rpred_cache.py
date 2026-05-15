#!/usr/bin/env python3
"""Check Stage-1 R_pred cache integrity."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Check Stage-1 R_pred cache")
    parser.add_argument("--artifact-dir", type=str, required=True)
    args = parser.parse_args()

    ad = Path(args.artifact_dir)
    checks = {}
    errors = []

    # Check files exist
    for fname in ["metadata.json", "rpred_cache.npz", "stage1_best_model.pt", "rpred_cache_summary.md"]:
        p = ad / fname
        checks[f"file:{fname}"] = p.exists()
        if not p.exists():
            errors.append(f"MISSING: {fname}")

    # Load metadata
    with open(ad / "metadata.json") as f:
        meta = json.load(f)
    checks["meta:scattering_convention"] = "R_in = B_ref" in meta.get("scattering_convention", "")
    checks["meta:encoder_config"] = "encoder_config" in meta
    checks["meta:checkpoint"] = "checkpoint" in meta

    # Load npz
    data = np.load(ad / "rpred_cache.npz")
    keys_needed = ["a", "omega", "u", "v", "y", "r", "lambda_",
                   "S_pred", "P", "h2", "R_pred", "R_pred_over_P", "mask_near_infinity"]
    for k in keys_needed:
        checks[f"npz:{k}"] = k in data
        if k not in data:
            errors.append(f"MISSING npz key: {k}")

    if not all(checks.get(f"npz:{k}", False) for k in ["a", "R_pred"]):
        print("SKIP: core arrays missing")
        for k, v in checks.items():
            print(f"  {'PASS' if v else 'FAIL'}: {k}")
        return

    a = data["a"]
    omega = data["omega"]
    n_param = len(a)
    n_y = data["y"].shape[1]
    print(f"n_param={n_param}, n_y={n_y}")

    S = data["S_pred"]
    P_arr = data["P"]
    h2_arr = data["h2"]
    R = data["R_pred"]
    R_over_P = data["R_pred_over_P"]
    mask_inf = data["mask_near_infinity"]

    # Shape checks
    checks["shape:S_pred"] = S.shape == (n_param, n_y)
    checks["shape:P"] = P_arr.shape == (n_param, n_y)
    checks["shape:h2"] = h2_arr.shape == (n_param,)
    checks["shape:R_pred"] = R.shape == (n_param, n_y)
    checks["shape:R_pred_over_P"] = R_over_P.shape == (n_param, n_y)
    checks["shape:a"] = len(a) == n_param
    checks["shape:omega"] = len(omega) == n_param
    checks["shape:mask"] = mask_inf.shape == (n_param, n_y)
    checks["shape:r"] = len(data["r"]) == n_y

    for k, v in list(checks.items()):
        if k.startswith("shape:") and not v:
            errors.append(f"SHAPE MISMATCH: {k}")

    # Finite checks
    checks["finite:S_pred"] = bool(np.all(np.isfinite(S)))
    checks["finite:P"] = bool(np.all(np.isfinite(P_arr)))
    checks["finite:R_pred"] = bool(np.all(np.isfinite(R)))
    checks["finite:R_pred_over_P"] = bool(np.all(np.isfinite(R_over_P)))

    for k, v in list(checks.items()):
        if k.startswith("finite:") and not v:
            errors.append(f"NON-FINITE: {k}")

    # R = P*h2*S check
    R_recon = P_arr * h2_arr[:, None] * S
    rel_diff = np.abs(R - R_recon) / np.maximum(np.abs(R_recon), 1e-14)
    max_reldiff = float(np.max(rel_diff))
    checks["consistency:R=Ph2S"] = max_reldiff < 1e-4
    if not checks["consistency:R=Ph2S"]:
        errors.append(f"R=Ph2S check FAILED: max_rel_diff={max_reldiff:.2e}")
    print(f"R = P*h2*S  max rel diff = {max_reldiff:.2e}")

    # R_pred_over_P ≈ R/P check
    with np.errstate(divide="ignore", invalid="ignore"):
        R_over_P_check = R / P_arr
    rel_diff2 = np.abs(R_over_P - R_over_P_check) / np.maximum(np.abs(R_over_P_check), 1e-14)
    max_reldiff2 = float(np.nanmax(rel_diff2))
    checks["consistency:RoverP"] = max_reldiff2 < 1e-4
    if not checks["consistency:RoverP"]:
        errors.append(f"R_over_P check FAILED: max_rel_diff={max_reldiff2:.2e}")
    print(f"R_over_P = R/P  max rel diff = {max_reldiff2:.2e}")

    # Infinity coverage
    n_inf = int(mask_inf.sum())
    checks["infinity:has_points"] = n_inf > 0
    if not checks["infinity:has_points"]:
        errors.append("No points near infinity")

    print(f"Near-infinity points: {n_inf}/{mask_inf.size}")

    all_pass = all(v for k, v in checks.items())
    n_fail = sum(1 for v in checks.values() if not v)

    print()
    print(f"{'='*50}")
    print(f"CHECKS: {len(checks)} total, {len(checks) - n_fail} pass, {n_fail} fail")
    if errors:
        print("ERRORS:")
        for e in errors:
            print(f"  - {e}")
    else:
        print("ALL CHECKS PASSED")
    print(f"{'='*50}")

    if not all_pass:
        sys.exit(1)


if __name__ == "__main__":
    main()
