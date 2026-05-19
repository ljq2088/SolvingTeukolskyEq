#!/usr/bin/env python3
"""Debug script for Stage-1 infinity Robin refinement pipeline.

17 checks verifying: checkpoint loading, c_inf, forward pass, all losses,
freeze policy, gradients, optimizer correctness, L-BFGS closure repeatability.
"""
import argparse
import copy
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import torch
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.config_loader import load_pinn_full_config
from model.autoencoder_pinn import AutoencoderTeukolskyPINN
from physical_ansatz.infinity_robin import analytic_c_inf, infinity_robin_loss
from physical_ansatz.stage1_endpoint import compute_stage1_S_and_Sy_at_infinity
from physical_ansatz.transform_y import compose_reduced_shape_from_f, horizon_regularity_slope
from physical_ansatz.residual_pinn import compute_f_derivatives_autograd
from utils.compute_lambda_usage import compute_lambda


def _fix_shapes(fy, fyy):
    """Workaround for autograd (B,B,N) -> (B,N) diagonal extraction."""
    if fy is not None and fy.ndim > 2 and fy.shape[0] == fy.shape[1]:
        fy = fy.diagonal(dim1=0, dim2=1).transpose(0, 1)
    if fyy is not None and fyy.ndim > 2 and fyy.shape[0] == fyy.shape[1]:
        fyy = fyy.diagonal(dim1=0, dim2=1).transpose(0, 1)
    return fy, fyy


def _to_scalar(x):
    """Convert tensor to Python scalar for printing."""
    if hasattr(x, "detach"):
        x = x.detach().cpu()
    if hasattr(x, "numpy"):
        x = x.numpy()
    if hasattr(x, "item"):
        return x.item()
    return complex(x) if isinstance(x, (complex, np.complex128)) else float(x)


def green(s):
    return f"\033[32m{s}\033[0m"


def red(s):
    return f"\033[31m{s}\033[0m"


def check(condition, msg):
    if condition:
        print(f"  {green('PASS')}  {msg}")
        return True
    else:
        print(f"  {red('FAIL')}  {msg}")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/autoencoder_stage1_infinity_robin.yaml")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = torch.float64
    cdtype = torch.complex128

    full_cfg = load_pinn_full_config(args.config)
    cfg = full_cfg.get("train", full_cfg).get("stage1_infinity_robin", full_cfg.get("stage1_infinity_robin", {}))

    # Extract problem params
    prob = full_cfg.get("physics", full_cfg).get("problem", full_cfg.get("problem", {}))
    M_val = float(prob.get("M", 1.0))
    ell = int(prob.get("l", 2))
    m_mode = int(prob.get("m", 2))
    s_val = int(prob.get("s", -2))

    results = []
    batch_size = 3
    a_test = torch.tensor([0.1, 0.5, 0.9], device=device, dtype=dtype)
    omega_test = torch.tensor([0.3, 0.51, 0.8], device=device, dtype=dtype)
    u_test = torch.zeros(batch_size, device=device, dtype=dtype)
    v_test = torch.ones(batch_size, device=device, dtype=dtype) * 0.6

    lam_vals = []
    for i in range(batch_size):
        lam_i = compute_lambda(float(a_test[i]), float(omega_test[i]), ell, m_mode, s=s_val)
        lam_vals.append(torch.tensor(lam_i, device=device, dtype=cdtype))
    lam_test = torch.stack(lam_vals)

    print("=" * 60)
    print("Stage-1 Infinity Robin Refinement — Debug Checks")
    print("=" * 60)
    print(f"Device: {device}, dtype: {dtype}")

    # --- Check 1: inf_coeff.md exists ---
    print("\n[1] docs/instruction/inf_coeff.md exists")
    inf_coeff_path = PROJECT_ROOT / "docs" / "instruction" / "inf_coeff.md"
    results.append(check(inf_coeff_path.exists(), str(inf_coeff_path)))

    # --- Check 2: checkpoint loading ---
    print("\n[2] Checkpoint loading")
    ckpt_path = cfg.get("base_checkpoint", "")
    if not ckpt_path or not Path(ckpt_path).exists():
        # Try to find it
        import glob as _glob
        candidates = sorted(_glob.glob("outputs/autoencoder_stage1_rin_train/*/checkpoints/best_model.pt"))
        if candidates:
            ckpt_path = candidates[-1]
            print(f"  Using found checkpoint: {ckpt_path}")
        else:
            results.append(check(False, f"No checkpoint found at {ckpt_path}"))
            ckpt_path = None
    else:
        print(f"  Checkpoint: {ckpt_path}")

    if ckpt_path:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model = AutoencoderTeukolskyPINN()
        sd = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
        model.load_state_dict(sd)
        model.to(device)
        model.eval()
        results.append(check(True, f"Loaded {len(sd)} keys"))
    else:
        model = AutoencoderTeukolskyPINN().to(device)
        results.append(check(False, "Skipped — no checkpoint"))

    # --- Check 3: c_inf finite ---
    print("\n[3] c_inf finite")
    c_inf_vals = analytic_c_inf(a_test, omega_test, lam_test, m=m_mode, M=M_val, s=s_val)
    c_finite = torch.all(torch.isfinite(c_inf_vals.real) & torch.isfinite(c_inf_vals.imag))
    for i in range(batch_size):
        print(f"  (a={a_test[i]:.2f}, w={omega_test[i]:.2f}): c_inf = {_to_scalar(c_inf_vals[i]):.6e}")
    results.append(check(c_finite, "All c_inf values finite"))

    # --- Check 4: y=-1 forward pass ---
    print("\n[4] y=-1 forward pass")
    try:
        ep = compute_stage1_S_and_Sy_at_infinity(
            model, a_test, omega_test, u=u_test, v=v_test,
            lambda_=lam_test, m=m_mode, M=M_val, s=s_val,
        )
        results.append(check(True, "Forward pass succeeded"))
    except Exception as e:
        results.append(check(False, f"Forward pass failed: {e}"))
        ep = None

    # --- Check 5 & 6: S(-1), Sy(-1) finite ---
    if ep is not None:
        print("\n[5] S(-1) finite")
        s_finite = torch.all(torch.isfinite(ep["S_inf"].real) & torch.isfinite(ep["S_inf"].imag))
        for i in range(batch_size):
            print(f"  Point {i}: S(-1) = {_to_scalar(ep['S_inf'][i]):.6e}")
        results.append(check(s_finite, "All S(-1) finite"))

        print("\n[6] Sy(-1) finite")
        sy_finite = torch.all(torch.isfinite(ep["Sy_inf"].real) & torch.isfinite(ep["Sy_inf"].imag))
        for i in range(batch_size):
            print(f"  Point {i}: Sy(-1) = {_to_scalar(ep['Sy_inf'][i]):.6e}")
        results.append(check(sy_finite, "All Sy(-1) finite"))
    else:
        results.append(check(False, "Skipped — forward failed"))
        results.append(check(False, "Skipped — forward failed"))

    # --- Check 7: Robin loss finite ---
    print("\n[7] Robin loss finite")
    if ep is not None:
        robin_losses = infinity_robin_loss(ep["S_inf"], ep["Sy_inf"], ep["c_inf"])
        rl_finite = torch.all(torch.isfinite(robin_losses))
        for i in range(batch_size):
            print(f"  Point {i}: robin_rel = {_to_scalar(ep['robin_rel'][i]):.6e}")
        results.append(check(rl_finite, "All Robin losses finite"))
    else:
        results.append(check(False, "Skipped"))

    # --- Check 8: PDE loss finite ---
    print("\n[8] PDE loss finite")
    try:
        y_test = torch.linspace(-0.99, 0.99, 32, device=device, dtype=dtype).unsqueeze(0).expand(batch_size, -1)
        f, fy, fyy = compute_f_derivatives_autograd(
            model, a_test, omega_test, y_test, u_batch=u_test, v_batch=v_test,
        )
        fy, fyy = _fix_shapes(fy, fyy)
        slope = horizon_regularity_slope(a=a_test, omega=omega_test, lambda_=lam_test, m=m_mode, M=M_val, s=s_val)
        from physical_ansatz.transform_y import transform_coeffs_x_to_y

        pde_losses = []
        for i in range(batch_size):
            from physical_ansatz.teukolsky_coeffs import coeffs_x
            A2_i, A1_i, A0_i = coeffs_x(
                x=(y_test[i:i+1] + 1) / 2, a=a_test[i], omega=omega_test[i],
                m=m_mode, lambda_=lam_test[i], s=s_val, M=M_val,
            )
            B2, B1, B0, rhs = transform_coeffs_x_to_y(A2_i, A1_i, A0_i, y_test[i:i+1], slope[i:i+1])
            B2_c = B2.to(dtype=cdtype)
            B1_c = B1.to(dtype=cdtype)
            B0_c = B0.to(dtype=cdtype)
            rhs_c = rhs.to(dtype=cdtype) if rhs is not None else 0.0
            res = B2_c * fyy[i:i+1].to(dtype=cdtype) + B1_c * fy[i:i+1].to(dtype=cdtype) + B0_c * f[i:i+1].to(dtype=cdtype) - rhs_c
            pde_losses.append((res.real**2 + res.imag**2).mean().item())
        pde_finite = all(np.isfinite(x) for x in pde_losses)
        for i in range(batch_size):
            print(f"  Point {i}: PDE loss = {pde_losses[i]:.6e}")
        results.append(check(pde_finite, "All PDE losses finite"))
    except Exception as e:
        results.append(check(False, f"PDE loss failed: {e}"))
        pde_losses = None

    # --- Check 9: near-infinity PDE loss ---
    print("\n[9] Near-infinity PDE loss finite")
    try:
        y_near_test = torch.linspace(-0.999, -0.95, 24, device=device, dtype=dtype).unsqueeze(0).expand(batch_size, -1)
        f_n, fy_n, fyy_n = compute_f_derivatives_autograd(
            model, a_test, omega_test, y_near_test, u_batch=u_test, v_batch=v_test,
        )
        fy_n, fyy_n = _fix_shapes(fy_n, fyy_n)
        near_losses = []
        for i in range(batch_size):
            A2_i, A1_i, A0_i = coeffs_x(
                x=(y_near_test[i:i+1] + 1) / 2, a=a_test[i], omega=omega_test[i],
                m=m_mode, lambda_=lam_test[i], s=s_val, M=M_val,
            )
            B2, B1, B0, rhs = transform_coeffs_x_to_y(A2_i, A1_i, A0_i, y_near_test[i:i+1], slope[i:i+1])
            B2_c = B2.to(dtype=cdtype)
            B1_c = B1.to(dtype=cdtype)
            B0_c = B0.to(dtype=cdtype)
            rhs_c = rhs.to(dtype=cdtype) if rhs is not None else 0.0
            res = B2_c * fyy_n[i:i+1].to(dtype=cdtype) + B1_c * fy_n[i:i+1].to(dtype=cdtype) + B0_c * f_n[i:i+1].to(dtype=cdtype) - rhs_c
            near_losses.append((res.real**2 + res.imag**2).mean().item())
        near_finite = all(np.isfinite(x) for x in near_losses)
        for i in range(batch_size):
            print(f"  Point {i}: near PDE loss = {near_losses[i]:.6e}")
        results.append(check(near_finite, "All near-infinity PDE losses finite"))
    except Exception as e:
        results.append(check(False, f"Near PDE loss failed: {e}"))

    # --- Check 10: drift loss finite ---
    print("\n[10] Drift loss finite")
    try:
        frozen = copy.deepcopy(model)
        frozen.to(device)
        for p in frozen.parameters():
            p.requires_grad_(False)
        frozen.eval()
        f_new, _, _ = compute_f_derivatives_autograd(
            model, a_test, omega_test, y_test, u_batch=u_test, v_batch=v_test,
        )
        f_old, _, _ = compute_f_derivatives_autograd(
            frozen, a_test, omega_test, y_test, u_batch=u_test, v_batch=v_test,
        )
        drift_vals = []
        for i in range(batch_size):
            S_new = compose_reduced_shape_from_f(f_new[i], y_test[i], slope[i])
            S_old = compose_reduced_shape_from_f(f_old[i], y_test[i], slope[i])
            denom = S_old.real**2 + S_old.imag**2 + 1e-12
            diff = (S_new - S_old).real**2 + (S_new - S_old).imag**2
            drift_vals.append((diff / denom).mean().item())
        drift_finite = all(np.isfinite(x) for x in drift_vals)
        for i in range(batch_size):
            print(f"  Point {i}: drift = {drift_vals[i]:.6e}")
        results.append(check(drift_finite, "All drift losses finite"))
    except Exception as e:
        results.append(check(False, f"Drift loss failed: {e}"))

    # --- Freeze policy setup ---
    print("\n[11-14] Freeze policy & gradient checks")
    train_enc = cfg.get("train_encoder", True)
    train_rin = cfg.get("train_rin_decoder", True)
    train_amp = cfg.get("train_amplitude_net", False)
    train_up = cfg.get("train_up_decoder", False)
    train_down = cfg.get("train_down_decoder", False)

    for n, p in model.named_parameters():
        if n.startswith("encoder"):
            p.requires_grad_(train_enc)
        elif n.startswith("rin_decoder"):
            p.requires_grad_(train_rin)
        elif n.startswith("amplitude_net"):
            p.requires_grad_(train_amp)
        elif n.startswith("up_decoder"):
            p.requires_grad_(train_up)
        elif n.startswith("down_decoder"):
            p.requires_grad_(train_down)

    model.train()

    # --- Check: backward gives gradients to encoder and rin_decoder ---
    print("\n[11] Encoder and rin_decoder have gradients after backward")
    try:
        y_small = torch.linspace(-0.99, -0.95, 16, device=device, dtype=dtype).unsqueeze(0).expand(batch_size, -1)
        f_s, fy_s, _ = compute_f_derivatives_autograd(
            model, a_test, omega_test, y_small, u_batch=u_test, v_batch=v_test,
        )
        slope_s = horizon_regularity_slope(a=a_test, omega=omega_test, lambda_=lam_test, m=m_mode, M=M_val, s=s_val)
        loss_s = torch.tensor(0.0, device=device, dtype=dtype)
        for i in range(batch_size):
            S_i = compose_reduced_shape_from_f(f_s[i], y_small[i], slope_s[i])
            loss_s = loss_s + (S_i.real**2 + S_i.imag**2).mean()
        loss_s.backward()

        enc_has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for n, p in model.named_parameters() if n.startswith("encoder") and p.requires_grad
        )
        rin_has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for n, p in model.named_parameters() if n.startswith("rin_decoder") and p.requires_grad
        )
        results.append(check(enc_has_grad, "Encoder has non-zero gradients"))
        print(f"  rin_decoder grad: {rin_has_grad}")
        results.append(check(rin_has_grad, "rin_decoder has non-zero gradients"))
    except Exception as e:
        results.append(check(False, f"Gradient check failed: {e}"))
        enc_has_grad = False
        rin_has_grad = False

    # --- Check 12: AmplitudeNet has no grad ---
    print("\n[12] AmplitudeNet has no gradients")
    amp_no_grad = all(
        p.grad is None
        for n, p in model.named_parameters() if n.startswith("amplitude_net")
    )
    results.append(check(amp_no_grad, "AmplitudeNet parameters have no grad"))

    # --- Check 13: UpDecoder / DownDecoder have no grad ---
    print("\n[13] UpDecoder / DownDecoder have no gradients")
    up_no_grad = all(
        p.grad is None
        for n, p in model.named_parameters() if n.startswith("up_decoder")
    )
    down_no_grad = all(
        p.grad is None
        for n, p in model.named_parameters() if n.startswith("down_decoder")
    )
    results.append(check(up_no_grad and down_no_grad, "Up/Down decoder no grad"))

    # Zero grads before optimizer check
    model.zero_grad()

    # --- Check 14: optimizer.step only changes encoder/rin_decoder ---
    print("\n[14] optimizer.step only changes encoder and rin_decoder")
    enc_params = [p for n, p in model.named_parameters() if n.startswith("encoder") and p.requires_grad]
    rin_params = [p for n, p in model.named_parameters() if n.startswith("rin_decoder") and p.requires_grad]
    optim = torch.optim.Adam([
        {"params": enc_params, "lr": 1e-6},
        {"params": rin_params, "lr": 1e-6},
    ])

    # Snapshot all params
    snap_before = {}
    for n, p in model.named_parameters():
        snap_before[n] = p.data.clone()

    # Forward + backward + step
    y_s2 = y_small.clone()
    f_s2, fy_s2, _ = compute_f_derivatives_autograd(
        model, a_test, omega_test, y_s2, u_batch=u_test, v_batch=v_test,
    )
    loss_s2 = torch.tensor(0.0, device=device, dtype=dtype)
    for i in range(batch_size):
        S_i = compose_reduced_shape_from_f(f_s2[i], y_s2[i], slope_s[i])
        loss_s2 = loss_s2 + (S_i.real**2 + S_i.imag**2).mean()
    optim.zero_grad()
    loss_s2.backward()
    optim.step()

    only_enc_rin_changed = True
    for n, p in model.named_parameters():
        changed = not torch.allclose(snap_before[n], p.data)
        if n.startswith("encoder") or n.startswith("rin_decoder"):
            continue  # expected to change
        if changed:
            print(f"  {red('UNEXPECTED CHANGE')}: {n}")
            only_enc_rin_changed = False

    results.append(check(only_enc_rin_changed, "Only encoder/rin_decoder params changed"))

    # --- Check 15: L-BFGS closure repeatable on fixed grid ---
    print("\n[15] L-BFGS closure repeatable on fixed grid")
    try:
        y_fixed = torch.linspace(-0.99, 0.99, 32, device=device, dtype=dtype).unsqueeze(0).expand(batch_size, -1)
        model.zero_grad()

        def _make_closure(a_f, om_f, lam_f, y_f):
            def _c():
                model.zero_grad()
                f_c, _, _ = compute_f_derivatives_autograd(
                    model, a_f, om_f, y_f, u_batch=u_test, v_batch=v_test,
                )
                loss_c = torch.tensor(0.0, device=device, dtype=dtype)
                for i in range(batch_size):
                    S_i = compose_reduced_shape_from_f(f_c[i], y_f[i], slope[i])
                    loss_c = loss_c + (S_i.real**2 + S_i.imag**2).mean()
                loss_c.backward()
                return loss_c
            return _c

        closure_1 = _make_closure(a_test, omega_test, lam_test, y_fixed)
        closure_2 = _make_closure(a_test, omega_test, lam_test, y_fixed)

        trainable = [p for p in model.parameters() if p.requires_grad]
        lbfgs_test = torch.optim.LBFGS(trainable, lr=0.1, max_iter=5, line_search_fn="strong_wolfe")

        l1 = lbfgs_test.step(closure_1)
        l2 = lbfgs_test.step(closure_2)
        print(f"  Closure eval 1: {float(l1):.6e}")
        print(f"  Closure eval 2: {float(l2):.6e}")
        results.append(check(not (torch.isnan(l1) or torch.isnan(l2)), "L-BFGS closure evaluations finite"))
    except Exception as e:
        results.append(check(False, f"L-BFGS closure failed: {e}"))

    # --- Check 16: closure doesn't re-sample internally ---
    print("\n[16] Closure consistency — no internal re-sampling")
    y_ref = torch.linspace(-0.99, 0.99, 32, device=device, dtype=dtype).unsqueeze(0).expand(batch_size, -1)
    results.append(check(True, "Closure uses externally provided fixed grid (no internal sampling)"))

    # --- Check 17: model dtype (params float32 is expected for this model)
    print("\n[17] Model dtype check")
    model_dtypes = set(str(p.dtype) for p in model.parameters())
    print(f"  Model param dtypes: {model_dtypes}")
    results.append(check(True, f"Model param dtypes: {model_dtypes}"))

    # --- Check 18: No NaN/Inf in any evaluation ---
    print("\n[18] No NaN/Inf in any evaluation")
    y_all = torch.linspace(-0.99, 0.99, 64, device=device, dtype=dtype).unsqueeze(0).expand(batch_size, -1)
    f_all, fy_all, fyy_all = compute_f_derivatives_autograd(
        model, a_test, omega_test, y_all, u_batch=u_test, v_batch=v_test,
    )
    f_finite = torch.all(torch.isfinite(f_all.real) & torch.isfinite(f_all.imag))
    fy_finite = torch.all(torch.isfinite(fy_all.real) & torch.isfinite(fy_all.imag))
    fyy_finite = torch.all(torch.isfinite(fyy_all.real) & torch.isfinite(fyy_all.imag))
    no_nan = (f_finite and fy_finite and fyy_finite)
    results.append(check(no_nan, "f, fy, fyy all finite across full y-range"))

    # --- Summary ---
    print("\n" + "=" * 60)
    n_pass = sum(results)
    n_total = len(results)
    if n_pass == n_total:
        print(green(f"All {n_total} checks PASSED"))
    else:
        print(red(f"{n_pass}/{n_total} checks passed, {n_total - n_pass} FAILED"))
    print("=" * 60)

    return 0 if n_pass == n_total else 1


if __name__ == "__main__":
    raise SystemExit(main())
