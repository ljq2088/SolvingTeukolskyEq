#!/usr/bin/env python3
"""
Stage-4 preflight consistency diagnostic.

Checks whether frozen Stage-1 R_pred, Stage-2 B_inc/B_ref, Stage-3 u_up/u_down
already satisfy:
    R_pred / P ≈ (B_ref * u_up * A_up + B_inc * u_down * A_down) / P

near infinity, WITHOUT any training.

Usage:
  python scripts/diagnose_stage4_consistency.py \
    --stage1-artifact outputs/stage1_artifacts/patch_000_logw_v2 \
    --stage2-checkpoint outputs/autoencoder_stage2_amplitude_train/.../best_model.pt \
    --stage3-checkpoint outputs/autoencoder_stage3_ubasis_refine/.../best_model.pt \
    --config config/autoencoder_stage3_ubasis_refine.yaml \
    --device cuda
"""
import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from config.config_loader import load_pinn_full_config
from model.autoencoder_pinn import AutoencoderTeukolskyPINN
from physical_ansatz.u_basis import compose_u_from_f, infinity_slopes_y, A_up, A_down


def _get_dtype(dtype_name):
    return torch.float32 if dtype_name == "float32" else torch.float64


def main():
    parser = argparse.ArgumentParser(description="Stage-4 consistency preflight")
    parser.add_argument("--stage1-artifact", type=str, required=True)
    parser.add_argument("--stage2-checkpoint", type=str, required=True)
    parser.add_argument("--stage3-checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default="config/autoencoder_stage3_ubasis_refine.yaml")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--n-param", type=int, default=32)
    parser.add_argument("--n-y", type=int, default=128)
    parser.add_argument("--y-min", type=float, default=-0.999)
    parser.add_argument("--y-max", type=float, default=-0.95)
    parser.add_argument("--output-dir", type=str, default="outputs/stage4_preflight/patch_000_logw_v2")
    parser.add_argument("--no-plots", action="store_true", default=False)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # ---- Load config ----
    full_cfg = load_pinn_full_config(args.config)
    runtime_cfg = full_cfg.get("runtime", {})
    dtype_name = runtime_cfg.get("dtype", "float64")
    dtype = _get_dtype(dtype_name)
    cdtype = torch.complex128 if dtype == torch.float64 else torch.complex64

    physics = full_cfg.get("physics", full_cfg)
    prob = physics.get("problem", {})
    M = float(prob.get("M", 1.0))
    s = int(prob.get("s", -2))
    m = int(prob.get("m", 2))

    # ---- Load artifact ----
    artifact_dir = Path(args.stage1_artifact)
    data = np.load(artifact_dir / "rpred_cache.npz")
    valid = np.isfinite(data["a"]) & np.isfinite(data["omega"]) & np.isfinite(data["lambda_"])
    a_pool = data["a"][valid]
    omega_pool = data["omega"][valid]
    lam_pool = data["lambda_"][valid]
    y_artifact = data["y"][valid]  # (N_pool, N_y_artifact)
    R_pred_over_P_artifact = data["R_pred_over_P"][valid]
    P_artifact = data["P"][valid]
    n_pool = len(a_pool)
    print(f"[preflight] Artifact: {n_pool} samples, y_artifact shape: {y_artifact.shape}")

    # ---- Load Stage-2 checkpoint (for amplitude_net comparison) ----
    ckpt2 = torch.load(args.stage2_checkpoint, map_location="cpu", weights_only=False)

    # ---- Load Stage-3 checkpoint (main model) ----
    ckpt3 = torch.load(args.stage3_checkpoint, map_location="cpu", weights_only=False)

    # ---- Compare amplitude_net between Stage-2 and Stage-3 ----
    amp_match = True
    for key in ckpt2["model_state_dict"]:
        if key.startswith("amplitude_net."):
            v2 = ckpt2["model_state_dict"][key]
            v3 = ckpt3["model_state_dict"].get(key)
            if v3 is None:
                amp_match = False
                print(f"  [WARN] amplitude_net key '{key}' missing in Stage-3 checkpoint")
            elif not torch.equal(v2, v3):
                amp_match = False
                print(f"  [WARN] amplitude_net key '{key}' differs between Stage-2 and Stage-3")
    if amp_match:
        print("[preflight] amplitude_net matches between Stage-2 and Stage-3 checkpoints")
    else:
        print("[preflight] WARNING: amplitude_net differs — using Stage-3 as authoritative")

    # ---- Build model from Stage-3 checkpoint ----
    full_cfg_raw = ckpt3.get("full_cfg", full_cfg)
    model_cfg = full_cfg_raw.get("model", full_cfg.get("model", {}))
    model = AutoencoderTeukolskyPINN(
        hidden_dims=model_cfg.get("hidden_dims", [128, 128, 128, 128]),
        activation=str(model_cfg.get("activation", "silu")),
        param_embed_dim=int(model_cfg.get("param_embed_dim", 64)),
        fourier_num_freqs=int(model_cfg.get("fourier_num_freqs", 2)),
        fourier_base_scale=float(model_cfg.get("fourier_base_scale", 1.0)),
        use_film=bool(model_cfg.get("use_film", True)),
        use_residual=bool(model_cfg.get("use_residual", True)),
        amp_hidden_dim=int(model_cfg.get("amp_hidden_dim", 128)),
        amp_n_blocks=int(model_cfg.get("amp_n_blocks", 3)),
        **model_cfg.get("encoder_kwargs", {}),
    )
    model.load_state_dict(ckpt3["model_state_dict"], strict=False)
    model.to(device=device, dtype=dtype)
    model.eval()
    print(f"[preflight] Model loaded from Stage-3 checkpoint (epoch={ckpt3.get('epoch', '?')})")

    # ---- Sample parameters ----
    n_param = min(args.n_param, n_pool)
    idx = np.linspace(0, n_pool - 1, n_param, dtype=int)
    print(f"[preflight] Sampling {n_param} parameter sets")

    # ---- Build diagnostic y-grid ----
    y_diag = torch.linspace(args.y_min, args.y_max, args.n_y, device=device, dtype=dtype)
    print(f"[preflight] Diagnostic y-grid: [{args.y_min}, {args.y_max}], {args.n_y} points")

    # ---- Output dir ----
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    # ---- Collect results ----
    all_cases = []
    all_rel_err = []
    all_rel_err_near_inf = []
    all_abs_err = []
    all_term_up_mag = []
    all_term_down_mag = []

    for i_case, i in enumerate(idx):
        a_val = torch.tensor([[a_pool[i]]], device=device, dtype=dtype)
        omega_val = torch.tensor([[omega_pool[i]]], device=device, dtype=dtype)
        lam_val = torch.tensor([complex(lam_pool[i].real, lam_pool[i].imag)],
                               device=device, dtype=cdtype)

        # ---- Interpolate R_pred_over_P from artifact to diagnostic y-grid ----
        y_art_i = y_artifact[i]  # (N_y_art,)
        rpop_art_i = R_pred_over_P_artifact[i]  # (N_y_art,) complex
        P_art_i = P_artifact[i]  # (N_y_art,) real

        # Interpolate real and imag parts separately
        y_art_sort = np.argsort(y_art_i)
        y_art_sorted = y_art_i[y_art_sort]
        rpop_art_sorted = rpop_art_i[y_art_sort]
        P_art_sorted = P_art_i[y_art_sort]

        y_diag_np = y_diag.cpu().numpy()
        rpop_interp = (
            np.interp(y_diag_np, y_art_sorted, rpop_art_sorted.real)
            + 1j * np.interp(y_diag_np, y_art_sorted, rpop_art_sorted.imag)
        )
        P_interp = np.interp(y_diag_np, y_art_sorted, P_art_sorted)

        # ---- Model forward ----
        y_t = y_diag.unsqueeze(0)  # (1, N_y)
        with torch.no_grad():
            f_up = model.predict_u_up(a_val, omega_val, y_t)
            f_down = model.predict_u_down(a_val, omega_val, y_t)
            B_inc, B_ref, _ = model.predict_amplitudes(a_val, omega_val)

        c_up, c_down = infinity_slopes_y(a_val, omega_val, lam_val, m=m, M=M)
        u_up = compose_u_from_f(f_up, y_t, c_up)
        u_down = compose_u_from_f(f_down, y_t, c_down)

        # r from y: y = 2*r_plus/r - 1 => r = 2*r_plus/(y+1)
        from physical_ansatz.mapping import r_plus as rp_fn
        rp = rp_fn(a_val, M)
        r_t = 2.0 * rp / (y_t + 1.0)
        A_up_val = A_up(r_t, a_val, omega_val, M=M)
        A_down_val = A_down(r_t, a_val, omega_val, M=M)

        # ---- Construct R_combo_over_P ----
        # R_combo = B_ref * u_up * A_up + B_inc * u_down * A_down
        B_ref_b = B_ref.unsqueeze(-1)  # (1, 1)
        B_inc_b = B_inc.unsqueeze(-1)
        term_up = B_ref_b * u_up * A_up_val  # (1, N_y)
        term_down = B_inc_b * u_down * A_down_val
        R_combo = term_up + term_down

        # R_combo_over_P
        P_t = torch.tensor(P_interp.real, device=device, dtype=dtype).unsqueeze(0)
        R_combo_over_P = R_combo / P_t

        # R_pred_over_P from interpolation
        R_pred_over_P_t = torch.tensor(rpop_interp, device=device, dtype=cdtype).unsqueeze(0)

        # ---- Errors ----
        abs_err = torch.abs(R_combo_over_P - R_pred_over_P_t)
        rel_err = abs_err / (torch.abs(R_pred_over_P_t) + 1e-12)

        abs_err_np = abs_err.cpu().numpy().ravel()
        rel_err_np = rel_err.cpu().numpy().ravel()

        # Near-infinity: y < -0.95 (closer to infinity)
        near_inf_mask = y_diag_np < -0.95

        term_up_abs = torch.abs(term_up / P_t).detach().cpu().numpy().ravel()
        term_down_abs = torch.abs(term_down / P_t).detach().cpu().numpy().ravel()

        case = {
            "idx": int(i),
            "a": float(a_pool[i]),
            "omega": float(omega_pool[i]),
            "median_rel_err": float(np.median(rel_err_np)),
            "max_rel_err": float(np.max(rel_err_np)),
            "median_rel_err_near_inf": float(np.median(rel_err_np[near_inf_mask])) if near_inf_mask.any() else np.nan,
            "max_rel_err_near_inf": float(np.max(rel_err_np[near_inf_mask])) if near_inf_mask.any() else np.nan,
            "median_abs_err": float(np.median(abs_err_np)),
            "max_abs_err": float(np.max(abs_err_np)),
            "median_term_up_over_P": float(np.median(term_up_abs)),
            "median_term_down_over_P": float(np.median(term_down_abs)),
            "term_up_down_ratio": float(np.median(term_up_abs) / max(np.median(term_down_abs), 1e-30)),
            "rel_err_np": rel_err_np,
            "abs_err_np": abs_err_np,
            "term_up_abs": term_up_abs,
            "term_down_abs": term_down_abs,
            "R_combo_over_P": R_combo_over_P.detach().cpu().numpy().ravel(),
            "R_pred_over_P": rpop_interp,
            "u_up": u_up.detach().cpu().numpy().ravel(),
            "u_down": u_down.detach().cpu().numpy().ravel(),
        }
        all_cases.append(case)
        all_rel_err.append(case["median_rel_err"])
        all_rel_err_near_inf.append(case["median_rel_err_near_inf"])
        all_abs_err.append(case["median_abs_err"])
        all_term_up_mag.append(case["median_term_up_over_P"])
        all_term_down_mag.append(case["median_term_down_over_P"])

        print(f"  [{i_case+1}/{n_param}] a={a_pool[i]:.4f} omega={omega_pool[i]:.6e} | "
              f"med_rel_err={case['median_rel_err']:.4e} near_inf={case['median_rel_err_near_inf']:.4e} | "
              f"up/down_ratio={case['term_up_down_ratio']:.2e}")

    # ---- Aggregate statistics ----
    median_rel_err_all = np.median(all_rel_err)
    max_rel_err_all = np.max([c["max_rel_err"] for c in all_cases])
    median_rel_err_ni = np.median([x for x in all_rel_err_near_inf if np.isfinite(x)])
    max_rel_err_ni = np.max([c["max_rel_err_near_inf"] for c in all_cases if np.isfinite(c["max_rel_err_near_inf"])])

    median_term_up = np.median(all_term_up_mag)
    median_term_down = np.median(all_term_down_mag)
    up_down_ratio = median_term_up / max(median_term_down, 1e-30)

    # Worst 5 cases
    sorted_cases = sorted(all_cases, key=lambda c: c["median_rel_err"], reverse=True)
    worst5 = sorted_cases[:5]

    # ---- Plots ----
    has_plots = False
    if not args.no_plots:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            has_plots = True
        except ImportError:
            pass

    if has_plots:
        y_plot = y_diag_np
        # Plot worst 5 + best 2 cases
        plot_indices = [wc["idx"] for wc in worst5]
        best_cases = sorted_cases[-2:]
        for bc in best_cases:
            if bc["idx"] not in plot_indices:
                plot_indices.append(bc["idx"])

        for case in all_cases:
            if case["idx"] not in plot_indices:
                continue
            a_str = f"{case['a']:.4f}"
            omega_str = f"{case['omega']:.6e}"

            # rel_err vs y
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.semilogy(y_plot, case["rel_err_np"], "b-", lw=1)
            ax.axvline(x=-0.95, color="gray", ls="--", alpha=0.5, label="near-inf boundary")
            ax.set_xlabel("y"); ax.set_ylabel("relative error")
            ax.set_title(f"Relative error |R_pred/P - R_combo/P| — a={a_str}, $\\omega$={omega_str}")
            ax.grid(True, alpha=0.3); ax.legend()
            fig.savefig(plots_dir / f"rel_err_vs_y_a{a_str}_omega{omega_str}.png", dpi=100)
            plt.close(fig)

            # abs terms vs y
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.semilogy(y_plot, case["term_up_abs"], "b-", lw=1, label="|B_ref*u_up*A_up/P|")
            ax.semilogy(y_plot, case["term_down_abs"], "r-", lw=1, label="|B_inc*u_down*A_down/P|")
            ax.set_xlabel("y"); ax.set_ylabel("magnitude")
            ax.set_title(f"Term magnitudes — a={a_str}, $\\omega$={omega_str}")
            ax.grid(True, alpha=0.3); ax.legend()
            fig.savefig(plots_dir / f"abs_terms_vs_y_a{a_str}_omega{omega_str}.png", dpi=100)
            plt.close(fig)

            # combo vs rpred
            fig, axes = plt.subplots(2, 1, figsize=(10, 6))
            ax = axes[0]
            ax.plot(y_plot, np.abs(case["R_combo_over_P"]), "b-", lw=1, label="|R_combo/P|")
            ax.plot(y_plot, np.abs(case["R_pred_over_P"]), "r--", lw=1, label="|R_pred/P|")
            ax.set_ylabel("magnitude"); ax.legend(); ax.grid(True, alpha=0.3)
            ax.set_title(f"Combo vs R_pred — a={a_str}, $\\omega$={omega_str}")
            ax = axes[1]
            ax.plot(y_plot, np.angle(case["R_combo_over_P"]), "b-", lw=1, label="phase(R_combo/P)")
            ax.plot(y_plot, np.angle(case["R_pred_over_P"]), "r--", lw=1, label="phase(R_pred/P)")
            ax.set_xlabel("y"); ax.set_ylabel("phase (rad)"); ax.legend(); ax.grid(True, alpha=0.3)
            fig.savefig(plots_dir / f"combo_vs_rpred_a{a_str}_omega{omega_str}.png", dpi=100)
            plt.close(fig)

            # u_up and u_down near infinity
            u_up_np = case["u_up"].ravel()
            u_down_np = case["u_down"].ravel()
            fig, axes = plt.subplots(2, 2, figsize=(14, 8))
            fig.suptitle(f"u_up / u_down near infinity — a={a_str}, $\\omega$={omega_str}")
            # |u_up|
            ax = axes[0, 0]
            ax.plot(y_plot, np.abs(u_up_np), "b-", lw=1)
            ax.set_xlabel("y"); ax.set_ylabel("|u_up|"); ax.grid(True, alpha=0.3)
            # Re/Im u_up
            ax = axes[0, 1]
            ax.plot(y_plot, u_up_np.real, "b-", lw=1, label="Re(u_up)")
            ax.plot(y_plot, u_up_np.imag, "r--", lw=1, label="Im(u_up)")
            ax.set_xlabel("y"); ax.legend(); ax.grid(True, alpha=0.3)
            # |u_down|
            ax = axes[1, 0]
            ax.plot(y_plot, np.abs(u_down_np), "r-", lw=1)
            ax.set_xlabel("y"); ax.set_ylabel("|u_down|"); ax.grid(True, alpha=0.3)
            # Re/Im u_down
            ax = axes[1, 1]
            ax.plot(y_plot, u_down_np.real, "b-", lw=1, label="Re(u_down)")
            ax.plot(y_plot, u_down_np.imag, "r--", lw=1, label="Im(u_down)")
            ax.set_xlabel("y"); ax.legend(); ax.grid(True, alpha=0.3)
            plt.tight_layout()
            fig.savefig(plots_dir / f"u_up_down_near_inf_a{a_str}_omega{omega_str}.png", dpi=100)
            plt.close(fig)

    # ---- Save results ----
    # JSON summary
    summary = {
        "stage1_artifact": str(artifact_dir.resolve()),
        "stage2_checkpoint": str(Path(args.stage2_checkpoint).resolve()),
        "stage3_checkpoint": str(Path(args.stage3_checkpoint).resolve()),
        "amplitude_net_match": amp_match,
        "y_grid_range": [args.y_min, args.y_max],
        "n_y": args.n_y,
        "n_param": n_param,
        "R_in_convention": "B_ref*u_up*A_up + B_inc*u_down*A_down",
        "median_rel_err_all": float(median_rel_err_all),
        "max_rel_err_all": float(max_rel_err_all),
        "median_rel_err_near_inf": float(median_rel_err_ni),
        "max_rel_err_near_inf": float(max_rel_err_ni),
        "median_term_up_over_P": float(median_term_up),
        "median_term_down_over_P": float(median_term_down),
        "up_down_ratio": float(up_down_ratio),
        "dominant_term": "up" if up_down_ratio > 10 else ("down" if up_down_ratio < 0.1 else "mixed"),
        "worst5_cases": [{"a": c["a"], "omega": c["omega"], "median_rel_err": c["median_rel_err"],
                          "max_rel_err": c["max_rel_err"], "up_down_ratio": c["term_up_down_ratio"]}
                         for c in worst5],
        "all_cases": [{k: v for k, v in c.items() if k not in ("rel_err_np", "abs_err_np",
                        "term_up_abs", "term_down_abs", "R_combo_over_P", "R_pred_over_P",
                        "u_up", "u_down")}
                      for c in all_cases],
    }
    with open(output_dir / "consistency_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # CSV
    csv_path = output_dir / "consistency_cases.csv"
    with open(csv_path, "w") as f:
        f.write("idx,a,omega,median_rel_err,max_rel_err,median_rel_err_near_inf,"
                "max_rel_err_near_inf,median_term_up,median_term_down,up_down_ratio\n")
        for c in all_cases:
            f.write(f"{c['idx']},{c['a']:.6f},{c['omega']:.6e},{c['median_rel_err']:.6e},"
                    f"{c['max_rel_err']:.6e},{c['median_rel_err_near_inf']:.6e},"
                    f"{c['max_rel_err_near_inf']:.6e},{c['median_term_up_over_P']:.6e},"
                    f"{c['median_term_down_over_P']:.6e},{c['term_up_down_ratio']:.6e}\n")

    # ---- Criteria check ----
    criteria_pass = True
    criteria_notes = []

    if median_rel_err_ni >= 0.1:
        criteria_pass = False
        criteria_notes.append(f"median_rel_err_near_inf={median_rel_err_ni:.4e} >= 0.1")
    elif median_rel_err_ni >= 0.05:
        criteria_notes.append(f"median_rel_err_near_inf={median_rel_err_ni:.4e} in [0.05, 0.1) — borderline")

    if max_rel_err_ni > 1.0:
        criteria_pass = False
        criteria_notes.append(f"max_rel_err_near_inf={max_rel_err_ni:.4e} > 1 (exploding error)")

    if not np.isfinite(median_rel_err_ni):
        criteria_pass = False
        criteria_notes.append("NaN/Inf detected in errors")

    recommendation = "ENTER_STAGE4" if criteria_pass else "STAGE3_REFINE_NEEDED"
    if not criteria_pass:
        criteria_notes.append(f"Up term dominates (ratio={up_down_ratio:.2e}) — u_up is the bottleneck" if up_down_ratio > 10 else "")

    # ---- Markdown report ----
    md_path = output_dir / "consistency_summary.md"
    with open(md_path, "w") as f:
        f.write("# Stage-4 Preflight Consistency Diagnostic\n\n")
        f.write(f"- **Stage-1 artifact**: `{artifact_dir.resolve()}`\n")
        f.write(f"- **Stage-2 checkpoint**: `{Path(args.stage2_checkpoint).resolve()}`\n")
        f.write(f"- **Stage-3 checkpoint**: `{Path(args.stage3_checkpoint).resolve()}`\n")
        f.write(f"- **amplitude_net match**: {'YES' if amp_match else 'NO (using Stage-3 as authoritative)'}\n")
        f.write(f"- **y-grid range**: [{args.y_min}, {args.y_max}]\n")
        f.write(f"- **n_param**: {n_param}, **n_y**: {args.n_y}\n")
        f.write(f"- **R_in convention**: `B_ref*u_up*A_up + B_inc*u_down*A_down`\n")
        f.write(f"- **R_pred_over_P source**: interpolated from artifact y-grid to diagnostic y-grid\n\n")

        f.write("## Consistency Error\n\n")
        f.write("| Metric | Value |\n")
        f.write("|--------|-------|\n")
        f.write(f"| median_rel_err_all | {median_rel_err_all:.4e} |\n")
        f.write(f"| max_rel_err_all | {max_rel_err_all:.4e} |\n")
        f.write(f"| median_rel_err_near_inf | {median_rel_err_ni:.4e} |\n")
        f.write(f"| max_rel_err_near_inf | {max_rel_err_ni:.4e} |\n\n")

        f.write("## Up/Down Contribution\n\n")
        f.write("| Term | median magnitude |\n")
        f.write("|------|-----------------|\n")
        f.write(f"| |term_up/P| | {median_term_up:.4e} |\n")
        f.write(f"| |term_down/P| | {median_term_down:.4e} |\n")
        f.write(f"| up/down ratio | {up_down_ratio:.2e} |\n")
        f.write(f"| dominant | {'up (u_up bottleneck)' if up_down_ratio > 10 else ('down' if up_down_ratio < 0.1 else 'mixed')} |\n\n")

        f.write("## Worst 5 Cases\n\n")
        f.write("| a | omega | median_rel_err | max_rel_err | up/down_ratio |\n")
        f.write("|---|-------|---------------|------------|---------------|\n")
        for c in worst5:
            f.write(f"| {c['a']:.4f} | {c['omega']:.6e} | {c['median_rel_err']:.4e} | {c['max_rel_err']:.4e} | {c['term_up_down_ratio']:.2e} |\n")

        f.write("\n## Recommendation\n\n")
        f.write(f"**Decision: {recommendation}**\n\n")
        if criteria_notes:
            for note in criteria_notes:
                if note:
                    f.write(f"- {note}\n")
        f.write("\n")

    # ---- Print summary ----
    print(f"\n{'='*60}")
    print("Stage-4 Preflight Consistency Summary")
    print(f"{'='*60}")
    print(f"  median_rel_err_all:       {median_rel_err_all:.4e}")
    print(f"  max_rel_err_all:          {max_rel_err_all:.4e}")
    print(f"  median_rel_err_near_inf:  {median_rel_err_ni:.4e}")
    print(f"  max_rel_err_near_inf:     {max_rel_err_ni:.4e}")
    print(f"  median |term_up/P|:       {median_term_up:.4e}")
    print(f"  median |term_down/P|:     {median_term_down:.4e}")
    print(f"  up/down ratio:            {up_down_ratio:.2e}")
    print(f"  dominant term:            {'up' if up_down_ratio > 10 else ('down' if up_down_ratio < 0.1 else 'mixed')}")
    print(f"  amplitude_net match:      {'YES' if amp_match else 'NO'}")
    print(f"  worst 5:")
    for c in worst5:
        print(f"    a={c['a']:.4f} omega={c['omega']:.6e} med_err={c['median_rel_err']:.4e} max_err={c['max_rel_err']:.4e}")
    print(f"\n  Recommendation: {recommendation}")
    for note in criteria_notes:
        if note:
            print(f"    - {note}")
    print(f"\n  Output: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
