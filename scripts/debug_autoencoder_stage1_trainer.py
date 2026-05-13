#!/usr/bin/env python3
"""
Stage-1 autoencoder trainer smoke test.

Verifies:
    - AtlasPatchTrainer constructs with model_type="autoencoder"
    - Freeze policy: only encoder + rin_decoder trainable
    - train_one_step() works end-to-end
    - Checkpoint save/load round-trips
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Point to original project's domain files
DOMAIN_DIR = Path("/home/ljq/code/PINN/SolvingTeukolsky/outputs/domain")

import torch
torch.manual_seed(42)
torch.set_default_dtype(torch.float64)


def check(desc, condition):
    assert condition, f"FAIL: {desc}"
    print(f"   OK — {desc}")


def main():
    from trainer.atlas_patch_trainer import AtlasPatchTrainer
    from model.autoencoder_pinn import AutoencoderTeukolskyPINN

    cfg_path = str(PROJECT_ROOT / "config/autoencoder_stage1_rin.yaml")
    probe_json = str(DOMAIN_DIR / "probe_l2_m2.json")
    atlas_json = str(DOMAIN_DIR / "atlas_l2_m2.json")
    patch_json = str(DOMAIN_DIR / "patch_cover_l2_m2.json")

    print("1. Construct trainer with model_type='autoencoder'...")
    trainer = AtlasPatchTrainer(
        cfg_path=cfg_path,
        probe_json=probe_json,
        atlas_json=atlas_json,
        patch_json=patch_json,
        patch_id=0,
        device="cpu",
        anchor_enabled=False,
        verbose=True,
        output_root="outputs/autoencoder_stage1_rin",
        model_type="autoencoder",
    )
    check("trainer created", trainer is not None)
    check("model is AutoencoderTeukolskyPINN", isinstance(trainer.model, AutoencoderTeukolskyPINN))

    model = trainer.model
    n_total = sum(p.numel() for p in model.parameters())
    print(f"   Total params: {n_total}")

    print("2. Freeze policy: only encoder + rin_decoder trainable...")
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    enc_total = sum(p.numel() for p in model.encoder.parameters())
    enc_trainable = sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)
    check("encoder fully trainable", enc_trainable == enc_total)

    rin_total = sum(p.numel() for p in model.rin_decoder.parameters())
    rin_trainable = sum(p.numel() for p in model.rin_decoder.parameters() if p.requires_grad)
    check("rin_decoder fully trainable", rin_trainable == rin_total)

    amp_trainable = sum(p.numel() for p in model.amplitude_net.parameters() if p.requires_grad)
    check("amplitude_net fully frozen", amp_trainable == 0)

    up_trainable = sum(p.numel() for p in model.up_decoder.parameters() if p.requires_grad)
    check("up_decoder fully frozen", up_trainable == 0)

    dn_trainable = sum(p.numel() for p in model.down_decoder.parameters() if p.requires_grad)
    check("down_decoder fully frozen", dn_trainable == 0)

    check("trainable == enc + rin", n_trainable == enc_total + rin_total)
    print(f"   Trainable/Total: {n_trainable}/{n_total}")

    print("3. train_one_step() ...")
    info = trainer.train_one_step()
    check("loss_pde finite", torch.isfinite(torch.tensor(info["loss_pde"])))
    check("total_loss finite", torch.isfinite(torch.tensor(info["total_loss"])))
    print(f"   loss_pde={info['loss_pde']:.6e}, total_loss={info['total_loss']:.6e}")

    print("4. Second train_one_step() — loss should change...")
    info2 = trainer.train_one_step()
    loss_diff = abs(info2["total_loss"] - info["total_loss"])
    check("loss changed", loss_diff > 1e-16)
    print(f"   loss_pde={info2['loss_pde']:.6e}, total_loss={info2['total_loss']:.6e}")

    print("5. Checkpoint save/load round-trip...")
    ckpt_dir = Path(trainer.run_dir) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "smoke_test.pt"

    model.eval()

    # Save state dict + encoder config (local coord params not in state_dict)
    sd_cpu = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    enc_cfg = {
        "local_coord_mode": model.encoder.local_coord_mode,
        "a_center_local": model.encoder.a_center_local,
        "a_half_range_local": model.encoder.a_half_range_local,
        "omega_min_local": model.encoder.omega_min_local,
        "omega_max_local": model.encoder.omega_max_local,
        "u_center_local": model.encoder.u_center_local,
        "v_center_local": model.encoder.v_center_local,
        "u_half_range_local": model.encoder.u_half_range_local,
        "v_half_range_local": model.encoder.v_half_range_local,
    }
    torch.save({"model_state_dict": sd_cpu, "encoder_config": enc_cfg, "step": 1}, ckpt_path)
    check("checkpoint saved", ckpt_path.exists())

    # Load with matching encoder config
    ckpt_data = torch.load(ckpt_path, map_location="cpu")
    ec = ckpt_data["encoder_config"]
    model2 = AutoencoderTeukolskyPINN(
        local_coord_mode=ec["local_coord_mode"],
        a_center_local=ec["a_center_local"],
        a_half_range_local=ec["a_half_range_local"],
        omega_min_local=ec["omega_min_local"],
        omega_max_local=ec["omega_max_local"],
        u_center_local=ec["u_center_local"],
        v_center_local=ec["v_center_local"],
        u_half_range_local=ec["u_half_range_local"],
        v_half_range_local=ec["v_half_range_local"],
    ).double()
    model2.load_state_dict(ckpt_data["model_state_dict"], strict=True)
    model2.eval()

    # Verify state dict match
    sd1 = model.state_dict()
    sd2 = model2.state_dict()
    max_sd_err = max((sd1[k].cpu() - sd2[k].cpu()).abs().max().item() for k in sd1)
    print(f"   max state_dict diff: {max_sd_err:.2e}")

    # Fixed inputs for comparison
    a_test = torch.tensor([0.3, 0.5, 0.7, 0.9], dtype=torch.float64)
    omega_test = torch.tensor([0.1, 1.0, 3.0, 8.0], dtype=torch.float64)
    u_test = torch.tensor([0.5, 0.5, 0.5, 0.5], dtype=torch.float64)
    v_test = torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float64)
    y_grid = torch.linspace(-1.0, 1.0, 64, dtype=torch.float64)

    with torch.no_grad():
        f1 = model(a_test, omega_test, y_grid, u=u_test, v=v_test)
        f2 = model2(a_test, omega_test, y_grid, u=u_test, v=v_test)
    max_err = (f1 - f2).abs().max().item()
    check(f"round-trip max_err={max_err:.2e}", max_err < 1e-10)
    print(f"   ckpt size: {ckpt_path.stat().st_size / 1024:.1f} KB")

    print()
    print("=" * 60)
    print("ALL STAGE-1 TRAINER SMOKE TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
