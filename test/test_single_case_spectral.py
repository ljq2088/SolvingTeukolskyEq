from __future__ import annotations
import sys
sys.path.append("/home/ljq/code/PINN/SolvingTeukolsky")
import os
import sys
import yaml
import torch
from tqdm import tqdm
import matplotlib
matplotlib.use("TkAgg")   # 如果你装的是 Qt，也可以换成 "QtAgg"
import matplotlib.pyplot as plt
# 让脚本能从项目根目录导入
sys.path.insert(0, os.getcwd())

from dataset.grids import chebyshev_grid_bundle
from physical_ansatz.residual import (
    AuxCache,
    get_lambda_from_cfg,
    get_ramp_and_p_from_cfg,
    teukolsky_residual_loss_coeff,
    diagnose_operator_scales,
    residual_from_nodes,   # 新增
)
from physical_ansatz.mapping import r_plus   # 新增
import numpy as np                           # 新增


def load_cfg(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def cheb_eval_with_derivs(coeff: torch.Tensor, y: torch.Tensor):
    """
    coeff: (Nc,) complex
    y:     (Ny,) real
    返回:
        f(y), df/dy, d2f/dy2   都是 (Ny,) complex
    """
    if coeff.ndim != 1:
        raise ValueError(f"coeff must be 1D, got {tuple(coeff.shape)}")
    if y.ndim != 1:
        raise ValueError(f"y must be 1D, got {tuple(y.shape)}")

    device = coeff.device
    dtype = coeff.dtype
    y_c = y.to(device=device, dtype=dtype)

    Nc = coeff.shape[0]
    Ny = y.shape[0]

    T   = torch.zeros((Nc, Ny), dtype=dtype, device=device)
    dT  = torch.zeros((Nc, Ny), dtype=dtype, device=device)
    d2T = torch.zeros((Nc, Ny), dtype=dtype, device=device)

    T[0] = 1.0 + 0.0j
    if Nc >= 2:
        T[1] = y_c
        dT[1] = 1.0 + 0.0j

    for n in range(2, Nc):
        T[n]   = 2.0 * y_c * T[n - 1]   - T[n - 2]
        dT[n]  = 2.0 * T[n - 1] + 2.0 * y_c * dT[n - 1]  - dT[n - 2]
        d2T[n] = 4.0 * dT[n - 1] + 2.0 * y_c * d2T[n - 1] - d2T[n - 2]

    f   = coeff @ T
    fy  = coeff @ dT
    fyy = coeff @ d2T
    return f, fy, fyy
def evaluate_residual_on_r_grid(
    cfg,
    coeff_re,
    coeff_im,
    a_batch,
    omega_batch,
    lambda_batch,
    ramp_batch,
    p,
    rmin,
    rmax,
    npts=200,
):
    """
    在给定 r 网格上评估 residual，返回:
        r_np, res_np, stat_dict
    """
    device = coeff_re.device
    dtype_real = coeff_re.dtype
    dtype_cplx = torch.complex128

    M = float(cfg["problem"].get("M", 1.0))

    # 单样本训练：取 batch 里的第一个
    a_t = a_batch[0]
    omega_t = omega_batch[0]

    rp = r_plus(a_t, M)
    rp_val = float(rp.detach().cpu().item())

    # 避免监控点跑到视界内
    rmin_eff = max(float(rmin), rp_val + 1e-8)
    r_t = torch.linspace(rmin_eff, float(rmax), npts, dtype=dtype_real, device=device)

    # r -> x -> y
    x_t = rp / r_t
    y_t = 2.0 * x_t - 1.0

    coeff = torch.complex(coeff_re[0], coeff_im[0]).to(dtype=dtype_cplx)

    # 任意 y 点上直接算 f, fy, fyy
    f_t, fy_t, fyy_t = cheb_eval_with_derivs(coeff, y_t)

    # 用训练时同一个 operator 算 residual
    res_t = residual_from_nodes(
        y_nodes=y_t,
        f=f_t.unsqueeze(0),
        fy=fy_t.unsqueeze(0),
        fyy=fyy_t.unsqueeze(0),
        a_batch=a_batch,
        omega_batch=omega_batch,
        lambda_batch=lambda_batch,
        p=p,
        ramp_batch=ramp_batch,
        cfg=cfg,
    )[0]

    abs_res = torch.abs(res_t)
    stat = {
        "mean_abs": float(abs_res.mean().detach().cpu().item()),
        "max_abs": float(abs_res.max().detach().cpu().item()),
        "r_at_max": float(r_t[torch.argmax(abs_res)].detach().cpu().item()),
    }

    return (
        r_t.detach().cpu().numpy(),
        res_t.detach().cpu().numpy(),
        stat,
    )

def init_live_monitor():
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8))

    line_res, = ax1.semilogy([], [], lw=1.8, label="|res(r)|")
    ax1.set_xlabel("r / M")
    ax1.set_ylabel("|residual|")
    ax1.set_title("Residual on r-grid")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    line_max, = ax2.semilogy([], [], "o-", ms=3, label="max |res|")
    line_mean, = ax2.semilogy([], [], "o-", ms=3, label="mean |res|")
    line_loss, = ax2.semilogy([], [], "-", lw=1.2, label="train loss")
    ax2.set_xlabel("step")
    ax2.set_ylabel("monitor / loss")
    ax2.set_title("Training progress")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    fig.tight_layout()
    fig.show()

    return {
        "fig": fig,
        "ax1": ax1,
        "ax2": ax2,
        "line_res": line_res,
        "line_max": line_max,
        "line_mean": line_mean,
        "line_loss": line_loss,
    }

def update_live_monitor(
    monitor,
    r_np,
    res_np,
    step,
    current_loss,
    steps_monitor,
    max_hist,
    mean_hist,
    loss_steps,
    loss_hist,
    M=1.0,
):
    fig = monitor["fig"]
    ax1 = monitor["ax1"]
    ax2 = monitor["ax2"]

    monitor["line_res"].set_data(r_np / M, np.abs(res_np))
    ax1.relim()
    ax1.autoscale_view()

    monitor["line_max"].set_data(steps_monitor, max_hist)
    monitor["line_mean"].set_data(steps_monitor, mean_hist)
    monitor["line_loss"].set_data(loss_steps, loss_hist)
    ax2.relim()
    ax2.autoscale_view()

    fig.suptitle(
        f"step={step}   train_loss={current_loss:.3e}   "
        f"max|res|={max_hist[-1]:.3e}   mean|res|={mean_hist[-1]:.3e}"
    )
    fig.canvas.draw_idle()
    fig.canvas.flush_events()
    plt.pause(0.001)
def main():
    # -----------------------------
    # 1. 基本设置
    # -----------------------------
    cfg = load_cfg("config/teukolsky_radial.yaml")

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device=torch.device("cpu")
    print(f"Using device: {device}")
    dtype = torch.float64
    # 早停设置
    patience = 10000              # 连续多少步没有显著改善则停止
    min_delta_ratio = 0.0001      # 最小改善比例
    best_loss = float('inf')     # 历史最佳损失
    wait = 0                     # 连续未改善步数
    # 这里先固定一个单参数点
    a0 = 0.1
    omega0 = 0.1

    # 谱阶数
    order = 16

    n_boundary_drop = 2
    n_collocation_train = 64   # 每步随机抽 64 个内点
    eval_every = 1000           # 每 1000 步做一次全点验证

    # 训练步数和学习率
    MONITOR=False
    n_steps = 1000000
    lr = 1e-3
    print_every = 100
    monitor_every = 2000      # 每 2000 步评估一次 r-grid residual
    monitor_npts = 50
    M = float(cfg["problem"].get("M", 1.0))
    r_monitor_min = 2.0 * M
    r_monitor_max = 10.0 * M
    sample_rs = [2.0 * M, 4.0 * M, 6.0 * M, 8.0 * M, 10.0 * M]
    # -----------------------------
    # 2. 构造谱网格
    # -----------------------------
    grid = chebyshev_grid_bundle(order=order, dtype=dtype, device=device)

    y_nodes = grid.y_nodes
    D = grid.D
    D2 = grid.D2
    Tmat = grid.Tmat

    # -----------------------------
    # 3. 单样本 batch
    # -----------------------------
    a_batch = torch.tensor([a0], dtype=dtype, device=device)
    omega_batch = torch.tensor([omega0], dtype=dtype, device=device)
    cache = AuxCache()

    lam_list = []
    ramp_list = []
    p_val = None

    for i in range(a_batch.shape[0]):
        lam_i = get_lambda_from_cfg(cfg, cache, a_batch[i], omega_batch[i])
        p_i, ramp_i = get_ramp_and_p_from_cfg(cfg, cache, a_batch[i], omega_batch[i])

        lam_list.append(lam_i)
        ramp_list.append(ramp_i)

        if p_val is None:
            p_val = p_i
        else:
            if p_i != p_val:
                raise ValueError(f"Inconsistent p across batch: {p_val} vs {p_i}")

    lambda_batch = torch.stack(lam_list).to(device=device, dtype=torch.complex128)
    ramp_batch   = torch.stack(ramp_list).to(device=device, dtype=torch.complex128)
    p = int(p_val or 5)  # 默认 p=5
    diag0 = diagnose_operator_scales(
        cfg=cfg,
        y_nodes=y_nodes,
        a_batch=a_batch,
        omega_batch=omega_batch,
        lambda_batch=lambda_batch,
        ramp_batch=ramp_batch,
        p=p,
        n_boundary_report=n_boundary_drop,
    )

    print("\n=== Pre-train operator diagnostics ===")
    for name, info in diag0.items():
        print(
            f"{name:>4s} : "
            f"max={info['max_abs']:.6e}, "
            f"left={info.get('left_boundary_max', float('nan')):.6e}, "
            f"right={info.get('right_boundary_max', float('nan')):.6e}, "
            f"interior={info.get('interior_max', float('nan')):.6e}, "
            f"argmax_j={info['argmax_j']}, "
            f"y={info['y_at_argmax']:.6e}, "
            f"x={info['x_at_argmax']:.6e}"
        )
    print("======================================\n")
    # -----------------------------
    # 4. 可训练谱系数
    #    shape = (B, Nc) = (1, order+1)
    # -----------------------------
    coeff_re = torch.nn.Parameter(
        1e-3 * torch.randn(1, order + 1, dtype=dtype, device=device)
    )
    coeff_im = torch.nn.Parameter(
        1e-3 * torch.randn(1, order + 1, dtype=dtype, device=device)
    )

    # 也可以把低阶初始化得更明显一点，比如：
    with torch.no_grad():
        coeff_re[0, 0] = 0.0
        coeff_im[0, 0] = 0.0

#     optimizer = torch.optim.SGD(
#     [coeff_re, coeff_im],
#     lr=lr,
#     momentum=0.9,
#     nesterov=True,
# )
    optimizer = torch.optim.Adam([coeff_re, coeff_im], lr=1e-4)

    


    # 在 main 函数中，创建保存目录（在训练前）
    loss_dir = "/home/ljq/code/PINN/SolvingTeukolsky/outputs/loss_curve"
    os.makedirs(loss_dir, exist_ok=True)


    # -----------------------------
    # 5. 训练循环
    # -----------------------------
    Ny = y_nodes.shape[0]
    interior_idx = torch.arange(
        n_boundary_drop, Ny - n_boundary_drop,
        device=device, dtype=torch.long
    )
    best_loss = float('inf')
    best_step = 0
    wait = 0

    best_coeff_re = coeff_re.detach().clone()
    best_coeff_im = coeff_im.detach().clone()

    loss_history = []
    step_history = []
    monitor_steps = []
    monitor_max_history = []
    monitor_mean_history = []

    live_monitor = init_live_monitor()
    with tqdm(total=n_steps, desc="Training", unit="step") as pbar:
        for step in range(1, n_steps + 1):
            optimizer.zero_grad()

            # -----------------------------
            # A. 每一步随机抽一批 collocation 点
            # -----------------------------
            perm = torch.randperm(interior_idx.numel(), device=device)
            batch_idx = interior_idx[perm[:n_collocation_train]]

            train_loss, train_diag = teukolsky_residual_loss_coeff(
                cfg=cfg,
                y_nodes=y_nodes,
                D=D,
                D2=D2,
                Tmat=Tmat,
                coeff_re=coeff_re,
                coeff_im=coeff_im,
                a_batch=a_batch,
                omega_batch=omega_batch,
                lambda_batch=lambda_batch,
                ramp_batch=ramp_batch,
                p=p,
                n_boundary_drop=n_boundary_drop,
                collocation_idx=batch_idx,   # 关键
            )

            # 轻微谱系数正则
            k = torch.arange(order + 1, dtype=dtype, device=device).reshape(1, -1)
            reg = 1e-8 * torch.mean((1.0 + k**2) * (coeff_re**2 + coeff_im**2))

            total_train_loss = train_loss + reg
            total_train_loss.backward()
            torch.nn.utils.clip_grad_norm_([coeff_re, coeff_im], max_norm=1.0)
            optimizer.step()

            # -----------------------------
            # B. 只每隔 eval_every 步做一次全点验证
            # -----------------------------
            if step % eval_every == 0 or step == 1:
                with torch.no_grad():
                    val_loss, val_diag = teukolsky_residual_loss_coeff(
                        cfg=cfg,
                        y_nodes=y_nodes,
                        D=D,
                        D2=D2,
                        Tmat=Tmat,
                        coeff_re=coeff_re,
                        coeff_im=coeff_im,
                        a_batch=a_batch,
                        omega_batch=omega_batch,
                        lambda_batch=lambda_batch,
                        ramp_batch=ramp_batch,
                        p=p,
                        n_boundary_drop=n_boundary_drop,
                        collocation_idx=None,   # 全部内点验证
                    )

                    current_val_loss = float(val_loss.item())
                    

                improved = current_val_loss < best_loss * (1.0 - min_delta_ratio)

                if improved:
                    best_loss = current_val_loss
                    best_step = step
                    best_coeff_re = coeff_re.detach().clone()
                    best_coeff_im = coeff_im.detach().clone()
                    wait = 0
                else:
                    wait += 1

                loss_history.append(current_val_loss)
                step_history.append(step)

                pbar.set_postfix({
                    "train": f"{float(total_train_loss.item()):.6e}",
                    "val": f"{current_val_loss:.6e}",
                    "best": f"{best_loss:.6e}",
                    "wait": wait,
                    "Ns": int(batch_idx.numel()),
                })

                if wait >= patience:
                    pbar.write(
                        f"Early stopping triggered at step {step}, "
                        f"best_step={best_step}, best_val_loss={best_loss:.6e}"
                    )
                    pbar.update(1)
                    break
            if MONITOR == True:
                if step % monitor_every == 0 or step == 1:
                    with torch.no_grad():
                        r_np, res_np, stat = evaluate_residual_on_r_grid(
                            cfg=cfg,
                            coeff_re=coeff_re,
                            coeff_im=coeff_im,
                            a_batch=a_batch,
                            omega_batch=omega_batch,
                            lambda_batch=lambda_batch,
                            ramp_batch=ramp_batch,
                            p=p,
                            rmin=r_monitor_min,
                            rmax=r_monitor_max,
                            npts=monitor_npts,
                        )

                    monitor_steps.append(step)
                    monitor_max_history.append(stat["max_abs"])
                    monitor_mean_history.append(stat["mean_abs"])

                    # 打印几个代表性的 r 点
                    msg_parts = [
                        f"[monitor] step={step}",
                        f"train_loss={train_loss:.6e}",
                        f"mean|res|={stat['mean_abs']:.6e}",
                        f"max|res|={stat['max_abs']:.6e}",
                        f"r_at_max={stat['r_at_max']:.6f}",
                    ]

                    abs_res_np = np.abs(res_np)
                    for rs in sample_rs:
                        idx = int(np.argmin(np.abs(r_np - rs)))
                        msg_parts.append(f"|res({rs/M:.1f}M)|={abs_res_np[idx]:.3e}")

                    pbar.write("  ".join(msg_parts))

                    update_live_monitor(
                        monitor=live_monitor,
                        r_np=r_np,
                        res_np=res_np,
                        step=step,
                        current_loss=train_loss,
                        steps_monitor=monitor_steps,
                        max_hist=monitor_max_history,
                        mean_hist=monitor_mean_history,
                        loss_steps=step_history,
                        loss_hist=loss_history,
                        M=M,
                    )   

            pbar.update(1)


    with torch.no_grad():
        coeff_re.copy_(best_coeff_re)
        coeff_im.copy_(best_coeff_im)
    # -----------------------------
    # 6. 训练结束后保存系数
    # -----------------------------
    os.makedirs("outputs", exist_ok=True)
    save_path = "outputs/single_case_coeffs.pt"
    torch.save(
        {
            "a": a0,
            "omega": omega0,
            "order": order,
            "coeff_re": coeff_re.detach().cpu(),
            "coeff_im": coeff_im.detach().cpu(),
        },
        save_path,
    )
    print(f"\nSaved coefficients to: {save_path}")
    # 训练结束后，绘制 loss 曲线
    plt.figure(figsize=(10, 5))
    plt.semilogy(step_history, loss_history, label='Total Loss')  # 使用对数坐标更直观
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    # 保存图片
    save_path = os.path.join(loss_dir, f"loss_a{a0}_omega{omega0}_order{order}.png")
    plt.savefig(save_path, dpi=150)
    plt.close()  # 关闭图形，释放内存
    print(f"Loss curve saved to: {save_path}")


if __name__ == "__main__":
    main()