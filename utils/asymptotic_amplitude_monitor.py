from __future__ import annotations

from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils.mode import KerrMode
from utils.amplitude import r_star
from pybhpt_usage.compute_solution import compute_pybhpt_solution


def _complex_rel_error(a: np.ndarray, b: np.ndarray, eps: float = 1.0e-30):
    return np.abs(a - b) / np.maximum(np.abs(b), eps)


def _best_complex_scale(y: np.ndarray, x: np.ndarray, eps: float = 1.0e-30) -> complex:
    """Find c minimizing ||c*x - y||_2."""
    denom = np.vdot(x, x)
    if abs(denom) < eps:
        return 1.0 + 0.0j
    return np.vdot(x, y) / denom


def compute_R_asym_from_B(
    r_grid: np.ndarray,
    Binc: complex,
    Bref: complex,
    M: float,
    a: float,
    omega: float,
    ell: int,
    m: int,
    s: int = -2,
    lam: complex | None = None,
):
    mode = KerrMode(M=M, a=a, omega=omega, ell=ell, m=m, lam=lam, s=s)
    r_grid = np.asarray(r_grid, dtype=np.float64)
    rs = r_star(r_grid, mode)
    return (
        Binc * r_grid ** (-1.0) * np.exp(-1j * omega * rs)
        + Bref * r_grid ** 3.0 * np.exp(+1j * omega * rs)
    )


@torch.no_grad()
def monitor_network_amplitudes_vs_pybhpt(
    model,
    physics_cfg: dict,
    a: float,
    omega: float,
    u: float,
    v: float,
    lam: complex | None,
    out_path: str | Path,
    r_points=None,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
    pybhpt_timeout: float = 20.0,
):
    """Compare network-predicted asymptotic R_amp with pybhpt R_in."""
    problem_cfg = physics_cfg["problem"]
    M = float(problem_cfg.get("M", 1.0))
    s = int(problem_cfg.get("s", -2))
    ell = int(problem_cfg.get("l", problem_cfg.get("ell", 2)))
    m = int(problem_cfg.get("m", 2))

    if r_points is None:
        r_points = np.array([200.0, 300.0, 500.0, 800.0, 1000.0], dtype=np.float64)
    else:
        r_points = np.asarray(r_points, dtype=np.float64)

    dev = torch.device(device)

    a_t = torch.tensor([a], device=dev, dtype=dtype)
    omega_t = torch.tensor([omega], device=dev, dtype=dtype)
    u_t = torch.tensor([u], device=dev, dtype=dtype)
    v_t = torch.tensor([v], device=dev, dtype=dtype)

    if not hasattr(model, "predict_asymptotic_amplitudes"):
        raise AttributeError("model has no predict_asymptotic_amplitudes method.")

    Binc_t, Bref_t, amp_raw = model.predict_asymptotic_amplitudes(
        a_t, omega_t, u=u_t, v=v_t
    )

    Binc = complex(Binc_t.detach().cpu().numpy()[0])
    Bref = complex(Bref_t.detach().cpu().numpy()[0])

    R_amp = compute_R_asym_from_B(
        r_grid=r_points,
        Binc=Binc,
        Bref=Bref,
        M=M,
        a=a,
        omega=omega,
        ell=ell,
        m=m,
        s=s,
        lam=lam,
    )

    _, R_py = compute_pybhpt_solution(
        a=a,
        omega=omega,
        ell=ell,
        m=m,
        r_grid=r_points,
        timeout=pybhpt_timeout,
    )

    raw_rel = _complex_rel_error(R_amp, R_py)

    c_fit = _best_complex_scale(R_py, R_amp)
    R_amp_scaled = c_fit * R_amp
    scaled_rel = _complex_rel_error(R_amp_scaled, R_py)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(4, 1, figsize=(10, 13), sharex=True)

    axes[0].plot(r_points, np.abs(R_py), "o-", label="|R_in| pybhpt")
    axes[0].plot(r_points, np.abs(R_amp), "s--", label="|R_amp| raw")
    axes[0].plot(r_points, np.abs(R_amp_scaled), "d--", label="|c R_amp| scaled")
    axes[0].set_ylabel("abs")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(r_points, R_py.real, "o-", label="Re pybhpt")
    axes[1].plot(r_points, R_amp.real, "s--", label="Re R_amp raw")
    axes[1].plot(r_points, R_amp_scaled.real, "d--", label="Re c R_amp")
    axes[1].set_ylabel("real")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    axes[2].plot(r_points, R_py.imag, "o-", label="Im pybhpt")
    axes[2].plot(r_points, R_amp.imag, "s--", label="Im R_amp raw")
    axes[2].plot(r_points, R_amp_scaled.imag, "d--", label="Im c R_amp")
    axes[2].set_ylabel("imag")
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    axes[3].semilogy(r_points, raw_rel, "s--", label="raw rel error")
    axes[3].semilogy(r_points, scaled_rel, "d--", label="scaled rel error")
    axes[3].set_xlabel("r")
    axes[3].set_ylabel("relative error")
    axes[3].legend()
    axes[3].grid(alpha=0.3)

    fig.suptitle(
        f"Network amplitude asymptotic monitor vs pybhpt\n"
        f"a={a:.8g}, omega={omega:.8g}, ell={ell}, m={m}\n"
        f"Binc={Binc:.4e}, Bref={Bref:.4e}, c_fit={c_fit:.4e}"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)

    metrics = {
        "a": float(a),
        "omega": float(omega),
        "u": float(u),
        "v": float(v),
        "Binc_re": float(Binc.real),
        "Binc_im": float(Binc.imag),
        "Bref_re": float(Bref.real),
        "Bref_im": float(Bref.imag),
        "scale_fit_re": float(c_fit.real),
        "scale_fit_im": float(c_fit.imag),
        "raw_rel_mean": float(np.mean(raw_rel)),
        "raw_rel_max": float(np.max(raw_rel)),
        "scaled_rel_mean": float(np.mean(scaled_rel)),
        "scaled_rel_max": float(np.max(scaled_rel)),
        "plot_path": str(out_path),
    }
    return metrics
