from __future__ import annotations


def stage_weights(stage: int, step: int, total_steps: int) -> dict[str, float]:
    if stage <= 0:
        return {"inner": 0.0, "outer": 0.0, "R_anchor": 1.0, "amp_anchor": 0.0, "residual": 5.0e-2, "weak": 0.0}
    if stage <= 1:
        return {"inner": 1.0, "outer": 1.0, "R_anchor": 0.5, "amp_anchor": 0.2, "residual": 1.0e-3, "weak": 0.0}
    ramp = min(1.0, step / max(total_steps, 1))
    return {"inner": 1.0, "outer": 1.0, "R_anchor": 0.5 - 0.4 * ramp, "amp_anchor": 0.2, "residual": 1.0e-3 + (1.0 - 1.0e-3) * ramp, "weak": 0.0}
