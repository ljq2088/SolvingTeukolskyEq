"""
Causal spatial weighting for PINN training (Wang et al. 2022).

Supports multiple causal directions and protected regions.
"""
import torch


def compute_causal_weights(
    y: torch.Tensor,
    pointwise_residual: torch.Tensor,
    n_chunks: int = 16,
    epsilon: float = 1.0,
    direction: str = "horizon_first",
    y_protect_above: float | None = None,
):
    """
    Compute causal weights for each collocation point.

    Args:
        y: (Ny,) collocation points, sorted ascending [-1, +1].
        pointwise_residual: (B, Ny) absolute PDE residual at each point.
        n_chunks: number of spatial chunks for the causal region.
        epsilon: causal strength (larger = stricter).
        direction: "horizon_first", "infinity_first", or unused when y_protect_above set.
        y_protect_above: if set, points with y > this value get weight 1.0.
            Causal propagation goes from this boundary toward the other end.

    Returns:
        w: (Ny,) causal weights (broadcastable over batch).
        chunk_info: list of per-chunk diagnostics.
    """
    Ny = y.numel()

    # Normalize each sample's residual by its spatial mean so causal weights
    # depend on the relative spatial distribution, not absolute magnitude.
    # This prevents a few large-residual points (e.g. near infinity) from
    # zeroing out all downstream chunk weights.
    sample_mean = pointwise_residual.mean(dim=-1, keepdim=True).clamp(min=1e-30)
    pointwise_residual = pointwise_residual / sample_mean

    # Build full weight tensor
    w = torch.ones(Ny, device=y.device, dtype=y.dtype)
    chunk_losses = []

    if y_protect_above is not None:
        # Protected region: y > y_protect_above → weight 1.0
        # Causal region: y <= y_protect_above → causal from y_protect_above toward -1
        causal_mask = y <= y_protect_above
        if not causal_mask.any():
            return w, chunk_losses

        y_causal = y[causal_mask]
        causal_indices = torch.nonzero(causal_mask, as_tuple=False).squeeze(-1)

        # Sort from y_protect_above boundary (highest y in causal region) down to -1
        sort_idx_local = torch.argsort(y_causal, descending=True)
        y_sorted = y_causal[sort_idx_local]
        res_sorted = pointwise_residual[:, causal_indices[sort_idx_local]]
        Ny_causal = y_causal.numel()
    else:
        if direction == "infinity_first":
            sort_idx = torch.argsort(y, descending=False)
        else:
            sort_idx = torch.argsort(y, descending=True)
        res_sorted = pointwise_residual[:, sort_idx]
        Ny_causal = Ny

    # Split into equal chunks
    chunk_size = max(1, Ny_causal // n_chunks)
    actual_chunks = max(2, Ny_causal // chunk_size)

    # Build weights for causal region
    w_causal = torch.ones(Ny_causal, device=y.device, dtype=y.dtype)
    cumulative_loss = 0.0

    for i in range(actual_chunks):
        start = i * chunk_size
        end = start + chunk_size if i < actual_chunks - 1 else Ny_causal
        chunk_res = res_sorted[:, start:end]
        L_i = float(chunk_res.mean().detach().cpu().item())

        if i == 0:
            w_i = 1.0
        else:
            w_i = float(torch.exp(torch.tensor(-epsilon * cumulative_loss)).item())

        w_causal[start:end] = w_i
        chunk_losses.append({"chunk": i, "L": L_i, "w": w_i, "cum_loss": cumulative_loss})
        cumulative_loss += L_i

    if y_protect_above is not None:
        # Map back to original indices
        unsort_local = torch.argsort(sort_idx_local)
        w[causal_indices] = w_causal[unsort_local]
    else:
        # Unsort back to original y order
        unsort_idx = torch.argsort(sort_idx)
        w = w_causal[unsort_idx]

    return w, chunk_losses
