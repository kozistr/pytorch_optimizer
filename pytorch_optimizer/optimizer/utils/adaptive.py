import torch


def update_adaptive_distance(
    d: torch.Tensor,
    estimate: torch.Tensor,
    lr: torch.Tensor,
    active: torch.Tensor,
    growth_rate: float = float('inf'),
) -> torch.Tensor:
    """Bound an adaptive distance estimate without reading a device scalar."""
    updated = torch.fmax(d, torch.minimum(estimate, d * growth_rate))
    return torch.where(active & (lr > 0.0), updated, d)
