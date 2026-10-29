import torch


def can_reduce_batch(tensors: list[torch.Tensor]) -> bool:
    """Check that FP32 reductions fit a concatenation buffer of at most four MiB."""
    return tensors[0].dtype == torch.float32 and len(tensors) > 1 and sum(t.numel() for t in tensors) <= 2 ** 20


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
