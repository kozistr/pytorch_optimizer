import torch

from pytorch_optimizer.optimizer.utils import unit_norm


def agc(
    p: torch.Tensor, grad: torch.Tensor, agc_eps: float = 1e-3, agc_clip_val: float = 1e-2, eps: float = 1e-6
) -> torch.Tensor:
    """Clip gradients relative to their parameter unit norms.

    Args:
        p: Parameter tensor.
        grad: Gradient tensor with the same shape as `p`.
        agc_eps: Lower bound for parameter unit norms.
        agc_clip_val: Maximum gradient-to-parameter norm ratio.
        eps: Lower bound for gradient unit norms.

    Returns:
        torch.Tensor: Clipped gradient tensor. The input gradient remains unchanged.

    """
    max_norm = unit_norm(p).clamp_min_(agc_eps).mul_(agc_clip_val)
    g_norm = unit_norm(grad).clamp_min_(eps)

    clipped_grad = grad * (max_norm / g_norm)

    return torch.where(g_norm > max_norm, clipped_grad, grad)
