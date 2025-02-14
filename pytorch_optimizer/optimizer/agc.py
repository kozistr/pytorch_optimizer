import torch

from pytorch_optimizer.optimizer.utils import unit_norm


def agc(
    p: torch.Tensor, grad: torch.Tensor, agc_eps: float = 1e-3, agc_clip_val: float = 1e-2, eps: float = 1e-6
) -> torch.Tensor:
    r"""Clip gradient values in excess of the unit wise norm.

    :param p: torch.Tensor. parameter.
    :param grad: torch.Tensor, gradient.
    :param agc_eps: float. agc epsilon to clip the norm of parameter.
    :param agc_clip_val: float. norm clip.
    :param eps: float. simple stop from div by zero and no relation to standard optimizer eps.
    """
    max_norm = unit_norm(p).clamp_min_(agc_eps).mul_(agc_clip_val)
    g_norm = unit_norm(grad).clamp_min_(eps)

    clipped_grad = grad * (max_norm / g_norm)

    return torch.where(g_norm > max_norm, clipped_grad, grad)
