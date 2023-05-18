from typing import Optional

import torch

from pytorch_optimizer.optimizer.utils import unit_norm


def agc(
    p: torch.Tensor,
    grad: torch.Tensor,
    agc_eps: float,
    agc_clip_val: float,
    eps: float = 1e-6,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    r"""Clip gradient values in excess of the unit wise norm.

    :param p: torch.Tensor. parameter.
    :param grad: torch.Tensor, gradient.
    :param agc_eps: float. agc epsilon to clip the norm of parameter.
    :param agc_clip_val: float. norm clip.
    :param eps: float. simple stop from div by zero and no relation to standard optimizer eps.
    :param out: Optional[torch.Tensor]. variable to save clipped gradient.
    """
    p_norm = unit_norm(p).clamp_(agc_eps)
    g_norm = unit_norm(grad)

    max_norm = p_norm * agc_clip_val

    clipped_grad = grad * (max_norm / g_norm.clamp_min_(eps))

    if out is None:
        return torch.where(g_norm > max_norm, clipped_grad, grad)

    torch.where(g_norm > max_norm, clipped_grad, grad, out=out)
