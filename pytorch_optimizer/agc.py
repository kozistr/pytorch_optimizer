import torch

from pytorch_optimizer.utils import unit_norm


def agc(p, agc_eps: float, agc_clip_val: float, eps: float = 1e-6):
    """clip gradient values in excess of the unit-wise norm.
    :param p: parameter.
    :param agc_eps: float.
    :param agc_clip_val: float.
    :param eps: float. simple stop from div by zero and no relation to standard optimizer eps
    """
    p_norm = unit_norm(p).clamp_(agc_eps)
    g_norm = unit_norm(p.grad)

    max_norm = p_norm * agc_clip_val

    clipped_grad = p.grad * (max_norm / g_norm.clamp(min=eps))

    new_grads = torch.where(g_norm > max_norm, clipped_grad, p.grad)
    p.grad.detach().copy_(new_grads)
