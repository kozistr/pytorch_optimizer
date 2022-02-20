import math
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn
from torch.distributed import all_reduce
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_

from pytorch_optimizer.types import PARAMETERS


def is_valid_parameters(parameters: PARAMETERS) -> bool:
    return isinstance(parameters, (list, tuple)) and len(parameters) > 0 and isinstance(parameters[0], dict)


def has_overflow(grad_norm: torch.Tensor) -> bool:
    """Detect inf and NaN in grad_norm."""
    return grad_norm != grad_norm or grad_norm == float('inf')  # pylint: disable=comparison-with-itself


def normalize_gradient(x: torch.Tensor, use_channels: bool = False, epsilon: float = 1e-8) -> torch.Tensor:
    """normalize gradient with stddev
    :param x: torch.Tensor. gradient
    :param use_channels: bool. channel-wise normalization
    :param epsilon: float. eps
    :return: torch.Tensor. normalized gradient.
    """
    size: int = x.dim()
    if size > 1 and use_channels:
        s = x.std(dim=tuple(range(1, size)), keepdim=True) + epsilon
        x.div_(s)
    elif torch.numel(x) > 2:
        s = x.std() + epsilon
        x.div_(s)
    return x


def flatten_grad(grads: List[torch.Tensor]) -> torch.Tensor:
    return torch.cat([g.flatten() for g in grads])


def un_flatten_grad(grads: torch.Tensor, shapes: List[int]) -> List[torch.Tensor]:
    idx: int = 0
    un_flatten_grad: List[torch.Tensor] = []
    for shape in shapes:
        length = np.prod(shape)
        un_flatten_grad.append(grads[idx : idx + length].view(shape).clone())
        idx += length
    return un_flatten_grad


def channel_view(x: torch.Tensor) -> torch.Tensor:
    return x.view(x.size()[0], -1)


def layer_view(x: torch.Tensor) -> torch.Tensor:
    return x.view(1, -1)


def cosine_similarity_by_view(
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float,
    view_func: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    x = view_func(x)
    y = view_func(y)
    return F.cosine_similarity(x, y, dim=1, eps=eps).abs_()


def clip_grad_norm(parameters: PARAMETERS, max_norm: float = 0, sync: bool = False) -> Union[torch.Tensor, float]:
    """Clips grad norms.
    During combination with FSDP, will also ensure that grad norms are aggregated
    across all workers, since each worker only stores their shard of the gradients
    :param parameters: Parameters whose gradients we wish to clip
    :param max_norm: Maximum norm we wish the gradients to have. If non-positive, then
        we will not perform clipping
    :param sync: Boolean indicating whether we should aggregate across the distributed
        group. Used only in combination with FSDP
    :returns: The gradient norm across all parameters, before clipping.
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]

    # make sure any generators are expanded
    parameters = list(parameters)

    # if syncing we need to manually perform the clipping so that we aggregate properly
    if max_norm > 0 and not sync:
        return clip_grad_norm_(parameters, max_norm)

    norm_sq = sum(p.grad.norm() ** 2 for p in parameters if p.grad is not None)
    if sync:
        # also need to get the norms from all the other sharded works in FSDP
        all_reduce(norm_sq)

    grad_norm = math.sqrt(norm_sq)
    if max_norm > 0:
        clip_coefficient = max_norm / (grad_norm + 1e-6)
        for p in parameters:
            p.grad.detach().mul_(clip_coefficient)

    return grad_norm


def projection(
    p,
    grad,
    perturb: torch.Tensor,
    delta: float,
    wd_ratio: float,
    eps: float,
) -> Tuple[torch.Tensor, float]:
    wd: float = 1.0
    expand_size: List[int] = [-1] + [1] * (len(p.shape) - 1)
    for view_func in (channel_view, layer_view):
        cosine_sim = cosine_similarity_by_view(grad, p, eps, view_func)

        if cosine_sim.max() < delta / math.sqrt(view_func(p).size()[1]):
            p_n = p / view_func(p).norm(dim=1).view(expand_size).add_(eps)
            perturb -= p_n * view_func(p_n * perturb).sum(dim=1).view(expand_size)
            wd = wd_ratio
            return perturb, wd

    return perturb, wd


def unit_norm(x: torch.Tensor, norm: float = 2.0) -> torch.Tensor:
    keep_dim: bool = True
    dim: Optional[Union[int, Tuple[int, ...]]] = None

    x_len: int = len(x.shape)
    if x_len <= 1:
        keep_dim = False
    elif x_len in (2, 3):  # linear layers
        dim = 1
    elif x_len == 4:  # conv kernels
        dim = (1, 2, 3)
    else:
        dim = tuple(range(1, x_len))

    return x.norm(dim=dim, keepdim=keep_dim, p=norm)


def get_optimizer_parameters(
    model: nn.Module, weight_decay: float, wd_ban_list: List[str] = ('bias', 'LayerNorm.bias', 'LayerNorm.weight')
) -> PARAMETERS:
    param_optimizer: List[Tuple[str, nn.Parameter]] = list(model.named_parameters())

    return [
        {
            'params': [p for n, p in param_optimizer if not any(nd in n for nd in wd_ban_list)],
            'weight_decay': weight_decay,
        },
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in wd_ban_list)], 'weight_decay': 0.0},
    ]
