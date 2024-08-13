import math
import warnings
from importlib.util import find_spec
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn
from torch.distributed import all_reduce
from torch.nn.functional import cosine_similarity
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.utils import clip_grad_norm_

from pytorch_optimizer.base.types import PARAMETERS

HAS_TRANSFORMERS: bool = find_spec('transformers') is not None

if HAS_TRANSFORMERS:  # pragma: no cover
    try:
        from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
    except ImportError:
        from transformers.deepspeed import is_deepspeed_zero3_enabled
else:

    def is_deepspeed_zero3_enabled() -> bool:
        r"""Check if DeepSpeed zero3 is enabled."""
        if HAS_TRANSFORMERS:
            return is_deepspeed_zero3_enabled()  # pragma: no cover

        warnings.warn(
            'you need to install `transformers` to use `is_deepspeed_zero3_enabled` function. it\'ll return False.',
            category=ImportWarning,
            stacklevel=2,
        )

        return False


def is_valid_parameters(parameters: PARAMETERS) -> bool:
    r"""Check where the parameters are valid."""
    return isinstance(parameters, (list, tuple)) and len(parameters) > 0 and isinstance(parameters[0], dict)


def has_overflow(grad_norm: torch.Tensor) -> bool:
    r"""Detect inf and NaN in grad_norm."""
    return bool(torch.logical_or(torch.isnan(grad_norm), torch.isinf(grad_norm)).any())


def to_real(x: torch.Tensor) -> torch.Tensor:
    r"""Return real value of tensor."""
    return x.real if torch.is_complex(x) else x


def normalize_gradient(x: torch.Tensor, use_channels: bool = False, epsilon: float = 1e-8) -> None:
    r"""Normalize gradient with stddev.

    :param x: torch.Tensor. gradient.
    :param use_channels: bool. channel-wise normalization.
    :param epsilon: float. eps.
    """
    size: int = x.dim()
    if size > 1 and use_channels:
        s = x.std(dim=tuple(range(1, size)), keepdim=True).add_(epsilon)
        x.div_(s)
    elif torch.numel(x) > 2:
        s = x.std().add_(epsilon)
        x.div_(s)


def flatten_grad(grads: List[torch.Tensor]) -> torch.Tensor:
    r"""Flatten the gradient."""
    return torch.cat([grad.flatten() for grad in grads])


def un_flatten_grad(grads: torch.Tensor, shapes: List[int]) -> List[torch.Tensor]:
    r"""Unflatten the gradient."""
    idx: int = 0
    un_flatten_grads: List[torch.Tensor] = []
    for shape in shapes:
        length = np.prod(shape)
        un_flatten_grads.append(grads[idx:idx + length].view(shape).clone())  # fmt: skip
        idx += length
    return un_flatten_grads


def channel_view(x: torch.Tensor) -> torch.Tensor:
    r"""Do channel view."""
    return x.view(x.size()[0], -1)


def layer_view(x: torch.Tensor) -> torch.Tensor:
    r"""Do layer view."""
    return x.view(1, -1)


def cosine_similarity_by_view(
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float,
    view_func: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    r"""Calculate cosine similarity by the view.

    :param x: torch.Tensor. src.
    :param y: torch.Tensor. dst.
    :param eps: float. epsilon.
    :param view_func: Callable. view (channel or layer) function.
    """
    x = view_func(x)
    y = view_func(y)
    return cosine_similarity(x, y, dim=1, eps=eps).abs_()


def clip_grad_norm(
    parameters: PARAMETERS,
    max_norm: float = 0.0,
    sync: bool = False,
) -> Union[torch.Tensor, float]:  # pragma: no cover
    r"""Clip gradient norms.

        During combination with FSDP, will also ensure that grad norms are aggregated across all workers,
        since each worker only stores their shard of the gradients.

    :param parameters: PARAMETERS. Parameters whose gradients we wish to clip.
    :param max_norm: float. Maximum norm we wish the gradients to have. If non-positive, then we will not perform
        clipping.
    :param sync: bool. Boolean indicating whether we should aggregate across the distributed group. Used only in
        combination with FSDP.
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
    p: torch.Tensor,
    grad: torch.Tensor,
    perturb: torch.Tensor,
    delta: float,
    wd_ratio: float,
    eps: float,
) -> Tuple[torch.Tensor, float]:
    r"""Project to remove the radial component from the update vector."""
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
    r"""Get norm of unit."""
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

    return x.norm(p=norm, dim=dim, keepdim=keep_dim)


def get_optimizer_parameters(
    model_or_parameter: Union[nn.Module, List],
    weight_decay: float,
    wd_ban_list: List[str] = ('bias', 'LayerNorm.bias', 'LayerNorm.weight'),
) -> PARAMETERS:
    r"""Get optimizer parameters while filtering specified modules.

    :param model_or_parameter: Union[nn.Module, List]. model or parameters.
    :param weight_decay: float. weight_decay.
    :param wd_ban_list: List[str]. ban list not to set weight decay.
    :returns: PARAMETERS. new parameter list.
    """
    if isinstance(model_or_parameter, nn.Module):
        model_or_parameter = list(model_or_parameter.named_parameters())

    return [
        {
            'params': [p for n, p in model_or_parameter if p.requires_grad and not any(nd in n for nd in wd_ban_list)],
            'weight_decay': weight_decay,
        },
        {
            'params': [p for n, p in model_or_parameter if p.requires_grad and any(nd in n for nd in wd_ban_list)],
            'weight_decay': 0.0,
        },
    ]


def neuron_norm(x: torch.Tensor) -> torch.Tensor:
    r"""Get norm of the tensor."""
    if x.dim() <= 1:
        return x.abs()

    view_shape: List[int] = [x.shape[0]] + [1] * (x.dim() - 1)

    return channel_view(x).norm(dim=1).view(*view_shape)


def neuron_mean(x: torch.Tensor) -> torch.Tensor:
    r"""Get mean of the tensor."""
    if x.dim() <= 1:
        raise ValueError('[-] neuron_mean not defined on 1D tensors.')

    view_shape: List[int] = [x.shape[0]] + [1] * (x.dim() - 1)

    return channel_view(x).mean(dim=1).view(*view_shape)


def disable_running_stats(model):
    r"""Disable running stats (momentum) of BatchNorm."""

    def _disable(module):
        if isinstance(module, _BatchNorm):
            module.backup_momentum = module.momentum
            module.momentum = 0

    model.apply(_disable)


def enable_running_stats(model):
    r"""Enable running stats (momentum) of BatchNorm."""

    def _enable(module):
        if isinstance(module, _BatchNorm) and hasattr(module, 'backup_momentum'):
            module.momentum = module.backup_momentum

    model.apply(_enable)


@torch.no_grad()
def l2_projection(parameters: PARAMETERS, max_norm: float = 1e2):
    r"""Get l2 normalized parameter."""
    global_norm = torch.sqrt(sum(p.norm().pow(2) for p in parameters))
    if global_norm > max_norm:
        ratio = max_norm / global_norm
        for param in parameters:
            param.mul_(ratio)


@torch.no_grad()
def get_global_gradient_norm(param_groups: List[Dict]) -> torch.Tensor:
    r"""Get global gradient norm."""
    global_grad_norm = torch.zeros(1, dtype=torch.float32, device=param_groups[0]['params'][0].device)

    for group in param_groups:
        for p in group['params']:
            if p.grad is not None:
                global_grad_norm.add_(p.grad.norm().pow(2))

    return global_grad_norm


@torch.no_grad()
def reduce_max_except_dim(x: torch.Tensor, dim: int) -> torch.Tensor:
    r"""Perform reduce-max along all dimensions except the given dim.

    :param x: torch.Tensor. tensor to reduce-max.
    :param dim: int. dimension to exclude.
    """
    rank: int = len(x.shape)
    if rank == 0:
        return x

    if dim >= rank:
        raise ValueError(f'[-] given dim is bigger than rank. {dim} >= {rank}')

    for d in range(rank):
        if d != dim:
            x = x.max(dim=d, keepdim=True).values
    return x


@torch.no_grad()
def reg_noise(
    network1: nn.Module, network2: nn.Module, num_data: int, lr: float, eta: float = 8e-3, temperature: float = 1e-4
) -> Union[torch.Tensor, float]:
    r"""Entropy-MCMC: Sampling from flat basins with ease.

    usage: https://github.com/lblaoke/EMCMC/blob/master/exp/cifar10_emcmc.py

    :param network1: nn.Module. network.
    :param network2: nn.Module. network.
    :param num_data: int. number of training data.
    :param lr: float. learning rate.
    :param eta: float. eta.
    :param temperature: float. temperature.
    """
    reg_coef: float = 0.5 / (eta * num_data)
    noise_coef: float = math.sqrt(2.0 / lr / num_data * temperature)

    loss = torch.tensor(0.0, device=next(network1.parameters()).device)

    for param1, param2 in zip(network1.parameters(), network2.parameters()):
        reg = (param1 - param2).pow_(2).mul_(reg_coef).sum()

        noise = param1 * torch.randn_like(param1)
        noise.add_(param2 * torch.randn_like(param2))

        loss.add_(reg - noise.mul_(noise_coef).sum())

    return loss
