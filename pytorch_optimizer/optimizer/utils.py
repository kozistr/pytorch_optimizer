import math
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn
from torch.distributed import all_reduce
from torch.nn import functional as f
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.utils import clip_grad_norm_

from pytorch_optimizer.base.types import PARAMETERS


def is_valid_parameters(parameters: PARAMETERS) -> bool:
    return isinstance(parameters, (list, tuple)) and len(parameters) > 0 and isinstance(parameters[0], dict)


def has_overflow(grad_norm: torch.Tensor) -> bool:
    r"""Detect inf and NaN in grad_norm."""
    return grad_norm != grad_norm or grad_norm == float('inf')  # pylint: disable=comparison-with-itself


def normalize_gradient(x: torch.Tensor, use_channels: bool = False, epsilon: float = 1e-8) -> torch.Tensor:
    r"""normalize gradient with stddev

    :param x: torch.Tensor. gradient.
    :param use_channels: bool. channel-wise normalization.
    :param epsilon: float. eps.
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
    un_flatten_grads: List[torch.Tensor] = []
    for shape in shapes:
        length = np.prod(shape)
        un_flatten_grads.append(grads[idx:idx + length].view(shape).clone())  # fmt: skip
        idx += length
    return un_flatten_grads


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
    return f.cosine_similarity(x, y, dim=1, eps=eps).abs_()


def clip_grad_norm(parameters: PARAMETERS, max_norm: float = 0, sync: bool = False) -> Union[torch.Tensor, float]:
    r"""Clips gradient norms. During combination with FSDP, will also ensure that grad norms are aggregated
        across all workers, since each worker only stores their shard of the gradients.

    :param parameters: PARAMETERS. Parameters whose gradients we wish to clip.
    :param max_norm: float. Maximum norm we wish the gradients to have. If non-positive, then
        we will not perform clipping.
    :param sync: bool. Boolean indicating whether we should aggregate across the distributed group.
        Used only in combination with FSDP.
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
    r"""get optimizer parameters while filtering specified modules.

    :param model: nn.Module. model.
    :param weight_decay: float. weight_decay.
    :param wd_ban_list: List[str]. ban list not to set weight decay.
    :returns: PARAMETERS. new parameter list.
    """
    param_optimizer: List[Tuple[str, nn.Parameter]] = list(model.named_parameters())

    return [
        {
            'params': [p for n, p in param_optimizer if not any(nd in n for nd in wd_ban_list)],
            'weight_decay': weight_decay,
        },
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in wd_ban_list)], 'weight_decay': 0.0},
    ]


def neuron_norm(x: torch.Tensor) -> torch.Tensor:
    if x.dim() <= 1:
        return x.abs()

    view_shape = [x.shape[0]] + [1] * (x.dim() - 1)
    x = x.view(x.shape[0], -1)

    return x.norm(dim=1).view(*view_shape)


def neuron_mean(x: torch.Tensor) -> torch.Tensor:
    if x.dim() <= 1:
        raise ValueError('[-] neuron_mean not defined on 1D tensors.')

    view_shape = [x.shape[0]] + [1] * (x.dim() - 1)
    x = x.view(x.shape[0], -1)

    return x.mean(dim=1).view(*view_shape)


def disable_running_stats(model):
    r"""disable running stats (momentum) of BatchNorm"""

    def _disable(module):
        if isinstance(module, _BatchNorm):
            module.backup_momentum = module.momentum
            module.momentum = 0

    model.apply(_disable)


def enable_running_stats(model):
    r"""enable running stats (momentum) of BatchNorm"""

    def _enable(module):
        if isinstance(module, _BatchNorm) and hasattr(module, 'backup_momentum'):
            module.momentum = module.backup_momentum

    model.apply(_enable)


@torch.no_grad()
def power_iter(
    mat_g: torch.Tensor, error_tolerance: float = 1e-6, num_iters: int = 100
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    r"""Power iteration. Compute the maximum eigenvalue of mat, for scaling.
        v is a random vector with values in (-1, 1)

    :param mat_g: torch.Tensor. the symmetric PSD matrix.
    :param error_tolerance: float. Iterative exit condition.
    :param num_iters: int. Number of iterations.
    """
    v: torch.Tensor = torch.rand(list(mat_g.shape)[0], device=mat_g.device) * 2 - 1

    error: torch.Tensor = 1.0
    iters: int = 0
    singular_val: torch.Tensor = 0
    while error > error_tolerance and iters < num_iters:
        v.div_(v.norm())
        mat_v = torch.mv(mat_g, v)
        s_v = torch.dot(v, mat_v)
        error = torch.abs(s_v - singular_val)
        v.copy_(mat_v)
        singular_val = s_v
        iters += 1

    return singular_val, v / torch.norm(v), iters


@torch.no_grad()
def mat_power(mat_m: torch.Tensor, p: int) -> torch.Tensor:
    r"""Computes mat_m^p, for p a positive integer.

    :param mat_m: torch.Tensor. a square matrix.
    :param p: int. a positive integer.
    """
    if p in (1, 2, 4, 8, 16, 32):
        p_done: int = 1
        res = mat_m
        while p_done < p:
            res = torch.matmul(res, res)
            p_done *= 2
        return res

    power = None
    while p > 0:
        if p % 2 == 1:
            power = torch.matmul(mat_m, power) if power is not None else mat_m
        p //= 2
        mat_m = torch.matmul(mat_m, mat_m)
    return power


@torch.no_grad()
def compute_power(
    mat_g: torch.Tensor, p: int, iter_count: int = 100, error_tolerance: float = 1e-6, ridge_epsilon: float = 1e-6
) -> torch.Tensor:
    r"""A method to compute G^{-1/p} using a coupled Newton iteration. See for example equation 3.2 on page 9 of:
        A Schur-Newton Method for the Matrix p-th Root and its Inverse by Chun-Hua Guo and Nicholas J. Higham
        SIAM Journal on Matrix Analysis and Applications, 2006, Vol. 28, No. 3 : pp. 788-804
        https://pdfs.semanticscholar.org/0abe/7f77433cf5908bfe2b79aa91af881da83858.pdf

    :param mat_g: torch.Tensor. A square positive semi-definite matrix.
    :param p: int. a positive integer.
    :param iter_count: int. Stop iterating after this many rounds.
    :param error_tolerance: float. Threshold for stopping iteration.
    :param ridge_epsilon: float. We add this times I to G, to make is positive definite. For scaling,
        we multiply it by the largest eigenvalue of G.
    """
    shape: List[int] = list(mat_g.shape)
    if len(shape) == 1:
        return torch.pow(mat_g + ridge_epsilon, -1 / p)

    identity = torch.eye(shape[0], device=mat_g.device)
    if shape[0] == 1:
        return identity

    max_ev, _, _ = power_iter(mat_g)
    ridge_epsilon *= max_ev
    mat_g += ridge_epsilon * identity

    z: torch.Tensor = (1 + p) / (2 * torch.norm(mat_g))

    mat_root = identity * torch.pow(z, 1.0 / p)
    mat_m = mat_g * z

    alpha: float = -1.0 / p
    error = torch.max(torch.abs(mat_m - identity))
    count: int = 0
    while error > error_tolerance and count < iter_count:
        tmp_mat_m = (1 - alpha) * identity + alpha * mat_m
        new_mat_root = torch.matmul(mat_root, tmp_mat_m)
        mat_m = torch.matmul(mat_power(tmp_mat_m, p), mat_m)

        new_error = torch.max(torch.abs(mat_m - identity))
        if new_error > error * 1.2:
            break

        mat_root = new_mat_root
        error = new_error
        count += 1

    return mat_root


def merge_small_dims(shape_to_merge: List[int], max_dim: int) -> List[int]:
    r"""Merge small dimensions. If there are some small dimensions, we collapse them:
        e.g. [1, 2, 512, 1, 2048, 1, 3, 4] --> [1024, 2048, 12] if max_dim = 1024
        [1, 2, 768, 1, 2048] --> [2, 768, 2048]

    :param shape_to_merge: List. Shape to merge small dimensions.
    :param max_dim: int. Maximal dimension of output shape used in merging.
    """
    resulting_shape: List[int] = []

    product: int = 1
    for d in shape_to_merge:
        if product * d <= max_dim:
            product *= d
        else:
            if product > 1:
                resulting_shape.append(product)
            product = d

    if product > 1:
        resulting_shape.append(product)

    return resulting_shape
