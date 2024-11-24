import functools
import math
import operator
import re
import warnings
from importlib.util import find_spec
from typing import Callable, Dict, List, Optional, Tuple, Type, Union

import torch
from torch import nn
from torch.distributed import all_reduce
from torch.nn.functional import cosine_similarity
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.utils import clip_grad_norm_

from pytorch_optimizer.base.types import CLOSURE, LOSS, PARAMETERS


def parse_pytorch_version(version_string: str) -> List[int]:
    r"""Parse Pytorch version."""
    match = re.match(r'(\d+\.\d+\.\d+)', version_string)
    if not match:
        raise ValueError(f'invalid version string format: {version_string}')

    return [int(x) for x in match.group(1).split('.')]


def compare_versions(v1: str, v2: str) -> bool:
    r"""Compare two Pytorch versions."""
    v1_parts: List[int] = parse_pytorch_version(v1)
    v2_parts: List[int] = parse_pytorch_version(v2)
    return (v1_parts > v2_parts) - (v1_parts < v2_parts)


HAS_TRANSFORMERS: bool = find_spec('transformers') is not None
TORCH_VERSION_AT_LEAST_2_4: bool = compare_versions(torch.__version__, '2.4.0')

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


class CPUOffloadOptimizer:  # pragma: no cover
    """Offload optimizer to CPU for single-GPU training. This will reduce GPU memory by the size of optimizer state.

    Reference: https://github.com/pytorch/ao/blob/main/torchao/prototype/low_bit_optim/cpu_offload.py

    :param params: PARAMETERS. a list of parameters or parameter groups.
    :param optimizer_class: Type[torch.optim.Optimizer]. constructor of the base optimizer. Defaults to
        :class:`torch.optim.AdamW`.
    :param offload_gradients: bool. free GPU gradients once they are moved to CPU. Not compatible with gradient
        accumulation.
    :param kwargs: other keyword arguments to be passed to the base optimizer e.g. `lr`, `weight_decay`.
    """

    def __init__(
        self,
        params: PARAMETERS,
        optimizer_class: Type[torch.optim.Optimizer] = torch.optim.AdamW,
        *,
        offload_gradients: bool = False,
        **kwargs,
    ) -> None:
        if optimizer_class is torch.optim.AdamW and TORCH_VERSION_AT_LEAST_2_4 and 'fused' not in kwargs:
            kwargs.update(fused=True)

        param_groups = list(params)
        if len(param_groups) == 0:
            raise ValueError('optimizer got an empty parameter list')
        if not isinstance(param_groups[0], dict):
            param_groups = [{'params': param_groups}]

        self.param_cuda2cpu_map = {}
        self.optim_dict = {}
        self.stream = torch.cuda.Stream()

        self.queue = {}

        def backward_hook(p_cuda: torch.Tensor) -> None:
            if p_cuda.grad is None:
                return

            p_cpu = self.param_cuda2cpu_map[p_cuda]

            self.stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self.stream):
                p_cpu.grad.copy_(p_cuda.grad, non_blocking=True)

            if p_cuda in self.queue:
                del self.queue[p_cuda]

            self.queue[p_cuda] = self.stream.record_event()

            if offload_gradients:
                p_cuda.grad.record_stream(self.stream)
                p_cuda.grad = None

        for param_group in param_groups:
            params = param_group.pop('params')

            for p_cuda in params:
                p_cpu = torch.empty_like(p_cuda, device='cpu', pin_memory=True)
                p_cpu.grad = torch.empty_like(p_cpu, pin_memory=True)

                p_cpu.copy_(p_cuda.detach(), non_blocking=True)
                self.param_cuda2cpu_map[p_cuda] = p_cpu

                p_cuda.register_post_accumulate_grad_hook(backward_hook)
                self.optim_dict[p_cuda] = optimizer_class([{'params': p_cpu, **param_group}], **kwargs)

    @torch.no_grad()
    def step(self, closure: CLOSURE = None) -> LOSS:
        loss = None
        if closure is not None:
            loss = closure()

        for p_cuda, grad_d2h_event in self.queue.items():
            grad_d2h_event.synchronize()
            self.optim_dict[p_cuda].step()

            p_cpu = self.param_cuda2cpu_map[p_cuda]
            with torch.cuda.stream(self.stream):
                p_cuda.copy_(p_cpu, non_blocking=True)

        self.queue.clear()

        return loss

    def zero_grad(self, _: bool = True) -> None:
        for p_cuda in self.param_cuda2cpu_map:
            p_cuda.grad = None

    @property
    def param_groups(self):
        return functools.reduce(operator.add, (optim.param_groups for optim in self.optim_dict.values()), [])

    def state_dict(self):
        return [optim.state_dict() for optim in self.optim_dict.values()]

    def load_state_dict(self, state_dict):
        for optim, optim_state_dict in zip(self.optim_dict.values(), state_dict):
            optim.load_state_dict(optim_state_dict)


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
    elif x_len in (2, 3):
        dim = 1
    elif x_len == 4:
        dim = (1, 2, 3)
    else:
        dim = tuple(range(1, x_len))

    return x.norm(p=norm, dim=dim, keepdim=keep_dim)


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
def get_global_gradient_norm(param_groups: List[Dict]) -> torch.Tensor:
    r"""Get global gradient norm."""
    global_grad_norm = torch.zeros(1, dtype=torch.float32, device=param_groups[0]['params'][0].device)

    for group in param_groups:
        for p in group['params']:
            if p.grad is not None:
                global_grad_norm.add_(p.grad.norm().pow(2))

    return global_grad_norm


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
