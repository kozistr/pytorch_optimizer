import functools
import math
import operator
import re
import warnings
from importlib.util import find_spec
from typing import cast

import torch
from torch import nn
from torch.distributed import all_reduce
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.utils import clip_grad_norm_
from torch.optim.optimizer import Optimizer

from pytorch_optimizer.base.type import Closure, Loss, ParamGroup, ParamsT


def parse_pytorch_version(version_string: str) -> list[int]:
    """Parse the major, minor, and patch numbers of a PyTorch version string."""
    match = re.match(r'(\d+\.\d+\.\d+)', version_string)
    if not match:
        raise ValueError(f'invalid version string format: {version_string}')

    return [int(x) for x in match.group(1).split('.')]


def compare_versions(v1: str, v2: str) -> bool:
    """Return whether PyTorch version `v1` is at least `v2`."""
    return parse_pytorch_version(v1) >= parse_pytorch_version(v2)


HAS_TRANSFORMERS: bool = find_spec('transformers') is not None
TORCH_VERSION_AT_LEAST_2_4: bool = compare_versions(torch.__version__, '2.4.0')
TORCH_VERSION_AT_LEAST_2_8: bool = compare_versions(torch.__version__, '2.8.0')

if HAS_TRANSFORMERS:  # pragma: no cover
    try:
        from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
    except ImportError:
        from transformers.deepspeed import is_deepspeed_zero3_enabled
else:

    def is_deepspeed_zero3_enabled() -> bool:
        """Check if DeepSpeed zero3 is enabled."""
        if HAS_TRANSFORMERS:
            return is_deepspeed_zero3_enabled()  # pragma: no cover

        warnings.warn(
            'you need to install `transformers` to use `is_deepspeed_zero3_enabled` function. it will return False.',
            category=ImportWarning,
            stacklevel=2,
        )

        return False


class CPUOffloadOptimizer:  # pragma: no cover
    """Offload optimizer states and updates to the CPU for single GPU training.

    Transfers gradients to pinned CPU memory and copies updated parameters back to the GPU.

    Reference: https://github.com/pytorch/ao/blob/main/torchao/prototype/low_bit_optim/cpu_offload.py

    Args:
        params: Parameters to optimize or dictionaries defining parameter groups.
        optimizer_class: Base optimizer class. Defaults to `torch.optim.AdamW`.
        offload_gradients: Free GPU gradients after transfer. Incompatible with gradient accumulation.
        **kwargs (dict): Options for the base optimizer, such as `lr` and `weight_decay`.

    """

    def __init__(
        self,
        params: ParamsT,
        optimizer_class: type[Optimizer] = torch.optim.AdamW,
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

        param_groups = cast(list[ParamGroup], param_groups)

        self.param_cuda2cpu_map: dict[torch.Tensor, torch.Tensor] = {}
        self.optim_dict: dict[torch.Tensor, Optimizer] = {}
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
            group_params = param_group.get('params', None)
            if group_params is None:
                continue

            for p_cuda in group_params:
                p_cpu = torch.empty_like(p_cuda, device='cpu', pin_memory=True)
                p_cpu.grad = torch.empty_like(p_cpu, pin_memory=True)

                p_cpu.copy_(p_cuda.detach(), non_blocking=True)
                self.param_cuda2cpu_map[p_cuda] = p_cpu

                p_cuda.register_post_accumulate_grad_hook(backward_hook)
                self.optim_dict[p_cuda] = optimizer_class([{**param_group, 'params': [p_cpu]}], **kwargs)

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
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


class StochasticAccumulator:
    """Accumulate bfloat16 gradients with stochastic rounding.

    Attach hooks once, then restore the accumulated gradient buffers before each optimizer step.

    Examples:
        ```python
        hooks = StochasticAccumulator.assign_hooks(model)
        optimizer.zero_grad()
        for inputs, targets in microbatches:
            loss = loss_fn(model(inputs), targets) / len(microbatches)
            loss.backward()
        StochasticAccumulator.reassign_grad_buffer(model)
        optimizer.step()
        optimizer.zero_grad()
        ```

    """

    @staticmethod
    def stochastic_grad_accum(p: torch.Tensor) -> None:
        if hasattr(p, 'acc_grad'):
            acc_grad_fp32 = p.acc_grad.clone().to(torch.float32)
            acc_grad_fp32.add_(p.grad.to(torch.float32))

            copy_stochastic(p.acc_grad, acc_grad_fp32)

            del acc_grad_fp32
        else:
            p.acc_grad = p.grad.clone().to(torch.bfloat16)  # ty: ignore[invalid-assignment]

        del p.grad

    @staticmethod
    def reassign_grad_buffer(model: nn.Module) -> None:
        for _, p in model.named_parameters():
            if p.requires_grad and hasattr(p, 'acc_grad'):
                p.grad = p.acc_grad  # ty: ignore[invalid-assignment]
                del p.acc_grad

    @staticmethod
    def assign_hooks(model: nn.Module) -> list:
        return [
            p.register_post_accumulate_grad_hook(StochasticAccumulator.stochastic_grad_accum)
            for _, p in model.named_parameters()
            if p.requires_grad
        ]


def is_valid_parameters(parameters: ParamsT) -> bool:
    """Check for a nonempty list or tuple whose first entry is a parameter group dictionary."""
    return isinstance(parameters, (list, tuple)) and len(parameters) > 0 and isinstance(parameters[0], dict)


def has_overflow(grad_norm: torch.Tensor) -> bool:
    """Return whether a tensor contains NaN or infinite values."""
    return bool(torch.logical_or(torch.isnan(grad_norm), torch.isinf(grad_norm)).any())


def to_real(x: torch.Tensor) -> torch.Tensor:
    """Return the real part of a complex tensor, or the original real tensor."""
    return x.real if torch.is_complex(x) else x


def normalize_gradient(x: torch.Tensor, use_channels: bool = False, epsilon: float = 1e-8) -> None:
    """Divide gradients by their standard deviation in place.

    Args:
        x: Gradient tensor to normalize.
        use_channels: If True, perform channel wise normalization.
        epsilon: Small constant added for numerical stability.

    """
    size: int = x.dim()
    if size > 1 and use_channels:
        s = x.std(dim=tuple(range(1, size)), keepdim=True).add_(epsilon)
        x.div_(s)
    elif torch.numel(x) > 2:
        s = x.std().add_(epsilon)
        x.div_(s)


def clip_grad_norm(
    parameters: ParamsT | torch.Tensor,
    max_norm: float = 0.0,
    sync: bool = False,
) -> torch.Tensor | float:
    """Compute the global L2 gradient norm and optionally clip gradients in place.

    Args:
        parameters: Tensor or iterable of tensors whose gradients to inspect.
        max_norm: Maximum gradient norm. A nonpositive value disables clipping.
        sync: Sum squared norms across the distributed process group for sharded gradients.

    Returns:
        torch.Tensor | float: Global gradient norm before clipping.

    """
    if parameters is None:
        raise ValueError('ParamsT cannot be None.')

    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]

    # make sure any generators are expanded
    parameters = cast(list, list(parameters))

    # if syncing we need to manually perform the clipping so that we aggregate properly
    if max_norm > 0 and not sync:
        return clip_grad_norm_(parameters, max_norm)

    norm_sq = sum(p.grad.norm() ** 2 for p in parameters if p.grad is not None)
    if sync:  # pragma: no cover
        # also need to get the norms from all the other sharded works in FSDP
        all_reduce(norm_sq)

    grad_norm: float = math.sqrt(norm_sq)
    if max_norm > 0:  # pragma: no cover
        clip_coefficient = max_norm / (grad_norm + 1e-6)
        for p in parameters:
            if p.grad is not None:
                p.grad.detach().mul_(clip_coefficient)

    return grad_norm


def unit_norm(x: torch.Tensor, norm: float = 2.0) -> torch.Tensor:
    """Compute parameter unit norms for adaptive gradient clipping.

    Uses the full norm for scalars and vectors, dimension 1 for 2D and 3D tensors,
    and all dimensions after the first for tensors with four or more dimensions.

    Args:
        x: Parameter or gradient tensor.
        norm: Order of the vector norm.

    Returns:
        torch.Tensor: Unit norms, with reduced dimensions retained for multidimensional inputs.

    """
    keep_dim: bool = True
    dim: int | tuple[int, ...] | None = None

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


def disable_running_stats(model: nn.Module):
    """Pause BatchNorm running statistic updates by setting momentum to zero."""

    def _disable(module):
        if isinstance(module, _BatchNorm):
            module.backup_momentum = module.momentum
            module.momentum = 0

    model.apply(_disable)


def enable_running_stats(model: nn.Module):
    """Restore BatchNorm momentum after pausing running statistic updates."""

    def _enable(module):
        if isinstance(module, _BatchNorm) and hasattr(module, 'backup_momentum'):
            module.momentum = module.backup_momentum

    model.apply(_enable)


@torch.no_grad()
def get_global_gradient_norm(param_groups: list[dict]) -> torch.Tensor:
    """Return the sum of squared L2 gradient norms across parameter groups.

    Args:
        param_groups: Nonempty optimizer parameter groups.

    Returns:
        torch.Tensor: Squared global norm as a single element float32 tensor.

    """
    global_grad_norm = torch.zeros(1, dtype=torch.float32, device=param_groups[0]['params'][0].device)

    for group in param_groups:
        for p in group['params']:
            if p.grad is not None:
                global_grad_norm.add_(p.grad.norm().pow(2))

    return global_grad_norm


def reg_noise(
    network1: nn.Module, network2: nn.Module, num_data: int, lr: float, eta: float = 8e-3, temperature: float = 1e-4
) -> torch.Tensor | float:
    """Compute the Entropy-MCMC coupling and noise term for two networks.

    Usage example and detailed implementation can be found at:
    https://github.com/lblaoke/EMCMC/blob/master/exp/cifar10_emcmc.py

    Args:
        network1: First neural network.
        network2: Second neural network.
        num_data: Number of training data points.
        lr: Learning rate.
        eta: Eta parameter controlling auxiliary guiding variable.
        temperature: Temperature parameter for sampling.

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


@torch.no_grad()
def copy_stochastic(target: torch.Tensor, source: torch.Tensor) -> None:
    """Copy float32 values to bfloat16 with stochastic rounding.

    reference: https://github.com/pytorch/pytorch/issues/120376#issuecomment-1974828905

    Args:
        target: A tensor in bfloat16 format to copy to.
        source: A tensor in float32 format to copy from.

    """
    result = torch.randint_like(
        source,
        dtype=torch.int32,
        low=0,
        high=1 << 16,
    )

    result.add_(source.view(dtype=torch.int32))

    result.bitwise_and_(-65536)

    target.copy_(result.view(dtype=torch.float32))
