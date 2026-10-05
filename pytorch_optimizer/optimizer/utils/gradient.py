import math
from typing import cast

import torch
from torch.distributed import all_reduce
from torch.nn.utils import clip_grad_norm_

from pytorch_optimizer.base.type import ParamsT


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


@torch.no_grad()
def get_global_gradient_norm(param_groups: list[dict], device: torch.device | None = None) -> torch.Tensor:
    """Return the sum of squared L2 gradient norms across parameter groups.

    Args:
        param_groups: Nonempty optimizer parameter groups.
        device: Device. If None, it will use the device of the first param of the paramter group.

    Returns:
        torch.Tensor: Squared global norm as a single element float32 tensor.

    """
    if device is None:
        device = param_groups[0]['params'][0].device

    global_grad_norm = torch.zeros(1, dtype=torch.float32, device=device)

    for group in param_groups:
        for p in group['params']:
            if p.grad is not None:
                global_grad_norm.add_(p.grad.norm().pow(2))

    return global_grad_norm
