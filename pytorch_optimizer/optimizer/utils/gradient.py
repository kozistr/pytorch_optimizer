import math
from typing import cast

import torch
from torch.distributed import all_reduce
from torch.nn.utils import clip_grad_norm_

from pytorch_optimizer.base.type import ParamGroup, ParamsT
from pytorch_optimizer.optimizer.utils.foreach import has_foreach_support


def normalize_gradient(x: torch.Tensor, use_channels: bool = False, epsilon: float = 1e-8) -> None:
    """Divide gradients by their standard deviation in place.

    Args:
        x: Gradient tensor to normalize.
        use_channels: If True, perform channel wise normalization.
        epsilon: Small constant added for numerical stability.

    """
    size: int = x.dim()
    if size > 1 and use_channels:
        if math.prod(x.shape[1:]) <= 1:
            return

        s = x.std(dim=tuple(range(1, size)), keepdim=True).add_(epsilon)
        x.div_(s)
    elif torch.numel(x) > 2:
        s = x.std().add_(epsilon)
        x.div_(s)


def clip_grad_norm(
    parameters: ParamsT | torch.Tensor,
    max_norm: float = 0.0,
    sync: bool = False,
) -> torch.Tensor:
    """Compute the global L2 gradient norm and optionally clip gradients in place.

    Args:
        parameters: Tensor or iterable of tensors whose gradients to inspect.
        max_norm: Maximum gradient norm. A nonpositive value disables clipping.
        sync: Sum squared norms across the distributed process group for sharded gradients.

    Returns:
        torch.Tensor: Global gradient norm before clipping.

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

    norm_sq = get_global_gradient_norm([{'params': parameters}])
    if sync:  # pragma: no cover
        # also need to get the norms from all the other sharded works in FSDP
        all_reduce(norm_sq)

    grad_norm = norm_sq.sqrt_().squeeze_(0)
    if max_norm > 0:  # pragma: no cover
        clip_coefficient = (max_norm / (grad_norm + 1e-6)).clamp_(max=1.0)
        for p in parameters:
            if p.grad is not None:
                p.grad.detach().mul_(clip_coefficient.to(p.grad.device))

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
def get_global_gradient_norm(
    param_groups: list[ParamGroup] | None,
    device: torch.device | None = None,
    *,
    weight_adaptive: bool = False,
) -> torch.Tensor:
    """Return the sum of squared L2 gradient norms across parameter groups.

    Args:
        param_groups: Optimizer parameter groups, or None for no parameters.
        device: Output device. Defaults to the first parameter's device, or CPU for empty groups.
        weight_adaptive: Weight gradients by absolute parameter values for groups with `adaptive=True`.

    Returns:
        torch.Tensor: Squared global norm as a single element float32 tensor.

    """
    param_groups = param_groups or []
    device = device or next((p.device for group in param_groups for p in group['params']), torch.device('cpu'))
    grads: list[torch.Tensor] = []
    for group in param_groups:
        adaptive = weight_adaptive and group.get('adaptive', False)
        for p in group['params']:
            if p.grad is None:
                continue

            grad = p.grad
            if adaptive:
                if grad.dtype in (torch.float16, torch.bfloat16):
                    grad = grad.float()
                grad = grad * p.abs()

            grads.append(grad)

    if not grads:
        return torch.zeros(1, dtype=torch.float32, device=device)

    norms = (
        torch._foreach_norm(grads)
        if has_foreach_support(grads) and grads[0].dtype not in (torch.float16, torch.bfloat16)
        else [(grad.float() if grad.dtype in (torch.float16, torch.bfloat16) else grad).norm() for grad in grads]
    )
    squared_norms = torch.stack([norm.to(device) for norm in norms]).float().square_()
    return squared_norms.sum().reshape(1)
