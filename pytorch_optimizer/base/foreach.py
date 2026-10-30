from collections.abc import Callable, Sequence
from functools import wraps

import torch

from pytorch_optimizer.base.compatibility import TORCH_VERSION_AT_LEAST_2_8


def compile_foreach_step(step: Callable, compile_kwargs: dict | None = None) -> Callable:
    """Compile a foreach update while keeping changing scalars out of graph guards.

    Args:
        step: Bound foreach update accepting a parameter group, parameters, and update arguments.
        compile_kwargs: Options passed to `torch.compile`.

    Returns:
        Callable: Update with cached FP32 scalar tensors on each parameter device.

    """
    compiled_step = torch.compile(step, **{'dynamic': True, **(compile_kwargs or {})})
    scalars: dict[tuple[torch.device, int], tuple[torch.Tensor, float]] = {}

    def scalar_tensor(value: float | torch.Tensor, device: torch.device, index: int) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            return value.to(device=device, dtype=torch.float32)

        key = (device, index)
        if key not in scalars:
            scalars[key] = (torch.empty((), device=device, dtype=torch.float32), float('nan'))

        tensor, previous = scalars[key]
        if value != previous:
            tensor.fill_(value)
            scalars[key] = (tensor, value)

        return tensor

    @wraps(step)
    def update(group: dict, params: list[torch.Tensor], *args):
        device = params[0].device
        group = {**group, 'lr': scalar_tensor(group['lr'], device, -1)}
        args = tuple(
            scalar_tensor(value, device, index) if isinstance(value, (float, torch.Tensor)) else value
            for index, value in enumerate(args)
        )

        return compiled_step(group, params, *args)

    return update


def foreach_rsqrt(
    tensors: list[torch.Tensor] | tuple[torch.Tensor, ...],
) -> Sequence[torch.Tensor]:  # pragma: no cover
    """Compute reciprocal square roots with a fallback for PyTorch versions before 2.8.

    `torch._foreach_rsqrt` was introduced in PyTorch 2.8.0, so earlier versions
    use a reciprocal-of-sqrt fallback.
    """
    if TORCH_VERSION_AT_LEAST_2_8:
        return torch._foreach_rsqrt(tensors)

    return torch._foreach_reciprocal(torch._foreach_sqrt(tensors))


def foreach_rsqrt_(tensors: list[torch.Tensor] | tuple[torch.Tensor, ...]) -> None:  # pragma: no cover
    """Compute reciprocal square roots in place with a fallback for PyTorch versions before 2.8.

    `torch._foreach_rsqrt_` was introduced in PyTorch 2.8.0, so earlier versions
    use in place sqrt followed by reciprocal.

    """
    if TORCH_VERSION_AT_LEAST_2_8:
        torch._foreach_rsqrt_(tensors)
    else:
        torch._foreach_sqrt_(tensors)
        torch._foreach_reciprocal_(tensors)
