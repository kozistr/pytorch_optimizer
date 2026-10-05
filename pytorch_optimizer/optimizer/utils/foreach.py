from collections.abc import Callable, Sequence
from functools import wraps

import torch
from torch.utils._foreach_utils import _group_tensors_by_device_and_dtype

from pytorch_optimizer.base.compatibility import TORCH_VERSION_AT_LEAST_2_8


def has_foreach_support(tensors: list[torch.Tensor]) -> bool:
    """Check that a nonempty tensor list supports batched operations.

    Args:
        tensors: Tensors to inspect.

    Returns:
        bool: `True` if all tensors are dense and share a device and data type.

    """
    if len(tensors) == 0:
        return False

    first_device = tensors[0].device
    first_dtype = tensors[0].dtype

    for t in tensors:
        if t.device != first_device:
            return False
        if t.dtype != first_dtype:
            return False
        if t.is_sparse:
            return False

    return True


def group_tensors_by_device_and_dtype(
    params: list[torch.Tensor],
    grads: list[torch.Tensor],
    state_lists: dict[str, list[torch.Tensor]] | None = None,
) -> list[dict]:
    """Group aligned tensor lists by the parameters' device and data type.

    Args:
        params: Parameter tensors.
        grads: Corresponding gradients, in the same order as `params`.
        state_lists: Optional state names mapped to aligned lists of state tensors.

    Returns:
        list[dict]: Groups containing `params`, `grads`, original `indices`, and the requested state lists.
            Aligned input lists are reused when all parameters share a device and dtype.

    """
    if not params:
        return []

    state_lists = state_lists or {}
    groups = _group_tensors_by_device_and_dtype([params], with_indices=True)

    if len(groups) == 1:
        return [{'params': params, 'grads': grads, 'indices': next(iter(groups.values()))[1], **state_lists}]

    return [
        {
            'params': tensor_lists[0],
            'grads': [grads[index] for index in indices],
            'indices': indices,
            **{name: [values[index] for index in indices] for name, values in state_lists.items()},
        }
        for tensor_lists, indices in sorted(groups.values(), key=lambda group: group[1][0])
    ]


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
    def update(group: dict, params: list[torch.Tensor], *args) -> None:
        device = params[0].device
        group = {**group, 'lr': scalar_tensor(group['lr'], device, -1)}
        args = tuple(
            scalar_tensor(value, device, index) if isinstance(value, (float, torch.Tensor)) else value
            for index, value in enumerate(args)
        )
        compiled_step(group, params, *args)

    return update


def foreach_addcdiv_(
    tensors: list[torch.Tensor],
    numerators: Sequence[torch.Tensor],
    denominators: Sequence[torch.Tensor],
    value: float | torch.Tensor,
) -> None:
    """Apply an adaptive update with a scalar or a compilable tensor multiplier.

    Args:
        tensors: Parameters to update in place.
        numerators: Update numerators.
        denominators: Update denominators.
        value: Multiplier, applied with promoted arithmetic for tensor values.

    """
    if not isinstance(value, torch.Tensor):
        torch._foreach_addcdiv_(tensors, numerators, denominators, value=value)
        return

    dtype = torch.promote_types(tensors[0].dtype, value.dtype)
    updates = torch._foreach_div(
        [numerator.to(dtype=dtype) for numerator in numerators],
        [denominator.to(dtype=dtype) for denominator in denominators],
    )
    torch._foreach_mul_(updates, value)
    torch._foreach_add_(tensors, updates)


def foreach_add_(tensors: list[torch.Tensor], updates: list[torch.Tensor], alpha: float | torch.Tensor) -> None:
    """Add updates in place with a scalar or a compilable tensor multiplier.

    Args:
        tensors: Parameters to update in place.
        updates: Updates to add.
        alpha: Multiplier, applied with promoted arithmetic for tensor values.

    """
    if not isinstance(alpha, torch.Tensor):
        torch._foreach_add_(tensors, updates, alpha=alpha)
        return

    dtype = torch.promote_types(tensors[0].dtype, alpha.dtype)
    scaled_updates = torch._foreach_mul([update.to(dtype=dtype) for update in updates], alpha)
    torch._foreach_add_(tensors, scaled_updates)


def foreach_scalar_div_(tensors: Sequence[torch.Tensor], scalar: float | torch.Tensor) -> None:
    """Divide a scalar by tensors in place without allocating full-size numerators.

    Direct division avoids overflowing an intermediate FP16 reciprocal.

    Args:
        tensors: Nonempty tensors sharing a device and dtype, overwritten with their quotients.
        scalar: Numerator, rounded to the tensors' dtype before division.

    """
    numerator = (
        scalar.to(dtype=tensors[0].dtype, device=tensors[0].device)
        if isinstance(scalar, torch.Tensor)
        else torch.full((), fill_value=scalar, dtype=tensors[0].dtype, device=tensors[0].device)
    )

    for tensor in tensors:
        if isinstance(scalar, torch.Tensor):
            tensor.copy_(numerator / tensor)
        else:
            torch.div(numerator, tensor, out=tensor)


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
