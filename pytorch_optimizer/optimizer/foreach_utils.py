from collections.abc import Sequence

import torch

from pytorch_optimizer.optimizer.utils import TORCH_VERSION_AT_LEAST_2_8


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

    """
    if state_lists is None:
        state_lists = {}

    groups: dict[tuple[torch.device, torch.dtype], dict] = {}

    for idx, (p, g) in enumerate(zip(params, grads)):
        key = (p.device, p.dtype)

        if key not in groups:
            groups[key] = {
                'params': [],
                'grads': [],
                'indices': [],
                **{name: [] for name in state_lists},
            }

        groups[key]['params'].append(p)
        groups[key]['grads'].append(g)
        groups[key]['indices'].append(idx)

        for name, state_list in state_lists.items():
            groups[key][name].append(state_list[idx])

    return list(groups.values())


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
