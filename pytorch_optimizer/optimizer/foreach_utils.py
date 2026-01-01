from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch

from pytorch_optimizer.optimizer.utils import TORCH_VERSION_AT_LEAST_2_8


def has_foreach_support(tensors: List[torch.Tensor]) -> bool:
    """Check if foreach operations are supported for the given tensors.

    Foreach operations require:
    - All tensors on the same device
    - All tensors have the same dtype
    - No sparse tensors

    Args:
        tensors: List of tensors to check.

    Returns:
        True if foreach operations are supported, False otherwise.
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
    params: List[torch.Tensor],
    grads: List[torch.Tensor],
    state_lists: Optional[Dict[str, List[torch.Tensor]]] = None,
) -> List[Dict]:
    """Group tensors by device and dtype for efficient foreach operations.

    This function organizes parameters, gradients, and state tensors into groups
    where all tensors share the same device and dtype, enabling foreach operations.

    Args:
        params: List of parameter tensors.
        grads: List of gradient tensors (corresponding to params).
        state_lists: Optional dictionary mapping state names to lists of state tensors.

    Returns:
        List of dictionaries, each containing:
        - 'params': List of parameters in this group
        - 'grads': List of gradients in this group
        - 'indices': Original indices of parameters in this group
        - state_name: List of state tensors for each state in state_lists
    """
    if state_lists is None:
        state_lists = {}

    groups: Dict[Tuple[torch.device, torch.dtype], Dict] = {}

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
    tensors: Union[List[torch.Tensor], Tuple[torch.Tensor, ...]],
) -> Sequence[torch.Tensor]:  # pragma: no cover
    """foreach_rsqrt implementation by Pytorch version.

    `torch._foreach_rsqrt` is introduced since Pytorch 2.8.0, So, previous versions do not use this.
    """
    if TORCH_VERSION_AT_LEAST_2_8:
        return torch._foreach_rsqrt(tensors)

    return torch._foreach_reciprocal(torch._foreach_sqrt(tensors))


def foreach_rsqrt_(tensors: Union[List[torch.Tensor], Tuple[torch.Tensor, ...]]) -> None:  # pragma: no cover
    """foreach_rsqrt_ implementation by Pytorch version.

    `torch._foreach_rsqrt_` is introduced since Pytorch 2.8.0, So, previous versions do not use this.
    """
    if TORCH_VERSION_AT_LEAST_2_8:
        torch._foreach_rsqrt_(tensors)
    else:
        torch._foreach_sqrt_(tensors)
        torch._foreach_reciprocal_(tensors)
