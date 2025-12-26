from typing import Dict, List, Optional, Tuple, Union

import torch


def has_foreach_support(tensors: List[torch.Tensor]) -> bool:
    """Check if foreach operations are supported for the given tensors.

    Foreach operations require:
    - All tensors on the same device
    - All tensors have the same dtype
    - No sparse tensors
    - CUDA tensors (CPU falls back to for-loop)

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

    return first_device.type == 'cuda'


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


def foreach_add_(
    tensors: List[torch.Tensor],
    other: Union[torch.Tensor, List[torch.Tensor], float],
    *,
    alpha: float = 1.0,
    foreach: bool = True,
) -> None:
    """Apply in-place addition to a list of tensors.

    Args:
        tensors: List of tensors to add to.
        other: Scalar, tensor, or list of tensors to add.
        alpha: Scaling factor for other.
        foreach: Whether to use foreach operations.
    """
    if foreach and has_foreach_support(tensors):
        if isinstance(other, list):
            torch._foreach_add_(tensors, other, alpha=alpha)
        else:
            torch._foreach_add_(tensors, alpha * other if isinstance(other, (int, float)) else other)
    else:
        for i, t in enumerate(tensors):
            if isinstance(other, list):
                t.add_(other[i], alpha=alpha)
            else:
                t.add_(other, alpha=alpha)


def foreach_mul_(
    tensors: List[torch.Tensor],
    scalar: Union[float, List[torch.Tensor]],
    foreach: bool = True,
) -> None:
    """Apply in-place multiplication to a list of tensors.

    Args:
        tensors: List of tensors to multiply.
        scalar: Scalar value or list of tensors to multiply by.
        foreach: Whether to use foreach operations.
    """
    if foreach and has_foreach_support(tensors):
        torch._foreach_mul_(tensors, scalar)
    else:
        for i, t in enumerate(tensors):
            if isinstance(scalar, list):
                t.mul_(scalar[i])
            else:
                t.mul_(scalar)


def foreach_lerp_(
    tensors: List[torch.Tensor],
    other: List[torch.Tensor],
    weight: float,
    foreach: bool = True,
) -> None:
    """Apply in-place linear interpolation to a list of tensors.

    Computes: tensors = tensors + weight * (other - tensors)
            = (1 - weight) * tensors + weight * other

    Args:
        tensors: List of tensors to interpolate.
        other: List of tensors to interpolate towards.
        weight: Interpolation weight.
        foreach: Whether to use foreach operations.
    """
    if foreach and has_foreach_support(tensors):
        torch._foreach_lerp_(tensors, other, weight)
    else:
        for t, o in zip(tensors, other):
            t.lerp_(o, weight)


def foreach_addcmul_(
    tensors: List[torch.Tensor],
    tensor1: List[torch.Tensor],
    tensor2: List[torch.Tensor],
    value: float = 1.0,
    foreach: bool = True,
) -> None:
    """Apply in-place addcmul to a list of tensors.

    Computes: tensors += value * tensor1 * tensor2

    Args:
        tensors: List of tensors to add to.
        tensor1: First list of multiplicand tensors.
        tensor2: Second list of multiplicand tensors.
        value: Scalar multiplier.
        foreach: Whether to use foreach operations.
    """
    if foreach and has_foreach_support(tensors):
        torch._foreach_addcmul_(tensors, tensor1, tensor2, value=value)
    else:
        for t, t1, t2 in zip(tensors, tensor1, tensor2):
            t.addcmul_(t1, t2, value=value)


def foreach_addcdiv_(
    tensors: List[torch.Tensor],
    tensor1: List[torch.Tensor],
    tensor2: List[torch.Tensor],
    value: float = 1.0,
    foreach: bool = True,
) -> None:
    """Apply in-place addcdiv to a list of tensors.

    Computes: tensors += value * tensor1 / tensor2

    Args:
        tensors: List of tensors to add to.
        tensor1: List of dividend tensors.
        tensor2: List of divisor tensors.
        value: Scalar multiplier.
        foreach: Whether to use foreach operations.
    """
    if foreach and has_foreach_support(tensors):
        torch._foreach_addcdiv_(tensors, tensor1, tensor2, value=value)
    else:
        for t, t1, t2 in zip(tensors, tensor1, tensor2):
            t.addcdiv_(t1, t2, value=value)


def foreach_sqrt(
    tensors: List[torch.Tensor],
    foreach: bool = True,
) -> List[torch.Tensor]:
    """Compute element-wise square root of a list of tensors.

    Args:
        tensors: List of tensors.
        foreach: Whether to use foreach operations.

    Returns:
        List of tensors containing square roots.
    """
    if foreach and has_foreach_support(tensors):
        return list(torch._foreach_sqrt(tensors))
    return [t.sqrt() for t in tensors]


def foreach_sqrt_(
    tensors: List[torch.Tensor],
    foreach: bool = True,
) -> None:
    """Apply in-place element-wise square root to a list of tensors.

    Args:
        tensors: List of tensors.
        foreach: Whether to use foreach operations.
    """
    if foreach and has_foreach_support(tensors):
        torch._foreach_sqrt_(tensors)
    else:
        for t in tensors:
            t.sqrt_()


def foreach_maximum_(
    tensors: List[torch.Tensor],
    other: List[torch.Tensor],
    foreach: bool = True,
) -> None:
    """Apply in-place element-wise maximum to a list of tensors.

    Args:
        tensors: List of tensors to update in-place.
        other: List of tensors to compare against.
        foreach: Whether to use foreach operations.
    """
    if foreach and has_foreach_support(tensors):
        torch._foreach_maximum_(tensors, other)
    else:
        for t, o in zip(tensors, other):
            torch.maximum(t, o, out=t)


def foreach_sign_(
    tensors: List[torch.Tensor],
    foreach: bool = True,
) -> None:
    """Apply in-place sign operation to a list of tensors.

    Args:
        tensors: List of tensors.
        foreach: Whether to use foreach operations.
    """
    if foreach and has_foreach_support(tensors):
        torch._foreach_sign_(tensors)
    else:
        for t in tensors:
            t.sign_()


def foreach_neg_(
    tensors: List[torch.Tensor],
    foreach: bool = True,
) -> None:
    """Apply in-place negation to a list of tensors.

    Args:
        tensors: List of tensors.
        foreach: Whether to use foreach operations.
    """
    if foreach and has_foreach_support(tensors):
        torch._foreach_neg_(tensors)
    else:
        for t in tensors:
            t.neg_()


def foreach_sub_(
    tensors: List[torch.Tensor],
    other: Union[torch.Tensor, List[torch.Tensor], float],
    *,
    alpha: float = 1.0,
    foreach: bool = True,
) -> None:
    """Apply in-place subtraction to a list of tensors.

    Args:
        tensors: List of tensors to subtract from.
        other: Scalar, tensor, or list of tensors to subtract.
        alpha: Scaling factor for other.
        foreach: Whether to use foreach operations.
    """
    if foreach and has_foreach_support(tensors):
        if isinstance(other, list):
            torch._foreach_sub_(tensors, other, alpha=alpha)
        else:
            torch._foreach_sub_(tensors, alpha * other if isinstance(other, (int, float)) else other)
    else:
        for i, t in enumerate(tensors):
            if isinstance(other, list):
                t.sub_(other[i], alpha=alpha)
            else:
                t.sub_(other, alpha=alpha)


def foreach_div_(
    tensors: List[torch.Tensor],
    other: Union[float, List[torch.Tensor]],
    foreach: bool = True,
) -> None:
    """Apply in-place division to a list of tensors.

    Args:
        tensors: List of tensors to divide.
        other: Scalar value or list of tensors to divide by.
        foreach: Whether to use foreach operations.
    """
    if foreach and has_foreach_support(tensors):
        torch._foreach_div_(tensors, other)
    else:
        for i, t in enumerate(tensors):
            if isinstance(other, list):
                t.div_(other[i])
            else:
                t.div_(other)


def foreach_zero_(
    tensors: List[torch.Tensor],
    foreach: bool = True,
) -> None:
    """Zero out a list of tensors in-place.

    Args:
        tensors: List of tensors to zero.
        foreach: Whether to use foreach operations.
    """
    if foreach and has_foreach_support(tensors):
        torch._foreach_zero_(tensors)
    else:
        for t in tensors:
            t.zero_()


def foreach_copy_(
    tensors: List[torch.Tensor],
    src: List[torch.Tensor],
    foreach: bool = True,
) -> None:
    """Copy values from source tensors to destination tensors in-place.

    Args:
        tensors: List of destination tensors.
        src: List of source tensors.
        foreach: Whether to use foreach operations.
    """
    if foreach and has_foreach_support(tensors):
        torch._foreach_copy_(tensors, src)
    else:
        for t, s in zip(tensors, src):
            t.copy_(s)


def foreach_clamp_min_(
    tensors: List[torch.Tensor],
    min_val: float,
    foreach: bool = True,
) -> None:
    """Clamp tensors to minimum value in-place.

    Args:
        tensors: List of tensors to clamp.
        min_val: Minimum value.
        foreach: Whether to use foreach operations.
    """
    if foreach and has_foreach_support(tensors):
        torch._foreach_clamp_min_(tensors, min_val)
    else:
        for t in tensors:
            t.clamp_min_(min_val)
