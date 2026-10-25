from collections import defaultdict

import torch


def centralize_gradient(grad: torch.Tensor | list[torch.Tensor], gc_conv_only: bool = False) -> None:
    """Subtract the mean of each gradient channel in place.

    Args:
        grad: Gradient tensor or list of dense gradient tensors to centralize in batches.
        gc_conv_only: If False, apply GC to both convolutional and fully connected layers. If True, apply only to
            convolutional layers.

    """
    if isinstance(grad, torch.Tensor):
        size: int = grad.dim()
        if (gc_conv_only and size > 3) or (not gc_conv_only and size > 1):
            grad.add_(-grad.mean(dim=tuple(range(1, size)), keepdim=True))

        return

    groups: defaultdict[tuple[torch.device, torch.dtype, torch.Size], list[torch.Tensor]] = defaultdict(list)

    for gradient in grad:
        if gradient.ndim <= (3 if gc_conv_only else 1):
            continue

        key = (gradient.device, gradient.dtype, gradient.shape)
        groups[key].append(gradient)

    for tensors in groups.values():
        if len(tensors) == 1:
            centralize_gradient(tensors[0], gc_conv_only=gc_conv_only)
            continue

        batch = torch.stack(tensors)
        batch.sub_(batch.mean(dim=tuple(range(2, batch.ndim)), keepdim=True))
        torch._foreach_copy_(tensors, batch.unbind())
