import torch


def centralize_gradient(grad: torch.Tensor, gc_conv_only: bool = False) -> None:
    """Gradient Centralization (GC).

    Args:
        grad (torch.Tensor): Gradient tensor.
        gc_conv_only (bool): If False, apply GC to both convolutional and fully connected layers; if True, apply only
            to convolutional layers.
    """
    size: int = grad.dim()
    if (gc_conv_only and size > 3) or (not gc_conv_only and size > 1):
        grad.add_(-grad.mean(dim=tuple(range(1, size)), keepdim=True))
