import torch


def centralize_gradient(grad: torch.Tensor, gc_conv_only: bool = False) -> None:
    r"""Gradient Centralization (GC).

    :param grad: torch.Tensor. gradient.
    :param gc_conv_only: bool. 'False' for both conv & fc layers.
    """
    size: int = grad.dim()
    if (gc_conv_only and size > 3) or (not gc_conv_only and size > 1):
        grad.add_(-grad.mean(dim=tuple(range(1, size)), keepdim=True))
