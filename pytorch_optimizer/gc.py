import torch


def centralize_gradient(x: torch.Tensor, gc_conv_only: bool = False) -> torch.Tensor:
    """Gradient Centralization (GC)
    :param x: torch.Tensor. gradient
    :param gc_conv_only: bool. 'False' for both conv & fc layers
    :return: torch.Tensor. GC-ed gradient
    """
    size: int = x.dim()
    if (gc_conv_only and size > 3) or (not gc_conv_only and size > 1):
        x.add_(-x.mean(dim=tuple(range(1, size)), keepdim=True))
    return x
