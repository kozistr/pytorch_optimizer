import torch


def centralize_gradient(
    x: torch.Tensor, gc_conv_only: bool = False
) -> torch.Tensor:
    """Gradient Centralization (GC)
    :param x: torch.Tensor. gradient.
    :param gc_conv_only: bool. 'False' for both conv & fc layers.
    :return: torch.Tensor.GC-ed gradient
    """
    size: int = x.dim()

    if gc_conv_only:
        if size > 3:
            x.add_(-x.mean(dim=tuple(range(1, size)), keepdim=True))
    else:
        if size > 1:
            x.add_(-x.mean(dim=tuple(range(1, size)), keepdim=True))

    return x
