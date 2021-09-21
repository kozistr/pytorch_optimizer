import torch


def normalize_gradient(
    x: torch.Tensor, use_channels: bool = False, epsilon: float = 1e-8
) -> torch.Tensor:
    """ normalize gradient with stddev
    :param x: torch.Tensor. gradient.
    :param use_channels: bool. channel-wise normalization.
    :param epsilon: float. eps.
    :return: torch.Tensor. normalized gradient.
    """
    size: int = x.dim()

    if size > 1 and use_channels:
        s = x.std(dim=tuple(range(1, size)), keepdim=True) + epsilon
        x.div_(s)
    elif torch.numel(x) > 2:
        s = x.std() + epsilon
        x.div_(s)
    return x
