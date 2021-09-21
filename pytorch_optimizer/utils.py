from typing import Optional, Tuple, Union

import torch


def normalize_gradient(
    x: torch.Tensor, use_channels: bool = False, epsilon: float = 1e-8
) -> torch.Tensor:
    """normalize gradient with stddev
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


def unit_norm(x: torch.Tensor) -> torch.Tensor:
    keep_dim: bool = True
    dim: Optional[Union[int, Tuple[int, ...]]] = None

    x_len = len(x.shape)

    if x_len <= 1:
        keep_dim = False
    elif x_len in (2, 3):  # linear layers
        dim = 1
    elif x_len == 4:  # conv kernels
        dim = (1, 2, 3)
    else:
        dim = tuple([x for x in range(1, x_len)])

    return x.norm(dim=dim, keepdim=keep_dim, p=2.0)
