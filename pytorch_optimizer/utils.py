from typing import List, Optional, Tuple, Union

import torch
from torch import nn

from pytorch_optimizer.types import PARAMETERS


def is_valid_parameters(parameters: PARAMETERS) -> bool:
    return isinstance(parameters, (list, tuple)) and len(parameters) > 0 and isinstance(parameters[0], dict)


def normalize_gradient(x: torch.Tensor, use_channels: bool = False, epsilon: float = 1e-8) -> torch.Tensor:
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


def unit_norm(x: torch.Tensor, norm: float = 2.0) -> torch.Tensor:
    keep_dim: bool = True
    dim: Optional[Union[int, Tuple[int, ...]]] = None

    x_len: int = len(x.shape)
    if x_len <= 1:
        keep_dim = False
    elif x_len in (2, 3):  # linear layers
        dim = 1
    elif x_len == 4:  # conv kernels
        dim = (1, 2, 3)
    else:
        dim = tuple(range(1, x_len))

    return x.norm(dim=dim, keepdim=keep_dim, p=norm)


def get_optimizer_parameters(
    model: nn.Module, weight_decay: float, wd_ban_list: List[str] = ('bias', 'LayerNorm.bias', 'LayerNorm.weight')
) -> PARAMETERS:
    param_optimizer: List[Tuple[str, nn.Parameter]] = list(model.named_parameters())

    optimizer_parameters = [
        {
            'params': [p for n, p in param_optimizer if not any(nd in n for nd in wd_ban_list)],
            'weight_decay': weight_decay,
        },
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in wd_ban_list)], 'weight_decay': 0.0},
    ]

    return optimizer_parameters
