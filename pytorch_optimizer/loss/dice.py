from typing import Optional, Tuple

import torch


def soft_dice_score(
    output: torch.Tensor,
    target: torch.Tensor,
    label_smooth: float = 0.0,
    eps: float = 1e-6,
    dims: Optional[Tuple[int, ...]] = None,
) -> torch.Tensor:
    r"""Get soft dice score.

    :param output: torch.Tensor. predicted segments.
    :param target. torch.Tensor. ground truth segments.
    :param label_smooth: float. label smoothing factor.
    :param eps: float. epsilon.
    :param dims: Optional[Tuple[int, ...]]. target dimensions to reduce.
    """
    if dims is not None:
        intersection = torch.sum(output * target, dim=dims)
        cardinality = torch.sum(output + target, dim=dims)
    else:
        intersection = torch.sum(output * target)
        cardinality = torch.sum(output + target)

    return (2.0 * intersection + label_smooth) / (cardinality + label_smooth).clamp_min(eps)
