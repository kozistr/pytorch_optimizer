import fnmatch
from collections.abc import Sequence

from torch import nn

from pytorch_optimizer.loss.bi_tempered import (
    BinaryBiTemperedLogisticLoss,
    BiTemperedLogisticLoss,
    bi_tempered_logistic_loss,
)
from pytorch_optimizer.loss.cross_entropy import BCELoss
from pytorch_optimizer.loss.dice import DiceLoss, soft_dice_score
from pytorch_optimizer.loss.f1 import SoftF1Loss
from pytorch_optimizer.loss.focal import BCEFocalLoss, FocalCosineLoss, FocalLoss, FocalTverskyLoss
from pytorch_optimizer.loss.jaccard import JaccardLoss, soft_jaccard_score
from pytorch_optimizer.loss.ldam import LDAMLoss
from pytorch_optimizer.loss.lovasz import LovaszHingeLoss
from pytorch_optimizer.loss.tversky import TverskyLoss

__all__ = [
    'BCEFocalLoss',
    'BCELoss',
    'BiTemperedLogisticLoss',
    'BinaryBiTemperedLogisticLoss',
    'DiceLoss',
    'FocalCosineLoss',
    'FocalLoss',
    'FocalTverskyLoss',
    'JaccardLoss',
    'LDAMLoss',
    'LovaszHingeLoss',
    'SoftF1Loss',
    'TverskyLoss',
    'bi_tempered_logistic_loss',
    'get_supported_loss_functions',
    'soft_dice_score',
    'soft_jaccard_score',
]

LOSS_FUNCTION_LIST: list = [
    BCELoss,
    BCEFocalLoss,
    FocalLoss,
    SoftF1Loss,
    DiceLoss,
    LDAMLoss,
    FocalCosineLoss,
    JaccardLoss,
    BiTemperedLogisticLoss,
    BinaryBiTemperedLogisticLoss,
    TverskyLoss,
    FocalTverskyLoss,
    LovaszHingeLoss,
]
LOSS_FUNCTIONS: dict[str, nn.Module] = {
    str(loss_function.__name__).lower(): loss_function for loss_function in LOSS_FUNCTION_LIST
}


def get_supported_loss_functions(filters: str | list[str] | None = None) -> list[str]:
    """List registered loss function names in alphabetical order.

    Args:
        filters: Wildcard pattern or list of patterns, such as `'*focal*'`. `None` selects all names.

    Returns:
        list[str]: Matching names in lowercase, without duplicates.

    """
    if filters is None:
        return sorted(LOSS_FUNCTIONS.keys())

    include_filters: Sequence[str] = filters if isinstance(filters, (tuple, list)) else [filters]

    filtered_list: set[str] = set()
    for include_filter in include_filters:
        filtered_list.update(fnmatch.filter(LOSS_FUNCTIONS.keys(), include_filter))

    return sorted(filtered_list)
