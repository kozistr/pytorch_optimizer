import fnmatch
from typing import Dict, List, Optional, Sequence, Set, Union

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

LOSS_FUNCTION_LIST: List = [
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
LOSS_FUNCTIONS: Dict[str, nn.Module] = {
    str(loss_function.__name__).lower(): loss_function for loss_function in LOSS_FUNCTION_LIST
}


def get_supported_loss_functions(filters: Optional[Union[str, List[str]]] = None) -> List[str]:
    r"""Return list of available loss function names, sorted alphabetically.

    :param filters: Optional[Union[str, List[str]]]. wildcard filter string that works with fmatch. if None, it will
        return the whole list.
    """
    if filters is None:
        return sorted(LOSS_FUNCTIONS.keys())

    include_filters: Sequence[str] = filters if isinstance(filters, (tuple, list)) else [filters]

    filtered_list: Set[str] = set()
    for include_filter in include_filters:
        filtered_list.update(fnmatch.filter(LOSS_FUNCTIONS.keys(), include_filter))

    return sorted(filtered_list)
