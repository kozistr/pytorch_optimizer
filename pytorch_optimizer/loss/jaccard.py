from typing import List, Optional, Tuple

import torch
from torch.nn.functional import logsigmoid, one_hot
from torch.nn.modules.loss import _Loss

from pytorch_optimizer.base.types import CLASS_MODE


def soft_jaccard_score(
    output: torch.Tensor,
    target: torch.Tensor,
    label_smooth: float = 0.0,
    eps: float = 1e-6,
    dims: Optional[Tuple[int, ...]] = None,
) -> torch.Tensor:
    r"""Get soft jaccard score.

    :param output: torch.Tensor. predicted segments.
    :param target: torch.Tensor. ground truth segments.
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

    return (intersection + label_smooth) / (cardinality - intersection + label_smooth).clamp_min(eps)


class JaccardLoss(_Loss):
    r"""Jaccard loss for image segmentation task. It supports binary, multiclass and multilabel cases.

    Reference : https://github.com/BloodAxe/pytorch-toolbelt

    :param mode: CLASS_MODE. loss mode 'binary', 'multiclass', or 'multilabel.
    :param classes: Optional[List[int]]. List of classes that contribute in loss computation. By default,
        all channels are included.
    :param log_loss: If True, loss computed as `-log(jaccard)`; otherwise `1 - jaccard`
    :param from_logits: bool. If True, assumes input is raw logits.
    :param label_smooth: float. Smoothness constant for dice coefficient (a).
    :param eps: float. epsilon.
    """

    def __init__(
        self,
        mode: CLASS_MODE,
        classes: List[int] = None,
        log_loss: bool = False,
        from_logits: bool = True,
        label_smooth: float = 0.0,
        eps: float = 1e-6,
    ):
        super().__init__()

        if classes is not None:
            if mode == 'binary':
                raise ValueError('[-] Masking classes is not supported with mode=binary')

            classes = torch.LongTensor(classes)

        self.mode = mode
        self.classes = classes
        self.log_loss = log_loss
        self.from_logits = from_logits
        self.label_smooth = label_smooth
        self.eps = eps

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        if self.from_logits:
            # Apply activations to get [0..1] class probabilities
            # Using Log-Exp as this gives more numerically stable result and does not cause vanishing gradient on
            # extreme values 0 and 1
            y_pred = y_pred.log_softmax(dim=1).exp() if self.mode == 'multiclass' else logsigmoid(y_pred).exp()

        bs: int = y_true.size(0)
        num_classes: int = y_pred.size(1)

        dims: Tuple[int, ...] = (0, 2)

        if self.mode == 'binary':
            y_true = y_true.view(bs, 1, -1)
            y_pred = y_pred.view(bs, 1, -1)

        if self.mode == 'multiclass':
            y_true = y_true.view(bs, -1)
            y_pred = y_pred.view(bs, num_classes, -1)

            y_true = one_hot(y_true, num_classes)
            y_true = y_true.permute(0, 2, 1)

        if self.mode == 'multilabel':
            y_true = y_true.view(bs, num_classes, -1)
            y_pred = y_pred.view(bs, num_classes, -1)

        scores = soft_jaccard_score(
            y_pred, y_true.type(y_pred.dtype), label_smooth=self.label_smooth, eps=self.eps, dims=dims
        )

        loss = -torch.log(scores.clamp_min(self.eps)) if self.log_loss else 1.0 - scores

        # IoU loss is defined for non-empty classes
        # So we zero contribution of channel that does not have true pixels
        # NOTE: A better workaround would be to use loss term `mean(y_pred)`
        # for this case, however it will be a modified jaccard loss

        mask = y_true.sum(dims) > 0
        loss *= mask.float()

        if self.classes is not None:
            loss = loss[self.classes]

        return loss.mean()
