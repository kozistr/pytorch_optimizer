from typing import List, Optional, Tuple

import torch
from torch.nn.functional import logsigmoid, one_hot
from torch.nn.modules.loss import _Loss

from pytorch_optimizer.base.types import CLASS_MODE


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


class DiceLoss(_Loss):
    r"""Dice loss for image segmentation task. It supports binary, multiclass and multilabel cases.

    Reference : https://github.com/BloodAxe/pytorch-toolbelt

    :param mode: CLASS_MODE. loss mode 'binary', 'multiclass', or 'multilabel.
    :param classes: Optional[List[int]]. List of classes that contribute in loss computation. By default,
        all channels are included.
    :param log_loss: bool. If True, loss computed as `-log(dice_coeff)`, otherwise `1 - dice_coeff`.
    :param from_logits: bool. If True, assumes input is raw logits.
    :param label_smooth: float. Smoothness constant for dice coefficient (a).
    :param ignore_index: Optional[int]. Label that indicates ignored pixels (does not contribute to loss).
    :param eps: float. epsilon.
    """

    def __init__(
        self,
        mode: CLASS_MODE = 'binary',
        classes: Optional[List[int]] = None,
        log_loss: bool = False,
        from_logits: bool = True,
        label_smooth: float = 0.0,
        ignore_index: Optional[int] = None,
        eps: float = 1e-6,
    ):
        super().__init__()

        if classes is not None:
            if mode == 'binary':
                raise ValueError('[-] Masking classes is not supported with mode=binary')

            classes = torch.LongTensor(classes)

        self.mode = mode
        self.classes = classes
        self.from_logits = from_logits
        self.label_smooth = label_smooth
        self.eps = eps
        self.log_loss = log_loss
        self.ignore_index = ignore_index

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

            if self.ignore_index is not None:
                mask = y_true != self.ignore_index
                y_pred = y_pred * mask
                y_true = y_true * mask

        if self.mode == 'multiclass':
            y_true = y_true.view(bs, -1)
            y_pred = y_pred.view(bs, num_classes, -1)

            if self.ignore_index is not None:
                mask = y_true != self.ignore_index
                y_pred = y_pred * mask.unsqueeze(1)

                y_true = one_hot((y_true * mask).to(torch.long), num_classes)
                y_true = y_true.permute(0, 2, 1) * mask.unsqueeze(1)
            else:
                y_true = one_hot(y_true, num_classes)
                y_true = y_true.permute(0, 2, 1)

        if self.mode == 'multilabel':
            y_true = y_true.view(bs, num_classes, -1)
            y_pred = y_pred.view(bs, num_classes, -1)

            if self.ignore_index is not None:
                mask = y_true != self.ignore_index
                y_pred = y_pred * mask
                y_true = y_true * mask

        scores = self.compute_score(
            y_pred, y_true.type_as(y_pred), label_smooth=self.label_smooth, eps=self.eps, dims=dims
        )

        loss = -torch.log(scores.clamp_min(self.eps)) if self.log_loss else 1.0 - scores

        # Dice loss is undefined for non-empty classes
        # So we zero contribution of channel that does not have true pixels
        # NOTE: A better workaround would be to use loss term `mean(y_pred)`
        # for this case, however it will be a modified jaccard loss

        mask = y_true.sum(dims) > 0
        loss *= mask.to(loss.dtype)

        if self.classes is not None:
            loss = loss[self.classes]

        return self.aggregate_loss(loss)

    @staticmethod
    def aggregate_loss(loss: torch.Tensor) -> torch.Tensor:
        return loss.mean()

    @staticmethod
    def compute_score(
        output: torch.Tensor,
        target: torch.Tensor,
        label_smooth: float = 0.0,
        eps: float = 1e-6,
        dims: Optional[Tuple[int, ...]] = None,
    ) -> torch.Tensor:
        return soft_dice_score(output, target, label_smooth, eps, dims)
