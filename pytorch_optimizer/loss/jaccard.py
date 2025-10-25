from typing import List, Optional, Tuple

import torch
from torch.nn.functional import logsigmoid, one_hot
from torch.nn.modules.loss import _Loss

from pytorch_optimizer.base.type import CLASS_MODE


def soft_jaccard_score(
    output: torch.Tensor,
    target: torch.Tensor,
    label_smooth: float = 0.0,
    eps: float = 1e-6,
    dims: Optional[Tuple[int, ...]] = None,
) -> torch.Tensor:
    r"""Get soft Jaccard score.

    Args:
        output (torch.Tensor): Predicted segments (probabilities or logits).
        target (torch.Tensor): Ground truth segments.
        label_smooth (float): Label smoothing factor to avoid zero denominators.
        eps (float): Small epsilon for numerical stability.
        dims (Optional[Tuple[int, ...]]): Dimensions to reduce over when computing the score.
    """
    if dims is not None:
        intersection = torch.sum(output * target, dim=dims)
        cardinality = torch.sum(output + target, dim=dims)
    else:
        intersection = torch.sum(output * target)
        cardinality = torch.sum(output + target)

    return (intersection + label_smooth) / (cardinality - intersection + label_smooth).clamp_min(eps)


class JaccardLoss(_Loss):
    r"""Jaccard loss for image segmentation.

    Reference: https://github.com/BloodAxe/pytorch-toolbelt

    Args:
        mode (str): Loss mode, one of 'binary', 'multiclass', or 'multilabel'.
        classes (Optional[List[int]]): List of classes to include in the loss computation,
            defaults to all classes if None.
        log_loss (bool): If True, loss is computed as -log(jaccard);
            otherwise, 1 - jaccard.
        from_logits (bool): If True, input is raw logits, which will be converted to probabilities.
        label_smooth (float): Label smoothing constant.
        eps (float): Small number to prevent division by zero.
    """

    def __init__(
        self,
        mode: CLASS_MODE,
        classes: Optional[List[int]] = None,
        log_loss: bool = False,
        from_logits: bool = True,
        label_smooth: float = 0.0,
        eps: float = 1e-6,
    ):
        super().__init__()

        if classes is not None:
            if mode == 'binary':
                raise ValueError('masking classes is not supported with mode=binary')

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
