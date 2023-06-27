import torch
from torch import nn
from torch.nn.functional import binary_cross_entropy_with_logits

from pytorch_optimizer.loss.cross_entropy import BCELoss


class FocalLoss(nn.Module):
    r"""BCEFocal loss function w/ logit input.

    :param alpha: float. alpha.
    :param gamma: float. gamma.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        bce_loss = binary_cross_entropy_with_logits(y_pred, y_true, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()


class BCEFocalLoss(nn.Module):
    r"""BCEFocal loss function w/ probability input.

    :param alpha: float. alpha.
    :param gamma: float. gamma.
    :param label_smooth: float. Smoothness constant for dice coefficient (a).
    :param eps: float. epsilon.
    :param reduction: str. type of reduction.
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        label_smooth: float = 0.0,
        eps: float = 1e-6,
        reduction: str = 'mean',
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

        self.bce = BCELoss(label_smooth=label_smooth, eps=eps, reduction='none')

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(y_pred, y_true)
        focal_loss = (
            y_true * self.alpha * (1.0 - y_pred) ** self.gamma * bce_loss
            + (1.0 - y_true) ** self.gamma * bce_loss
        )  # fmt: skip

        return focal_loss.mean() if self.reduction == 'mean' else focal_loss.sum()
