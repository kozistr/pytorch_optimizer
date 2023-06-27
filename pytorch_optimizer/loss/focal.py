import torch
from torch import nn

from pytorch_optimizer.loss.cross_entropy import BCELoss


class BCEFocalLoss(nn.Module):
    r"""BCEFocal loss function w/ probability input.

    :param alpha: float. alpha.
    :param gamma: float. gamma.
    :param label_smoothing: float. label smoothing factor.
    :param eps: float. epsilon.
    :param reduction: str. type of reduction.
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        label_smoothing: float = 0.0,
        eps: float = 1e-6,
        reduction: str = 'mean',
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

        self.bce = BCELoss(label_smoothing=label_smoothing, eps=eps, reduction='none')

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(y_pred, y_true)
        focal_loss = (
            y_true * self.alpha * (1.0 - y_pred) ** self.gamma * bce_loss
            + (1.0 - y_true) ** self.gamma * bce_loss
        )  # fmt: skip

        return focal_loss.mean() if self.reduction == 'mean' else focal_loss.sum()
