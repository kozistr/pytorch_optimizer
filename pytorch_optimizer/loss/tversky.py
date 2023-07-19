import torch
from torch import nn


class TverskyLoss(nn.Module):
    r"""Tversky Loss w/ logits input.

    :param alpha: float. alpha.
    :param beta: float. beta.
    :param smooth: float. smooth factor.
    """

    def __init__(self, alpha: float = 0.5, beta: float = 0.5, smooth: float = 1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        y_pred = torch.sigmoid(y_pred)

        y_pred = y_pred.view(-1)
        y_true = y_true.view(-1)

        tp = (y_pred * y_true).sum()
        fp = ((1.0 - y_true) * y_pred).sum()
        fn = (y_true * (1.0 - y_pred)).sum()

        loss = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)

        return 1.0 - loss
