import torch
from torch import nn


class TverskyLoss(nn.Module):
    """Tversky Loss with logits input.

    Args:
        alpha (float): Weight of false positives.
        beta (float): Weight of false negatives.
        smooth (float): Small constant to avoid division by zero.

    """

    def __init__(self, alpha: float = 0.5, beta: float = 0.5, smooth: float = 1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        dtype = torch.promote_types(torch.promote_types(y_pred.dtype, y_true.dtype), torch.float32)
        y_pred = y_pred.to(dtype).sigmoid().view(-1)
        y_true = y_true.to(dtype).view(-1)

        tp = (y_pred * y_true).sum()
        fp = ((1.0 - y_true) * y_pred).sum()
        fn = (y_true * (1.0 - y_pred)).sum()

        loss = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)

        return 1.0 - loss
