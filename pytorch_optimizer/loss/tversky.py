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
        y_pred = torch.sigmoid(y_pred)

        y_pred = y_pred.view(-1)
        y_true = y_true.view(-1)

        tp = y_pred * y_true
        output_dtype = tp.dtype
        accumulation_dtype = torch.float32 if output_dtype in (torch.float16, torch.bfloat16) else output_dtype
        tp = tp.sum(dtype=accumulation_dtype)
        fp = ((1.0 - y_true) * y_pred).sum(dtype=accumulation_dtype)
        fn = (y_true * (1.0 - y_pred)).sum(dtype=accumulation_dtype)

        loss = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)

        return (1.0 - loss).to(output_dtype)
