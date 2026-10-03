import torch
from torch import nn


class SoftF1Loss(nn.Module):
    """Soft F-beta loss for binary prediction probabilities.

    Args:
        beta: Precision recall balance. Values above 1 give recall more weight.
        eps: Small epsilon value to avoid division by zero during calculation.

    """

    def __init__(self, beta: float = 1.0, eps: float = 1e-6):
        super().__init__()
        self.beta = beta
        self.eps = eps

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Compute the loss for predictions and targets.

        Args:
            y_pred: Binary probabilities, with the same shape as `y_true`.
            y_true: Binary target labels.

        Returns:
            torch.Tensor: Scalar loss, `1 - soft_f_beta`.

        """
        tp = (y_true * y_pred).sum().float()
        fp = ((1 - y_true) * y_pred).sum().float()
        fn = (y_true * (1 - y_pred)).sum().float()

        p = tp / (tp + fp + self.eps)
        r = tp / (tp + fn + self.eps)

        f1 = (1 + self.beta ** 2) * (p * r) / ((self.beta ** 2) * p + r + self.eps)  # fmt: skip
        f1 = torch.where(torch.isnan(f1), torch.zeros_like(f1), f1)

        return 1.0 - f1.mean()
