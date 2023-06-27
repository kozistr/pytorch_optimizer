import torch
from torch import nn
from torch.nn.functional import binary_cross_entropy


class BCELoss(nn.Module):
    r"""binary cross entropy with label smoothing + probability input.

    :param label_smooth: float. Smoothness constant for dice coefficient (a).
    :param eps: float. epsilon.
    :param reduction: str. type of reduction.
    """

    def __init__(self, label_smooth: float = 0.0, eps: float = 1e-6, reduction: str = 'mean'):
        super().__init__()
        self.label_smooth = label_smooth
        self.eps = eps
        self.reduction = reduction

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        if self.training and self.label_smooth > 0.0:
            y_true = (1.0 - self.label_smooth) * y_true + self.label_smooth / y_pred.size(-1)
        y_pred = torch.clamp(y_pred, self.eps, 1.0 - self.eps)
        return binary_cross_entropy(y_pred, y_true, reduction=self.reduction)
