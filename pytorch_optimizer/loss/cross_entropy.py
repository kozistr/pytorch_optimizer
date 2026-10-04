import torch
from torch import nn
from torch.nn.functional import binary_cross_entropy


class BCELoss(nn.Module):
    """Binary cross entropy for probabilities, with optional label smoothing.

    Args:
        label_smooth: Smoothness constant to soften target labels.
        eps: Small epsilon to avoid numerical instability.
        reduction: Specifies the reduction to apply to the output. 'none' | 'mean' | 'sum'.

    """

    def __init__(self, label_smooth: float = 0.0, eps: float = 1e-6, reduction: str = 'mean'):
        super().__init__()
        self.label_smooth = label_smooth
        self.eps = eps
        self.reduction = reduction

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Compute the loss for predictions and targets.

        Args:
            y_pred: Binary probabilities, with the same shape as `y_true`.
            y_true: Binary targets in `[0, 1]`.

        Returns:
            torch.Tensor: Elementwise loss or the requested scalar reduction.

        """
        if self.training and self.label_smooth > 0.0:
            y_true = (1.0 - self.label_smooth) * y_true + 0.5 * self.label_smooth
        y_pred = torch.clamp(y_pred, self.eps, 1.0 - self.eps)
        return binary_cross_entropy(y_pred, y_true, reduction=self.reduction)
