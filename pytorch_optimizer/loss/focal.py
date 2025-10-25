import torch
from torch import nn
from torch.nn.functional import (
    binary_cross_entropy_with_logits,
    cosine_embedding_loss,
    cross_entropy,
    normalize,
    one_hot,
)

from pytorch_optimizer.loss.cross_entropy import BCELoss
from pytorch_optimizer.loss.tversky import TverskyLoss


class FocalLoss(nn.Module):
    """Focal Loss function with logits input.

    Args:
        alpha (float): Weighting factor for class imbalance.
        gamma (float): Focusing parameter to down-weight easy examples and focus training on hard negatives.
    """

    def __init__(self, alpha: float = 1.0, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        bce_loss = binary_cross_entropy_with_logits(y_pred, y_true, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()


class FocalCosineLoss(nn.Module):
    """Focal Cosine Loss function with logits input.

    Args:
        alpha (float): Weighting factor for class imbalance.
        gamma (float): Focusing parameter to reduce loss contribution from easy examples.
        focal_weight (float): Weight of the focal loss component in the combined loss.
        reduction (str): Specifies the reduction to apply to the output: 'none', 'mean', or 'sum'.
    """

    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, focal_weight: float = 0.1, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.focal_weight = focal_weight
        self.reduction = reduction

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        cosine_loss = cosine_embedding_loss(
            y_pred,
            one_hot(y_true, num_classes=y_pred.size(-1)),
            torch.tensor([1], device=y_true.device),
            reduction=self.reduction,
        )

        ce_loss = cross_entropy(normalize(y_pred), y_true, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (self.alpha * (1 - pt) ** self.gamma * ce_loss).mean()

        return cosine_loss + self.focal_weight * focal_loss


class BCEFocalLoss(nn.Module):
    """BCEFocal loss function with probability input.

    Args:
        alpha (float): Weighting factor for class imbalance, commonly set to 0.25.
        gamma (float): Focusing parameter to reduce loss contribution of easy examples.
        label_smooth (float): Smoothness constant to regularize target labels.
        eps (float): Small epsilon to avoid numerical instability.
        reduction (str): Specifies reduction type to apply to output: 'none', 'mean' or 'sum'.
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


class FocalTverskyLoss(nn.Module):
    """Focal Tversky Loss with logits input.

    Args:
        alpha (float): Weight for false negatives in Tversky index.
        beta (float): Weight for false positives in Tversky index.
        gamma (float): Focusing parameter that shapes the loss to focus more on hard examples.
        smooth (float): Smoothing factor to avoid division by zero.
    """

    def __init__(self, alpha: float = 0.5, beta: float = 0.5, gamma: float = 1.0, smooth: float = 1e-6):
        super().__init__()
        self.gamma = gamma

        self.tversky = TverskyLoss(alpha, beta, smooth)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        return self.tversky(y_pred, y_true) ** self.gamma
