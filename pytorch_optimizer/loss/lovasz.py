import torch
from torch import nn
from torch.nn.functional import relu


def lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
    """Compute the gradient of the Lovasz extension with respect to sorted errors."""
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:  # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def lovasz_hinge_flat(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """Compute binary Lovasz hinge loss.

    Args:
        y_pred: Binary prediction logits, flattened to one dimension.
        y_true: Binary target labels, flattened to one dimension.

    """
    y_pred = y_pred.reshape(-1)
    y_true = y_true.reshape(-1)

    signs = 2.0 * y_true.float() - 1.0

    errors = 1.0 - y_pred * signs
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)

    grad = lovasz_grad(y_true[perm]).to(dtype=errors_sorted.dtype)

    return torch.dot(relu(errors_sorted), grad)


class LovaszHingeLoss(nn.Module):
    """Binary Lovasz hinge loss.

    Args:
        per_image: Compute the loss per image instead of per batch.

    """

    def __init__(self, per_image: bool = True):
        super().__init__()
        self.per_image = per_image

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Compute the loss for predictions and targets.

        Args:
            y_pred: Binary segmentation logits.
            y_true: Binary masks with the same shape as `y_pred`.

        Returns:
            torch.Tensor: Scalar loss, averaged over images when `per_image=True`.

        """
        if not self.per_image:
            return lovasz_hinge_flat(y_pred, y_true)

        losses = torch.stack([lovasz_hinge_flat(y_p, y_t) for y_p, y_t in zip(y_pred, y_true)])
        return losses.mean()
