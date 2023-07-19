import torch
from torch import nn
from torch.nn.functional import relu


def lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
    r"""Compute gradient of the Lovasz extension w.r.t sorted errors."""
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:  # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def lovasz_hinge_flat(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    r"""Binary Lovasz hinge loss.

    :param y_pred: torch.Tensor.
    :param y_true: torch.Tensor.
    """
    y_pred = y_pred.view(-1)
    y_true = y_true.view(-1)

    signs = 2.0 * y_true.float() - 1.0

    errors = 1.0 - y_pred * signs
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)

    grad = lovasz_grad(y_true[perm])

    return torch.dot(relu(errors_sorted), grad)


class LovaszHingeLoss(nn.Module):
    r"""Binary Lovasz hinge loss.

    :param per_image: bool. compute the loss per image instead of per batch.
    """

    def __init__(self, per_image: bool = True):
        super().__init__()
        self.per_image = per_image

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        if not self.per_image:
            return lovasz_hinge_flat(y_pred, y_true)
        return sum(lovasz_hinge_flat(y_p, y_t) for y_p, y_t in zip(y_pred, y_true)) / y_pred.size()[0]
