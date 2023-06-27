from typing import Optional, Tuple

import torch
from torch import nn


def log_t(u: torch.Tensor, t: float) -> torch.Tensor:
    r"""Compute log_t for `u'."""
    return u.log() if t == 1.0 else (u.pow(1.0 - t) - 1.0) / (1.0 - t)


def exp_t(u: torch.Tensor, t: float) -> torch.Tensor:
    r"""Compute exp_t for `u'."""
    return u.exp() if t == 1 else (1.0 + (1.0 - t) * u).relu().pow(1.0 / (1.0 - t))


def compute_normalization_fixed_point(activations: torch.Tensor, t: float, num_iters: int) -> torch.Tensor:
    r"""Return the normalization value for each example (t > 1.0).

    :param activations: torch.Tensor. A multidimensional tensor with last dimension `num_classes`.
    :param t: float. Temperature (> 1.0 for tail heaviness).
    :param num_iters: int. Number of iterations to run the method.
    """
    mu, _ = torch.max(activations, dim=-1, keepdim=True)

    normalized_activations_step_0 = activations - mu

    normalized_activations = normalized_activations_step_0

    for _ in range(num_iters):
        logt_partition = torch.sum(exp_t(normalized_activations, t), dim=-1, keepdim=True)
        normalized_activations = normalized_activations_step_0 * logt_partition.pow(1.0 - t)

    logt_partition = torch.sum(exp_t(normalized_activations, t), dim=-1, keepdim=True)

    return -log_t(1.0 / logt_partition, t) + mu


def compute_normalization_binary_search(activations: torch.Tensor, t: float, num_iters: int) -> torch.Tensor:
    """Compute normalization value for each example (t < 1.0).

    :param activations: torch.Tensor. A multidimensional tensor with last dimension `num_classes`.
    :param t: float. Temperature (> 1.0 for tail heaviness).
    :param num_iters: int. Number of iterations to run the method.
    """
    mu, _ = torch.max(activations, dim=-1, keepdim=True)
    normalized_activations = activations - mu

    effective_dim = torch.sum((normalized_activations > -1.0 / (1.0 - t)).to(torch.int32), dim=-1, keepdim=True).to(
        activations.dtype
    )

    shape_partition: Tuple[int, ...] = activations.shape[:-1] + (1,)

    lower = torch.zeros(shape_partition, dtype=activations.dtype, device=activations.device)
    upper = -log_t(1.0 / effective_dim, t) * torch.ones_like(lower)

    for _ in range(num_iters):
        logt_partition = (upper + lower) / 2.0
        sum_probs = torch.sum(exp_t(normalized_activations - logt_partition, t), dim=-1, keepdim=True)
        update = (sum_probs < 1.0).to(activations.dtype)
        lower = torch.reshape(lower * update + (1.0 - update) * logt_partition, shape_partition)
        upper = torch.reshape(upper * (1.0 - update) + update * logt_partition, shape_partition)

    logt_partition = (upper + lower) / 2.0

    return logt_partition + mu


class ComputeNormalization(torch.autograd.Function):
    r"""Custom backward pass for compute_normalization. See compute_normalization."""

    @staticmethod
    def forward(ctx, activations, t, num_iters):
        normalization_constants = (
            compute_normalization_binary_search(activations, t, num_iters)
            if t < 1.0
            else compute_normalization_fixed_point(activations, t, num_iters)
        )

        ctx.save_for_backward(activations, normalization_constants)
        ctx.t = t

        return normalization_constants

    @staticmethod
    def backward(ctx, grad_output):
        activations, normalization_constants = ctx.saved_tensors
        t = ctx.t

        normalized_activations = activations - normalization_constants
        probabilities = exp_t(normalized_activations, t)
        escorts = probabilities.pow(t)
        escorts = escorts / escorts.sum(dim=-1, keepdim=True)
        grad_input = escorts * grad_output

        return grad_input, None, None


def compute_normalization(activations: torch.Tensor, t: float, num_iters: int = 5) -> torch.Tensor:
    """Compute normalization value for each example.

    :param activations: torch.Tensor. A multidimensional tensor with last dimension `num_classes`.
    :param t: float. Temperature (> 1.0 for tail heaviness).
    :param num_iters: int. Number of iterations to run the method.
    """
    return ComputeNormalization.apply(activations, t, num_iters)


def tempered_softmax(activations: torch.Tensor, t: float, num_iters: int = 5) -> torch.Tensor:
    """Tempered softmax function.

    :param activations: torch.Tensor. A multidimensional tensor with last dimension `num_classes`.
    :param t: float. Temperature (> 1.0 for tail heaviness).
    :param num_iters: int. Number of iterations to run the method.
    """
    if t == 1.0:
        return activations.softmax(dim=-1)

    normalization_constants = compute_normalization(activations, t, num_iters)

    return exp_t(activations - normalization_constants, t)


def bi_tempered_logistic_loss(
    activations: torch.Tensor,
    labels: torch.Tensor,
    t1: float,
    t2: float,
    label_smooth: float = 0.0,
    num_iters: int = 5,
    reduction: str = 'mean',
) -> torch.Tensor:
    """Bi-Tempered Logistic Loss.

    :param activations: torch.Tensor. A multidimensional tensor with last dimension `num_classes`.
    :param labels: torch.Tensor. A tensor with shape and dtype as activations (onehot), or a long tensor of
        one dimension less than activations (pytorch standard)
    :param t1: float. Temperature 1 (< 1.0 for boundedness).
    :param t2: float. Temperature 2 (> 1.0 for tail heaviness, < 1.0 for finite support).
    :param label_smooth: float. Label smoothing parameter between [0, 1).
    :param num_iters: int. Number of iterations to run the method.
    :param reduction: str. type of reduction.
    """
    if len(labels.shape) < len(activations.shape):
        labels_onehot = torch.zeros_like(activations)
        labels_onehot.scatter_(1, labels[..., None], 1)
    else:
        labels_onehot = labels

    if label_smooth > 0:
        num_classes: int = labels_onehot.shape[-1]
        labels_onehot = (1.0 - label_smooth * num_classes / (num_classes - 1)) * labels_onehot + label_smooth / (
            num_classes - 1
        )

    probabilities = tempered_softmax(activations, t2, num_iters)

    loss_values = (
        labels_onehot * log_t(labels_onehot + 1e-10, t1)
        - labels_onehot * log_t(probabilities, t1)
        - labels_onehot.pow(2.0 - t1) / (2.0 - t1)
        + probabilities.pow(2.0 - t1) / (2.0 - t1)
    )
    loss_values = loss_values.sum(dim=-1)

    if reduction == 'sum':
        return loss_values.sum()
    if reduction == 'mean':
        return loss_values.mean()
    return loss_values


class BiTemperedLogisticLoss(nn.Module):
    """Bi-Tempered Log Loss.

    Reference : https://github.com/BloodAxe/pytorch-toolbelt/blob/develop/pytorch_toolbelt/losses/bitempered_loss.py

    :param t1: float. Temperature 1 (< 1.0 for boundedness).
    :param t2: float. Temperature 2 (> 1.0 for tail heaviness, < 1.0 for finite support).
    :param label_smooth: float. Label smoothing parameter between [0, 1).
    :param ignore_index: Optional[int]. Index to ignore.
    :param reduction: str. type of reduction.
    """

    def __init__(
        self,
        t1: float,
        t2: float,
        label_smooth: float = 0.0,
        ignore_index: Optional[int] = None,
        reduction: str = 'mean',
    ):
        super().__init__()
        self.t1 = t1
        self.t2 = t2
        self.label_smooth = label_smooth
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss = bi_tempered_logistic_loss(
            predictions, targets, t1=self.t1, t2=self.t2, label_smooth=self.label_smooth, reduction='none'
        )

        if self.ignore_index is not None:
            mask = ~targets.eq(self.ignore_index)
            loss *= mask

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss


class BinaryBiTemperedLogisticLoss(nn.Module):
    """Modification of BiTemperedLogisticLoss for binary classification case.

    :param t1: float. Temperature 1 (< 1.0 for boundedness).
    :param t2: float. Temperature 2 (> 1.0 for tail heaviness, < 1.0 for finite support).
    :param label_smooth: float. Label smoothing parameter between [0, 1).
    :param ignore_index: Optional[int]. Index to ignore.
    :param reduction: str. type of reduction.
    """

    def __init__(
        self,
        t1: float,
        t2: float,
        label_smooth: float = 0.0,
        ignore_index: Optional[int] = None,
        reduction: str = 'mean',
    ):
        super().__init__()
        self.t1 = t1
        self.t2 = t2
        self.label_smooth = label_smooth
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if predictions.size(1) != 1 or targets.size(1) != 1:
            raise ValueError('Channel dimension for predictions and targets must be equal to 1')

        loss = bi_tempered_logistic_loss(
            torch.cat((-predictions, predictions), dim=1).moveaxis(1, -1),
            torch.cat((1.0 - targets, targets), dim=1).moveaxis(1, -1),
            t1=self.t1,
            t2=self.t2,
            label_smooth=self.label_smooth,
            reduction='none',
        ).unsqueeze(dim=1)

        if self.ignore_index is not None:
            mask = targets.eq(self.ignore_index)
            loss = torch.masked_fill(loss, mask, value=0)

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss
