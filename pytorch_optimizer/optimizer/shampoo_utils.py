import itertools
from enum import IntEnum
from typing import List, Tuple

import numpy as np
import torch

from pytorch_optimizer.optimizer.utils import compute_power, merge_small_dims


class LayerWiseGrafting(IntEnum):
    r"""Layer-wise grafting.

    Grafting is a technique to fix the layer-wise scale of Shampoo optimizer.
    https://arxiv.org/pdf/2002.11803.pdf studies this in detail. This
    allows us to plugin the Shampoo optimizer into settings where SGD/AdaGrad
    is already well tuned. Grafting onto Shampoo means take the Shampoo direction,
    but use the step magnitude from the grafted optimizer such as Adagrad or SGD.
    """

    NONE = 0
    SGD = 1
    ADAGRAD = 2
    RMSPROP = 3
    SQRTN = 4


class Graft:
    r"""Base class to perform grafting onto Shampoo. This class does no grafting."""

    def __init__(self, *args):
        pass

    def add_statistics(self, grad: torch.Tensor, unused_beta2: float):
        r"""Add the statistics."""
        pass

    def precondition_gradient(self, grad: torch.Tensor) -> torch.Tensor:
        r"""Get preconditioned gradient."""
        return grad

    def update_momentum(self, update: torch.Tensor, unused_beta1: float) -> torch.Tensor:  # noqa: ARG002
        r"""Update momentum."""
        return update


class SGDGraft(Graft):
    r"""Graft using SGD + momentum. momentum maintains an exponentially weighted moving average of gradients."""

    def __init__(self, var: torch.Tensor):
        super().__init__(var)
        self.momentum: torch.Tensor = torch.zeros_like(var, device=var.device)

    def update_momentum(self, update: torch.Tensor, beta1: float) -> torch.Tensor:
        r"""Update momentum."""
        self.momentum.mul_(beta1).add_(update)
        return self.momentum


class SQRTNGraft(Graft):
    r"""Graft using SQRTN."""

    def __init__(self, var: torch.Tensor):
        super().__init__(var)

    def precondition_gradient(self, grad: torch.Tensor) -> torch.Tensor:
        r"""Get preconditioned gradient."""
        return torch.ones_like(grad) * torch.sign(grad)


class AdagradGraft(SGDGraft):
    r"""Graft using Adagrad. Essentially an implementation of Adagrad with momentum.

    :param var: torch.Tensor. variable.
    :param diagonal_eps: float. diagonal epsilon.
    """

    def __init__(self, var: torch.Tensor, diagonal_eps: float):
        super().__init__(var)
        self.diagonal_eps = diagonal_eps
        self.statistics: torch.Tensor = torch.zeros_like(var, device=var.device)

    def add_statistics(self, grad: torch.Tensor, _):
        r"""Add the statistics."""
        self.statistics.add_(grad.pow(2))

    def precondition_gradient(self, grad: torch.Tensor) -> torch.Tensor:
        r"""Get preconditioned gradient."""
        return grad / (torch.sqrt(self.statistics) + self.diagonal_eps)


class RMSPropGraft(SGDGraft):
    r"""Graft using RMSProp. Essentially an implementation of RMSProp with momentum.

    :param var: torch.Tensor. variable.
    :param diagonal_eps: float. diagonal epsilon.
    """

    def __init__(self, var: torch.Tensor, diagonal_eps: float):
        super().__init__(var)
        self.diagonal_eps = diagonal_eps
        self.statistics: torch.Tensor = torch.zeros_like(var, device=var.device)

    def add_statistics(self, grad: torch.Tensor, beta2: float):
        r"""Add the statistics."""
        self.statistics.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

    def precondition_gradient(self, grad: torch.Tensor) -> torch.Tensor:
        r"""Get preconditioned gradient."""
        return grad / (torch.sqrt(self.statistics) + self.diagonal_eps)


class BlockPartitioner:
    r"""Partition a tensor into smaller tensors for preconditioning.

        For example, if a variable has shape (4096, 512), we might split the 4096 into 4 blocks,
        so we effectively have 4 variables of size (1024, 512) each.

    :param var: torch.Tensor. tensor variable.
    :param block_size: int. block size.
    :param pre_conditioner_type: int type of pre-conditioner.
    """

    def __init__(self, var: torch.Tensor, block_size: int, pre_conditioner_type: int):
        self.shape: List[int] = var.shape

        self.splits: List[Tuple[int, np.ndarray]] = []
        self.split_sizes: List[Tuple[int, np.ndarray]] = []

        split_sizes: List[np.ndarray] = []

        # We split var into smaller blocks. Here we store the metadata to make that split.
        for i, d in enumerate(self.shape):
            if 0 < block_size < d:
                # d - 1, otherwise split appends a 0-size array.
                num_split: int = (d - 1) // block_size
                indices = (np.arange(num_split, dtype=np.int32) + 1) * block_size
                sizes = np.ones(num_split + 1, dtype=np.int32) * block_size
                sizes[-1] = d - indices[-1]
                self.splits.append((i, indices))
                self.split_sizes.append((i, sizes))
                split_sizes.append(sizes)
            else:
                split_sizes.append(np.array([d], dtype=np.int32))

        self.num_splits: int = len(split_sizes)
        self.pre_conditioner_shapes: List[List[int]] = []
        for t in itertools.product(*split_sizes):
            if not (pre_conditioner_type == PreConditionerType.ALL or self.num_splits <= 1):
                t = t[:-1]
            self.pre_conditioner_shapes.extend([[d, d] for d in t])

    def shapes_for_pre_conditioners(self) -> List[List[int]]:
        r"""Get shapes of pre-conditioner."""
        return self.pre_conditioner_shapes

    def partition(self, x: torch.Tensor) -> List[torch.Tensor]:
        r"""Partition tensor into blocks."""
        if x.shape != self.shape:
            raise ValueError(f'self._shape != x.shape ({self.shape} vs {x.shape})')

        tensors: List[torch.Tensor] = [x]
        for i, sizes in self.split_sizes:
            tensors_local: List[torch.Tensor] = []
            for t in tensors:
                tensors_local.extend(torch.split(t, list(sizes), dim=i))
            tensors = tensors_local
        return tensors

    def merge_partitions(self, partitions: List[torch.Tensor]) -> torch.Tensor:
        r"""Merge partitions back to original shape."""
        for i, indices in reversed(self.splits):
            n: int = len(indices) + 1

            partitions: List[torch.Tensor] = [
                torch.cat(partitions[idx:idx + n], axis=i) for idx in range(0, len(partitions), n)  # fmt: skip
            ]

        # TODO
        # when length of partitions is 1, raise error

        return partitions[0]


class PreConditionerType(IntEnum):
    r"""Type of PreConditioner.

    In default (ALL), computes pre-conditioner for each dim.
    INPUT is one-sided Shampoo, in this case only on input dim.
    Assumes last dim is always the output dim and everything else input dim.
    """

    ALL = 0
    INPUT = 1


class PreConditioner:
    r"""Compute statistics/shape from gradients for preconditioning.

    :param var: torch.Tensor. variable.
    :param beta2: float. beta2.
    :param inverse_exponent_override: int.
    :param block_size: int.
    :param shape_interpretation: bool.
    :param matrix_eps: float.
    :param pre_conditioner_type: int. type of pre-conditioner.
    """

    def __init__(
        self,
        var: torch.Tensor,
        beta2: float,
        inverse_exponent_override: int,
        block_size: int,
        shape_interpretation: bool,
        matrix_eps: float,
        pre_conditioner_type: int = PreConditionerType.ALL,
    ):
        self.beta2 = beta2
        self.inverse_exponent_override = inverse_exponent_override
        self.matrix_eps = matrix_eps
        self.pre_conditioner_type = pre_conditioner_type

        self.original_shape: List[int] = var.shape
        self.transformed_shape: List[int] = var.shape
        if shape_interpretation:
            self.transformed_shape = merge_small_dims(self.original_shape, block_size)

        self.statistics: List[torch.Tensor] = []
        self.pre_conditioners: List[torch.Tensor] = []
        if len(self.transformed_shape) > 1:
            reshaped_var = torch.reshape(var, self.transformed_shape)
            self.partitioner = BlockPartitioner(reshaped_var, block_size, self.pre_conditioner_type)

            shapes = self.partitioner.shapes_for_pre_conditioners()
            self.statistics = [self.matrix_eps * torch.eye(s[0], device=var.device) for s in shapes]
            self.pre_conditioners = [torch.eye(s[0], device=var.device) for s in shapes]

    def add_statistics(self, grad: torch.Tensor):
        r"""Compute statistics from gradients and add to the correct state entries.

        :param grad: torch.Tensor. gradient to compute statistics from.
        """
        if not self.statistics:
            return

        reshaped_grad: torch.Tensor = torch.reshape(grad, self.transformed_shape)
        partitioned_grads: List[torch.Tensor] = self.partitioner.partition(reshaped_grad)

        w2: float = 1.0 if self.beta2 == 1.0 else (1.0 - self.beta2)
        rank: int = sum(self.should_precondition_dims())
        for j, partitioned_grad in enumerate(partitioned_grads):
            for i in range(rank):
                axes: List[int] = [ax for ax in range(partitioned_grad.ndim) if ax != i]
                stat: torch.Tensor = torch.tensordot(partitioned_grad, partitioned_grad, [axes, axes])
                self.statistics[j * rank + i].mul_(self.beta2).add_(stat, alpha=w2)

    def should_precondition_dims(self) -> List[bool]:
        r"""Vector containing indicator indicating if the dim is preconditioned."""
        rank: int = len(self.transformed_shape)
        return (
            [True] * rank
            if self.pre_conditioner_type == PreConditionerType.ALL or rank <= 1
            else [True] * (rank - 1) + [False]
        )

    def exponent_for_pre_conditioner(self) -> int:
        r"""Return exponent to use for inverse-pth root M^{-1/p}."""
        return (
            self.inverse_exponent_override
            if self.inverse_exponent_override > 0
            else 2 * sum(self.should_precondition_dims())
        )

    def compute_pre_conditioners(self):
        r"""Compute L^{-1/exp} for each stats matrix L."""
        exp: int = self.exponent_for_pre_conditioner()
        for i, stat in enumerate(self.statistics):
            self.pre_conditioners[i] = compute_power(stat, exp, ridge_epsilon=self.matrix_eps)

    @staticmethod
    def precondition_block(
        partitioned_grad: torch.Tensor,
        should_preconditioned_dims: List[bool],
        pre_conditioners_for_grad: List[torch.Tensor],
    ) -> torch.Tensor:
        r"""Perform a preconditioning operation on a single gradient block.

        Loop invariant: the dimension to be preconditioned is first
        We keep all axes in the same cyclic order they were originally.
        """
        rank: int = len(partitioned_grad.shape)
        roll: Tuple[int, ...] = (*tuple(range(1, rank)), 0)
        for j, should_precondition in enumerate(should_preconditioned_dims):
            if not should_precondition:
                partitioned_grad = torch.permute(partitioned_grad, roll).contiguous()
                continue
            partitioned_grad = torch.tensordot(partitioned_grad, pre_conditioners_for_grad[j], dims=[[0], [0]])
        return partitioned_grad

    def preconditioned_grad(self, grad: torch.Tensor) -> torch.Tensor:
        r"""Precondition the gradient.

        :param grad: torch.Tensor. a gradient tensor to precondition.
        """
        if not self.pre_conditioners:
            return grad

        reshaped_grad = torch.reshape(grad, self.transformed_shape)
        partitioned_grads = self.partitioner.partition(reshaped_grad)

        should_precondition_dims: List[bool] = self.should_precondition_dims()
        num_pre_conditioners: int = sum(should_precondition_dims)

        pre_cond_partitioned_grads: List[torch.Tensor] = [
            self.precondition_block(
                partitioned_grad,
                should_precondition_dims,
                self.pre_conditioners[i * num_pre_conditioners:(i + 1) * num_pre_conditioners]  # fmt: skip
            )
            for i, partitioned_grad in enumerate(partitioned_grads)
        ]

        merged_grad = self.partitioner.merge_partitions(pre_cond_partitioned_grads)

        return torch.reshape(merged_grad, self.original_shape)
