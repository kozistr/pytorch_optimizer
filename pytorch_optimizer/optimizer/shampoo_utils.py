import itertools
from enum import IntEnum
from typing import List, Tuple

import numpy as np
import torch

from pytorch_optimizer.optimizer.utils import compute_power, merge_small_dims


class LayerWiseGrafting(IntEnum):
    r"""layer-wise grafting
    Grafting is a technique to fix the layer-wise scale of Shampoo optimizer.
    https://arxiv.org/pdf/2002.11803.pdf studies this in detail. This
    allows us to plugin the Shampoo optimizer into settings where SGD/AdaGrad
    is already well tuned. Grafting onto Shampoo means take the Shampoo direction,
    but use the step magnitude from the grafted optimizer such as Adagrad or SGD.
    """
    NONE = 0
    SGD = 1
    ADAGRAD = 2


class Graft:
    r"""Base class to perform grafting onto Shampoo. This class does no grafting."""

    def __init__(self, *args):
        pass

    def add_statistics(self, grad: torch.Tensor):
        pass

    def precondition_gradient(self, grad: torch.Tensor) -> torch.Tensor:
        return grad

    def update_momentum(self, update: torch.Tensor, unused_beta1: float) -> torch.Tensor:  # noqa: ARG002
        return update


class SGDGraft(Graft):
    r"""Graft using SGD + momentum. momentum maintains an exponentially weighted moving average of gradients."""

    def __init__(self, var: torch.Tensor):
        super().__init__(var)
        self.momentum: torch.Tensor = torch.zeros_like(var, device=var.device)

    def update_momentum(self, update: torch.Tensor, beta1: float) -> torch.Tensor:
        self.momentum.mul_(beta1).add_(update)
        return self.momentum


class AdagradGraft(SGDGraft):
    r"""Graft using Adagrad. Essentially an implementation of Adagrad with momentum."""

    def __init__(self, var: torch.Tensor, diagonal_eps: float):
        super().__init__(var)
        self.diagonal_eps = diagonal_eps
        self.statistics: torch.Tensor = torch.zeros_like(var, device=var.device)

    def add_statistics(self, grad: torch.Tensor):
        self.statistics.add_(grad.pow(2))

    def precondition_gradient(self, grad: torch.Tensor) -> torch.Tensor:
        return grad / (torch.sqrt(self.statistics) + self.diagonal_eps)


class BlockPartitioner:
    r"""Partitions a tensor into smaller tensors for preconditioning.
        For example, if a variable has shape (4096, 512), we might split the 4096 into 4 blocks,
        so we effectively have 4 variables of size (1024, 512) each.

    :param var: torch.Tensor. tensor variable.
    :param block_size: int. block size.
    """

    def __init__(self, var: torch.Tensor, block_size: int):
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
            self.pre_conditioner_shapes.extend([[d, d] for d in t])

    def shapes_for_pre_conditioners(self) -> List[List[int]]:
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

        # if len(partitions) == 1:
        #     raise ValueError('[-] num of partitions is 1')

        return partitions[0]


class PreConditioner:
    r"""Compute statistics/shape from gradients for preconditioning."""

    def __init__(
        self,
        var: torch.Tensor,
        beta2: float,
        inverse_exponent_override: int,
        block_size: int,
        shape_interpretation: bool,
        matrix_eps: float,
    ):
        self.beta2 = beta2
        self.inverse_exponent_override = inverse_exponent_override
        self.matrix_eps = matrix_eps

        self.original_shape: List[int] = var.shape
        self.transformed_shape: List[int] = var.shape
        if shape_interpretation:
            self.transformed_shape = merge_small_dims(self.original_shape, block_size)

        self.statistics: List[torch.Tensor] = []
        self.pre_conditioners: List[torch.Tensor] = []
        if len(self.transformed_shape) > 1:
            reshaped_var = torch.reshape(var, self.transformed_shape)
            self.partitioner = BlockPartitioner(reshaped_var, block_size)

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
        rank: int = len(self.transformed_shape)
        for j, partitioned_grad in enumerate(partitioned_grads):
            for i in range(rank):
                axes: List[int] = list(range(i)) + list(range(i + 1, rank))
                stat: torch.Tensor = torch.tensordot(partitioned_grad, partitioned_grad, [axes, axes])
                self.statistics[j * rank + i].mul_(self.beta2).add_(stat, alpha=w2)

    def exponent_for_pre_conditioner(self) -> int:
        r"""Returns exponent to use for inverse-pth root M^{-1/p}."""
        return (
            self.inverse_exponent_override if self.inverse_exponent_override > 0 else 2 * len(self.transformed_shape)
        )

    def compute_pre_conditioners(self):
        r"""Compute L^{-1/exp} for each stats matrix L."""
        exp: int = self.exponent_for_pre_conditioner()
        for i, stat in enumerate(self.statistics):
            self.pre_conditioners[i] = compute_power(stat, exp, ridge_epsilon=self.matrix_eps)

    def preconditioned_grad(self, grad: torch.Tensor) -> torch.Tensor:
        r"""Precondition the gradient.

        :param grad: torch.Tensor. a gradient tensor to precondition.
        """
        if not self.pre_conditioners:
            return grad

        reshaped_grad = torch.reshape(grad, self.transformed_shape)
        partitioned_grads = self.partitioner.partition(reshaped_grad)

        num_splits: int = self.partitioner.num_splits
        pre_cond_partitioned_grads: List[torch.Tensor] = []
        for i, partitioned_grad in enumerate(partitioned_grads):
            pre_conditioners_for_grad = self.pre_conditioners[i * num_splits:(i + 1) * num_splits]  # fmt: skip
            rank: int = len(partitioned_grad.shape)

            pre_cond_grad = partitioned_grad
            for j in range(rank):
                pre_cond_grad = torch.tensordot(pre_cond_grad, pre_conditioners_for_grad[j], [[0], [0]])

            pre_cond_partitioned_grads.append(pre_cond_grad)

        merged_grad = self.partitioner.merge_partitions(pre_cond_partitioned_grads)

        return torch.reshape(merged_grad, self.original_shape)
