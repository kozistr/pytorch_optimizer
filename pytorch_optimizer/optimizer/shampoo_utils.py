import itertools
from enum import IntEnum
from typing import List, Optional, Tuple, Union

import numpy as np
import torch


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

    def add_statistics(self, grad: torch.Tensor, unused_beta2: float) -> None:
        r"""Add the statistics."""
        pass

    def precondition_gradient(self, grad: torch.Tensor) -> torch.Tensor:
        r"""Get preconditioned gradient."""
        return grad

    def update_momentum(self, update: torch.Tensor, unused_beta1: float) -> torch.Tensor:
        r"""Update momentum."""
        return update


class SGDGraft(Graft):
    r"""Graft using SGD + momentum. momentum maintains an exponentially weighted moving average of gradients."""

    def __init__(self, var: torch.Tensor):
        super().__init__(var)
        self.momentum: torch.Tensor = torch.zeros_like(var)

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
        return grad.sign()


class AdaGradGraft(SGDGraft):
    r"""Graft using AdaGrad. Essentially an implementation of AdaGrad with momentum.

    :param var: torch.Tensor. variable.
    :param diagonal_eps: float. diagonal epsilon.
    """

    def __init__(self, var: torch.Tensor, diagonal_eps: float):
        super().__init__(var)
        self.diagonal_eps = diagonal_eps
        self.statistics: torch.Tensor = torch.zeros_like(var)

    def add_statistics(self, grad: torch.Tensor, _) -> None:
        r"""Add the statistics."""
        self.statistics.add_(grad.pow(2))

    def precondition_gradient(self, grad: torch.Tensor) -> torch.Tensor:
        r"""Get preconditioned gradient."""
        return grad.div(self.statistics.sqrt().add_(self.diagonal_eps))


class RMSPropGraft(SGDGraft):
    r"""Graft using RMSProp. Essentially an implementation of RMSProp with momentum.

    :param var: torch.Tensor. variable.
    :param diagonal_eps: float. diagonal epsilon.
    """

    def __init__(self, var: torch.Tensor, diagonal_eps: float):
        super().__init__(var)
        self.diagonal_eps = diagonal_eps
        self.statistics: torch.Tensor = torch.zeros_like(var)

    def add_statistics(self, grad: torch.Tensor, beta2: float) -> None:
        r"""Add the statistics."""
        self.statistics.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

    def precondition_gradient(self, grad: torch.Tensor) -> torch.Tensor:
        r"""Get preconditioned gradient."""
        return grad.div(self.statistics.sqrt().add_(self.diagonal_eps))


class BlockPartitioner:
    r"""Partition a tensor into smaller tensors for preconditioning.

        For example, if a variable has shape (4096, 512), we might split the 4096 into 4 blocks,
        so we effectively have 4 variables of size (1024, 512) each.

    :param var: torch.Tensor. tensor variable.
    :param rank: int. rank.
    :param block_size: int. block size.
    :param pre_conditioner_type: int type of pre-conditioner.
    """

    def __init__(self, var: torch.Tensor, rank: int, block_size: int, pre_conditioner_type: int):
        self.shape: torch.Size = var.shape

        self.splits: List[Tuple[int, torch.Tensor]] = []
        self.split_sizes: List[Tuple[int, torch.Tensor]] = []

        split_sizes: List[torch.Tensor] = []

        # We split var into smaller blocks. Here we store the metadata to make that split.
        for i, d in enumerate(self.shape):
            if block_size <= 0 or block_size >= d:
                split_sizes.append(torch.tensor([d], dtype=torch.int32))
                continue

            # d - 1, otherwise split appends a 0-size array.
            num_split: int = (d - 1) // block_size
            indices = (torch.arange(num_split, dtype=torch.int32) + 1) * block_size

            sizes: torch.Tensor = torch.full((num_split + 1,), block_size, dtype=torch.int32)
            sizes[-1] = d - indices[-1]

            self.splits.append((i, indices))
            self.split_sizes.append((i, sizes))
            split_sizes.append(sizes)

        self.num_splits: int = len(split_sizes)
        self.pre_conditioner_shapes: List[List[torch.Tensor]] = self.build_pre_conditioner_shapes(
            split_sizes, pre_conditioner_type, rank
        )

    @staticmethod
    def build_pre_conditioner_shapes(
        split_sizes: List[torch.Tensor], pre_conditioner_type: int, rank: int
    ) -> List[List[torch.Tensor]]:
        r"""Build pre-conditioner shapes."""
        pre_conditioner_shapes: List[List[torch.Tensor]] = []
        for t in itertools.product(*split_sizes):
            t_shape: List[Optional[List[torch.Tensor]]] = [[d, d] for d in t]
            if pre_conditioner_type == PreConditionerType.INPUT:
                t_shape[-1] = None
            elif pre_conditioner_type == PreConditionerType.OUTPUT:
                t_shape = [None] * (rank - 1) + t_shape[-1:]
            pre_conditioner_shapes.extend(t_shape)
        return pre_conditioner_shapes

    def shapes_for_pre_conditioners(self) -> List[List[torch.Tensor]]:
        r"""Get shapes of pre-conditioner."""
        return self.pre_conditioner_shapes

    @torch.no_grad()
    def partition(self, x: torch.Tensor) -> List[torch.Tensor]:
        r"""Partition tensor into blocks."""
        if x.shape != self.shape:
            raise ValueError(f'self.shape != x.shape ({self.shape} vs {x.shape})')

        tensors = [x]
        for i, sizes in self.split_sizes:
            tensors = [torch.split(t, list(sizes), dim=i) for t in tensors]
            tensors = [t for tensor in tensors for t in tensor]
        return tensors

    def merge_partitions(self, partitions: List[torch.Tensor]) -> torch.Tensor:
        r"""Merge partitions back to original shape."""
        for i, indices in reversed(self.splits):
            n: int = len(indices) + 1

            partitions: List[torch.Tensor] = [
                torch.cat(partitions[idx:idx + n], dim=i) for idx in range(0, len(partitions), n)  # fmt: skip
            ]

        return partitions[0]


class PreConditionerType(IntEnum):
    r"""Type of PreConditioner.

    In default (ALL), computes pre-conditioner for each dim.
    INPUT/OUTPUT is one-sided Shampoo, in this case only on input/output dim.
    Assumes last dim is always the output dim and everything else input dim.
    """

    ALL = 0
    INPUT = 1
    OUTPUT = 2


class PreConditioner:
    r"""Compute statistics & shape from gradients for preconditioning.

    :param var: torch.Tensor. variable.
    :param beta2: float. beta2.
    :param inverse_exponent_override: int. override inv exp.
    :param block_size: int. size of block.
    :param skip_preconditioning_rank_lt: int. skip low-rank parameter.
    :param no_preconditioning_for_layers_with_dim_gt: int. skip large size of dim of parameter.
    :param shape_interpretation: bool. reshaping parameter.
    :param pre_conditioner_type: int. type of pre-conditioner.
    :param matrix_eps: float. epsilon of matrix.
    :param use_svd: bool. use SVD instead of Schur-Newton method to calculate M^{-1/p}.
    """

    def __init__(
        self,
        var: torch.Tensor,
        beta2: float,
        inverse_exponent_override: int,
        block_size: int,
        skip_preconditioning_rank_lt: int,
        no_preconditioning_for_layers_with_dim_gt: int,
        shape_interpretation: bool,
        pre_conditioner_type: int,
        matrix_eps: float = 1e-6,
        use_svd: bool = False,
    ):
        self.beta2 = beta2
        self.inverse_exponent_override = inverse_exponent_override
        self.skip_preconditioning_rank_lt = skip_preconditioning_rank_lt
        self.no_preconditioning_for_layers_with_dim_gt = no_preconditioning_for_layers_with_dim_gt
        self.pre_conditioner_type = pre_conditioner_type
        self.matrix_eps = matrix_eps
        self.use_svd = use_svd

        self.w2: float = 1.0 if self.beta2 == 1.0 else (1.0 - self.beta2)

        self.original_shape: torch.Size = var.shape
        self.transformed_shape: List[int] = (
            merge_small_dims(self.original_shape, block_size) if shape_interpretation else var.shape
        )

        self.should_precondition_dims: List[bool] = self.get_should_precondition_dims()
        self.rank: int = sum(self.should_precondition_dims)
        self.exponent_for_pre_conditioner: int = (
            self.inverse_exponent_override if self.inverse_exponent_override > 0 else 2 * self.rank
        )

        self.statistics: Union[List[torch.Tensor], torch.Tensor] = []
        self.pre_conditioners: Union[List[torch.Tensor], torch.Tensor] = []

        self.is_same_shapes: bool = False
        if len(self.transformed_shape) > 1 and not self.skip_precondition(var):
            self.partitioner = BlockPartitioner(
                var=torch.reshape(var, self.transformed_shape),
                rank=self.rank,
                block_size=block_size,
                pre_conditioner_type=self.pre_conditioner_type,
            )

            shapes: List[Optional[List[torch.Tensor]]] = self.partitioner.shapes_for_pre_conditioners()
            self.statistics = [self.matrix_eps * torch.eye(shape[0], device=var.device) for shape in shapes if shape]
            self.pre_conditioners = [torch.eye(shape[0], device=var.device) for shape in shapes if shape]
            self.is_same_shapes = None not in shapes and len(np.unique(shapes)) == 1

        if self.is_same_shapes:
            self.statistics = torch.stack(self.statistics, dim=0)
            self.pre_conditioners = torch.stack(self.pre_conditioners, dim=0)

    def get_should_precondition_dims(self) -> List[bool]:
        r"""Get pre-condition dimensions by the type of conditioner."""
        if self.pre_conditioner_type == PreConditionerType.ALL or len(self.transformed_shape) <= 1:
            return [True] * len(self.transformed_shape)
        if self.pre_conditioner_type == PreConditionerType.INPUT:
            return [True] * (len(self.transformed_shape) - 1) + [False]
        if self.pre_conditioner_type == PreConditionerType.OUTPUT:
            return [False] * (len(self.transformed_shape) - 1) + [True]
        raise ValueError

    def skip_precondition(self, x: torch.Tensor) -> bool:
        return (len(x.shape) < self.skip_preconditioning_rank_lt) or any(
            dim > self.no_preconditioning_for_layers_with_dim_gt for dim in x.shape
        )

    def add_statistics(self, grad: torch.Tensor) -> None:
        r"""Compute statistics from gradients and add to the correct state entries.

        :param grad: torch.Tensor. gradient to compute statistics from.
        """
        if len(self.statistics) == 0:
            return

        reshaped_grad: torch.Tensor = torch.reshape(grad, self.transformed_shape)
        partitioned_grads: List[torch.Tensor] = self.partitioner.partition(reshaped_grad)

        for j, partitioned_grad in enumerate(partitioned_grads):
            for i in range(self.rank):
                axes: List[int] = [ax for ax in range(partitioned_grad.ndim) if ax != i]
                stat: torch.Tensor = torch.tensordot(partitioned_grad, partitioned_grad, dims=[axes, axes])
                self.statistics[j * self.rank + i].mul_(self.beta2).add_(stat, alpha=self.w2)

    def compute_pre_conditioners(self) -> None:
        r"""Compute L^{-1/exp} for each stats matrix L.

        If `self.use_svd` is enabled and where all shapes of statistics & pre-conditioners are same, perform batch SVD.
        else, SVD one by one.
        If `self.use_svd` is disabled, use Schur-Newton method, which is usually much faster.
        """
        if self.use_svd and self.is_same_shapes:
            self.pre_conditioners = compute_power_svd(matrix=self.statistics, power=self.exponent_for_pre_conditioner)
            return

        for i, statistic in enumerate(self.statistics):
            self.pre_conditioners[i] = (
                compute_power_svd(matrix=statistic, power=self.exponent_for_pre_conditioner)
                if self.use_svd
                else compute_power_schur_newton(
                    mat_g=statistic, p=self.exponent_for_pre_conditioner, ridge_epsilon=self.matrix_eps
                )
            )

    @staticmethod
    def precondition_block(
        partitioned_grad: torch.Tensor,
        should_preconditioned_dims: List[bool],
        pre_conditioners_for_grad: Union[List[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        r"""Perform a preconditioning operation on a single gradient block.

        Loop invariant: the dimension to be preconditioned is first
        We keep all axes in the same cyclic order they were originally.
        """
        rank: int = len(partitioned_grad.shape)
        roll: Tuple[int, ...] = (*range(1, rank), 0)

        i: int = 0
        for should_precondition_dim in should_preconditioned_dims:
            if not should_precondition_dim:
                partitioned_grad = torch.permute(partitioned_grad, roll)
                continue

            partitioned_grad = torch.tensordot(partitioned_grad, pre_conditioners_for_grad[i], dims=[[0], [0]])
            i += 1

        return partitioned_grad

    def preconditioned_grad(self, grad: torch.Tensor) -> torch.Tensor:
        r"""Precondition the gradient.

        :param grad: torch.Tensor. a gradient tensor to precondition.
        """
        if len(self.pre_conditioners) == 0:
            return grad

        reshaped_grad = torch.reshape(grad, self.transformed_shape)
        partitioned_grads = self.partitioner.partition(reshaped_grad)

        pre_cond_partitioned_grads: List[torch.Tensor] = [
            self.precondition_block(
                partitioned_grad,
                self.should_precondition_dims,
                self.pre_conditioners[i * self.rank:(i + 1) * self.rank]  # fmt: skip
            )
            for i, partitioned_grad in enumerate(partitioned_grads)
        ]

        merged_grad = self.partitioner.merge_partitions(pre_cond_partitioned_grads)

        return merged_grad.reshape(self.original_shape)


def build_graft(p: torch.Tensor, graft_type: int, diagonal_eps: float = 1e-10):
    r"""Build Graft by given graft_type."""
    if graft_type == LayerWiseGrafting.ADAGRAD:
        return AdaGradGraft(p, diagonal_eps)
    if graft_type == LayerWiseGrafting.RMSPROP:
        return RMSPropGraft(p, diagonal_eps)
    if graft_type == LayerWiseGrafting.SGD:
        return SGDGraft(p)
    if graft_type == LayerWiseGrafting.SQRTN:
        return SQRTNGraft(p)
    return Graft(p)


@torch.no_grad()
def power_iteration(mat_g: torch.Tensor, num_iters: int = 100) -> torch.Tensor:
    r"""Compute the maximum eigenvalue of matrix, for scaling.

        Mostly, power_iteration method is faster than torch.einval in case of the symmetric PSD matrix.
        Also, I removed the validation, error of singular value every iteration, so that boosting the speed.

    :param mat_g: torch.Tensor. the symmetric PSD matrix.
    :param num_iters: int. Number of iterations.
    """
    v = torch.randn(mat_g.shape[0], dtype=mat_g.dtype, device=mat_g.device)
    mat_v = torch.empty_like(v)

    for _ in range(num_iters):
        torch.mv(mat_g, v, out=mat_v)
        v.copy_(mat_v)
        v.div_(torch.linalg.norm(v))

    return (v.t() @ mat_g @ v).clamp_min_(1e-16)


@torch.no_grad()
def compute_power_schur_newton(
    mat_g: torch.Tensor,
    p: int,
    max_iters: int = 100,
    error_tolerance: float = 1e-3,
    ridge_epsilon: float = 1e-6,
    max_error_ratio: float = 1.2,
) -> torch.Tensor:
    r"""Compute G^{-1/p} using a coupled Newton iteration.

        See for example equation 3.2 on page 9 of:
            A Schur-Newton Method for the Matrix p-th Root and its Inverse by Chun-Hua Guo and Nicholas J. Higham
            SIAM Journal on Matrix Analysis and Applications, 2006, Vol. 28, No. 3 : pp. 788-804
            https://pdfs.semanticscholar.org/0abe/7f77433cf5908bfe2b79aa91af881da83858.pdf.

        The best value for z is (1 + p) * (c_max^{1/p} - c_min^{1/p}) / (c_max^{1+1/p} - c_min^{1+1/p})
        where c_max and c_min are the largest and smallest singular values of mat_g.
        The above estimate assumes that c_max > c_min * 2^p can replace above line by the one below,
        but it is less accurate, hence needs more iterations to converge.

        z = (1 + p) / tf.trace(mat_g)
        If we want the method to always converge, use z = 1 / norm(mat_g) or z = 1 / tf.trace(mat_g),
        but these can result in many extra iterations.

    :param mat_g: torch.Tensor. A square positive semi-definite matrix.
    :param p: int. a positive integer.
    :param max_iters: int. Stop iterating after this many rounds.
    :param error_tolerance: float. Threshold for stopping iteration.
    :param ridge_epsilon: float. We add this times I to G, to make is positive definite.
        For scaling, we multiply it by the largest eigenvalue of G.
    :param max_error_ratio: float. Sometimes error increases after an iteration before decreasing and converging.
        1.2 factor is used to bound the maximal allowed increase.
    """
    shape: List[int] = mat_g.shape
    if len(shape) == 1:
        return torch.pow(mat_g + ridge_epsilon, -1.0 / p)

    identity = torch.eye(shape[0], dtype=mat_g.dtype, device=mat_g.device)
    if shape[0] == 1:
        return identity

    mat_g += power_iteration(mat_g) * identity * ridge_epsilon

    z = (1 + p) / (2 * torch.linalg.norm(mat_g))

    mat_root = identity * torch.pow(z, 1.0 / p)

    mat_m = mat_g * z

    alpha: float = -1.0 / p
    alpha_identity = (1.0 - alpha) * identity

    prev_error = torch.max(torch.abs(mat_m - identity))

    for _ in range(max_iters):
        mat_m_i = alpha_identity + alpha * mat_m

        new_mat_root = torch.matmul(mat_root, mat_m_i)
        torch.matmul(torch.linalg.matrix_power(mat_m_i, p), mat_m, out=mat_m)

        error = torch.max(torch.abs(mat_m - identity))

        # NOTE
        # this is the main bottleneck that makes Scalable Shampoo slow.
        # because it is handled on the Python side so values need to be on the CPU
        # while XLA devices (e.g. TPU) doesn't seem to be affected.
        if torch.logical_or(error > prev_error * max_error_ratio, error <= error_tolerance):
            break

        mat_root = new_mat_root
        prev_error = error

    return mat_root


@torch.no_grad()
def compute_power_svd(matrix: torch.Tensor, power: float) -> torch.Tensor:
    r"""Compute G^{-1/p} using SVD.

        Calculate SVD on the GPU. Sometimes, SVD on the CPU is faster than GPU, but based on the several experiments,
        CUDA seems much faster than on CPU.

    :param matrix: torch.Tensor. a square positive semi-definite matrix.
    :param power: float. rank.
    """
    u, s, vh = torch.linalg.svd(matrix, full_matrices=False)
    s.pow_(-1.0 / power)
    return u @ (s.diag() if len(matrix.shape) == 2 else s.diag_embed()) @ vh


def merge_small_dims(shape_to_merge: Union[List[int], torch.Size], max_dim: int) -> List[int]:
    r"""Merge small dimensions.

        If there are some small dimensions, we collapse them
            e.g. [1, 2, 512, 1, 2048, 1, 3, 4] --> [1024, 2048, 12] if max_dim = 1024
            [1, 2, 768, 1, 2048] --> [2, 768, 2048].

    :param shape_to_merge: Union[List[int], torch.Size]. Shape to merge small dimensions.
    :param max_dim: int. Maximal dimension of output shape used in merging.
    """
    merged_shape: List[int] = []

    product: int = 1
    for dim in shape_to_merge:
        product *= dim
        if product > max_dim:
            merged_shape.append(product // dim)
            product = dim

    merged_shape.append(product)

    return merged_shape if len(merged_shape) > 1 else [1]


def zero_power_via_newton_schulz_5(
    g: torch.Tensor, num_steps: int = 10, eps: float = 1e-7, weights: Tuple[int, int, int] = (3.4445, -4.7750, 2.0315)
) -> torch.Tensor:
    r"""Compute the zeroth power / orthogonalization of G.

    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a quintic iteration
    whose coefficients are selected to maximize the slope at zero. For the purpose of minimizing steps, it turns out
    to be empirically effective to keep increasing the slope at zero even beyond the point where the iteration no
    longer converges all the way to one everywhere on the interval. This iteration therefore does not produce UV^T but
    rather something like US'V^T where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt
    model performance at all relative to UV^T, where USV^T = G is the SVD.

    :param g: torch.Tensor. matrix.
    :param num_steps: int. number of iterations.
    :param eps: float. add this times I to G, to make is positive definite. For scaling, we multiply it by the largest
        eigenvalue of G.
    :param weights: Tuple[int, int, int]. weights.
    """
    if len(g.shape) != 2:
        raise ValueError('shape of g must be 2-dimensional')

    x = g.bfloat16()
    x.div_(x.norm().add_(eps))

    if g.size(0) > g.size(1):
        x = x.T

    for _ in range(num_steps):
        a = x @ x.T
        b = weights[1] * a + weights[2] * a @ a
        x = weights[0] * x + b @ x

    if g.size(0) > g.size(1):
        x = x.T

    return x
